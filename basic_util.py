import os, sys, traceback, subprocess, signal, concurrent.futures, asyncio, functools, psutil
from typing import Tuple, List, Dict, Any, Set, Generator, Union, Callable, Optional, TypeVar, Union, cast
import time, threading, gc, attrs
from attrs import field
from concurrent.futures import ThreadPoolExecutor
from time import time as ttime

now_dir = os.getcwd()
ROOT = "GPT_SoVITS/pretrained_models/"

def func_timer(func):
    def wrapper(*args, **kwargs):
        st = ttime()
        result = func(*args, **kwargs)
        print(f'{func.__name__}()耗时{ttime() - st:.4f}秒')
        return result

    return wrapper


def profil(func):
    def wrapper(*args, **kwargs):
        from pyinstrument import Profiler
        p = Profiler()
        p.start()
        r = func(*args, **kwargs)
        p.stop()
        p.print()
        return r

    return wrapper


def perf_check(func):
    # from heartrate import trace, files
    # trace(browser=True, files=files.all)

    @func_timer
    @profil
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    return wrapper


T = TypeVar('T')
R = TypeVar('R')


class TaskExecutionError(Exception):
    """任务执行过程中出现的异常"""
    pass


def get_memory_usage_percent():
    return psutil.virtual_memory().percent

def estimate_safe_workers():
    """根据当前内存使用情况估算安全的工作线程数"""
    mem_usage = get_memory_usage_percent()
    cpu_count = os.cpu_count() or 4  # 保守策略：内存使用率越高，工作线程越少

    if mem_usage > 90:
        return max(1, cpu_count // 8)  # 内存严重不足，最小化线程数
    elif mem_usage > 80:
        return max(1, cpu_count // 4)  # 内存不足，四分之一线程
    elif mem_usage > 70:
        return max(2, cpu_count // 2)  # 内存偏紧，一半线程
    else:
        return max(2, cpu_count - 1)  # 内存充足，接近全部线程


def async_processor(func: Callable[..., T]):
    """
    将单线程函数转换为异步批处理的装饰器

    单项处理函数->异步上下文
    """

    @functools.wraps(func)
    async def process_async(data: List[List[Any]], *args, **kwargs) -> List[T]:
        """
        异步处理一批数据项

        Args:
            data: 要处理的数据列表，每个元素也是一个列表，包含传递给func的参数
            *args, **kwargs: 传递给每个函数调用的其他参数

        Returns:
            处理结果的列表
        """
        results = []
        total = len(data)
        batch_size_initial = 50  # 初始批次大小
        processed = 0

        while processed < total:
            mem_usage = get_memory_usage_percent()
            worker_count = estimate_safe_workers()

            if mem_usage > 85:
                batch_size = max(5, batch_size_initial // 4)  # 内存紧张时大幅减小批次
            elif mem_usage > 75:
                batch_size = max(10, batch_size_initial // 2)  # 内存偏高时减小批次
            else:
                batch_size = batch_size_initial

            end_idx = min(processed + batch_size, total)
            current_batch = data[processed:end_idx]
            print(f"批次 {processed}/{total} ({processed / total:.1%}) | 当前 {processed}~{end_idx} | 进程 {worker_count} | 内存 {mem_usage:.1f}% ")

            batch_results = []
            loop = asyncio.get_event_loop()

            with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) as executor:
                futures = []
                for item_args in current_batch:
                    # 解包参数并添加额外的共享参数
                    future = loop.run_in_executor(executor, functools.partial(func, *item_args, *args, **kwargs))
                    futures.append(future)

                # 等待所有任务完成并收集结果
                completed_futures = await asyncio.gather(*futures, return_exceptions=True)
                for result in completed_futures:
                    if isinstance(result, Exception):
                        print(f"处理中出现错误: {str(result)}")
                    elif result:
                        batch_results.append(result)

            results.extend(batch_results)
            processed = end_idx

            if mem_usage > 90:
                print(f"内存使用过高：{mem_usage:.1f}%，等待系统回收...")
                await asyncio.sleep(2)  # 给系统一些时间回收内存

        return results

    return process_async
