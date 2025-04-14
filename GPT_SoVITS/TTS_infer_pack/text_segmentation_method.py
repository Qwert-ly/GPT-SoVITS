from typing import Callable
from seg_utils import *


def get_method(name: str) -> Callable:
    method = METHODS.get(name, None)
    if method is None:
        raise ValueError(f"Method {name} not found")
    return method


def get_method_names() -> list:
    return list(METHODS.keys())


def split_big_text(text, max_len=510):
    # 定义全角和半角标点符号
    punctuation = "".join(splits)

    # 切割文本
    segments = re.split('([' + punctuation + '])', text)

    # 初始化结果列表和当前片段
    result = []
    current_segment = ''

    for segment in segments:
        # 如果当前片段加上新的片段长度超过max_len，就将当前片段加入结果列表，并重置当前片段
        if len(current_segment + segment) > max_len:
            result.append(current_segment)
            current_segment = segment
        else:
            current_segment += segment

    # 将最后一个片段加入结果列表
    if current_segment:
        result.append(current_segment)

    return result


if __name__ == '__main__':
    method = get_method("cut5")
    print(method("你好，我是小明。你好，我是小红。你好，我是小刚。你好，我是小张。"))
