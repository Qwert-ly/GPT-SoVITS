# -*- coding: utf-8 -*-

import torch, shutil
import os.path
from text.cleaner import clean_text
from transformers import AutoModelForMaskedLM, AutoTokenizer
from tools.my_utils import clean_path
from basic_util import *
from inf_utils import get_bert_feature

inp_text = os.environ.get("inp_text")
inp_wav_dir = os.environ.get("inp_wav_dir")
exp_name = os.environ.get("exp_name")
i_part = os.environ.get("i_part")
all_parts = os.environ.get("all_parts")
if "_CUDA_VISIBLE_DEVICES" in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["_CUDA_VISIBLE_DEVICES"]
opt_dir = os.environ.get("opt_dir")
bert_pretrained_dir = os.environ.get("bert_pretrained_dir")
is_half = eval(os.environ.get("is_half", "True")) and torch.cuda.is_available()
version = os.environ.get('version', None)
thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=estimate_safe_workers())

import tempfile, uuid


async def my_save(fea, path):  #####fix issue: torch.save doesn't support chinese path
    loop = asyncio.get_event_loop()

    dir_path = os.path.dirname(path)
    if dir_path and not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)

    temp_dir = tempfile.gettempdir()
    tmp_path = os.path.join(temp_dir, f"{int(ttime())}_{uuid.uuid4().hex[:8]}.pth")

    try:
        await loop.run_in_executor(
            thread_pool,
            functools.partial(torch.save, fea, tmp_path)
        )

        await loop.run_in_executor(
            thread_pool,
            functools.partial(shutil.move, tmp_path, path)
        )
    except Exception as e:
        print(f"保存中出错: {str(e)}")
        if os.path.exists(tmp_path):
            await loop.run_in_executor(
                thread_pool,
                functools.partial(os.remove, tmp_path)
            )
        raise


txt_path = f"{opt_dir}/2-name2text-{i_part}.txt"
if os.path.exists(txt_path):
    raise ()
bert_dir = f"{opt_dir}/3-bert"
os.makedirs(opt_dir, exist_ok=True)
os.makedirs(bert_dir, exist_ok=True)
device = "cuda:0" if torch.cuda.is_available() else "cpu"
if not os.path.exists(bert_pretrained_dir):
    raise FileNotFoundError(bert_pretrained_dir)
tokenizer = AutoTokenizer.from_pretrained(bert_pretrained_dir)
bert_model = AutoModelForMaskedLM.from_pretrained(bert_pretrained_dir)
if is_half:
    bert_model = bert_model.half().to(device)
else:
    bert_model = bert_model.to(device)


@async_processor
def process_single_item(name: str, text: str, lan: str, bert_dir: str, version: str) -> Tuple[str, str, str, str]:
    """处理单个数据项"""
    try:
        name = clean_path(name)
        name = os.path.basename(name)
        print(name)
        phones, word2ph, norm_text = clean_text(text.replace("%", "-").replace("￥", ","), lan, version)
        path_bert = f"{bert_dir}/{name}.pt"
        if not os.path.exists(path_bert) and lan == "zh":
            bert_feature = get_bert_feature(norm_text, word2ph, tokenizer, bert_model)
            assert bert_feature.shape[-1] == len(phones)
            asyncio.run(my_save(bert_feature, path_bert))
        phones_str = " ".join(phones)
        return name, phones_str, word2ph, norm_text
    except Exception as e:
        print(f"处理{name}：{text}发生错误")
        print(traceback.format_exc())
        return None


todo = []
res = []
with open(inp_text, "r", encoding="utf8") as f:
    lines = f.read().strip("\n").split("\n")

language_v1_to_language_v2 = {
    "ZH": "zh",
    "zh": "zh",
    "JP": "ja",
    "jp": "ja",
    "JA": "ja",
    "ja": "ja",
    "EN": "en",
    "en": "en",
    "En": "en",
    "KO": "ko",
    "Ko": "ko",
    "ko": "ko",
    "yue": "yue",
    "YUE": "yue",
    "Yue": "yue",
}
for line in lines[int(i_part)::int(all_parts)]:
    try:
        wav_name, spk_name, language, text = line.split("|")

        if language in language_v1_to_language_v2.keys():
            todo.append(
                [wav_name, text, language_v1_to_language_v2.get(language, language)]
            )
        else:
            print(f"\033[33m[Waring] 训练不支持{wav_name}的{language = }。\033[0m")
    except:
        print(line, traceback.format_exc())


async def main():
    res = await process_single_item(todo, bert_dir, version)
    opt = []
    for name, phones, word2ph, norm_text in res:
        opt.append(f"{name}\t{phones}\t{word2ph}\t{norm_text}")
    with open(txt_path, "w", encoding="utf8") as f:
        f.write("\n".join(opt) + "\n")


if __name__ == "__main__":
    asyncio.run(main())
