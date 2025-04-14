import os
import re
import pickle
import functools
from typing import List, Tuple
from pypinyin import lazy_pinyin, Style
from pypinyin.contrib.tone_convert import to_finals_tone3, to_initials
import jieba_fast, logging

jieba_fast.setLogLevel(logging.CRITICAL)
import jieba_fast.posseg as psg
from text.symbols import punctuation
from text.tone_sandhi import ToneSandhi, pre_merge_for_modify
from text.zh_normalization.text_normlization import TextNormalizer
from numba import njit


# Mapping for text replacements
rep_map = {
    "：": ",", "；": ",", "，": ",", "。": ".", "！": "!",
    "？": "?", "\n": ".", "·": ",", "、": ",", "...": "…",
    "$": ".", "/": ",", "—": "-", "~": "…", "～": "…",
}

# Cache commonly used operations
current_file_path = os.path.dirname(__file__)

# Read the pinyin-to-symbol map once and store it
with open(os.path.join(current_file_path, "opencpop-strict.txt")) as f:
    pinyin_to_symbol_map = {line.split("\t")[0]: line.strip().split("\t")[1] for line in f}

# Precompile frequently used regex patterns
PUNCTUATION_PATTERN = re.compile("|".join(re.escape(p) for p in rep_map.keys()))
NON_CHINESE_PUNCT_PATTERN = re.compile(r"[^\u4e00-\u9fa5" + "".join(punctuation) + r"]+")
NON_CHINESE_EN_PUNCT_PATTERN = re.compile(r"[^\u4e00-\u9fa5A-Za-z" + "".join(punctuation) + r"]+")
CONSECUTIVE_PUNCT_PATTERN = re.compile(
    f'([{"".join(re.escape(p) for p in punctuation)}])([{"".join(re.escape(p) for p in punctuation)}])+')
ENGLISH_PATTERN = re.compile("[a-zA-Z]+")
SPLIT_PATTERN = re.compile(r"(?<=[{0}])\s*".format("".join(punctuation)))

# Initialize reusable objects
tone_modifier = ToneSandhi()
text_normalizer = TextNormalizer()

# Sets for erhua processing - use frozenset for faster lookup
must_erhua = frozenset({
    "小院儿", "胡同儿", "范儿", "老汉儿", "撒欢儿", "寻老礼儿", "妥妥儿", "媳妇儿"
})
not_erhua = frozenset({
    "虐儿", "为儿", "护儿", "瞒儿", "救儿", "替儿", "有儿", "一儿", "我儿", "俺儿", "妻儿",
    "拐儿", "聋儿", "乞儿", "患儿", "幼儿", "孤儿", "婴儿", "婴幼儿", "连体儿", "脑瘫儿",
    "流浪儿", "体弱儿", "混血儿", "蜜雪儿", "舫儿", "祖儿", "美儿", "应采儿", "可儿", "侄儿",
    "孙儿", "侄孙儿", "女儿", "男儿", "红孩儿", "花儿", "虫儿", "马儿", "鸟儿", "猪儿", "猫儿",
    "狗儿", "少儿"
})

# Configuration for G2PW
is_g2pw = True
if is_g2pw:
    print("当前使用g2pw进行拼音推理")
    from text.g2pw import G2PWPinyin

    PP_DICT_PATH = os.path.join(current_file_path, r"g2pw\polyphonic.rep")
    PP_FIX_DICT_PATH = os.path.join(current_file_path, r"g2pw\polyphonic-fix.rep")
    CACHE_PATH = os.path.join(current_file_path, r"g2pw\polyphonic.pickle")


    def cache_dict(polyphonic_dict, file_path):
        with open(file_path, "wb") as pickle_file:
            pickle.dump(polyphonic_dict, pickle_file)


    def read_dict():
        polyphonic_dict = {}
        with open(PP_DICT_PATH, encoding="utf-8") as f:
            line = f.readline()
            while line:
                key, value_str = line.split(':')
                value = eval(value_str.strip())
                polyphonic_dict[key.strip()] = value
                line = f.readline()
        with open(PP_FIX_DICT_PATH, encoding="utf-8") as f:
            line = f.readline()
            while line:
                key, value_str = line.split(':')
                value = eval(value_str.strip())
                polyphonic_dict[key.strip()] = value
                line = f.readline()
        return polyphonic_dict


    def get_dict():
        if os.path.exists(CACHE_PATH):
            with open(CACHE_PATH, "rb") as pickle_file:
                polyphonic_dict = pickle.load(pickle_file)
        else:
            polyphonic_dict = read_dict()
            cache_dict(polyphonic_dict, CACHE_PATH)

        return polyphonic_dict

    pp_dict = get_dict()

    def correct_pronunciation(word, word_pinyins):
        new_pinyins = pp_dict.get(word, "")
        if new_pinyins == "":
            for idx, w in enumerate(word):
                w_pinyin = pp_dict.get(w, "")
                if w_pinyin != "":
                    word_pinyins[idx] = w_pinyin[0]
            return word_pinyins
        else:
            return new_pinyins

    parent_directory = os.path.dirname(current_file_path)
    g2pw = G2PWPinyin(model_dir="GPT_SoVITS/text/G2PWModel", model_source=os.environ.get("bert_path", "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large"),
                      v_to_u=False, neutral_tone_with_five=True)


@functools.lru_cache(maxsize=256)
def replace_punctuation(text):
    """Replace Chinese punctuation with English punctuation (cached)"""
    text = text.replace("嗯", "恩").replace("呣", "母")
    replaced_text = PUNCTUATION_PATTERN.sub(lambda x: rep_map[x.group()], text)
    return NON_CHINESE_PUNCT_PATTERN.sub("", replaced_text)


@functools.lru_cache(maxsize=256)
def replace_punctuation_with_en(text):
    """Replace punctuation while keeping English letters (cached)"""
    text = text.replace("嗯", "恩").replace("呣", "母")
    replaced_text = PUNCTUATION_PATTERN.sub(lambda x: rep_map[x.group()], text)
    return NON_CHINESE_EN_PUNCT_PATTERN.sub("", replaced_text)


@functools.lru_cache(maxsize=256)
def replace_consecutive_punctuation(text):
    """Replace consecutive punctuation with a single one (cached)"""
    return CONSECUTIVE_PUNCT_PATTERN.sub(r'\1', text)


def g2p(text):
    """Convert text to phones"""
    sentences = [i for i in SPLIT_PATTERN.split(text) if i.strip() != ""]
    return _g2p(sentences)


@functools.lru_cache(maxsize=128)
def _get_initials_finals(word):
    """Get initials and finals for a word (cached)"""
    orig_initials = lazy_pinyin(word, neutral_tone_with_five=True, style=Style.INITIALS)
    orig_finals = lazy_pinyin(word, neutral_tone_with_five=True, style=Style.FINALS_TONE3)

    return tuple(orig_initials), tuple(orig_finals)


def _merge_erhua(initials: List[str], finals: List[str], word: str, pos: str) -> Tuple[List[str], List[str]]:
    """Process Erhua in Chinese pronunciation"""
    # Fix er1
    if finals[-1] == 'er1' and word[-1] == "儿" and len(finals) > 0:
        finals[-1] = 'er2'

    # Skip processing for certain words and parts of speech
    if word not in must_erhua and (word in not_erhua or pos in {"a", "j", "nr"}):
        return initials, finals

    # Handle special cases
    if len(finals) != len(word):
        return initials, finals

    # Process erhua
    new_initials = []
    new_finals = []
    for i, phn in enumerate(finals):
        if (i == len(finals) - 1 and
                word[i] == "儿" and
                phn in {"er2", "er5"} and
                word[-2:] not in not_erhua and
                new_finals):
            phn = "er" + new_finals[-1][-1]

        new_initials.append(initials[i])
        new_finals.append(phn)

    return new_initials, new_finals


# Performance-critical function wrapped with numba for speed
@njit
def process_phones(c_list, v_list, word2ph_list):
    """Process phones with Numba JIT compilation for better performance"""
    phones_list = []

    for i in range(len(c_list)):
        c = c_list[i]
        v = v_list[i]

        if c == v:  # Punctuation case
            phones_list.append(c)
            word2ph_list.append(1)
        else:
            word2ph_list.append(2)  # Most cases have 2 phones
            if v[-1] in "12345":
                phones_list.append(c)
                phones_list.append(v)
            else:
                phones_list.append(c)
                phones_list.append(v)

    return phones_list


def _g2p(segments):
    """Convert segments to phones and word-to-phone mapping"""
    phones_list = []
    word2ph = []

    for seg in segments:
        # Remove English words
        seg = ENGLISH_PATTERN.sub("", seg)

        # Get word segmentation
        seg_cut = psg.lcut(seg)
        seg_cut = pre_merge_for_modify(seg_cut)

        initials = []
        finals = []

        if not is_g2pw:
            for word, pos in seg_cut:
                if pos == "eng":
                    continue

                sub_initials, sub_finals = _get_initials_finals(word)
                sub_finals = tone_modifier.modified_tone(word, pos, list(sub_finals))

                # Process erhua
                sub_initials, sub_finals = _merge_erhua(list(sub_initials), sub_finals, word, pos)

                initials.append(sub_initials)
                finals.append(sub_finals)

            initials = sum(initials, [])
            finals = sum(finals, [])
            print("pypinyin结果", initials, finals)
        else:
            pinyins = g2pw.lazy_pinyin(seg, neutral_tone_with_five=True, style=Style.TONE3)

            pre_word_length = 0
            for word, pos in seg_cut:
                sub_initials = []
                sub_finals = []
                now_word_length = pre_word_length + len(word)

                if pos == 'eng':
                    pre_word_length = now_word_length
                    continue

                # Get pinyins for the current word
                word_pinyins = pinyins[pre_word_length:now_word_length]

                # Correct pronunciation (cached)
                word_pinyins = correct_pronunciation(word, tuple(word_pinyins))

                # Process each pinyin
                for pinyin in word_pinyins:
                    if pinyin[0].isalpha():
                        sub_initials.append(to_initials(pinyin))
                        sub_finals.append(to_finals_tone3(pinyin, neutral_tone_with_five=True))
                    else:
                        sub_initials.append(pinyin)
                        sub_finals.append(pinyin)

                pre_word_length = now_word_length

                # Apply tone modification
                sub_finals = tone_modifier.modified_tone(word, pos, list(sub_finals))

                # Process erhua
                sub_initials, sub_finals = _merge_erhua(sub_initials, sub_finals, word, pos)

                initials.append(sub_initials)
                finals.append(sub_finals)

            initials = sum(initials, [])
            finals = sum(finals, [])

        # Create a lookup table for pinyin substitutions
        c_v_substitution = {}
        single_substitution = {
            "v": "yu", "e": "e", "i": "y", "u": "w",
        }
        pinyin_rep_map = {
            "ing": "ying", "i": "yi", "in": "yin", "u": "wu",
        }
        v_rep_map = {
            "uei": "ui", "iou": "iu", "uen": "un",
        }

        # Process phones
        for i, (c, v) in enumerate(zip(initials, finals)):
            if c == v:  # Punctuation
                phones_list.append(c)
                word2ph.append(1)
                continue

            v_without_tone = v[:-1]
            tone = v[-1]

            # Skip processing if not a valid tone
            if tone not in "12345":
                phones_list.extend([c, v])
                word2ph.append(2)
                continue

            pinyin = c + v_without_tone

            if c:  # Multi-syllable
                if v_without_tone in v_rep_map:
                    pinyin = c + v_rep_map[v_without_tone]
            else:  # Single syllable
                if pinyin in pinyin_rep_map:
                    pinyin = pinyin_rep_map[pinyin]
                elif pinyin[0] in single_substitution:
                    pinyin = single_substitution[pinyin[0]] + pinyin[1:]

            # Use the mapping to get the correct symbols
            symbol = pinyin_to_symbol_map.get(pinyin)
            if symbol:
                new_c, new_v = symbol.split(" ")
                new_v = new_v + tone
                phones_list.extend([new_c, new_v])
                word2ph.append(2)
            else:
                # Fallback for unexpected cases
                phones_list.extend([c, v])
                word2ph.append(2)

    return phones_list, word2ph


@functools.lru_cache(maxsize=128)
def text_normalize(text):
    """Normalize Chinese text (cached)"""
    sentences = text_normalizer.normalize(text)
    dest_text = ""
    for sentence in sentences:
        dest_text += replace_punctuation(sentence)

    # Remove consecutive punctuation
    dest_text = replace_consecutive_punctuation(dest_text)
    return dest_text


@functools.lru_cache(maxsize=128)
def mix_text_normalize(text):
    """Normalize mixed Chinese and English text (cached)"""
    sentences = text_normalizer.normalize(text)
    dest_text = ""
    for sentence in sentences:
        dest_text += replace_punctuation_with_en(sentence)

    # Remove consecutive punctuation
    dest_text = replace_consecutive_punctuation(dest_text)
    return dest_text


if __name__ == "__main__":
    text = "啊——但是《原神》是由,米哈\游自主，研发的一款全.新开放世界.冒险游戏"
    text = "呣呣呣～就是…大人的鼹鼠党吧？"
    text = "你好"
    text = text_normalize(text)
    print(g2p(text))
