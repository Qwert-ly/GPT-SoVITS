from TTS_infer_pack.text_segmentation_method import split_big_text, get_method
from tts_utils import *
from inf_utils import get_bert_feature, filter_text

punctuation = {'!', '?', '…', ',', '.', '-'}


def get_first(text: str) -> str:
    pattern = "[" + "".join(re.escape(sep) for sep in splits) + "]"
    return re.split(pattern, text)[0].strip()


def clean_text_inf(text: str, language: str, version: str = "v2"):
    language = language.replace("all_", "")
    phones, word2ph, norm_text = clean_text(text, language, version)
    return cleaned_text_to_sequence(phones, version), word2ph, norm_text


def replace_consecutive_punctuation(text):
    punctuations = ''.join(re.escape(p) for p in punctuation)
    pattern = f'([{punctuations}])([{punctuations}])+'
    return re.sub(pattern, r'\1', text)


class TextPreprocessor:
    def __init__(self, bert_model: AutoModelForMaskedLM,
                 tokenizer: AutoTokenizer, device: torch.device):
        self.bert_model = bert_model
        self.tokenizer = tokenizer
        self.device = device
        self.bert_lock = threading.RLock()

    def pre_seg_text(self, text: str, lang: str, text_split_method: str):
        text = text.strip("\n")
        if len(text) == 0:
            return []
        if text[0] not in splits and len(get_first(text)) < 4:
            text = "。" + text if lang != "en" else "." + text
        print(i18n("实际输入的目标文本:"))
        print(text)

        seg_method = get_method(text_split_method)
        text = seg_method(text)

        while "\n\n" in text:
            text = text.replace("\n\n", "\n")

        _texts = filter_text(text.split("\n"))
        _texts = merge_short_text_in_array(_texts, 5)
        texts = []

        for text in _texts:
            # 解决输入目标文本的空行导致报错的问题
            if len(text.strip()) == 0:
                continue
            if not re.sub("\W+", "", text):
                # 检测一下，如果是纯符号，就跳过。
                continue
            if text[-1] not in splits: text += "。" if lang != "en" else "."

            # 解决句子过长导致Bert报错的问题
            if len(text) > 510:
                texts.extend(split_big_text(text))
            else:
                texts.append(text)

        print(i18n("实际输入的目标文本(切句后):"))
        print(texts)
        return texts

    def preprocess(self, text: str, lang: str, text_split_method: str, version: str = "v2") -> List[Dict]:
        print(f'############ {i18n("切分文本")} ############')
        text = replace_consecutive_punctuation(text)
        texts = self.pre_seg_text(text, lang, text_split_method)
        result = []
        print(f'############ {i18n("提取文本Bert特征")} ############')
        for text in tqdm(texts):
            phones, bert_features, norm_text = self.segment_and_extract_feature_for_text(text, lang, version)
            if phones is None or norm_text == "":
                continue
            result.append({
                "phones": phones,
                "bert_features": bert_features,
                "norm_text": norm_text,
            })
        return result

    def segment_and_extract_feature_for_text(self, text: str, language: str, version: str = "v1") -> Tuple[
        list, torch.Tensor, str]:
        return self.get_phones_and_bert(text, language, version)

    def get_phones_and_bert(self, text: str, language: str, version: str, final: bool = False):
        with self.bert_lock:
            if language in {"en", "all_zh", "all_ja", "all_ko", "all_yue"}:
                # language = language.replace("all_","")
                formattext = text
                while "  " in formattext:
                    formattext = formattext.replace("  ", " ")
                if language == "all_zh":
                    if re.search(r'[A-Za-z]', formattext):
                        formattext = re.sub(r'[a-z]', lambda x: x.group(0).upper(), formattext)
                        formattext = chinese.mix_text_normalize(formattext)
                        return self.get_phones_and_bert(formattext, "zh", version)

                    else:
                        phones, word2ph, norm_text = clean_text_inf(formattext, language, version)
                        bert = get_bert_feature(norm_text, word2ph).to(self.device)

                elif language == "all_yue" and re.search(r'[A-Za-z]', formattext):
                    formattext = re.sub(r'[a-z]', lambda x: x.group(0).upper(), formattext)
                    formattext = chinese.mix_text_normalize(formattext)
                    return self.get_phones_and_bert(formattext, "yue", version)

                else:
                    phones, word2ph, norm_text = clean_text_inf(formattext, language, version)
                    bert = torch.zeros(
                        (1024, len(phones)),
                        dtype=torch.float32,
                    ).to(self.device)

            elif language in {"zh", "ja", "ko", "yue", "auto", "auto_yue"}:
                textlist = []
                langlist = []
                if language == "auto":
                    for tmp in LangSegmenter.getTexts(text):
                        langlist.append(tmp["lang"])
                        textlist.append(tmp["text"])

                elif language == "auto_yue":
                    for tmp in LangSegmenter.getTexts(text):
                        if tmp["lang"] == "zh":
                            tmp["lang"] = "yue"
                        langlist.append(tmp["lang"])
                        textlist.append(tmp["text"])

                else:
                    for tmp in LangSegmenter.getTexts(text):
                        if tmp["lang"] == "en":
                            langlist.append(tmp["lang"])

                        else:
                            # 因无法区别中日韩文汉字,以用户输入为准
                            langlist.append(language)
                        textlist.append(tmp["text"])

                # print(textlist)
                # print(langlist)
                phones_list = []
                bert_list = []
                norm_text_list = []
                for i in range(len(textlist)):
                    lang = langlist[i]
                    phones, word2ph, norm_text = clean_text_inf(textlist[i], lang, version)
                    bert = self.get_bert_inf(phones, word2ph, norm_text, lang)
                    phones_list.append(phones)
                    norm_text_list.append(norm_text)
                    bert_list.append(bert)

                bert = torch.cat(bert_list, dim=1)
                phones = sum(phones_list, [])
                norm_text = ''.join(norm_text_list)
            if not final and len(phones) < 6:
                return self.get_phones_and_bert("." + text, language, version, final=True)
            return phones, bert, norm_text

    def get_bert_inf(self, phones: list, word2ph: list, norm_text: str, language: str):
        language = language.replace("all_", "")
        if language == "zh":
            feature = get_bert_feature(norm_text, word2ph, self.tokenizer, self.bert_model).to(self.device)
        else:
            feature = torch.zeros((1024, len(phones)), dtype=torch.float32).to(self.device)

        return feature
