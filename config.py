import os, sys, torch.cuda


def get_env_bool(key: str, default: bool) -> bool:
    """从环境变量获取布尔值"""
    return os.environ.get(key, str(default)).lower() == 'true'


# 模型路径配置
sovits_path = ""
gpt_path = ""
is_half = get_env_bool("is_half", True)
is_share = get_env_bool("is_share", False)

# 预训练模型路径
GPT_ROOT = "GPT_SoVITS/pretrained_models/"
cnhubert_path = GPT_ROOT+"chinese-hubert-base"
bert_path = GPT_ROOT+"chinese-roberta-wwm-ext-large"
pretrained_sovits_path = GPT_ROOT+"s2G488k.pth"
pretrained_gpt_path = GPT_ROOT+"s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt"

# 系统配置
exp_root = "logs"
python_exec = sys.executable or "python"
infer_device = "cuda" if torch.cuda.is_available() else "cpu"

# 端口配置
webui_port_main = 9874
webui_port_uvr5 = 9873
webui_port_infer_tts = 9872
webui_port_subfix = 9871
api_port = 9880

# 硬件适配逻辑
if infer_device == "cuda":
    upper_gpu = torch.cuda.get_device_name(0).upper()
    # 需要关闭半精度的显卡型号匹配
    blacklist_terms = {"P40", "P10", "1060", "1070", "1080", "16"}
    if ("16" in upper_gpu and "V100" not in upper_gpu) or any(term in upper_gpu for term in blacklist_terms):
        is_half = False
elif infer_device == "cpu":
    is_half = False


class Config:
    __slots__ = (
        'sovits_path', 'gpt_path', 'is_half', 'cnhubert_path', 'bert_path',
        'pretrained_sovits_path', 'pretrained_gpt_path', 'exp_root',
        'python_exec', 'infer_device', 'webui_port_main', 'webui_port_uvr5',
        'webui_port_infer_tts', 'webui_port_subfix', 'api_port'
    )

    def __init__(self):
        self.sovits_path = sovits_path
        self.gpt_path = gpt_path
        self.is_half = is_half
        self.cnhubert_path = cnhubert_path
        self.bert_path = bert_path
        self.pretrained_sovits_path = pretrained_sovits_path
        self.pretrained_gpt_path = pretrained_gpt_path
        self.exp_root = exp_root
        self.python_exec = python_exec
        self.infer_device = infer_device
        self.webui_port_main = webui_port_main
        self.webui_port_uvr5 = webui_port_uvr5
        self.webui_port_infer_tts = webui_port_infer_tts
        self.webui_port_subfix = webui_port_subfix
        self.api_port = api_port
