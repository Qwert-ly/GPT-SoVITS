from basic_util import *
import re, platform, glob, shutil, psutil, warnings, torch
from subprocess import Popen
from multiprocessing import cpu_count

# 配置常量
_CONFIG = {
    "version": "v2",
    "paths": {
        "weights": {
            "sovits": ["SoVITS_weights", "SoVITS_weights_v2", "SoVITS_weights_v3"],
            "gpt": ["GPT_weights", "GPT_weights_v2", "GPT_weights_v3"]
        },
        "pretrained": {
            "sovits": [
                ROOT+"s2G488k.pth",
                ROOT+"gsv-v2final-pretrained/s2G2333k.pth",
                ROOT+"s2Gv3.pth"
            ],
            "gpt": [
                ROOT+"s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt",
                ROOT+"gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt",
                ROOT+"s1v3.ckpt"
            ]
        },
        "runtime_dirs": ["TEMP"]
    },
    "gpu_keywords": {"10", "16", "20", "30", "40", "A2", "A3", "A4", "P4", "A50",
                     "500", "A60", "70", "80", "90", "M4", "T4", "TITAN", "L4",
                     "4060", "H", "600", "506", "507", "508", "509"}
}

SoVITS_weight_root = ["SoVITS_weights", "SoVITS_weights_v2", "SoVITS_weights_v3"]
GPT_weight_root = ["GPT_weights", "GPT_weights_v2", "GPT_weights_v3"]
system = platform.system()
pretrained_sovits_name = [
    ROOT+"s2G488k.pth",
    ROOT+"gsv-v2final-pretrained/s2G2333k.pth",
    ROOT+"s2Gv3.pth"
]
pretrained_gpt_name = [
    ROOT+"s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt",
    ROOT+"gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt",
    ROOT+"s1v3.ckpt"
]


# 初始化环境配置
def setup_environment() -> None:
    """初始化环境变量和系统路径"""
    os.environ.update({
        "TORCH_DISTRIBUTED_DEBUG": "INFO",
        "no_proxy": "localhost, 127.0.0.1, ::1",
        "all_proxy": ""
    })

    # 设置随机种子
    torch.manual_seed(233)

    # 添加当前路径到系统路径
    sys.path.insert(0, os.getcwd())
    warnings.filterwarnings("ignore")


# GPU信息检测
def detect_gpu_info() -> Tuple[bool, List[str], List[int], Set[int]]:
    gpu_info = []
    mem_info = []
    gpu_available = False
    set_gpu_numbers = set()

    # 检测NVIDIA GPU
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            name = torch.cuda.get_device_name(i)
            if any(kw in name.upper() for kw in _CONFIG["gpu_keywords"]):
                gpu_available = True
                set_gpu_numbers.add(i)
                gpu_info.append(f"{i}\t{name}")
                mem_info.append(int(torch.cuda.get_device_properties(i).total_memory / (1024 ** 3) + 0.4))

    # 检测Apple MPS
    if not gpu_available and torch.backends.mps.is_available():
        gpu_available = True
        gpu_info.append("0\tApple GPU")
        mem_info.append(psutil.virtual_memory().total / (1024 ** 3))

    return gpu_available, gpu_info, mem_info, set_gpu_numbers


# 文件系统管理模块
def manage_temp_directory():
    """管理临时目录"""
    temp_dir = os.path.join(os.getcwd(), "TEMP")
    os.makedirs(temp_dir, exist_ok=True)
    os.environ["TEMP"] = temp_dir

    # 清理临时文件
    for entry in os.listdir(temp_dir):
        if entry == "jieba.cache":
            continue
        path = os.path.join(temp_dir, entry)
        try:
            os.remove(path) if os.path.isfile(path) else shutil.rmtree(path)
        except Exception as e:
            print(f"清理失败: {str(e)}")
    return temp_dir


# 模型权重管理模块
class WeightManager:
    @staticmethod
    def get_weights(weight_type: str) -> List[str]:
        """获取指定类型的权重文件"""
        valid_files = []
        for root in _CONFIG["paths"]["weights"][weight_type]:
            valid_files.extend(glob.glob(os.path.join(root, "*.pth" if weight_type == "sovits" else "*.ckpt")))
        return sorted(
            [f for f in _CONFIG["paths"]["pretrained"][weight_type] if os.path.exists(f)] + valid_files,
            key=lambda x: [int(p) if p.isdigit() else p for p in re.split(r'(\d+)', x)]
        )

    @staticmethod
    def create_weight_directories() -> None:
        """创建权重目录"""
        for d in _CONFIG["paths"]["weights"]["sovits"] + _CONFIG["paths"]["weights"]["gpt"]:
            os.makedirs(d, exist_ok=True)


# 工具函数模块
def custom_sort_key(s: str) -> list:
    """自定义排序键函数"""
    return [int(p) if p.isdigit() else p.lower() for p in re.split(r'(\d+)', s)]


def sta(i=False):
    if i:
        return {"__type__": "update", "visible": True}, {"__type__": "update", "visible": False}
    return {'__type__': 'update', 'visible': False}, {'__type__': 'update', 'visible': True}


def up(i=False):
    if i:
        return {"__type__": "update"}, {"__type__": "update"}, {"__type__": "update"}
    return {"__type__": "update"}, {"__type__": "update"}


def sync(text):
    return {"__type__": "update", "value": text}

# 辅助函数: 用于返回错误响应
def yield_error_response(error_message):
    return error_message, *sta(i=1), *up(i=1)

def yield_response(message):
    return message, *sta(), *up(i=1)


def cmd_(cmd, wait=False):
    print(cmd)
    p = Popen(cmd, shell=True)
    if wait:
        p.wait()
    return p


def change_choices() -> Tuple[Dict[str, Any], Dict[str, Any]]:
    return (
        {"choices": WeightManager.get_weights("sovits"), "__type__": "update"},
        {"choices": WeightManager.get_weights("gpt"), "__type__": "update"}
    )


# 初始化流程
setup_environment()
tmp = manage_temp_directory()
WeightManager.create_weight_directories()

# 运行时变量 (保留原始变量名)
version = _CONFIG["version"]
n_cpu = cpu_count()
if_gpu_ok, gpu_infos, mem, set_gpu_numbers = detect_gpu_info()
default_gpu_numbers = str(sorted(set_gpu_numbers))
fix_gpu_number = lambda i: default_gpu_numbers if str(i).isdigit() and int(i) not in set_gpu_numbers else i
fix_gpu_numbers = lambda i: ",".join(map(str, (fix_gpu_number(i) for i in i.split(",")))) if "," in i else i

# 生成模型路径列表
version_index = int(version[-1]) - 1
sovits_path = pretrained_sovits_name[version_index]
pretrained_model_list = (
    sovits_path,
    sovits_path.replace("s2G", "s2D"),
    pretrained_gpt_name[version_index],
    ROOT+"chinese-roberta-wwm-ext-large",
    ROOT+"chinese-hubert-base"
)
SoVITS_names, GPT_names = WeightManager.get_weights("sovits"), WeightManager.get_weights("gpt")
p_label = None
p_uvr5 = None
p_asr = None
p_denoise = None
p_tts_inference = None

