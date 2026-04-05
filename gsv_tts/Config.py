import torch

def get_cuda_device_info(idx: int):
    """获取 CUDA 设备信息"""
    if not torch.cuda.is_available() or idx >= torch.cuda.device_count():
        return None

    try:
        props = torch.cuda.get_device_properties(idx)
    except Exception:
        return None

    name = props.name
    major, minor = props.major, props.minor
    sm_version = major + minor / 10.0
    mem_gb = props.total_memory / (1024**3)

    if sm_version < 5.3:
        return None

    device = torch.device(f"cuda:{idx}")

    # 针对旧架构或特殊系列强制使用 float32
    is_16_series = (major == 7 and minor == 5) and ("16" in name)
    if sm_version == 6.1 or is_16_series:
        return device, torch.float32, sm_version, mem_gb

    # 针对 Ampere (sm 8.0) 及以上架构，优先使用 bfloat16
    if sm_version >= 8.0:
        return device, torch.bfloat16, sm_version, mem_gb

    # 针对 Volta (sm 7.0) 和 Turing (sm 7.5，除16系列外) 使用 float16
    if sm_version >= 7.0:
        return device, torch.float16, sm_version, mem_gb

    # 其他情况兜底使用 float32
    return device, torch.float32, sm_version, mem_gb

def get_mps_device_info():
    """获取 Apple Silicon MPS 设备信息"""
    if not torch.backends.mps.is_available():
        return None

    try:
        # MPS 设备
        device = torch.device("mps")
        # Apple Silicon 上 MPS 使用 float32 更稳定
        # 虽然 MPS 支持 float16，但在某些模型上可能有精度问题
        return device, torch.float32, 0.0, 0.0  # sm_version 和 mem_gb 对 MPS 不适用
    except Exception:
        return None


# 检测设备类型和配置
device = None
device_type = "cpu"
dtype = None

# 优先尝试 CUDA
if torch.cuda.is_available():
    GPU_COUNT = torch.cuda.device_count()
    available_devices = []
    for i in range(GPU_COUNT):
        info = get_cuda_device_info(i)
        if info is not None:
            available_devices.append(info)

    if available_devices:
        best_info = max(available_devices, key=lambda x: (x[2], x[3]))
        device = best_info[0]
        dtype = best_info[1]
        device_type = "cuda"

# 如果没有 CUDA，尝试 MPS (Apple Silicon)
if device is None:
    mps_info = get_mps_device_info()
    if mps_info is not None:
        device = mps_info[0]
        dtype = torch.float32  # MPS 使用 float32
        device_type = "mps"

# 如果没有可用的 GPU，使用 CPU
if device is None:
    device = torch.device("cpu")
    dtype = torch.float32  # CPU 使用 float32
    device_type = "cpu"


class Config:
    def __init__(self):
        self.dtype = dtype
        self.device = device
        self.device_type = device_type  # 'cuda', 'mps', 或 'cpu'

        self.use_flash_attn = False

        self.gpt_cache = None
        self.sovits_cache = None

        self.compile_mode = None

        self.cnroberta = None


class GlobalConfig:
    def __init__(self):
        self.models_dir = None

        self.use_jieba_fast = None

        self.chinese_g2p = None
        self.japanese_g2p = None
        self.english_g2p = None

global_config = GlobalConfig()