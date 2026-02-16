import torch

def get_device_info(idx: int):
    cpu_config = (torch.device("cpu"), torch.float32, 0.0, 0.0)
    
    if not torch.cuda.is_available() or idx >= torch.cuda.device_count():
        return cpu_config

    try:
        props = torch.cuda.get_device_properties(idx)
    except Exception:
        return cpu_config

    name = props.name
    major, minor = props.major, props.minor
    sm_version = major + minor / 10.0
    mem_gb = props.total_memory / (1024**3)

    # 显存小于 4GB 或算力太低（Pascal 架构之前的旧卡），回退到 CPU
    if mem_gb < 4 or sm_version < 5.3:
        return cpu_config

    device = torch.device(f"cuda:{idx}")
    
    # 针对 GTX 16 系列 (Turing 架构但不带 Tensor Cores) 或 Pascal (如 GTX 1080)
    # 这些卡虽然支持 fp16 计算，但速度可能不如 fp32，或者容易溢出，所以使用 fp32
    is_16_series = (major == 7 and minor == 5) and ("16" in name)
    if sm_version == 6.1 or is_16_series:
        return device, torch.float32, sm_version, mem_gb

    # 其他较新的卡（Volta, Turing RTX, Ampere, Hopper 等）使用 fp16
    if sm_version > 6.1:
        return device, torch.float16, sm_version, mem_gb

    return cpu_config

GPU_COUNT = torch.cuda.device_count()
available_devices = []

for i in range(GPU_COUNT):
    info = get_device_info(i)
    if info[0].type != "cpu":
        available_devices.append(info)

if available_devices:
    best_info = max(available_devices, key=lambda x: (x[2], x[3]))
    infer_device = best_info[0]
    
    is_half = any(dtype == torch.float16 for _, dtype, _, _ in available_devices)
else:
    infer_device = torch.device("cpu")
    is_half = False


class Config:
    def __init__(self):  
        self.is_half = is_half
        self.dtype = torch.float16 if is_half else torch.float32
        self.device = infer_device

        self.use_flash_attn = False

        self.gpt_cache = None
        self.sovits_cache = None

        self.cnroberta = None


class GlobalConfig:
    def __init__(self):
        self.models_dir = None

        self.chinese_g2p = None
        self.japanese_g2p = None
        self.korean_g2p = None
        self.cantonese_g2p = None
        self.english_g2p = None

        self.lang_segmenter = None

global_config = GlobalConfig()