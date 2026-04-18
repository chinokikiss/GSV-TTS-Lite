<div align="center">
  <a href="项目主页链接">
    <img src="huiyeji.gif" alt="Logo" width="240" height="254">
  </a>

  <h1>GSV-TTS-Lite</h1>

  <p>
    A high-performance inference engine specifically designed for the GPT-SoVITS text-to-speech model
  </p>

  <p align="center">
      <a href="LICENSE">
        <img src="https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge" alt="License">
      </a>
      <a href="https://www.python.org/">
        <img src="https://img.shields.io/badge/Python-3.10+-blue.svg?style=for-the-badge&logo=python&logoColor=white" alt="Python Version">
      </a>
      <a href="https://github.com/chinokikiss/GSV-TTS-Lite/stargazers">
        <img src="https://img.shields.io/github/stars/chinokikiss/GSV-TTS-Lite?style=for-the-badge&color=yellow&logo=github" alt="GitHub stars">
      </a>
  </p>

  <p>
    <a href="README_EN.md">
      <img src="https://img.shields.io/badge/English-66ccff?style=flat-square&logo=github&logoColor=white" alt="English">
    </a>
    &nbsp;
    <a href="README.md">
      <img src="https://img.shields.io/badge/简体中文-ff99cc?style=flat-square&logo=github&logoColor=white" alt="Chinese">
    </a>
    &nbsp;
    <a href="README_JA.md">
      <img src="https://img.shields.io/badge/日本語-ffcc66?style=flat-square&logo=github&logoColor=white" alt="Japanese">
    </a>
  </p>
</div>

<div align="center">
  <img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif">
</div>

## 关于项目 (About)

本项目诞生的初衷源于对极致性能的追求。我在原版 GPT-SoVITS 的使用过程中，受限于 RTX 3050 (Laptop) 的算力瓶颈，推理延迟往往难以满足实时交互的需求。

为了打破这一限制，**GSV-TTS-Lite** 应运而生，它是基于 **GPT-SoVITS V2Pro** 开发的推理后端。通过一些深度优化技术，本项目成功在低显存环境下实现了毫秒级的实时响应。

除了性能上的飞跃，**GSV-TTS-Lite** 还实现了**音色与风格的解耦**，支持独立控制说话人的音色与情感，并加入了**字级时间戳对齐**与**音色迁移**等特色功能。

为了便于开发者集成，**GSV-TTS-Lite** 大幅精简了代码架构，并已作为 `gsv-tts-lite` 库发布至 PyPI，支持通过 `pip` 一键安装。

目前支持的语言有 **中日英**，支持的模型有 **V2Pro**、**V2ProPlus**。
## 性能对比 (Performance)

> [!NOTE]
> **测试环境**：NVIDIA GeForce RTX 3050 (Laptop)

| 推理后端 (Backend)| 设置 (Settings) | 首包延迟 (TTFT) | 实时率 (RTF) | 显存 (VRAM) | 提升幅度 |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Original** | `streaming_mode=3` | 436 ms | 0.381 | 1.6 GB | - |
| **Lite Version** | `Flash_Attn=Off` | 150 ms | 0.125 | **0.8 GB** | ⚡ **2.9x** Speed |
| **Lite Version** | `Flash_Attn=On` | **133 ms** | **0.108** | **0.8 GB** | 🔥 **3.3x** Speed |

可以看到，**GSV-TTS-Lite** 实现了 **3x ~ 4x** 速度提升，且显存占用 **减半**！🚀
<br>

## 整合包下载 (One-click Download)

> [!TIP]
> 如果你是小白，想要快速体验，可以直接下载预配置好的整合包。

- **硬件要求**：
  - **操作系统**：仅限 Windows。
  - **显卡需求**：NVIDIA 显卡，显存需 **4GB** 及以上。
  - **显存说明**：默认集成了 `Qwen3-ASR` 模型。若显存不足，可在 `go-webui.bat` 中通过参数禁用 ASR 模块以节省空间。
- **下载地址**：
  - [推荐版](https://modelscope.cn/models/chinokiki/GPTSoVITS-RT/resolve/master/gsv-tts-lite-web-cu128.zip)
  - [通用版(兼容旧款 NVIDIA 显卡)](https://modelscope.cn/models/chinokiki/GPTSoVITS-RT/resolve/master/gsv-tts-lite-web-cu118.zip)
- **使用说明**：
  1. 下载并解压压缩包（建议路径不要包含中文）。
  2. 双击运行 `go-webui.bat` 等待网页推理界面跳出。
  3. 然后就可以开始体验语音合成了！
  4. 注意！风格参考和音色参考两个都需要上传，少一个都会合成失败！

## 开发者部署 (Deployment)

### 环境准备

- **CUDA Toolkit**
> [!IMPORTANT]
> 当前版本已全面支持 **CUDA**、**MPS (Apple Silicon)** 及 **CPU** 推理后端。
> 未来计划集成 **ONNX Runtime** 以进一步加速 CPU 与 MPS 的推理速度。

### 安装部署

#### 1. 环境配置
建议使用 Python>=3.10 创建虚拟环境。
```bash
# NVIDIA GPU (CUDA 12.8)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# Apple Silicon (MPS) 或 Linux/Windows (仅 CPU)
pip install torch torchvision torchaudio
```
#### 2. 安装 GSV-TTS-Lite
若已准备好上述基础环境，可直接执行以下命令完成集成：
```bash 
pip install gsv-tts-lite==0.3.13
```

### WebUI 可视化界面

1. **安装依赖**：
  ```bash
  cd WebUI
  pip install -r requirements.txt
  ```
2. **启动程序**：
  ```bash
  python web.py
  ```

### API 服务接口

1. **安装依赖**：
  ```bash
  cd API
  pip install -r requirements.txt
  ```
2. **核心文档**：
   [进入 API 详细指南目录 ➔](https://github.com/chinokikiss/GSV-TTS-Lite/tree/main/API)

### Python SDK 接口调用

> [!TIP]
> 首次运行时，程序会自动下载所需的预训练模型。

#### 1. 基础推理
```python
from gsv_tts import TTS

tts = TTS(use_bert=True)
# tts = TTS(use_flash_attn=True) 如果安装了Flash Attention，建议这样设置

# 将 GPT 模型权重从指定路径加载到内存中，这里加载默认模型。
tts.load_gpt_model()

# 将 SoVITS 模型权重从指定路径加载到内存中，这里加载默认模型。
tts.load_sovits_model()

# 预加载与缓存资源，可显著减少首次推理的延迟
# tts.init_language_module("ja")
# tts.cache_spk_audio("examples\laffey.mp3")
# tts.cache_prompt_audio(
#     prompt_audio_paths="examples\AnAn.ogg",
#     prompt_audio_texts="ちが……ちがう。レイア、貴様は間違っている。",
# )

# infer 是最简单、最原始的推理方式，只适用于短文本推理，一般建议用 infer_batched 替代 infer 推理。
audio = tts.infer(
    spk_audio_path="examples\laffey.mp3", # 音色参考音频
    prompt_audio_path="examples\AnAn.ogg", # 风格参考音频
    prompt_audio_text="ちが……ちがう。レイア、貴様は間違っている。", # 风格参考音频对应的文本
    text="へぇー、ここまでしてくれるんですね。", # 目标生成文本
    # gpt_model = None, # 用于推理的GPT模型路径，默认用第一个加载的GPT模型推理
    # sovits_model = None, # 用于推理的SoVITS模型路径，默认用第一个加载的SoVITS模型推理
)

audio.play()
tts.audio_queue.wait()
# tts.audio_queue.stop() 停止播放
```

#### 2. 流式推理 / 字幕同步
```python
import time
import queue
import threading
from gsv_tts import TTS

class SubtitlesQueue:
    def __init__(self):
        self.q = queue.Queue()
        self.t = None
    
    def process(self):
        last_i = 0
        last_t = time.time()

        while True:
            subtitles, text = self.q.get()
            
            if subtitles is None:
                break

            for subtitle in subtitles:
                if subtitle["start_s"] > time.time() - last_t:
                    while time.time() - last_t <= subtitle["start_s"]:
                        time.sleep(0.01)

                if subtitle["end_s"] and subtitle["end_s"] > time.time() - last_t:
                    if subtitle["orig_idx_end"] > last_i:
                        print(text[last_i:subtitle["orig_idx_end"]], end="", flush=True)
                        last_i = subtitle["orig_idx_end"]
                        while time.time() - last_t <= subtitle["end_s"]:
                            time.sleep(0.01)

        self.t = None
    
    def add(self, subtitles, text):
        self.q.put((subtitles, text))
        if self.t is None:
            self.t = threading.Thread(target=self.process, daemon=True)
            self.t.start()

tts = TTS(use_bert=True)

# infer、infer_stream、infer_batched、infer_vc 其实都支持字级时间戳的返回，这里只是通过 infer_stream 举个例子
subtitlesqueue = SubtitlesQueue()

# infer_stream 实现了 Token 级别的流式输出，显著降低了首字延迟，能够实现极低延迟的实时反馈体验。
generator = tts.infer_stream(
    spk_audio_path="examples\laffey.mp3",
    prompt_audio_path="examples\AnAn.ogg",
    prompt_audio_text="ちが……ちがう。レイア、貴様は間違っている。",
    text="へぇー、ここまでしてくれるんですね。",
    debug=False,
)

for audio in generator:
    audio.play()
    subtitlesqueue.add(audio.subtitles, audio.orig_text)

tts.audio_queue.wait()
subtitlesqueue.add(None, None)
```

#### 3. 批量推理
```python
from gsv_tts import TTS

tts = TTS(use_bert=True)

# infer_batched 专为长文本及多句合成场景优化。该模式不仅在处理效率上具有显著优势，更支持在同一批次（Batch）中为不同句子指定不同的参考音频，提供了极高的合成自由度与灵活性。
audios = tts.infer_batched(
    spk_audio_paths="examples\laffey.mp3",
    prompt_audio_paths="examples\AnAn.ogg",
    prompt_audio_texts="ちが……ちがう。レイア、貴様は間違っている。",
    texts=["へぇー、ここまでしてくれるんですね。", "The old map crinkled in Leo’s trembling hands."],
)

for i, audio in enumerate(audios):
    audio.save(f"audio{i}.wav")
```

#### 4. 音色迁移
```python
from gsv_tts import TTS

tts = TTS(use_bert=True, always_load_cnhubert=True)

# infer_vc 虽然支持 Zero-shot 音色迁移，在便捷性上有一定优势，但在转换质量上，相较于 RVC、SVC 等专门的变声模型仍有提升空间。
audio = tts.infer_vc(
    spk_audio_path="examples\laffey.mp3",
    prompt_audio_path="examples\AnAn.ogg",
    prompt_audio_text="ちが……ちがう。レイア、貴様は間違っている。",
)

audio.play()
tts.audio_queue.wait()
```

#### 5. 声纹识别
```python
from gsv_tts import TTS

tts = TTS(use_bert=True, always_load_sv=True)

# verify_speaker 用于对比两段音频的说话人特征，判断其是否为同一人。
similarity = tts.verify_speaker("examples\laffey.mp3", "examples\AnAn.ogg")
print("声纹相似度：", similarity)
```

<details>
<summary><strong>6. 其他函数接口</strong></summary>

### 1. 模型管理

#### `init_language_module(languages)`
预加载必要的语言处理模块。

#### `load_gpt_model(model_paths)`
将 GPT 模型权重从指定路径加载到内存中。

#### `load_sovits_model(model_paths)`
将 SoVITS 模型权重从指定路径加载到内存中。

#### `unload_gpt_model(model_paths)` / `unload_sovits_model(model_paths)`
从内存中卸载模型以释放资源。

#### `get_gpt_list()` / `get_sovits_list()`
获取当前已加载模型的列表。

#### `to_safetensors(checkpoint_path)`
将 PyTorch 格式的模型权重文件（.pth 或 .ckpt）转换为 safetensors 格式。

### 2. 音频缓存管理

#### `cache_spk_audio(spk_audio_paths)`
预处理并缓存音色参考音频数据。

#### `cache_prompt_audio(prompt_audio_paths, prompt_audio_texts, prompt_audio_languages)`
预处理并缓存风格参考音频数据。

#### `del_spk_audio(spk_audio_paths)` / `del_prompt_audio(prompt_audio_paths)`
从缓存中移除音频数据。

#### `get_spk_audio_list()` / `get_prompt_audio_list()`
获取缓存中的音频数据列表。

</details>

## Flash Attn
如果你追求**更低的延迟**和**更高的吞吐量**，强烈建议开启 `Flash Attention` 支持。
由于该库对编译环境有特定要求，请根据你的系统手动安装：

*   **🐧 Linux / 源码构建**
    *   官方仓库：[Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention)

*   **🪟 Windows 用户**
    *   预编译 Wheel 包：[lldacing/flash-attention-windows-wheel](https://huggingface.co/lldacing/flash-attention-windows-wheel/tree/main)

> [!TIP]
> 安装完成后，在TTS配置中设置 `use_flash_attn=True` 即可享受加速效果！🚀

## 致谢 (Credits)
特别感谢以下项目：
- [RVC-Boss/GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS)

## ⭐ Star History

[![Star History Chart](https://api.star-history.com/svg?repos=chinokikiss/GSV-TTS-Lite&type=Date)](https://star-history.com/#chinokikiss/GSV-TTS-Lite&Date)
