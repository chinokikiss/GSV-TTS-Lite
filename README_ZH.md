<div align="center">
  <a href="项目主页链接">
    <img src="awa.gif" alt="Logo" width="320" height="480">
  </a>

  <h1>GPT-SoVITS-RT</h1>

  <p>
    🚀 <b>GPT-SoVITS-RealTime</b> 
    <br>
    A high-performance inference engine specifically designed for the GPT-SoVITS text-to-speech model
  </p>

  <p align="center">
      <a href="LICENSE">
        <img src="https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge" alt="License">
      </a>
      <a href="https://www.python.org/">
        <img src="https://img.shields.io/badge/Python-3.10+-blue.svg?style=for-the-badge&logo=python&logoColor=white" alt="Python Version">
      </a>
      <a href="https://github.com/chinokikiss/GPT-SoVITS-RT/stargazers">
        <img src="https://img.shields.io/github/stars/chinokikiss/GPT-SoVITS-RT?style=for-the-badge&color=yellow&logo=github" alt="GitHub stars">
      </a>
  </p>

  <p>
    <a href="README.md">
      <img src="https://img.shields.io/badge/English-66ccff?style=flat-square&logo=github&logoColor=white" alt="English">
    </a>
    &nbsp;
    <a href="README_ZH.md">
      <img src="https://img.shields.io/badge/简体中文-ff99cc?style=flat-square&logo=github&logoColor=white" alt="Chinese">
    </a>
  </p>
</div>

<div align="center">
  <img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif">
</div>

## 关于项目 (About)

本项目诞生的初衷源于对极致性能的追求。我在原版 GPT-SoVITS 的使用过程中，受限于 RTX 3050 (Laptop) 的算力瓶颈，推理延迟往往难以满足实时交互的需求。

为了打破这一限制，**GPT-SoVITS-RT** 应运而生，它是基于 **V2Pro** 模型开发的推理后端。通过一些深度优化技术，本项目成功在低显存环境下实现了毫秒级的实时响应。

除了性能上的飞跃，**GPT-SoVITS-RT** 还实现了**音色与风格的解耦**，支持独立控制说话人的声线与情感，并加入了**音字对齐**与**音色迁移**等特色功能。

为了便于开发者集成，**GPT-SoVITS-RT** 大幅精简了代码架构，且体积被压缩至 **800MB**。

## 性能对比 (Performance)

> [!NOTE]
> **测试环境**：单卡 NVIDIA GeForce RTX 3050 (Laptop)

| 推理后端 (Backend)| 设置 (Settings) | 首包延迟 (TTFT) | 实时率 (RTF) | 显存 (VRAM) | 提升幅度 |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Original** | `streaming_mode=3` | 436 ms | 0.381 | 1.6 GB | - |
| **RT Version** | `Flash_Attn=Off` | 150 ms | 0.125 | **0.8 GB** | ⚡ **2.9x** Speed |
| **RT Version** | `Flash_Attn=On` | **133 ms** | **0.108** | **0.8 GB** | 🔥 **3.3x** Speed |

可以看到，**GPT-SoVITS-RT** 实现了 **3x ~ 4x** 速度提升，且显存占用 **减半**！🚀
<br>

## 环境准备 (Prerequisites)

- **Anaconda**
- **CUDA Toolkit**
- **Microsoft Visual C++**

## 快速开始 (Quick Start)

### 安装步骤

> [!IMPORTANT]
> 确保项目所在的路径是纯英文的。

```bash
git clone https://github.com/chinokikiss/GPT-SoVITS-RT
cd GPT-SoVITS-RT

conda create -n gsv-rt python=3.11
conda activate gsv-rt
conda install "ffmpeg"

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

### 快速使用

在项目根目录下创建一个 Python 脚本，即可开始体验。

> [!TIP]
> 首次运行时，程序会自动下载所需的预训练模型。

#### 1. 基础语音合成
```python
import sounddevice as sd
from GPT_SoVITS_RT.TTS import TTS

tts = TTS()

res = tts.infer(
    spk_audio_path="拉菲\日配.mp3",
    prompt_audio_path="anan\0102Adv17_AnAn001.ogg",
    prompt_audio_text="ミリアは……本当に刺されているのか？",
    prompt_audio_language="ja",
    text="へぇー、ここまでしてくれるんですね",
    text_language="auto",
)

print(res)
sd.play(res["audio_data"], res["samplerate"], blocking=True)
```

#### 2. 音色迁移
```python
import sounddevice as sd
from GPT_SoVITS_RT.TTS import TTS

tts = TTS()

res = tts.infer_vc(
    spk_audio_path="拉菲\日配.mp3",
    prompt_audio_path="anan\0102Adv17_AnAn001.ogg",
    prompt_audio_text="ミリアは……本当に刺されているのか？",
    prompt_audio_language="ja",
)

print(res)
sd.play(res["audio_data"], res["samplerate"], blocking=True)
```

#### 3. 流式推理
这是 GPT-SoVITS-RT 的核心功能，能够实现极低延迟的实时对话体验。
```python
import queue
import numpy as np
import sounddevice as sd
from GPT_SoVITS_RT.TTS import TTS

tts = TTS()

class AudioStreamer:
    def __init__(self):
        self.q = queue.Queue()
        self.buffer = np.zeros((0, 1), dtype='float32')

    def put(self, data):
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        self.q.put(data)

    def callback(self, outdata, frames, time, status):
        while len(self.buffer) < frames:
            try:
                self.buffer = np.concatenate((self.buffer, self.q.get_nowait()))
            except queue.Empty:
                break
        n = min(len(self.buffer), frames)
        outdata[:n] = self.buffer[:n]
        outdata[n:] = 0
        self.buffer = self.buffer[n:]

streamer = AudioStreamer()

stream = sd.OutputStream(
    samplerate=32000, 
    channels=1, 
    callback=streamer.callback,
    dtype='float32'
)
stream.start()

while True:
    text = input("infer text: ")

    generator = tts.infer_stream(
        spk_audio_path="拉菲\日配.mp3",
        prompt_audio_path="anan\0102Adv17_AnAn001.ogg",
        prompt_audio_text="ミリアは……本当に刺されているのか？",
        prompt_audio_language="ja",
        text=text,
        text_language="auto",
        boost_first_chunk=True, # 如果设置为“True”，可以减少首包延迟，但可能会在短音段中产生噪音；若希望合成更稳定，则应将其设置为“False”。
    )

    for audio_data in generator:
        print(audio_data)
        streamer.put(audio_data["audio_data"])

    while not streamer.q.empty() or len(streamer.buffer) > 0:
        sd.sleep(100)
```

<details>
<summary><strong>4. 其他接口</strong></summary>

### 1. 模型初始化与加载

#### `init_language_module(languages)`
预加载必要的语言处理模块。

#### `load_gpt_model(model_paths="pretrained_models/s1v3.ckpt")`
将 GPT 模型权重从指定路径加载到内存中。

#### `load_sovits_model(model_paths="pretrained_models/v2Pro/s2Gv2ProPlus.pth")`
将 SoVITS 模型权重从指定路径加载到内存中。

### 2. 模型卸载与列表获取

#### `unload_gpt_model(model_paths)` / `unload_sovits_model(model_paths)`
从内存中卸载模型以释放资源。

#### `get_gpt_list()` / `get_sovits_list()`
获取当前已加载模型的列表。

### 3. 音频缓存管理

#### `cache_spk_audio(spk_audio_paths)`
预处理并缓存说话人音频数据。

#### `cache_prompt_audio(prompt_audio_list)`
预处理并缓存提示音频数据。

#### `del_spk_audio(spk_audio_list)` / `del_prompt_audio(prompt_audio_list)`
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

## 未来计划 (Future Roadmap)
* [ ] **API & WebUI & 整合包**
* [ ] **批量推理**
* [ ] **新架构 GPT 模型**

## 致谢 (Credits)
特别感谢以下项目：
- [RVC-Boss/GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS)
- [High-Logic/Genie-TTS](https://github.com/High-Logic/Genie-TTS)

## ⭐ Star History

[![Star History Chart](https://api.star-history.com/svg?repos=chinokikiss/GPT-SoVITS-RT&type=Date)](https://star-history.com/#chinokikiss/GPT-SoVITS-RT&Date)
