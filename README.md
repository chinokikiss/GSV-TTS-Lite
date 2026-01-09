<div align="center">
  <a href="https://github.com/chinokikiss/GPT-SoVITS-RT">
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

## About

This project was born out of a pursuit for extreme performance. While using the original GPT-SoVITS, I encountered significant latency due to the computational limitations of my RTX 3050 (Laptop), making real-time interaction impractical.

To overcome this bottleneck, **GPT-SoVITS-RT** was created — an optimized inference backend built on the **V2Pro** model. Leveraging deep-level optimizations, it achieves **millisecond-level response times**, even on low-VRAM GPUs.

Beyond raw speed, **GPT-SoVITS-RT** introduces **voice-style disentanglement**, allowing independent control over speaker timbre and emotional tone. It also supports advanced features such as **phoneme alignment** and **voice conversion (voice cloning)**.

For ease of integration, the codebase has been significantly streamlined, with the entire package compressed to just **800MB**.        

## Performance Comparison

> [!NOTE]
> **Test Environment**: Single GPU — NVIDIA GeForce RTX 3050 (Laptop)

| Inference Backend | Settings | TTFT | RTF | VRAM | Speedup |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Original** | `streaming_mode=3` | 436 ms | 0.381 | 1.6 GB | - |
| **RT Version** | `Flash_Attn=Off` | 150 ms | 0.125 | **0.8 GB** | ⚡ **2.9x** Speed |
| **RT Version** | `Flash_Attn=On` | **133 ms** | **0.108** | **0.8 GB** | 🔥 **3.3x** Speed |

As shown above, **GPT-SoVITS-RT** delivers a **3x ~ 4x** speed improvement while halving VRAM usage! 🚀
<br>

## Prerequisites

Before getting started, ensure you have the following installed:
- **Anaconda**
- **CUDA Toolkit**
- **Microsoft Visual C++ Build Tools**

## Quick Start

### Installation

> [!IMPORTANT]
> Make sure the project path contains **only English characters** (no spaces or special symbols recommended).

```bash
git clone https://github.com/chinokikiss/GPT-SoVITS-RT
cd GPT-SoVITS-RT

conda create -n gsv-rt python=3.11
conda activate gsv-rt
conda install "ffmpeg"

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

### Usage Examples

Create a Python script in the project root directory to begin experimenting.

> [!TIP]
> On first run, required pre-trained models will be downloaded automatically.

#### 1. Basic Text-to-Speech

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

#### 2. Voice Conversion

```python
res = tts.infer_vc(
    spk_audio_path="拉菲\日配.mp3",
    prompt_audio_path="anan\0102Adv17_AnAn001.ogg",
    prompt_audio_text="ミリアは……本当に刺されているのか？",
    prompt_audio_language="ja",
)

print(res)
sd.play(res["audio_data"], res["samplerate"], blocking=True)
```

#### 3. Streaming Inference

Streaming inference is the core feature of GPT-SoVITS-RT, enabling ultra-low-latency interactive speech synthesis.

```python
import queue
import numpy as np

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
    )

    for audio_data in generator:
        print(audio_data)
        streamer.put(audio_data["audio_data"])

    while not streamer.q.empty() or len(streamer.buffer) > 0:
        sd.sleep(100)
```

## Flash Attn

For **even lower latency** and **higher throughput**, we strongly recommend enabling **Flash Attention**.

Due to compilation complexity, installation must be done manually based on your OS:

*   **🐧 Linux / Source Build**
    *   Official Repo: [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention)

*   **🪟 Windows Users**
    *   Pre-built Wheels: [lldacing/flash-attention-windows-wheel](https://huggingface.co/lldacing/flash-attention-windows-wheel/tree/main)

> [!TIP]
> After installation, set `use_flash_attn=True` in the TTS configuration to unlock peak performance! 🚀

## Future Roadmap
* [ ] Batch Inference Support
* [ ] Support for New GPT Architectures

## Credits

Special thanks to the following projects:
- [RVC-Boss/GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS)
- [High-Logic/Genie-TTS](https://github.com/High-Logic/Genie-TTS)

## ⭐ Star History

[![Star History Chart](https://api.star-history.com/svg?repos=chinokikiss/GPT-SoVITS-RT&type=Date)](https://star-history.com/#chinokikiss/GPT-SoVITS-RT&Date)
