<div align="center">
  <a href="Project_Link_Placeholder">
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
  </p>
</div>

<div align="center">
  <img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif">
</div>

## About

The original motivation for this project was the pursuit of ultimate performance. While using the original GPT-SoVITS, I found that the inference latency often struggled to meet the demands of real-time interaction due to the computing power bottlenecks of the RTX 3050 (Laptop).

To break through these limitations, **GSV-TTS-Lite** was developed as an inference backend based on **GPT-SoVITS V2Pro**. Through deep optimization techniques, this project successfully achieves millisecond-level real-time response in low-VRAM environments.

Beyond the leap in performance, **GSV-TTS-Lite** implements the **decoupling of timbre and style**, supporting independent control over the speaker's voice and emotion. It also features **subtitle timestamp alignment** and **voice conversion (timbre transfer)**.

Regarding multilingual support, this project inherits full synthesis capabilities for five languages: **Chinese, English, Japanese, Korean, and Cantonese**.

To facilitate integration for developers, **GSV-TTS-Lite** features a significantly streamlined code architecture and is available on PyPI as the `gsv-tts-lite` library, supporting one-click installation via `pip`.

## Performance Comparison

> [!NOTE]
> **Test Environment**: NVIDIA GeForce RTX 3050 (Laptop)

| Backend | Settings | TTFT (First Packet) | RTF (Real-time Factor) | VRAM | Speedup |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Original** | `streaming_mode=3` | 436 ms | 0.381 | 1.6 GB | - |
| **Lite Version** | `Flash_Attn=Off` | 150 ms | 0.125 | **0.8 GB** | ⚡ **2.9x** Speed |
| **Lite Version** | `Flash_Attn=On` | **133 ms** | **0.108** | **0.8 GB** | 🔥 **3.3x** Speed |

As shown, **GSV-TTS-Lite** achieves **3x ~ 4x** speed improvements while **halving** the VRAM usage! 🚀
<br>

## One-click Download (Pre-configured)

> [!TIP]
> If you are a beginner looking for a quick start, you can download the pre-configured integrated package.

- **Hardware Requirements**:
  - **OS**: Windows only.
  - **GPU**: NVIDIA GPU with at least **4GB** VRAM.
  - **VRAM Note**: The `Qwen3-ASR` model is integrated by default. If VRAM is insufficient, you can disable the ASR module via parameters in `run.bat` to save space.
- **Download Link**: [Placeholder]
- **Instructions**:
  1. Download and extract the package (ensure the path contains no Chinese characters).
  2. Double-click `run.bat` and wait for the web UI to launch.
  3. Start experiencing high-speed voice synthesis!

## Deployment (For Developers)

### Prerequisites

- **Anaconda**
- **CUDA Toolkit**
- **Microsoft Visual C++**

### Installation Steps

#### 1. Environment Configuration
It is recommended to create a virtual environment using Python >=3.10 and install the necessary system dependency `ffmpeg`.
```bash
conda create -n gsv-tts python=3.11
conda activate gsv-tts
conda install "ffmpeg"

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```
#### 2.	Install GSV-TTS-Lite
If you have prepared the above basic environment, you can directly execute the following command to complete the integration:
```bash
pip install gsv-tts-lite --prefer-binary
```

### Quick Start

> [!TIP]
> The program will automatically download the required pre-trained models upon the first run.

#### 1. Basic Inference
```python
from gsv_tts import TTS

tts = TTS()

# Load GPT model weights from the specified path into memory; here, the default model is loaded.
tts.load_gpt_model()

# 将Load SoVITS model weights from the specified path into memory; here, the default model is loaded.
tts.load_sovits_model()


# infer is the simplest and most primitive inference method, suitable for short text inference.
audio = tts.infer(
    spk_audio_path="examples\laffey.mp3", # Speaker reference audio
    prompt_audio_path="examples\AnAn.ogg", # Prompt reference audio
    prompt_audio_text="ちが……ちがう。レイア、貴様は間違っている。", # Text corresponding to the prompt audio
    prompt_audio_language="ja", # Language corresponding to the text
    text="へぇー、ここまでしてくれるんですね。", # Target text to generate
    text_language="ja",  # Language corresponding to the text
)

audio.play()
tts.audio_queue.wait()
```

#### 2. Stream Inference / Subtitle Synchronization
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
                print()
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

tts = TTS()

# infer, infer_stream, and infer_batched all support returning subtitle timestamps; infer_stream is used here just as an example.
subtitlesqueue = SubtitlesQueue()

# infer_stream implements token-level streaming output, significantly reducing first-token latency and enabling a ultra-low latency real-time feedback experience.
generator = tts.infer_stream(
    spk_audio_path="examples\laffey.mp3",
    prompt_audio_path="examples\AnAn.ogg",
    prompt_audio_text="ちが……ちがう。レイア、貴様は間違っている。",
    prompt_audio_language="ja",
    text="へぇー、ここまでしてくれるんですね。",
    text_language="ja",
    debug=False,
)

for audio in generator:
    audio.play()
    subtitlesqueue.add(audio.subtitles, audio.orig_text)

tts.audio_queue.wait()
subtitlesqueue.add(None, None)
```

#### 3. Batched Inference
```python
from gsv_tts import TTS

tts = TTS()

# infer_batched is optimized specifically for long-form text and multi-sentence synthesis scenarios. This mode not only offers significant advantages in processing efficiency but also supports assigning different reference audios to different sentences within the same batch, providing high synthesis freedom and flexibility.
audios = tts.infer_batched(
    spk_audio_paths="examples\laffey.mp3",
    prompt_audio_paths="examples\AnAn.ogg",
    prompt_audio_texts="ちが……ちがう。レイア、貴様は間違っている。",
    prompt_audio_languages="ja",
    texts=["へぇー、ここまでしてくれるんですね。", "The old map crinkled in Leo’s trembling hands."],
    texts_language=["ja", "en"],
)

for i, audio in enumerate(audios):
    audio.save(f"audio{i}.wav")
```

#### 4. Voice Conversion
```python
from gsv_tts import TTS

tts = TTS()

# Although infer_vc supports few-shot voice conversion and offers convenience, its conversion quality still has room for improvement compared to specialized voice conversion models like RVC or SVC.
audio = tts.infer_vc(
    spk_audio_path="examples\laffey.mp3",
    prompt_audio_path="examples\AnAn.ogg",
    prompt_audio_text="ちが……ちがう。レイア、貴様は間違っている。",
    prompt_audio_language="ja",
)

audio.play()
tts.audio_queue.wait()
```

#### 5. Speaker Verification
```python
from gsv_tts import TTS

tts = TTS(always_load_sv=True)

# verify_speaker is used to compare the speaker characteristics of two audio clips to determine if they are the same person.
similarity = tts.verify_speaker("examples\laffey.mp3", "examples\AnAn.ogg")
print("Speaker Similarity:", similarity)
```

<details>
<summary><strong>6. Other Function Interfaces</strong></summary>

### 1. Model Initialization and Loading

#### `init_language_module(languages)`
Preload necessary language processing modules.

#### `load_gpt_model(model_paths)`
Load GPT model weights from specified paths into memory.

#### `load_sovits_model(model_paths)`
Load SoVITS model weights from specified paths into memory.

### 2. Model Unloading and List Retrieval

#### `unload_gpt_model(model_paths)` / `unload_sovits_model(model_paths)`
Unload models from memory to free up resources.

#### `get_gpt_list()` / `get_sovits_list()`
Get the list of currently loaded models.

### 3. Audio Cache Management

#### `cache_spk_audio(spk_audio_paths)`
Preprocess and cache speaker reference audio data.

#### `cache_prompt_audio(prompt_audio_paths, prompt_audio_texts, prompt_audio_languages)`
Preprocess and cache prompt reference audio data.

#### `del_spk_audio(spk_audio_list)` / `del_prompt_audio(prompt_audio_paths)`
Remove audio data from the cache.

#### `get_spk_audio_list()` / `get_prompt_audio_list()`
Get the list of audio data in the cache.

#### `to_safetensors(checkpoint_path)`
Converts PyTorch checkpoint files (.pth or .ckpt) into the safetensors format.

</details>

## Flash Attn
If you are looking for **lower latency** and **higher throughput**, it is highly recommended to enable `Flash Attention` support.
Since this library has specific compilation requirements, please install it manually based on your system:

*   **🐧 Linux / Build from Source**
    *   Official Repo: [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention)

*   **🪟 Windows Users**
    *   Pre-compiled Wheels: [lldacing/flash-attention-windows-wheel](https://huggingface.co/lldacing/flash-attention-windows-wheel/tree/main)

> [!TIP]
> After installation, set `use_flash_attn=True` in your TTS configuration to enjoy the acceleration! 🚀

## Credits
Special thanks to the following projects:
- [RVC-Boss/GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS)

## ⭐ Star History

[![Star History Chart](https://api.star-history.com/svg?repos=chinokikiss/GSV-TTS-Lite&type=Date)](https://star-history.com/#chinokikiss/GSV-TTS-Lite&Date)
