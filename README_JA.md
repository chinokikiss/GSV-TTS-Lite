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

## プロジェクトについて (About)

本プロジェクトは、極限までのパフォーマンス追求を初衷として誕生しました。原版 GPT-SoVITS の使用中、RTX 3050 (Laptop) の計算能力のボトルネックにより、推論遅延がリアルタイム交互のニーズを満たすことが難しい場合がありました。

この制限を打破するため、**GSV-TTS-Lite** が生まれました。これは **GPT-SoVITS V2Pro** に基づいて開発された推論バックエンドです。いくつかの深層最適化技術により、本プロジェクトは低 VRAM 環境においてミリ秒級のリアルタイム応答を実現しました。

パフォーマンスの飛躍に加え、**GSV-TTS-Lite** は**音色とスタイルの分離**を実現し、話者の音色と感情を独立して制御可能にしました。さらに、**字幕タイムスタンプ同期**や**音色変換**などの特有機能を追加しました。

開発者の統合を容易にするため、**GSV-TTS-Lite** はコードアーキテクチャを大幅に簡素化し、`gsv-tts-lite` ライブラリとして PyPI に公開されました。`pip` によるワンクリックインストールをサポートしています。

現在サポートされている言語は**中国語、日本語、英語**で、サポートされているモデルは **V2Pro**、**V2ProPlus** です。
## パフォーマンス比較 (Performance)

> [!NOTE]
> **テスト環境**：NVIDIA GeForce RTX 3050 (Laptop)

| 推論バックエンド (Backend)| 設定 (Settings) | 初包遅延 (TTFT) | リアルタイム率 (RTF) | VRAM (VRAM) | 向上幅 |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Original** | `streaming_mode=3` | 436 ms | 0.381 | 1.6 GB | - |
| **Lite Version** | `Flash_Attn=Off` | 150 ms | 0.125 | **0.8 GB** | ⚡ **2.9x** 速度 |
| **Lite Version** | `Flash_Attn=On` | **133 ms** | **0.108** | **0.8 GB** | 🔥 **3.3x** 速度 |

ご覧の通り、**GSV-TTS-Lite** は **3x ~ 4x** の速度向上を実現し、VRAM 占有量も**半分**になりました！🚀
<br>

## 開発者向けデプロイ (Deployment)

### 環境準備

- **FFmpeg**
- **CUDA Toolkit**
> [!IMPORTANT]
> 現在のバージョンでは、CUDA、MPS (Apple Silicon)、および CPU 推論バックエンドを全面的にサポートしています。
> 今後は ONNX Runtime を統合し、CPU および MPS における推論速度のさらなる高速化を計画しています。

### インストールとデプロイ

#### 1. 環境設定
Python>=3.10 を使用して仮想環境を作成することを推奨します。
```bash
# NVIDIA GPU (CUDA 12.8) の場合
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# Apple Silicon (MPS) または Linux/Windows (CPU のみ) の場合
pip install torch torchvision torchaudio
```
#### 2. GSV-TTS-Lite のインストール
上記の基本環境が準備できれば、以下のコマンドを実行するだけで統合が完了します：
```bash 
pip install gsv-tts-lite==0.3.7
```

### WebUI 可視化インターフェース

1. **依存関係のインストール**：
  ```bash
  cd WebUI
  pip install -r requirements.txt
  ```
2. **プログラムの起動**：
  ```bash
  python web.py
  ```

### API サービスインターフェース

1. **依存関係のインストール**：
  ```bash
  cd API
  pip install -r requirements.txt
  ```
2. **コアドキュメント**：
   [API 詳細ガイドディレクトリへ ➔](https://github.com/chinokikiss/GSV-TTS-Lite/tree/main/API)

### Python SDK インターフェース呼び出し

> [!TIP]
> 初回実行時、プログラムは必要な事前学習済みモデルを自動的にダウンロードします。

#### 1. 基本推論
```python
from gsv_tts import TTS

tts = TTS()
# tts = TTS(use_bert=True) より優れた中国語合成効果を得たい場合、この設定を推奨します
# tts = TTS(use_flash_attn=True) Flash Attention をインストール済みの場合、この設定を推奨します

# GPT モデルの重みを指定されたパスからメモリにロードします。ここではデフォルトモデルをロードします。
tts.load_gpt_model()

# SoVITS モデルの重みを指定されたパスからメモリにロードします。ここではデフォルトモデルをロードします。
tts.load_sovits_model()

# リソースを事前ロードおよびキャッシュし、初回推論の遅延を大幅に削減できます
# tts.init_language_module("ja")
# tts.cache_spk_audio("examples\laffey.mp3")
# tts.cache_prompt_audio(
#     prompt_audio_paths="examples\AnAn.ogg",
#     prompt_audio_texts="ちが……ちがう。レイア、貴様は間違っている。",
# )

# infer は最もシンプルで原始的な推論方式であり、短文の推論にのみ適しています。通常、infer の代わりに infer_batched を使用することが推奨されます。
audio = tts.infer(
    spk_audio_path="examples\laffey.mp3", # 音色参照オーディオ
    prompt_audio_path="examples\AnAn.ogg", # スタイル参照オーディオ
    prompt_audio_text="ちが……ちがう。レイア、貴様は間違っている。", # スタイル参照オーディオに対応するテキスト
    text="へぇー、ここまでしてくれるんですね。", # 生成対象テキスト
    # gpt_model = None, # 推論に使用する GPT モデルのパス。デフォルトでは最初にロードされた GPT モデルで推論します
    # sovits_model = None, # 推論に使用する SoVITS モデルのパス。デフォルトでは最初にロードされた SoVITS モデルで推論します
)

audio.play()
tts.audio_queue.wait()
# tts.audio_queue.stop() 再生を停止
```

#### 2. ストリーミング推論 / 字幕同期
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

tts = TTS()

# infer、infer_stream、infer_batched、infer_vc は実際すべて字幕タイムスタンプの返却をサポートしていますが、ここでは infer_stream を例に挙げています
subtitlesqueue = SubtitlesQueue()

# infer_stream は Token レベルのストリーミング出力を実装し、初字遅延を大幅に低減し、极低遅延のリアルタイムフィードバック体験を実現します。
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

#### 3. バッチ推論
```python
from gsv_tts import TTS

tts = TTS()

# infer_batched は長テキストおよび多文合成シーン向けに最適化されています。このモードは処理効率において顕著な優位性を持つだけでなく、同一バッチ（Batch）内で異なる文に対して異なる参照オーディオを指定することも支持し、極めて高い合成の自由度と柔軟性を提供します。
audios = tts.infer_batched(
    spk_audio_paths="examples\laffey.mp3",
    prompt_audio_paths="examples\AnAn.ogg",
    prompt_audio_texts="ちが……ちがう。レイア、貴様は間違っている。",
    texts=["へぇー、ここまでしてくれるんですね。", "The old map crinkled in Leo's trembling hands."],
)

for i, audio in enumerate(audios):
    audio.save(f"audio{i}.wav")
```

#### 4. 音色変換
```python
from gsv_tts import TTS

tts = TTS(always_load_cnhubert=True)

# infer_vc は Few-shot（少数サンプル）音色変換をサポートしていますが、利便性において一定の優位性があるものの、変換品質においては RVC、SVC などの専用変声モデルと比較するとまだ向上の余地があります。
audio = tts.infer_vc(
    spk_audio_path="examples\laffey.mp3",
    prompt_audio_path="examples\AnAn.ogg",
    prompt_audio_text="ちが……ちがう。レイア、貴様は間違っている。",
)

audio.play()
tts.audio_queue.wait()
```

#### 5. 声紋認識
```python
from gsv_tts import TTS

tts = TTS(always_load_sv=True)

# verify_speaker は 2 つのオーディオの話者特徴を比較し、同一人物かどうかを判断するために使用します。
similarity = tts.verify_speaker("examples\laffey.mp3", "examples\AnAn.ogg")
print("声紋類似度：", similarity)
```

<details>
<summary><strong>6. その他の関数インターフェース</strong></summary>

### 1. モデル管理

#### `init_language_module(languages)`
必要な言語処理モジュールを事前にロードします。

#### `load_gpt_model(model_paths)`
GPT モデルの重みを指定されたパスからメモリにロードします。

#### `load_sovits_model(model_paths)`
SoVITS モデルの重みを指定されたパスからメモリにロードします。

#### `unload_gpt_model(model_paths)` / `unload_sovits_model(model_paths)`
リソースを解放するためにメモリからモデルをアンロードします。

#### `get_gpt_list()` / `get_sovits_list()`
現在ロードされているモデルのリストを取得します。

#### `to_safetensors(checkpoint_path)`
PyTorch 形式のモデル重みファイル（.pth または .ckpt）を safetensors 形式に変換します。

### 2. オーディオキャッシュ管理

#### `cache_spk_audio(spk_audio_paths)`
音色参照オーディオデータを前処理し、キャッシュします。

#### `cache_prompt_audio(prompt_audio_paths, prompt_audio_texts, prompt_audio_languages)`
スタイル参照オーディオデータを前処理し、キャッシュします。

#### `del_spk_audio(spk_audio_paths)` / `del_prompt_audio(prompt_audio_paths)`
キャッシュからオーディオデータを削除します。

#### `get_spk_audio_list()` / `get_prompt_audio_list()`
キャッシュ内のオーディオデータリストを取得します。

</details>

## Flash Attn
**より低い遅延**と**より高いスループット**を追求する場合、`Flash Attention` サポートを有効にすることを強く推奨します。
このライブラリはコンパイル環境に特定の要件があるため、システムに応じて手動でインストールしてください：

*   **🐧 Linux / ソースコードビルド**
    *   公式リポジトリ：[Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention)

*   **🪟 Windows ユーザー**
    *   事前コンパイル済み Wheel パッケージ：[lldacing/flash-attention-windows-wheel](https://huggingface.co/lldacing/flash-attention-windows-wheel/tree/main)

> [!TIP]
> インストール完了後、TTS 設定で `use_flash_attn=True` を設定するだけで加速効果を楽しめます！🚀

## 謝辞 (Credits)
以下のプロジェクトに特別な感謝を表します：
- [RVC-Boss/GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS)

## ⭐ Star History

[![Star History Chart](https://api.star-history.com/svg?repos=chinokikiss/GSV-TTS-Lite&type=Date)](https://star-history.com/#chinokikiss/GSV-TTS-Lite&Date)
