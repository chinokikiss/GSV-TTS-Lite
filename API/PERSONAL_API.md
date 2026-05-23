# GSV-TTS 个人化应用 API

简单、功能全的 TTS API，支持流式和批量两种推理模式。

## 快速开始

```bash
cd API
python personal_api.py
```

服务启动后访问 http://localhost:8000/docs 查看交互式文档。

## API 端点

### 1. 流式推理 `/tts/stream`

使用 SSE (Server-Sent Events) 实时推送音频片段，适用于实时对话、长文本生成等需要低延迟响应的场景。

**请求参数：**

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| text | string | 是 | - | 要合成的文本 |
| speaker_audio | string | 是 | - | 说话人参考音频路径或URL |
| prompt_audio | string | 是 | - | 提示音频路径或URL |
| prompt_text | string | 否 | - | 提示音频文本，为空时自动ASR识别 |
| is_cut_text | bool | 否 | true | 是否按标点切分文本 |
| cut_minlen | int | 否 | 10 | 文本切分最小长度 |
| cut_mute | float | 否 | 0.3 | 切分后的静音时长(秒) |
| stream_mode | string | 否 | "token" | 流式模式: token 或 sentence |
| stream_chunk | int | 否 | 25 | token模式下每次生成的token数 |
| overlap_len | int | 否 | 5 | 重叠长度，用于平滑拼接 |
| boost_first_chunk | bool | 否 | true | 是否加速首个chunk生成 |
| top_k | int | 否 | 15 | GPT采样top_k |
| top_p | float | 否 | 1.0 | GPT采样top_p |
| temperature | float | 否 | 1.0 | GPT采样温度 |
| repetition_penalty | float | 否 | 1.35 | 重复惩罚 |
| noise_scale | float | 否 | 0.5 | 噪声强度 |
| speed | float | 否 | 1.0 | 语速 |

**返回格式 (SSE)：**

```
event: audio
data: {"audio": "<base64>", "sample_rate": 32000, "duration": 0.5, "subtitles": [...], "text": "..."}

event: done
data: {"total_duration": 5.2}

event: error
data: {"error": "错误信息"}
```

**调用示例：**

```python
import httpx
import json
import base64

async def stream_tts():
    async with httpx.AsyncClient(timeout=60.0) as client:
        async with client.stream("POST", "http://localhost:8000/tts/stream", json={
            "text": "你好，这是一段测试文本。今天天气真不错。",
            "speaker_audio": "examples/AnAn.ogg",
            "prompt_audio": "examples/AnAn.ogg",
            "prompt_text": "ちが……ちがう。レイア、貴様は間違っている。"
        }) as response:
            async for line in response.aiter_lines():
                if line.startswith("event: audio"):
                    continue
                if line.startswith("data: "):
                    data = json.loads(line[6:])
                    if "audio" in data:
                        audio_bytes = base64.b64decode(data["audio"])
                        print(f"收到音频片段: {data['duration']:.2f}秒")
```

### 2. 批量推理 `/tts/batched`

一次请求生成多个音频，适用于批量生成、离线处理等不需要实时响应的场景。

**请求参数：**

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| texts | [string] | 是 | - | 要合成的文本列表 |
| speaker_audio | string | 是 | - | 说话人参考音频路径或URL |
| prompt_audio | string | 是 | - | 提示音频路径或URL |
| prompt_text | string | 否 | - | 提示音频文本，为空时自动ASR识别 |
| is_cut_text | bool | 否 | true | 是否按标点切分文本 |
| cut_minlen | int | 否 | 10 | 文本切分最小长度 |
| cut_mute | float | 否 | 0.3 | 切分后的静音时长(秒) |
| return_subtitles | bool | 否 | false | 是否返回字幕时间戳 |
| top_k | int | 否 | 15 | GPT采样top_k |
| top_p | float | 否 | 1.0 | GPT采样top_p |
| temperature | float | 否 | 1.0 | GPT采样温度 |
| repetition_penalty | float | 否 | 1.35 | 重复惩罚 |
| noise_scale | float | 否 | 0.5 | 噪声强度 |
| speed | float | 否 | 1.0 | 语速 |

**返回格式 (JSON)：**

```json
{
    "success": true,
    "count": 2,
    "filenames": ["tts_abc12345.wav", "tts_def67890.wav"],
    "prompt_text_used": "ちが……ちがう。レイア、貴様は間違っている。",
    "subtitles": [
        [{"text": "你好", "start_s": 0.0, "end_s": 0.3}],
        [{"text": "世界", "start_s": 0.0, "end_s": 0.4}]
    ]
}
```

**调用示例：**

```python
import httpx

response = httpx.post("http://localhost:8000/tts/batched", json={
    "texts": ["第一段文本。", "第二段文本。"],
    "speaker_audio": "examples/AnAn.ogg",
    "prompt_audio": "examples/AnAn.ogg",
    "prompt_text": "ちが……ちがう。レイア、貴様は間違っている。",
    "return_subtitles": True
}, timeout=60.0)

result = response.json()
print(f"生成了 {result['count']} 个音频文件")
for filename in result["filenames"]:
    print(f"文件: {filename}")
```

### 3. 获取音频 `/audio/{filename}`

获取生成的音频文件。

**示例：**
```
GET /audio/tts_abc12345.wav
```

## 功能特性

### 外链音频URL支持

所有音频参数支持HTTP/HTTPS URL，API会自动下载：

```json
{
    "speaker_audio": "https://example.com/speaker.wav",
    "prompt_audio": "https://example.com/prompt.mp3"
}
```

### ASR自动识别

当 `prompt_text` 为空时，自动使用ASR模型识别提示音频的文本内容。

可通过环境变量控制：
```bash
USE_ASR=true python personal_api.py   # 启用ASR (默认)
USE_ASR=false python personal_api.py  # 禁用ASR
```

### 流式模式选择

- **token模式**：按token数量切分，延迟更低，适合实时对话
- **sentence模式**：按句子切分，音频更连贯，适合长文本朗读

## 两种模式对比

| 特性 | Stream | Batched |
|------|--------|---------|
| 适用场景 | 实时对话、长文本 | 批量生成、离线处理 |
| 响应方式 | SSE实时推送 | 一次性返回 |
| 首字延迟 | 低 | 高 |
| GPU利用率 | 中 | 高 |
| 音频返回 | base64编码 | 文件 |

## 环境要求

- Python 3.10+
- CUDA 11.x+ (推荐)
- 依赖见 requirements.txt
