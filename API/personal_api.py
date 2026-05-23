"""
个人化应用 API
核心目标：追求简单、功能全的交互体验
调用方式：支持 infer_stream、infer_batched 两种推理模式，中间无需复杂的调度逻辑
"""

import sys
import argparse
from pathlib import Path
import asyncio
import base64
import json
import logging
import os
import subprocess
import tempfile
import uuid
import wave
from io import BytesIO
from typing import Optional, Union, Any

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import httpx
import numpy as np
import soundfile as sf
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, FileResponse, Response, JSONResponse
from pydantic import BaseModel, Field

from gsv_tts import TTS

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

app = FastAPI(
    title="GSV-TTS 个人化应用 API",
    description="简单、功能全的TTS API，支持流式和批量两种推理模式",
    version="1.0"
)

output_dir = project_root / "output"
output_dir.mkdir(exist_ok=True)

tts: Optional[TTS] = None
asr = None
temp_dir = tempfile.mkdtemp(prefix="gsv_tts_personal_")
use_asr_flag = True


def is_url(path: str) -> bool:
    return path.startswith("http://") or path.startswith("https://")


def resolve_audio_path(path: str) -> str:
    if is_url(path):
        return path
    p = Path(path)
    if p.is_absolute():
        return path
    resolved = project_root / path
    if resolved.exists():
        return str(resolved)
    return path


async def download_audio(url: str) -> str:
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.get(url, follow_redirects=True)
        response.raise_for_status()
    
    ext = ".wav"
    content_type = response.headers.get("content-type", "")
    if "mp3" in content_type or url.lower().endswith(".mp3"):
        ext = ".mp3"
    elif "ogg" in content_type or url.lower().endswith(".ogg"):
        ext = ".ogg"
    elif "flac" in content_type or url.lower().endswith(".flac"):
        ext = ".flac"
    
    temp_path = os.path.join(temp_dir, f"download_{uuid.uuid4().hex}{ext}")
    with open(temp_path, "wb") as f:
        f.write(response.content)
    
    logging.info(f"下载音频到: {temp_path}")
    return temp_path


def transcribe_audio(audio_path: str) -> str:
    global asr
    if asr is None:
        raise HTTPException(status_code=500, detail="ASR模型未启用，请提供 prompt_text")
    
    results = asr.transcribe(audio_path)
    if results and len(results) > 0:
        result = results[0]
        if hasattr(result, 'text'):
            text = result.text
        elif isinstance(result, dict):
            text = result.get("text", "")
        else:
            text = str(result)
        logging.info(f"ASR识别结果: {text}")
        return text
    return ""


def pack_raw(io_buffer: BytesIO, data: np.ndarray):
    audio = np.asarray(data)
    if audio.dtype != np.int16:
        audio = np.clip(audio, -1.0, 1.0)
        audio = (audio * 32767).astype(np.int16)
    io_buffer.write(audio.tobytes())
    io_buffer.seek(0)
    return io_buffer


def pack_wav(io_buffer: BytesIO, data: np.ndarray, rate: int):
    sf.write(io_buffer, np.asarray(data), rate, format="wav")
    io_buffer.seek(0)
    return io_buffer


def pack_ogg(io_buffer: BytesIO, data: np.ndarray, rate: int):
    with sf.SoundFile(io_buffer, mode="w", samplerate=rate, channels=1, format="ogg") as audio_file:
        audio_file.write(np.asarray(data))
    io_buffer.seek(0)
    return io_buffer


def pack_aac(io_buffer: BytesIO, data: np.ndarray, rate: int):
    audio = np.asarray(data)
    if audio.dtype != np.int16:
        audio = np.clip(audio, -1.0, 1.0)
        audio = (audio * 32767).astype(np.int16)
    process = subprocess.Popen(
        [
            "ffmpeg",
            "-f",
            "s16le",
            "-ar",
            str(rate),
            "-ac",
            "1",
            "-i",
            "pipe:0",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-vn",
            "-f",
            "adts",
            "pipe:1",
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    out, _ = process.communicate(input=audio.tobytes())
    io_buffer.write(out)
    io_buffer.seek(0)
    return io_buffer


def pack_audio(data: np.ndarray, rate: int, media_type: str) -> bytes:
    io_buffer = BytesIO()
    if media_type == "wav":
        return pack_wav(io_buffer, data, rate).getvalue()
    if media_type == "ogg":
        return pack_ogg(io_buffer, data, rate).getvalue()
    if media_type == "aac":
        return pack_aac(io_buffer, data, rate).getvalue()
    return pack_raw(io_buffer, data).getvalue()


def wave_header_chunk(frame_input=b"", channels=1, sample_width=2, sample_rate=32000):
    wav_buf = BytesIO()
    with wave.open(wav_buf, "wb") as vfout:
        vfout.setnchannels(channels)
        vfout.setsampwidth(sample_width)
        vfout.setframerate(sample_rate)
        vfout.writeframes(frame_input)
    wav_buf.seek(0)
    return wav_buf.read()


async def switch_models(gpt_weights: Optional[str] = None, sovits_weights: Optional[str] = None):
    if gpt_weights:
        gpt_weights = resolve_audio_path(gpt_weights)
        for model_path in tts.get_gpt_list():
            tts.unload_gpt_model(model_path)
        tts.load_gpt_model(gpt_weights)
    if sovits_weights:
        sovits_weights = resolve_audio_path(sovits_weights)
        for model_path in tts.get_sovits_list():
            tts.unload_sovits_model(model_path)
        tts.load_sovits_model(sovits_weights)

CUT_METHOD2MINLEN = {
    "cut0": 999,
    "cut1": 50,
    "cut2": 50,
    "cut3": 20,
    "cut4": 20,
    "cut5": 10,
}

async def build_api_v2_inputs(req: dict[str, Any]):
    ref_audio_path = req.get("ref_audio_path")
    if not ref_audio_path:
        raise HTTPException(status_code=400, detail="ref_audio_path is required")

    aux_audio_paths = req.get("aux_ref_audio_paths") or []

    if aux_audio_paths:
        speaker_audio: str | dict[str, float] = {path: 1.0 for path in ([ref_audio_path] + aux_audio_paths)}
    else:
        speaker_audio = ref_audio_path

    prompt_text = req.get("prompt_text") or ""
    if prompt_text == "":
        prompt_text = transcribe_audio(ref_audio_path)
        if not prompt_text:
            raise HTTPException(status_code=400, detail="无法自动识别prompt_audio文本，请手动提供prompt_text")

    cut_method = req.get("text_split_method", "cut1")
    cut_minlen = CUT_METHOD2MINLEN[cut_method]

    return speaker_audio, ref_audio_path, prompt_text, cut_minlen


async def handle_api_v2_request(req: dict[str, Any]):
    speaker_audio, prompt_audio, prompt_text, cut_minlen = await build_api_v2_inputs(req)

    text_input = req.get("text")
    if text_input in [None, ""]:
        raise HTTPException(status_code=400, detail="text is required")

    media_type = req.get("media_type", "wav")
    if media_type not in {"wav", "raw", "ogg", "aac"}:
        raise HTTPException(status_code=400, detail=f"media_type not supported: {media_type}")

    streaming_mode = req.get("streaming_mode", False)
    stream_enabled = streaming_mode not in [False, 0, "0", None]

    batch_infer = bool(req.get("batch_infer", False))

    if batch_infer and stream_enabled:
        raise HTTPException(status_code=400, detail="batch_infer is not supported with streaming_mode")

    speed = float(req.get("speed_factor", 1.0))
    top_k = int(req.get("top_k", 15))
    top_p = float(req.get("top_p", 1.0))
    temperature = float(req.get("temperature", 1.0))
    repetition_penalty = float(req.get("repetition_penalty", 1.35))
    noise_scale = float(req.get("noise_scale", 0.5))
    overlap_len = int(req.get("overlap_length", 5))
    stream_chunk = int(req.get("min_chunk_length", 25))
    is_cut_text = req.get("text_split_method", "cut1") not in {"", "cut0", "none"}
    cut_mute = req.get("fragment_interval", 0.3)

    if batch_infer and not stream_enabled:
        audio_clips = await tts.infer_batched_async(
            spk_audio_paths=speaker_audio,
            prompt_audio_paths=prompt_audio,
            prompt_audio_texts=prompt_text,
            texts=text_input,
            return_subtitles=False,
            is_cut_text=is_cut_text,
            cut_mute=cut_mute,
            cut_minlen=cut_minlen,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            noise_scale=noise_scale,
            speed=speed,
        )
        if not audio_clips:
            raise HTTPException(status_code=500, detail="tts failed")
        sample_rate = audio_clips[0].samplerate
        merged_audio = np.concatenate([np.asarray(clip.audio_data) for clip in audio_clips])
        return Response(pack_audio(merged_audio, sample_rate, media_type), media_type=f"audio/{media_type}")

    if stream_enabled:
        mode_value = 1 if streaming_mode is True else int(streaming_mode)
        if mode_value not in {1, 2, 3}:
            raise HTTPException(status_code=400, detail="streaming_mode must be 0/1/2/3 or bool")

        async def stream_generator():
            first_chunk = True
            async for clip in tts.infer_stream_async(
                spk_audio_path=speaker_audio,
                prompt_audio_path=prompt_audio,
                prompt_audio_text=prompt_text,
                text=text_input,
                is_cut_text=is_cut_text,
                cut_mute=cut_mute,
                cut_minlen=cut_minlen,
                stream_mode="token",
                stream_chunk=max(1, stream_chunk),
                overlap_len=max(1, overlap_len),
                boost_first_chunk=(mode_value == 1),
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
                repetition_penalty=repetition_penalty,
                noise_scale=noise_scale,
                speed=speed,
                debug=False,
            ):
                if first_chunk and media_type == "wav":
                    yield wave_header_chunk(sample_rate=clip.samplerate)
                    first_chunk = False
                    yield pack_audio(clip.audio_data, clip.samplerate, "raw")
                else:
                    yield pack_audio(clip.audio_data, clip.samplerate, "raw" if media_type == "wav" else media_type)

        return StreamingResponse(stream_generator(), media_type=f"audio/{media_type}")

    clip = await tts.infer_async(
        spk_audio_path=speaker_audio,
        prompt_audio_path=prompt_audio,
        prompt_audio_text=prompt_text,
        text=text_input,
        return_subtitles=False,
        top_k=top_k,
        top_p=top_p,
        temperature=temperature,
        repetition_penalty=repetition_penalty,
        noise_scale=noise_scale,
        speed=speed,
    )
    return Response(pack_audio(clip.audio_data, clip.samplerate, media_type), media_type=f"audio/{media_type}")


class TTSStreamRequest(BaseModel):
    text: str = Field(..., description="要合成的文本")
    speaker_audio: str = Field(..., description="说话人参考音频路径或URL")
    prompt_audio: str = Field(..., description="提示音频路径或URL")
    prompt_text: Optional[str] = Field(None, description="提示音频文本，为空时自动ASR识别")
    
    is_cut_text: bool = Field(True, description="是否按标点切分文本")
    cut_minlen: int = Field(10, description="文本切分最小长度")
    cut_mute: float = Field(0.3, description="切分后的静音时长(秒)")
    
    stream_mode: str = Field("token", description="流式模式: token 或 sentence")
    stream_chunk: int = Field(25, description="token模式下每次生成的token数")
    overlap_len: int = Field(5, description="重叠长度，用于平滑拼接")
    boost_first_chunk: bool = Field(True, description="是否加速首个chunk生成")
    
    top_k: int = Field(15, description="GPT采样top_k")
    top_p: float = Field(1.0, description="GPT采样top_p")
    temperature: float = Field(1.0, description="GPT采样温度")
    repetition_penalty: float = Field(1.35, description="重复惩罚")
    noise_scale: float = Field(0.5, description="噪声强度")
    speed: float = Field(1.0, description="语速")


class TTSBatchedRequest(BaseModel):
    texts: list[str] = Field(..., description="要合成的文本列表")
    speaker_audio: str = Field(..., description="说话人参考音频路径或URL")
    prompt_audio: str = Field(..., description="提示音频路径或URL")
    prompt_text: Optional[str] = Field(None, description="提示音频文本，为空时自动ASR识别")
    
    is_cut_text: bool = Field(True, description="是否按标点切分文本")
    cut_minlen: int = Field(10, description="文本切分最小长度")
    cut_mute: float = Field(0.3, description="切分后的静音时长(秒)")
    
    return_subtitles: bool = Field(False, description="是否返回字幕时间戳")
    
    top_k: int = Field(15, description="GPT采样top_k")
    top_p: float = Field(1.0, description="GPT采样top_p")
    temperature: float = Field(1.0, description="GPT采样温度")
    repetition_penalty: float = Field(1.35, description="重复惩罚")
    noise_scale: float = Field(0.5, description="噪声强度")
    speed: float = Field(1.0, description="语速")


class APIV2Request(BaseModel):
    text: Union[str, list[str], None] = None
    text_lang: Optional[str] = None
    ref_audio_path: Optional[str] = None
    aux_ref_audio_paths: Optional[list[str]] = None
    prompt_lang: Optional[str] = None
    prompt_text: Optional[str] = ""
    top_k: int = 15
    top_p: float = 1.0
    temperature: float = 1.0
    text_split_method: str = "cut5"
    batch_size: int = 1
    batch_threshold: float = 0.75
    split_bucket: bool = True
    speed_factor: float = 1.0
    fragment_interval: float = 0.3
    seed: int = -1
    media_type: str = "wav"
    streaming_mode: Union[bool, int] = False
    parallel_infer: bool = True
    repetition_penalty: float = 1.35
    sample_steps: int = 32
    super_sampling: bool = False
    overlap_length: int = 5
    min_chunk_length: int = 25
    batch_infer: bool = False


@app.on_event("startup")
async def startup_event():
    global tts, asr
    print("正在加载 TTS 模型...")
    
    max_cache_len = 1024
    batch_sizes = [1, 4, 8]
    cache_lens = []
    length = 512
    while length <= max_cache_len:
        cache_lens.append(length)
        length *= 2
    gpt_cache = [(b, c) for b in batch_sizes for c in cache_lens]
    
    tts = TTS(
        models_dir=models_dir,
        gpt_cache=gpt_cache,
        sovits_cache=[50],
    )
    print("TTS 模型加载完成！")
    
    if use_asr_flag:
        try:
            import torch
            from huggingface_hub import snapshot_download
            
            local_model_path = models_dir / "qwen3_asr"
            repo_id = "Qwen/Qwen3-ASR-0.6B"
            
            if not (local_model_path.exists() and (local_model_path / "config.json").exists()):
                print(f"本地未找到ASR模型，正在下载: {repo_id}")
                snapshot_download(
                    repo_id=repo_id,
                    local_dir=str(local_model_path),
                    local_dir_use_symlinks=False,
                )
                print("ASR模型下载完成！")
            
            from qwen_asr import Qwen3ASRModel
            print("正在加载 ASR 模型...")
            asr = Qwen3ASRModel.from_pretrained(
                str(local_model_path),
                dtype=torch.bfloat16,
                device_map="cuda:0",
                local_files_only=True
            )
            print("ASR 模型加载完成！")
        except Exception as e:
            print(f"ASR 模型加载失败: {e}")
            print("提示：如果没有提供 prompt_text，请求将会失败")
            asr = None
    else:
        print("ASR 模型已禁用")


@app.get("/")
async def root():
    return {
        "message": "GSV-TTS 个人化应用 API",
        "version": "1.0",
        "endpoints": {
            "stream": "/tts/stream - 流式推理 (SSE)",
            "batched": "/tts/batched - 批量推理",
            "audio": "/audio/{filename} - 获取音频文件"
        },
        "features": {
            "url_support": True,
            "auto_asr": asr is not None
        }
    }

# 兼容官方api_v2的调用格式，不和本项目兼容的参数默认被忽略
@app.get("/tts")
async def tts_v2_get(
    text: str = None,
    text_lang: str = None,
    ref_audio_path: str = None,
    aux_ref_audio_paths: list[str] = None,
    prompt_lang: str = None,
    prompt_text: str = "",
    top_k: int = 15,
    top_p: float = 1.0,
    temperature: float = 1.0,
    text_split_method: str = "cut5",
    batch_size: int = 1,
    batch_threshold: float = 0.75,
    split_bucket: bool = True,
    speed_factor: float = 1.0,
    fragment_interval: float = 0.3,
    seed: int = -1,
    media_type: str = "wav",
    streaming_mode: Union[bool, int] = False,
    parallel_infer: bool = True,
    repetition_penalty: float = 1.35,
    sample_steps: int = 32,
    super_sampling: bool = False,
    overlap_length: int = 5,
    min_chunk_length: int = 25,
    batch_infer: bool = False,
):
    req = {
        "text": text,
        "text_lang": text_lang,
        "ref_audio_path": ref_audio_path,
        "aux_ref_audio_paths": aux_ref_audio_paths,
        "prompt_lang": prompt_lang,
        "prompt_text": prompt_text,
        "top_k": top_k,
        "top_p": top_p,
        "temperature": temperature,
        "text_split_method": text_split_method,
        "batch_size": batch_size,
        "batch_threshold": batch_threshold,
        "split_bucket": split_bucket,
        "speed_factor": speed_factor,
        "fragment_interval": fragment_interval,
        "seed": seed,
        "media_type": media_type,
        "streaming_mode": streaming_mode,
        "parallel_infer": parallel_infer,
        "repetition_penalty": repetition_penalty,
        "sample_steps": sample_steps,
        "super_sampling": super_sampling,
        "overlap_length": overlap_length,
        "min_chunk_length": min_chunk_length,
        "batch_infer": batch_infer,
    }
    return await handle_api_v2_request(req)


@app.post("/tts")
async def tts_v2_post(request: APIV2Request):
    req = request.model_dump() if hasattr(request, "model_dump") else request.dict()
    return await handle_api_v2_request(req)


@app.get("/set_gpt_weights")
async def set_gpt_weights(weights_path: str = None):
    try:
        if weights_path in ["", None]:
            return JSONResponse(status_code=400, content={"message": "gpt weight path is required"})
        await switch_models(gpt_weights=weights_path)
        return JSONResponse(status_code=200, content={"message": "success"})
    except Exception as e:
        return JSONResponse(status_code=400, content={"message": "change gpt weight failed", "Exception": str(e)})


@app.get("/set_sovits_weights")
async def set_sovits_weights(weights_path: str = None):
    try:
        if weights_path in ["", None]:
            return JSONResponse(status_code=400, content={"message": "sovits weight path is required"})
        await switch_models(sovits_weights=weights_path)
        return JSONResponse(status_code=200, content={"message": "success"})
    except Exception as e:
        return JSONResponse(status_code=400, content={"message": "change sovits weight failed", "Exception": str(e)})


@app.post("/tts/stream")
async def tts_stream(request: TTSStreamRequest):
    """
    流式推理API - 使用SSE实时推送音频片段
    
    适用场景：
    - 实时对话
    - 长文本生成
    - 需要低延迟响应
    
    返回格式：Server-Sent Events (SSE)
    - event: audio - 音频片段(base64编码)
    - event: subtitle - 字幕信息
    - event: done - 生成完成
    - event: error - 错误信息
    """
    try:
        speaker_audio = resolve_audio_path(request.speaker_audio)
        prompt_audio = resolve_audio_path(request.prompt_audio)
        prompt_text = request.prompt_text
        
        if is_url(speaker_audio):
            speaker_audio = await download_audio(speaker_audio)
        
        if is_url(prompt_audio):
            prompt_audio = await download_audio(prompt_audio)
        
        if prompt_text is None or prompt_text == "":
            prompt_text = transcribe_audio(prompt_audio)
            if not prompt_text:
                raise HTTPException(
                    status_code=400, 
                    detail="无法自动识别prompt_audio文本，请手动提供prompt_text"
                )
        
        final_prompt_text = prompt_text
        final_speaker_audio = speaker_audio
        final_prompt_audio = prompt_audio
        
        async def generate():
            try:
                loop = asyncio.get_running_loop()
                
                def stream_infer():
                    return list(tts.infer_stream(
                        spk_audio_path=final_speaker_audio,
                        prompt_audio_path=final_prompt_audio,
                        prompt_audio_text=final_prompt_text,
                        text=request.text,
                        is_cut_text=request.is_cut_text,
                        cut_minlen=request.cut_minlen,
                        cut_mute=request.cut_mute,
                        stream_mode=request.stream_mode,
                        stream_chunk=request.stream_chunk,
                        overlap_len=request.overlap_len,
                        boost_first_chunk=request.boost_first_chunk,
                        top_k=request.top_k,
                        top_p=request.top_p,
                        temperature=request.temperature,
                        repetition_penalty=request.repetition_penalty,
                        noise_scale=request.noise_scale,
                        speed=request.speed,
                        debug=False,
                    ))
                
                clips = await loop.run_in_executor(None, stream_infer)
                
                total_len = 0
                for clip in clips:
                    audio_bytes = clip.audio_data.tobytes()
                    audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')
                    
                    total_len += len(clip.audio_data)
                    
                    chunk_data = {
                        "audio": audio_b64,
                        "sample_rate": clip.samplerate,
                        "duration": clip.audio_len_s,
                        "subtitles": clip.subtitles,
                        "text": clip.orig_text
                    }
                    
                    yield f"event: audio\ndata: {json.dumps(chunk_data, ensure_ascii=False)}\n\n"
                
                yield f"event: done\ndata: {json.dumps({'total_duration': total_len / 32000}, ensure_ascii=False)}\n\n"
                
            except Exception as e:
                logging.error(f"流式推理错误: {e}")
                yield f"event: error\ndata: {json.dumps({'error': str(e)}, ensure_ascii=False)}\n\n"
        
        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/tts/batched")
async def tts_batched(request: TTSBatchedRequest):
    """
    批量推理API - 一次请求生成多个音频
    
    适用场景：
    - 批量生成
    - 离线处理
    - 不需要实时响应
    
    返回格式：JSON
    - success: 是否成功
    - count: 生成的音频数量
    - filenames: 生成的音频文件名列表
    - subtitles: 字幕信息(可选)
    """
    try:
        speaker_audio = resolve_audio_path(request.speaker_audio)
        prompt_audio = resolve_audio_path(request.prompt_audio)
        prompt_text = request.prompt_text
        
        if is_url(speaker_audio):
            speaker_audio = await download_audio(speaker_audio)
        
        if is_url(prompt_audio):
            prompt_audio = await download_audio(prompt_audio)
        
        if prompt_text is None or prompt_text == "":
            prompt_text = transcribe_audio(prompt_audio)
            if not prompt_text:
                raise HTTPException(
                    status_code=400, 
                    detail="无法自动识别prompt_audio文本，请手动提供prompt_text"
                )
        
        audio_clips = await tts.infer_batched_async(
            spk_audio_paths=speaker_audio,
            prompt_audio_paths=prompt_audio,
            prompt_audio_texts=prompt_text,
            texts=request.texts,
            return_subtitles=request.return_subtitles,
            is_cut_text=request.is_cut_text,
            cut_minlen=request.cut_minlen,
            cut_mute=request.cut_mute,
            top_k=request.top_k,
            top_p=request.top_p,
            temperature=request.temperature,
            repetition_penalty=request.repetition_penalty,
            noise_scale=request.noise_scale,
            speed=request.speed,
        )
        
        filenames = []
        subtitles_list = []
        
        for clip in audio_clips:
            filename = f"tts_{uuid.uuid4().hex[:8]}.wav"
            output_path = output_dir / filename
            clip.save(str(output_path))
            filenames.append(filename)
            
            if request.return_subtitles and clip.subtitles:
                subtitles_list.append(clip.subtitles)
        
        result = {
            "success": True,
            "count": len(audio_clips),
            "filenames": filenames,
            "prompt_text_used": prompt_text,
        }
        
        if request.return_subtitles:
            result["subtitles"] = subtitles_list
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/audio/{filename}")
async def get_audio(filename: str):
    """获取生成的音频文件"""
    file_path = output_dir / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="文件未找到")
    return FileResponse(file_path, media_type="audio/wav")


if __name__ == "__main__":
    import uvicorn
    parser = argparse.ArgumentParser(description="GSV-TTS personal API")
    parser.add_argument("--models_dir", type=str, default="models", help="预训练模型目录")
    parser.add_argument("-p", "--port", type=int, default=9880, help="server port")
    parser.add_argument("--use_asr", action="store_true", help="使用ASR自动识别音频文本")
    args = parser.parse_args()
    models_dir = args.models_dir
    use_asr_flag = args.use_asr
    uvicorn.run(app, host="0.0.0.0", port=args.port)
