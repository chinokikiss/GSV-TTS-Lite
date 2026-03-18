import re
import os
import time
import json
import uuid
import torch
import logging
import argparse
import gradio as gr
import numpy as np
from datetime import datetime
from pedalboard import Pedalboard, Compressor, HighpassFilter, PeakFilter, Reverb, Gain
import pyloudnorm as pyln
from pathlib import Path
from huggingface_hub import snapshot_download
import platform
if platform.system() == "Windows":
    import psutil
    p = psutil.Process(os.getpid())
    p.nice(psutil.HIGH_PRIORITY_CLASS)

from gsv_tts import TTS, AudioClip

logging.getLogger('asyncio').setLevel(logging.CRITICAL)
logging.getLogger('httpx').setLevel(logging.CRITICAL)


# Copied from https://github.com/Icelinea/BetterAIVoice/blob/main/process.py
def enhance_audio(audio_data, sample_rate):
    # 1. 构建美化链
    board = Pedalboard([
        # 去除低频浑浊
        HighpassFilter(cutoff_frequency_hz=80),
        
        # 增加女声磁性：250Hz-350Hz 提升
        PeakFilter(cutoff_frequency_hz=300, gain_db=2.5, q=1.0),
        
        # 压制 AI 齿音：6kHz-8kHz 微微削减
        PeakFilter(cutoff_frequency_hz=7000, gain_db=-3.0, q=2.0),
        
        # 稳定动态：防止有声书音量忽大忽小
        Compressor(threshold_db=-18, ratio=3.5),
        
        # 赋予录音棚空间感
        # 使用内建 Reverb 模拟 Ambience 预设 (Mix 3%, 极小衰减)
        Reverb(room_size=0.1, dry_level=0.97, wet_level=0.03, damping=0.5),
        
        # 最终增益补偿
        Gain(gain_db=2)
    ])

    # 2. 执行处理
    effected = board(audio_data, sample_rate)
    input_for_norm = effected.reshape(-1, 1)

    # 3. 响度标准化
    # 测量当前响度
    meter = pyln.Meter(sample_rate) 
    loudness = meter.integrated_loudness(input_for_norm)
    # 将响度统一调整至 -18.0 LUFS (播客标准)
    normalized_audio = pyln.normalize.loudness(input_for_norm, loudness, -18.0).T

    return normalized_audio.flatten()


def upload_gpt(new_gpt):
    if not new_gpt is None:
        for gpt in tts.get_gpt_list():
            tts.unload_gpt_model(gpt)
        
        tts.load_gpt_model(new_gpt.strip('"“”'))

def upload_sovits(new_sovits):
    if not new_sovits is None:
        for sovits in tts.get_sovits_list():
            tts.unload_sovits_model(sovits)
        
        tts.load_sovits_model(new_sovits.strip('"“”'))


def update_spk_weights(files, weights):
    if not files:
        return "1.0"

    weights = re.split(r'[：:]\s*', weights)
    weights = [weight for weight in weights if weight]

    f_len = len(files)
    w_len = len(weights)
    if f_len <= w_len:
        new_weights = weights[:f_len]
    else:
        new_weights = weights + ["1.0"]*(f_len-w_len)

    return ": ".join(new_weights)


ignore_transcribe = False
def audio_transcriber(audio_file):
    global ignore_transcribe

    if ignore_transcribe:
        ignore_transcribe = False
        audio_file = None

    if not audio_file is None and not asr is None:
        results = asr.transcribe(audio_file)
        text = results[0].text

        return text
    
    return gr.update()


def parse_tagged_text(text):
    parts = re.split(r'(<(?!(?:break))[^>]+>.*?</[^>]+>)', text)

    cut_texts = []
    tags = []
    for part in parts:
        if not part: continue

        match = re.search(r'<([^>]+)>(.*?)</[^>]+>', part)
        if match:
            tag_name = match.group(1)
            content = match.group(2)
            sub_parts = re.split(r'(<break:.*?>)', content)
            sub_parts = [p for p in sub_parts if p]
            tags.extend([tag_name]*len(sub_parts))
        else:    
            sub_parts = re.split(r'(<break:.*?>)', part)
            sub_parts = [p for p in sub_parts if p]
            tags.extend([None]*len(sub_parts))

        cut_texts.extend(sub_parts)
    
    for i in range(len(cut_texts)-1, -1, -1):
        if len(re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9]', '', cut_texts[i])) == 0:
            cut_texts.pop(i)
            tags.pop(i)

    return cut_texts, tags

def parse_speaker_weights(multi_spk_files, spk_weights):
    spk_weights = re.split(r'[：:]\s*', spk_weights)
    spk_audio = {multi_spk_files[i]: float(item) for i, item in enumerate(spk_weights) if item}
    return spk_audio


# 预设存储
presets = {}
def save_preset(name, prompt_audio, prompt_text, multi_spk_files, spk_weights):
    if not name:
        return gr.update(choices=list(presets.keys())), "请输入预设名称"
    presets[name] = {
        "prompt_audio": prompt_audio,
        "prompt_text": prompt_text,
        "multi_spk_files": multi_spk_files,
        "spk_weights": spk_weights
    }
    return gr.update(choices=list(presets.keys()), value=name), f"预设 '{name}' 已保存"

def load_preset(name):
    global ignore_transcribe
    ignore_transcribe = True

    if name not in presets:
        return None, "", None, "1.0"
    p = presets[name]
    return p["prompt_audio"], p["prompt_text"], p["multi_spk_files"], p["spk_weights"]


def vc_request(
    multi_spk_files, spk_weights,
    prompt_audio, prompt_text,
):
    try:
        start_time = time.time()

        audio = tts.infer_vc(
            spk_audio_path=parse_speaker_weights(multi_spk_files, spk_weights),
            prompt_audio_path=prompt_audio,
            prompt_audio_text=prompt_text,
        )

        end_time = time.time()

        infer_duration = end_time - start_time

        msg = (
            f"成功！\n"
            f"音频时长: {audio.audio_len_s:.2f}s | "
            f"推理耗时: {infer_duration:.2f}s"
        )

        return (audio.samplerate, audio.audio_data), msg

    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, f"异常: {str(e)}"

def tts_request(
    multi_spk_files, spk_weights,
    prompt_audio, prompt_text,
    text,
    top_k, top_p, temperature, rep_penalty, noise_scale, speed,
    enable_enhance,
    is_cut_text, cut_minlen, cut_mute, cut_mute_scale_map,
    sovits_batch_size,
):
    try:
        start_time = time.time()

        spk_audio = parse_speaker_weights(multi_spk_files, spk_weights)

        #cut_punds = set(cut_punds)
        cut_mute_scale_map = json.loads(cut_mute_scale_map)

        cut_texts, tags = parse_tagged_text(text)

        orig_idx = []
        spk_audio_paths = []
        prompt_audio_paths = []
        prompt_audio_texts = []
        texts = []

        for i in range(len(cut_texts)):
            result = re.search(r'<break:(.*?)/>', cut_texts[i])
            if result:
                cut_texts[i] = float(result.group(1))
                tags[i] = 'break'
            else:
                orig_idx.append(i)

                if tags[i] is None or tags[i] not in presets:
                    spk_audio_paths.append(spk_audio)
                    prompt_audio_paths.append(prompt_audio)
                    prompt_audio_texts.append(prompt_text)
                else:
                    p = presets[tags[i]]
                    spk_audio_paths.append(parse_speaker_weights(p["multi_spk_files"], p["spk_weights"]))
                    prompt_audio_paths.append(p["prompt_audio"])
                    prompt_audio_texts.append(p["prompt_text"])
                
                texts.append(cut_texts[i])
                    
        audios = tts.infer_batched(
            spk_audio_paths=spk_audio_paths,
            prompt_audio_paths=prompt_audio_paths,
            prompt_audio_texts=prompt_audio_texts,
            texts=texts,
            is_cut_text=is_cut_text,
            #cut_punds=cut_punds,
            cut_minlen=cut_minlen,
            cut_mute=cut_mute,
            cut_mute_scale_map=cut_mute_scale_map,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            repetition_penalty=rep_penalty,
            noise_scale=noise_scale,
            speed=speed,
            sovits_batch_size=sovits_batch_size,
        )

        samplerate = audios[0].samplerate

        audio_data = []
        audio_len_s = 0
        for i in range(len(cut_texts)):
            if tags[i] == "break":
                audio_data.append(np.zeros((int(cut_texts[i] * samplerate),)))
                audio_len_s += cut_texts[i]
            else:
                tmp_audio = audios[orig_idx.index(i)]
                audio_data.append(tmp_audio.audio_data)
                audio_len_s += tmp_audio.audio_len_s
        
        audio_data = np.concatenate(audio_data)
        
        audio = AudioClip(None, audio_data, samplerate, audio_len_s, None, None)
        
        end_time = time.time()
        
        if enable_enhance:
            audio.audio_data = enhance_audio(audio.audio_data, audio.samplerate)

        infer_duration = end_time - start_time
        rtf = infer_duration / audio.audio_len_s

        msg = (
            f"成功！\n"
            f"音频时长: {audio.audio_len_s:.2f}s | "
            f"推理耗时: {infer_duration:.2f}s | "
            f"RTF: {rtf:.3f}"
        )

        filename = f"result_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:4]}.wav"
        save_path = os.path.join(HISTORY_DIR, filename)
        audio.save(save_path)
        history_entry = [datetime.now().strftime("%H:%M:%S"), text[:20] + "...", save_path]

        return (audio.samplerate, audio.audio_data), msg, history_entry
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, f"异常: {str(e)}", None


# --- UI 界面 ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# GSV-TTS")

    with gr.Tabs():
        with gr.TabItem("文本转语音 (TTS)"):

            history_state = gr.State([])
            
            with gr.Group():
                gr.Markdown("### 第一步：加载模型文件")
                with gr.Row():
                    gpt_path = gr.Textbox(label="1. GPT 模型路径 (.ckpt)", placeholder="留空则默认模型", scale=1)
                    sovits_path = gr.Textbox(label="2. SoVITS 模型路径 (.pth)", placeholder="留空则默认模型", scale=1)

            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 第二步：合成内容（支持多说话人，支持停顿标签）")
                    text = gr.Textbox(label="合成目标文本", lines=5, value="谁罕见?啊？骂谁罕见！")
                    enable_enhance = gr.Checkbox(label="启用音频增强", value=True)
                    
                    with gr.Accordion("生成参数", open=False):
                        speed = gr.Slider(0.5, 2.0, 1.0, step=0.1, label="语速")
                        noise_scale = gr.Slider(0.1, 1.0, 0.5, step=0.05, label="噪声比例")
                        temperature = gr.Slider(0.1, 1.5, 1.0, label="温度")
                        top_k = gr.Slider(1, 50, 15, step=1, label="Top K")
                        top_p = gr.Slider(0.1, 1.0, 1.0, label="Top P")
                        rep_penalty = gr.Slider(1.0, 2.0, 1.35, label="重复惩罚")
                        sovits_batch_size = gr.Number(label="SoVITS最大并行推理大小", value=10)
                        is_cut_text = gr.Checkbox(label="是否切分文本", value=True)
                        #cut_punds = gr.Textbox(label="切分标点", value='{"。", ".", "?", "？", "!", "！", ",", "，", ":", "：", ";", "；", "、"}')
                        cut_minlen = gr.Number(label="最小切分长度", value=10)
                        cut_mute = gr.Number(label="切分静音时长(s)", value=0.2)
                        cut_mute_scale_map = gr.Textbox(label="标点静音缩放映射", value='{".": 1.5, "。": 1.5, "?": 1.5, "？": 1.5, "!": 1.5, "！": 1.5, ",": 0.8, "，": 0.8, "、": 0.6}')

                with gr.Column(scale=1):
                    gr.Markdown("### 第三步：风格与音色参考")
                    
                    with gr.Row():
                        preset_dropdown = gr.Dropdown(choices=[], label="加载预设", scale=2)
                        preset_name = gr.Textbox(label="预设名称", placeholder="保存当前设置为...", scale=2)
                        save_btn = gr.Button("💾 保存预设", scale=1)

                    with gr.Tab("风格参考"):
                        prompt_audio = gr.Audio(label="风格参考音频 (决定语气、情感)", type="filepath")
                        prompt_text = gr.Textbox(label="风格参考音频对应文本", placeholder="输入参考音频中的文本内容")

                    with gr.Tab("音色参考（支持多音色融合）"):
                        multi_spk_files = gr.File(label="可上传多个音色参考音频", file_count="multiple")
                        spk_weights = gr.Textbox(label="音色权重 (用冒号分隔)", value="1.0", placeholder="例如: 1.0: 1.0")

            with gr.Group():
                btn = gr.Button("🔥 开始语音合成", variant="primary", size="lg")
                with gr.Row():
                    with gr.Column(scale=2):
                        output_audio = gr.Audio(label="生成的音频结果")
                        log_output = gr.Textbox(label="系统状态信息")
                    
                    with gr.Column(scale=1):
                        gr.Markdown("### 🕒 最近生成历史")
                        history_display = gr.Dataset(
                            components=[gr.Textbox(visible=False)],
                            label="点击下方条目可重新加载音频",
                            samples=[],
                            type="values"
                        )


        with gr.TabItem("音色迁移 (VC)"):
            gr.Markdown("### 将一段音频的内容迁移到另一个人的音色上")
            
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("#### 1. 源音频参考")
                    vc_source_audio = gr.Audio(label="上传源音频", type="filepath")
                    vc_source_text = gr.Textbox(label="源音频对应文本", placeholder="输入源音频中的文本内容", lines=2)
                    
                    gr.Markdown("#### 2. 目标音色参考（支持多音色融合）")
                    vc_multi_spk_files = gr.File(label="可上传多个音色参考音频", file_count="multiple")
                    vc_spk_weights = gr.Textbox(label="音色权重 (用冒号分隔)", value="1.0", placeholder="例如: 1.0: 1.0")
                
                with gr.Column(scale=1):
                    gr.Markdown("#### 3. 执行与输出")
                    vc_btn = gr.Button("🚀 开始音色迁移", variant="primary", size="lg")
                    
                    vc_output_audio = gr.Audio(label="音色迁移结果", interactive=False)
                    vc_log_output = gr.Textbox(label="处理日志", lines=5)


    def update_history(history_entry, current_history):
        if history_entry is None:
            return current_history, gr.update(samples=current_history)
        
        current_history.insert(0, history_entry)
        current_history = current_history[:10]
        
        return current_history, gr.update(samples=current_history)

    def load_from_history(selected_row_data):
        if selected_row_data and len(selected_row_data) > 0:
            audio_path = selected_row_data[-1] 
            return audio_path
        return None

    save_btn.click(
        fn=save_preset,
        inputs=[preset_name, prompt_audio, prompt_text, multi_spk_files, spk_weights],
        outputs=[preset_dropdown, log_output]
    )
    
    preset_dropdown.change(
        fn=load_preset,
        inputs=[preset_dropdown],
        outputs=[prompt_audio, prompt_text, multi_spk_files, spk_weights]
    )

    multi_spk_files.change(
        fn=update_spk_weights,
        inputs=[multi_spk_files, spk_weights],
        outputs=spk_weights
    )

    vc_multi_spk_files.change(
        fn=update_spk_weights,
        inputs=[vc_multi_spk_files, vc_spk_weights],
        outputs=vc_spk_weights
    )

    prompt_audio.change(
        fn=audio_transcriber,
        inputs=prompt_audio,
        outputs=prompt_text
    )

    vc_source_audio.change(
        fn=audio_transcriber,
        inputs=vc_source_audio,
        outputs=vc_source_text
    )

    gpt_path.change(
        fn=upload_gpt,
        inputs=gpt_path
    )

    sovits_path.change(
        fn=upload_sovits,
        inputs=sovits_path
    )

    temp_history_entry = gr.State()

    btn.click(
        fn=tts_request,
        inputs=[
            multi_spk_files, spk_weights,
            prompt_audio, prompt_text,
            text,
            top_k, top_p, temperature, rep_penalty, noise_scale, speed,
            enable_enhance,
            is_cut_text, cut_minlen, cut_mute, cut_mute_scale_map,
            sovits_batch_size,
        ],
        outputs=[output_audio, log_output, temp_history_entry]
    ).then(
        fn=update_history,
        inputs=[temp_history_entry, history_state],
        outputs=[history_state, history_display]
    )

    vc_btn.click(
        fn=vc_request,
        inputs=[
            vc_multi_spk_files, vc_spk_weights,
            vc_source_audio, vc_source_text,
        ],
        outputs=[vc_output_audio, vc_log_output]
    )

    history_display.click(
        fn=load_from_history,
        inputs=[history_display],
        outputs=[output_audio]
    )


if __name__ == "__main__":
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() == 'true':
            return True
        elif v.lower() == 'false':
            return False
        
    parser = argparse.ArgumentParser(description="GSV-TTS")
    parser.add_argument("--gpt_cache_len", type=int, default=512, help="GPT KV cache 上下文长度")
    parser.add_argument("--gpt_batch_size", type=int, default=8, help="GPT 最大并行推理大小")
    parser.add_argument("--use_bert", type=str2bool, default=True, help="使用BERT提升中文语义理解能力")
    parser.add_argument("--use_flash_attn", type=str2bool, default=False, help="使用Flash Attn加速推理")
    parser.add_argument("--use_asr", type=str2bool, default=True, help="使用ASR自动识别音频文本")
    parser.add_argument("--port", type=int, default=9881, help="Gradio 端口号")
    parser.add_argument("--share", action="store_true", help="是否开启公网分享")
    
    args, _ = parser.parse_known_args()

    HISTORY_DIR = "output_history"
    os.makedirs(HISTORY_DIR, exist_ok=True)

    if args.use_asr:
        from qwen_asr import Qwen3ASRModel

        base_dir = Path(__file__).parent.resolve()
        local_model_path = base_dir / "models" / "qwen3_asr"
        
        # 可改1.7B
        repo_id = "Qwen/Qwen3-ASR-0.6B"

        # 检查本地是否已有
        if not (local_model_path.exists() and (local_model_path / "config.json").exists()):
            print(f"⬇️ 本地未找到模型，正在从 Hugging Face 下载: {repo_id}")
            print(f"📂 保存路径: {local_model_path}")
            
            try:
               
                snapshot_download(
                    repo_id=repo_id,
                    local_dir=str(local_model_path),
                    local_dir_use_symlinks=False,
                    # 下载慢的话可以用下面这个镜像
                    # endpoint="https://hf-mirror.com" 
                )
                print("✅ 模型下载完成！")
            except Exception as e:
                print(f"❌ 下载失败: {e}")
                print("💡 可以用https://hf-mirror.com镜像尝试")
                raise e
        else:
            print(f"✅ 检测到本地模型已存在: {local_model_path}")

        # 4. 加载模型 (始终使用绝对路径)
        print(f"🚀 正在加载 ASR 模型...")
        asr = Qwen3ASRModel.from_pretrained(
            str(local_model_path),  # 传入绝对路径字符串
            dtype=torch.bfloat16,
            device_map="cuda:0",
            attn_implementation="flash_attention_2" if args.use_flash_attn else None,
            local_files_only=True 
        )
    else:
        asr = None




    tts = TTS(
        gpt_cache=[(1, args.gpt_cache_len)] + [(B, args.gpt_cache_len) for B in range(4, args.gpt_batch_size-1, 4)] + [(args.gpt_batch_size, args.gpt_cache_len)],
        sovits_cache=[],
        use_bert=args.use_bert,
        use_flash_attn=args.use_flash_attn,
        models_dir="models",
    )
    
    demo.launch(
        server_port=args.port,
        share=args.share,
        inbrowser=True
    )