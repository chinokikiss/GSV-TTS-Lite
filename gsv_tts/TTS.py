import gc
import os

# 让 CUDA 算子同步执行，这样才能找到报错位置
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import torch
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(filename)s - %(levelname)s: %(message)s'
)
import torchaudio
import numpy as np
from tqdm import tqdm
from pathlib import Path
try:
    import sounddevice as sd
except:
    pass
from typing import Literal
from torch.nn import functional as F

from .Loader import get_gpt_weights, get_sovits_weights, Gpt, Sovits
from .download import check_pretrained_models, download_model
from .TextProcessor import get_phones_and_bert, cut_text, sub2text_index
from .GPT_SoVITS.Featurizer import CNHubert, CNRoberta
from .GPT_SoVITS.SV import ERes2Net
from .GPT_SoVITS.SoVITS.module.mel_processing import spectrogram_torch
from .GPT_SoVITS.G2P import text_to_phonemes
from .Player import AudioQueue, AudioClip
from .langsegmenter import LangSegmenter
from .config import Config, global_config
from .GPT_SoVITS.G2P import symbol_to_id


class TTS:
    def __init__(
        self,
        gpt_cache: list[tuple[int, int]] = [(1, 500), (4, 500)],
        sovits_cache: list[int] = [50],
        models_dir: str = None,
        device: str = None,
        is_half: bool = None,
        use_flash_attn: bool = False,
        use_bert: bool = False,
        always_load_cnhubert: bool = False,
        always_load_sv: bool = False,
    ):
        """
        Initializes GSV TTS engine.

        Args:
            gpt_cache (list[tuple[int, int]]): Static cache sizes for the GPT model's CUDA graph. Each tuple represents (batch_size, sequence_length).
            sovits_cache (list[int]): Static cache sizes for the SoVITS model's CUDA graph.
            models_dir (str): The directory path containing the pretrained model files.
            device (str): The device to run the model on.
            is_half (bool): Whether to use half-precision (FP16) inference.
            use_flash_attn (bool): Whether to enable Flash Attention for faster inference.
            use_bert (bool): Whether to use BERT for enhanced Chinese semantic understanding.
            always_load_cnhubert (bool): Whether to keep the CNHubert model loaded in VRAM. Set to True to accelerate Voice Conversion.
            always_load_sv (bool): Whether to keep the Speaker Verification model loaded in VRAM. Set to True to accelerate Speaker Verification.
        """

        self.tts_config = Config()
        
        if not device is None:
            self.tts_config.device = device
        if not is_half is None:
            self.tts_config.is_half = is_half
            self.tts_config.dtype = torch.float16 if is_half else torch.float32
        
        self.always_load_cnhubert = always_load_cnhubert
        self.always_load_sv = always_load_sv

        if models_dir is None: models_dir = Path.home() / ".cache" / "gsv"
        self.models_dir = models_dir
        if global_config.models_dir is None: global_config.models_dir = models_dir
        self.tts_config.use_flash_attn = use_flash_attn
        self.tts_config.gpt_cache = gpt_cache
        self.tts_config.sovits_cache = sovits_cache

        self.gpt_models: dict[str, Gpt] = {}
        self.sovits_models: dict[str, Sovits] = {}
        self.resample_transform_dict = {}
        self.spk_audio_cache = {}
        self.prompt_audio_cache = {}

        self.cnhubert_path = Path(self.models_dir) / "chinese-hubert-base"
        self.cnroberta_path = Path(self.models_dir) / "chinese-roberta-wwm-ext-large"
        self.sv_path = Path(self.models_dir) / "sv\pretrained_eres2netv2w24s4ep4.ckpt"
        self.default_gpt_path = Path(self.models_dir) / "s1v3.ckpt"
        self.default_sovits_path = Path(self.models_dir) / "v2Pro\s2Gv2ProPlus.pth"

        check_pretrained_models(self.models_dir)
        if use_bert and not os.path.exists(self.cnroberta_path):
            download_model(
                url="/GPTSoVITS-RT/resolve/master/chinese-roberta.zip",
                zip_filename=Path(self.models_dir) / "chinese-roberta-wwm-ext-large.zip"
            )

        if use_bert: self.tts_config.cnroberta = CNRoberta(self.cnroberta_path, self.tts_config)
        self.cnhubert_model = None
        self.sv_model = None

        if global_config.lang_segmenter is None: global_config.lang_segmenter = LangSegmenter()

        self.dict_language = {
            "中文": "all_zh",
            "粤语": "all_yue",
            "英文": "en",
            "日文": "all_ja",
            "韩文": "all_ko",
            "中英混合": "zh",
            "粤英混合": "yue",
            "日英混合": "ja",
            "韩英混合": "ko",
            "多语种混合": "auto",
            "多语种混合(粤语)": "auto_yue",
            "all_zh": "all_zh",
            "all_yue": "all_yue",
            "en": "en",
            "all_ja": "all_ja",
            "all_ko": "all_ko",
            "zh": "zh",
            "yue": "yue",
            "ja": "ja",
            "ko": "ko",
            "auto": "auto",
            "auto_yue": "auto_yue",
        }

        self.punctuation = {".", "。", "?", "？", "!", "！", ",", "，", ":", "：", ";", "；", "、"}

        self.samplerate = 32000
        self.gpt_hz = 25
        self.sovits_hz = 50

        self.audio_queue = AudioQueue()
        try:
            self.stream = sd.OutputStream(
                samplerate=self.samplerate,
                channels=1,
                callback=self.audio_queue.callback,
                dtype='float32'
            )
            self.stream.start()
        except:
            pass
        
        logging.info(f"Device: {self.tts_config.device}")
        logging.info(f"Half: {self.tts_config.is_half}, dtype: {self.tts_config.dtype}")

    
    @torch.inference_mode()
    def infer(
        self,
        spk_audio_path: str | dict,
        prompt_audio_path: str,
        prompt_audio_text: str,
        prompt_audio_language: str,
        text: str,
        text_language: str,
        top_k: int = 15,
        top_p: float = 1.0,
        temperature: float = 1.0,
        repetition_penalty: float = 1.35,
        noise_scale: float = 0.5,
        speed: float = 1.0,
        gpt_cache: int = -1,
        gpt_model: str = None,
        sovits_model: str = None,
    ):
        """
        Performs standard Text-to-Speech (TTS) inference to generate audio from text.

        Args:
            spk_audio_path (str | dict): Path(s) to the target speaker's reference audio file(s).
                - If a `str`, it's a single audio file path for the target speaker.
                - If a `dict`, it enables multi-speaker fusion. The format is `{"audio_file_path.wav": weight}`,
            prompt_audio_path (str): Path to the prompt audio file (reference audio for tone/style).
            prompt_audio_text (str): The transcription (text content) of the prompt audio.
            prompt_audio_language (str): The language of the prompt audio (e.g., "en", "zh", "ja", "yue", "ko").
            text (str): The target text to be synthesized into speech.
            text_language (str): The language of the target text (e.g., "en", "zh", "ja", "yue", "ko").
            top_k (int, optional): Sampling parameter for the GPT model. Limits the next token selection to the top K most probable tokens.
            top_p (float, optional): Sampling parameter for the GPT model. Limits the next token selection to a cumulative probability of P.
            temperature (float, optional): Sampling temperature for the GPT model. Higher values make the output more random/expressive; lower values make it more deterministic.
            repetition_penalty (float, optional): Penalty factor for repetition in the GPT model. Values > 1.0 penalize repetition.
            noise_scale (float, optional): Controls the standard deviation of the acoustic distribution in the SoVITS decoder. A certain amount of noise can enhance audio naturalness.
            speed (float, optional): Speed factor for the generated audio. 1.0 is normal speed, >1.0 is faster, <1.0 is slower.
            gpt_cache (int, optional): The size of the pre-allocated key-value (KV) cache to use for the GPT model, helping to speed up inference.
            gpt_model (str, optional): The GPT model to use for the inference.
            sovits_model (str, optional): The SoVITS model to use for the inference.

        Returns:
            AudioClip: An object encapsulating the generation results, which includes:
                - audio_data (np.ndarray, float32): The generated raw audio waveform data.
                - samplerate (int): The sample rate of the generated audio.
                - audio_len_s (float): The duration of the generated audio in seconds.
                - subtitles (list): Subtitle data corresponding to the generated audio.
        """

        try:
            if text[-1] not in self.punctuation:
                text += "."

            if len(text) > 20:
                logging.info(f"Starting inference for text: '{text[:20]}...'")
            else:
                logging.info(f"Starting inference for text: '{text}'")

            prompt_audio_language = self.dict_language[prompt_audio_language]
            text_language = self.dict_language[text_language]

            if gpt_model is None:
                if len(self.gpt_models) > 0:
                    gpt_model = list(self.gpt_models.keys())[0]
                else:
                    gpt_model = self.default_gpt_path
            if sovits_model is None:
                if len(self.sovits_models) > 0:
                    sovits_model = list(self.sovits_models.keys())[0]
                else:
                    sovits_model = self.default_sovits_path

            logging.info(f"Using GPT model: {gpt_model}")
            logging.info(f"Using SoVITS model: {sovits_model}")

            sovits, ge = self.prepare_sovits_resources(sovits_model, spk_audio_path)
            gpt, prompt, phones1, bert1 = self.prepare_gpt_resources(gpt_model, prompt_audio_path, prompt_audio_language, prompt_audio_text)
            t2s_model = gpt.t2s_model
            vq_model = sovits.vq_model

            logging.info("Processing text to phones and BERT features...")
            phones2, word2ph, bert2, norm_text = get_phones_and_bert(text, text_language, self.tts_config)
            all_phoneme_ids = torch.LongTensor(phones1 + phones2).to(self.tts_config.device).unsqueeze(0)
            bert = torch.cat([bert1, bert2]).unsqueeze(0)

            logging.info("Running GPT inference (Text-to-Semantic)...")
            pred_semantic = t2s_model.infer(
                all_phoneme_ids,
                prompt,
                bert,
                max_kv_cache=gpt_cache,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
                repetition_penalty=repetition_penalty,
            )

            logging.info("Running SoVITS inference (Semantic-to-Waveform)...")
            phones2_tensor = torch.LongTensor(phones2).to(self.tts_config.device).unsqueeze(0)
            encoded_text, text_mask = vq_model.enc_p.text_encode(phones2_tensor)

            audio, attn = vq_model.decode(
                pred_semantic, encoded_text, text_mask, ge, noise_scale=noise_scale, speed=speed
            )

            audio = audio[0, 0, :].cpu().numpy()
            assign = self.viterbi_monotonic(attn)
            subtitles = self.get_subtitles(word2ph, assign, speed)

            if subtitles[-1]['text'] not in self.punctuation:
                subtitles.append({
                    "text": word2ph['word'][-1],
                    "start_s": subtitles[-1]['end_s'],
                    "end_s": subtitles[-1]['end_s']
                })
            subtitles[-1]['end_s'] += 0.2

            subtitles = sub2text_index(subtitles, norm_text, text)

            max_audio = np.abs(audio).max()
            if max_audio > 1:
                audio = audio / max_audio
            audio = np.concatenate([audio, np.zeros((int(0.2*self.samplerate),), dtype=audio.dtype)])
            
            audio = audio.astype(np.float32)
            
            audio_len_s = len(audio) / self.samplerate

            logging.info(f"Inference complete. Generated {audio_len_s:.2f}s of audio.")

            return AudioClip(self.audio_queue, audio, self.samplerate, audio_len_s, subtitles, text)
        
        finally:
            self.empty_cache()

    @torch.inference_mode()
    def infer_stream(
        self,
        spk_audio_path: str | dict,
        prompt_audio_path: str,
        prompt_audio_text: str,
        prompt_audio_language: str,
        text: str,
        text_language: str,
        is_cut_text: bool = True,
        cut_punds: dict = {".", "。", "?", "？", "!", "！", ",", "，", ":", "：", ";", "；", "、", " "},
        cut_minlen: int = 10,
        cut_mute: int = 0.2,
        cut_mute_scale_map: dict = {".": 1.5, "。": 1.5, "?": 1.5, "？": 1.5, "!": 1.5, "！": 1.5,",": 0.8, "，": 0.8, "、": 0.6, " ": 0.1},
        stream_mode: Literal["token", "sentence"] = "token",
        stream_chunk: int = 25,
        overlap_len: int = 10,
        boost_first_chunk: bool = True,
        history_size: int = -1,
        top_k: int = 15,
        top_p: float = 1.0,
        temperature: float = 1.0,
        repetition_penalty: float = 1.35,
        noise_scale: float = 0.5,
        speed: float = 1.0,
        gpt_cache: int = -1,
        gpt_model: str = None,
        sovits_model: str = None,
        debug: str = True,
    ):
        """
        Performs streaming Text-to-Speech (TTS) inference, yielding audio chunks in real-time.

        Args:
            spk_audio_path (str | dict): Path(s) to the target speaker's reference audio file(s).
                - If a `str`, it's a single audio file path for the target speaker.
                - If a `dict`, it enables multi-speaker fusion. The format is `{"audio_file_path.wav": weight}`,
            prompt_audio_path (str): Path to the prompt audio file (reference audio for tone/style).
            prompt_audio_text (str): The transcription (text content) of the prompt audio.
            prompt_audio_language (str): The language of the prompt audio (e.g., "en", "zh", "ja", "yue", "ko").
            text (str): The target text to be synthesized into speech.
            text_language (str): The language of the target text (e.g., "en", "zh", "ja", "yue", "ko").
            is_cut_text (bool, optional): Whether to split the input text into smaller segments based on punctuation.
            cut_punds (set, optional): A set of punctuation marks used to split the text into segments for processing.
            cut_minlen (int, optional): The minimum length of a text segment. Segments shorter than this will be merged.
            cut_mute (float, optional): Duration of silence (in seconds) to insert between text segments.
            cut_mute_scale_map (dict, optional): A mapping to scale the mute duration (cut_mute) for specific punctuation marks.
            stream_mode (str, optional): The strategy for streaming. "token" yields audio as a specific chunk size of GPT tokens is accumulated; "sentence" yields audio after completing full sentences.
            stream_chunk (int, optional): The number of tokens to process in one chunk when using 'token' mode.
            overlap_len (int, optional): The number of overlapping tokens between chunks to ensure smooth audio transitions.
            boost_first_chunk (bool, optional): If True, reduces initial latency but may introduce noise in short audio.
            history_size (int, optional): The amount of historical context (in terms of characters) to retain from previous segments when processing the current one.
            top_k (int, optional): Sampling parameter for the GPT model. Limits the next token selection to the top K most probable tokens.
            top_p (float, optional): Sampling parameter for the GPT model. Limits the next token selection to a cumulative probability of P.
            temperature (float, optional): Sampling temperature for the GPT model. Higher values make the output more random/expressive; lower values make it more deterministic.
            repetition_penalty (float, optional): Penalty factor for repetition in the GPT model. Values > 1.0 penalize repetition.
            noise_scale (float, optional): Controls the standard deviation of the acoustic distribution in the SoVITS decoder. A certain amount of noise can enhance audio naturalness.
            speed (float, optional): Speed factor for the generated audio. 1.0 is normal speed, >1.0 is faster, <1.0 is slower.
            gpt_cache (int, optional): The size of the pre-allocated key-value (KV) cache to use for the GPT model, helping to speed up inference.
            gpt_model (str, optional): The GPT model to use for the inference.
            sovits_model (str, optional): The SoVITS model to use for the inference.
            debug (bool, optional): When set to “False”, certain outputs can be suppressed.

        Yields:
            AudioClip: An object encapsulating a chunk of the generated audio stream, which includes:
                - audio_data (np.ndarray, float32): The generated raw audio waveform data.
                - samplerate (int): The sample rate of the generated audio.
                - audio_len_s (float): The duration of the generated audio in seconds.
                - subtitles (list): Subtitle data specific to this chunk.
        """

        try:
            if text[-1] not in self.punctuation:
                text += "."

            if len(text) > 20:
                logging.info(f"Starting Stream inference for text: '{text[:20]}...'")
            else:
                logging.info(f"Starting Stream inference for text: '{text}'")

            if stream_mode == "sentence": stream_chunk: int = 10000
            if not is_cut_text: cut_minlen = 10000
            cut_mute = cut_mute / speed
            
            prompt_audio_language = self.dict_language[prompt_audio_language]
            text_language = self.dict_language[text_language]

            if gpt_model is None:
                if len(self.gpt_models) > 0:
                    gpt_model = list(self.gpt_models.keys())[0]
                else:
                    gpt_model = self.default_gpt_path
            if sovits_model is None:
                if len(self.sovits_models) > 0:
                    sovits_model = list(self.sovits_models.keys())[0]
                else:
                    sovits_model = self.default_sovits_path

            logging.info(f"Using GPT model: {gpt_model}")
            logging.info(f"Using SoVITS model: {sovits_model}")

            sovits, ge = self.prepare_sovits_resources(sovits_model, spk_audio_path)
            gpt, prompt, phones1, bert1 = self.prepare_gpt_resources(gpt_model, prompt_audio_path, prompt_audio_language, prompt_audio_text)
            t2s_model = gpt.t2s_model
            vq_model = sovits.vq_model

            overlap_samples = overlap_len * vq_model.samples_per_frame

            history = {"prompt":None, "phones":None, "bert":None}

            cur_text_l = 0
            audio_len_s = 0
            last_end_s = 0

            text_cuts = cut_text(text, cut_punds, cut_minlen)
            for i, text_cut in enumerate(text_cuts):
                if debug: logging.info(f"Processing segment {i+1}/{len(text_cuts)}: '{text_cut}'")

                phones2, word2ph, bert2, norm_text = get_phones_and_bert(text_cut, text_language, self.tts_config)

                phones2[-1] = symbol_to_id['.']

                if history["prompt"] is None:
                    curr_phoneme_ids = torch.LongTensor(phones1 + phones2).to(self.tts_config.device).unsqueeze(0)
                    curr_prompt = prompt
                    curr_bert = torch.cat([bert1, bert2]).unsqueeze(0)
                else:
                    curr_phoneme_ids = torch.LongTensor(phones1 + history["phones"] + phones2).to(self.tts_config.device).unsqueeze(0)
                    curr_prompt = torch.cat([prompt, history["prompt"]], dim=1)
                    curr_bert = torch.cat([bert1, history["bert"], bert2]).unsqueeze(0)
                
                generator = t2s_model.infer_stream(
                    curr_phoneme_ids,
                    curr_prompt,
                    curr_bert,
                    max_kv_cache=gpt_cache,
                    top_k=top_k,
                    top_p=top_p,
                    temperature=temperature,
                    repetition_penalty=repetition_penalty,
                    stream_chunk=stream_chunk,
                    boost_first_chunk=boost_first_chunk if i == 0 else False,
                    debug=debug,
                )

                phones2_tensor = torch.LongTensor(phones2).to(self.tts_config.device).unsqueeze(0)
                encoded_text, text_mask = vq_model.enc_p.text_encode(phones2_tensor)

                last_subtitles_end = 0
                last_overlap_audio = None
                valid_start_idx = 0
                chunk_idx = 0
                for pred_semantic, is_final in generator:
                    audio, attn = vq_model.decode(
                        pred_semantic,
                        encoded_text,
                        text_mask,
                        ge,
                        noise_scale=noise_scale,
                        speed=speed,
                        stream_mode=True,
                        valid_start_idx=valid_start_idx,
                        overlap_len=overlap_len,
                    )

                    if not last_overlap_audio is None:
                        audio, offset = self.sola_algorithm(last_overlap_audio, audio, overlap_samples)
                    last_overlap_audio = audio[:, :, -overlap_samples:].clone()

                    if not is_final:
                        audio = audio[:, :, :-overlap_samples]
                        attn = attn[:, :-overlap_len, :]
                        valid_start_idx = attn.shape[1]
                    
                    audio = audio[0, 0, :]

                    assign = self.viterbi_monotonic(attn)
                    subtitles = self.get_subtitles(word2ph, assign, speed, last_end_s=last_end_s)

                    if is_final:
                        if history_size != -1:
                            # 取后 history_size 个字对应的prompt、phones、bert
                            n = sum(word2ph["ph"][-history_size:])
                            m = sum(word2ph["ph"])
                            idx = torch.where(assign == m-n)[0]
                            if idx.shape[0] != 0:
                                idx = idx[0].item()
                                k = int((m - idx) / 2)
                                history["prompt"] = pred_semantic[0, :, -k:]
                                history["phones"] = phones2[-n:]
                                history["bert"] = bert2[-n:]
                            else:
                                history["prompt"] = None
                                history["phones"] = None
                                history["bert"] = None

                        tail_offset = self.find_threshold_offsets(audio)
                        audio = audio[:-tail_offset]
                        subtitles[-1]['end_s'] -= tail_offset / self.samplerate

                        if text_cut[-1] in cut_mute_scale_map:
                            cut_mute_scale = cut_mute_scale_map[text_cut[-1]]
                        else:
                            cut_mute_scale = 1.0

                        silence = torch.zeros((int(cut_mute * cut_mute_scale * self.samplerate),), dtype=audio.dtype, device=audio.device)
                        audio = torch.concatenate([audio, silence])

                        if subtitles[-1]['text'] not in self.punctuation:
                            subtitles.append({
                                "text": word2ph['word'][-1],
                                "start_s": subtitles[-1]['end_s'],
                                "end_s": subtitles[-1]['end_s']
                            })
                        subtitles[-1]['end_s'] += cut_mute * cut_mute_scale
                        last_end_s = subtitles[-1]['end_s']

                    if subtitles:
                        subtitles = sub2text_index(subtitles, norm_text, text, start_idx=cur_text_l, end_idx=cur_text_l+len(text_cut))
                        new_subtitles = subtitles[last_subtitles_end:]
                        last_subtitles_end = len(subtitles)-1
                        if not is_final and new_subtitles: new_subtitles[-1]['end_s'] = None
                    else:
                        new_subtitles = []

                    audio = audio.cpu().numpy().astype(np.float32)

                    audio_len_s += len(audio) / self.samplerate

                    yield AudioClip(self.audio_queue, audio, self.samplerate, audio_len_s, new_subtitles, text)

                    chunk_idx += 1
                
                vq_model.enc_p.y_overlap = None
                cur_text_l += len(text_cut)
            
            if debug: logging.info(f"Stream inference complete. Generated {audio_len_s:.2f}s of audio.")
        
        finally:
            self.empty_cache()
    
    torch.inference_mode()
    def infer_batched(
        self,
        spk_audio_paths: str | dict | list[str | dict],
        prompt_audio_paths: str | list[str],
        prompt_audio_texts: str | list[str],
        prompt_audio_languages: str | list[str],
        texts: str | list[str],
        texts_language: str | list[str],
        return_subtitles: bool = False,
        is_cut_text: bool = True,
        cut_punds: dict = {".", "。", "?", "？", "!", "！", ",", "，", ":", "：", ";", "；", "、", " "},
        cut_minlen: int = 10,
        cut_mute: int = 0.2,
        cut_mute_scale_map: dict = {".": 1.5, "。": 1.5, "?": 1.5, "？": 1.5, "!": 1.5, "！": 1.5,",": 0.8, "，": 0.8, "、": 0.6, " ": 0.1},
        top_k: int = 15,
        top_p: float = 1.0,
        temperature: float = 1.0,
        repetition_penalty: float = 1.35,
        noise_scale: float = 0.5,
        speed: float = 1.0,
        gpt_batch_size: int = -1,
        sovits_batch_size: int = -1,
        gpt_cache: int = -1,
        gpt_model: str = None,
        sovits_model: str = None,
    ) -> tuple[AudioClip]:
        """
        Performs batched Text-to-Speech (TTS) inference to generate multiple audio clips simultaneously.

        Args:
            spk_audio_paths (str | dict | list[str | dict]): Path(s) to the target speaker's reference audio file(s).
                - If a `str`, it's a single audio file path for the target speaker.
                - If a `dict`, it enables multi-speaker fusion. The format is `{"audio_file_path.wav": weight}`,
            prompt_audio_paths (str | list[str]): Path to the prompt audio file (reference audio for tone/style).
            prompt_audio_texts (str | list[str]): The transcription (text content) of the prompt audio.
            prompt_audio_languages (str | list[str]): The language of the prompt audio (e.g., "en", "zh", "ja", "yue", "ko").
            texts (str | list[str]): The target text to be synthesized into speech.
            texts_language (str | list[str]): The language of the target text (e.g., "en", "zh", "ja", "yue", "ko").
            return_subtitles (bool, optional): Whether to return subtitle information (timestamps) for the generated audio.
            is_cut_text (bool, optional): Whether to split the input text into smaller segments based on punctuation.
            cut_punds (set, optional): A set of punctuation marks used to split the text into segments for processing.
            cut_minlen (int, optional): The minimum length of a text segment. Segments shorter than this will be merged.
            cut_mute (float, optional): Duration of silence (in seconds) to insert between text segments.
            cut_mute_scale_map (dict, optional): A mapping to scale the mute duration (cut_mute) for specific punctuation marks.
            top_k (int, optional): Sampling parameter for the GPT model. Limits the next token selection to the top K most probable tokens.
            top_p (float, optional): Sampling parameter for the GPT model. Limits the next token selection to a cumulative probability of P.
            temperature (float, optional): Sampling temperature for the GPT model. Higher values make the output more random/expressive; lower values make it more deterministic.
            repetition_penalty (float, optional): Penalty factor for repetition in the GPT model. Values > 1.0 penalize repetition.
            noise_scale (float, optional): Controls the standard deviation of the acoustic distribution in the SoVITS decoder. A certain amount of noise can enhance audio naturalness.
            speed (float, optional): Speed factor for the generated audio. 1.0 is normal speed, >1.0 is faster, <1.0 is slower.
            gpt_batch_size (int, optional): Number of samples to process in one GPT forward pass.
            sovits_batch_size (int, optional): Number of samples to process in one SoVITS forward pass.
            gpt_cache (int, optional): The size of the pre-allocated key-value (KV) cache to use for the GPT model, helping to speed up inference.
            gpt_model (str, optional): The GPT model to use for the inference.
            sovits_model (str, optional): The SoVITS model to use for the inference.

        Returns:
            Tuple: A tuple of AudioClip objects in the same order as the input `texts`.
                Each AudioClip contains:
                - audio_data (np.ndarray, float32): The generated raw audio waveform data.
                - samplerate (int): The sample rate of the generated audio.
                - audio_len_s (float): The duration of the generated audio in seconds.
        """
        
        try:
            if isinstance(texts, str):
                texts = [texts]
            texts = [t if t.endswith(tuple(self.punctuation)) else t + "." for t in texts]

            if not is_cut_text: cut_minlen = 10000

            n = len(texts)

            logging.info(f"Starting batched TTS inference: processing {n} text segments.")

            if isinstance(spk_audio_paths, str):
                spk_audio_paths = [spk_audio_paths]*n
            if isinstance(prompt_audio_paths, str):
                prompt_audio_paths = [prompt_audio_paths]*n
            if isinstance(prompt_audio_texts, str):
                prompt_audio_texts = [prompt_audio_texts]*n
            if isinstance(prompt_audio_languages, str):
                prompt_audio_languages = [prompt_audio_languages]*n
            if isinstance(texts_language, str):
                texts_language = [texts_language]*n

            prompt_audio_languages = [self.dict_language[prompt_audio_language] for prompt_audio_language in prompt_audio_languages]
            texts_language = [self.dict_language[text_language] for text_language in texts_language]

            if gpt_model is None:
                if len(self.gpt_models) > 0:
                    gpt_model = list(self.gpt_models.keys())[0]
                else:
                    gpt_model = self.default_gpt_path
            if sovits_model is None:
                if len(self.sovits_models) > 0:
                    sovits_model = list(self.sovits_models.keys())[0]
                else:
                    sovits_model = self.default_sovits_path

            logging.info(f"Using GPT model: {gpt_model}")
            logging.info(f"Using SoVITS model: {sovits_model}")

            if gpt_model not in self.gpt_models:
                self.load_gpt_model(gpt_model)
            if sovits_model not in self.sovits_models:
                self.load_sovits_model(sovits_model)

            gpt = self.gpt_models[gpt_model]
            sovits = self.sovits_models[sovits_model]

            t2s_model = gpt.t2s_model
            vq_model = sovits.vq_model

            all_segments = []
            segment_to_original_map = []
            
            for idx, text in enumerate(texts):
                cuts = cut_text(text, cut_punds, cut_minlen)
                for cut in cuts:
                    all_segments.append(cut)
                    segment_to_original_map.append(idx)
            
            n_orig = len(texts)
            n_segs = len(all_segments)

            def expand_input(inp):
                return [inp[segment_to_original_map[i]] for i in range(n_segs)]

            spk_audio_paths = expand_input(spk_audio_paths)
            prompt_audio_paths = expand_input(prompt_audio_paths)
            prompt_audio_texts = expand_input(prompt_audio_texts)
            prompt_audio_languages = expand_input(prompt_audio_languages)
            texts_language = expand_input(texts_language)

            orig_texts = texts
            texts = all_segments

            if gpt_batch_size == -1:
                gpt_batch_size = len(texts)
            if sovits_batch_size == -1:
                sovits_batch_size = len(texts)

            logging.info("Processing text to phones and BERT features...")
            all_phoneme_ids = []
            all_prompts = []
            all_bert_features = []
            all_ge = []

            all_phones2, all_word2ph, all_bert2, all_norm_text = get_phones_and_bert(texts, texts_language, self.tts_config)

            for items in zip(texts, spk_audio_paths, prompt_audio_paths, prompt_audio_texts, prompt_audio_languages, all_phones2, all_bert2):
        
                (text, spk_audio_path, prompt_audio_path, prompt_audio_text, prompt_audio_language, phones2, bert2) = items

                if prompt_audio_path not in self.prompt_audio_cache:
                    self.cache_prompt_audio(prompt_audio_paths=prompt_audio_path, prompt_audio_texts=prompt_audio_text, prompt_audio_languages=prompt_audio_language)

                prompt = self.prompt_audio_cache[prompt_audio_path]["prompt"]
                phones1 = self.prompt_audio_cache[prompt_audio_path]["phones1"]
                bert1 = self.prompt_audio_cache[prompt_audio_path]["bert1"]
            
                if isinstance(spk_audio_path, dict):
                    weight_sum = sum(spk_audio_path.values())

                    ge = None
                    for audio_path, weight in spk_audio_path.items():
                        if audio_path not in self.spk_audio_cache:
                            self.cache_spk_audio(audio_path)

                        if ge is None:
                            ge = self.spk_audio_cache[audio_path]["ge"] * (weight / weight_sum)
                        else:
                            ge += self.spk_audio_cache[audio_path]["ge"] * (weight / weight_sum)
                else:
                    if spk_audio_path not in self.spk_audio_cache:
                        self.cache_spk_audio(spk_audio_path)

                    ge = self.spk_audio_cache[spk_audio_path]["ge"]
                
                phoneme_ids = torch.LongTensor(phones1 + phones2).to(self.tts_config.device)
                bert = torch.cat([bert1, bert2])
                
                all_phoneme_ids.append(phoneme_ids)
                all_prompts.append(prompt.squeeze(0))
                all_bert_features.append(bert)
                all_ge.append(ge.squeeze(0))

            logging.info("Running GPT batched inference (Text-to-Semantic)...")
            pred_semantic, semantic_orig_idx = t2s_model.infer_batched(
                all_phoneme_ids,
                all_prompts,
                all_bert_features,
                batch_size=gpt_batch_size,
                max_kv_cache=gpt_cache,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
                repetition_penalty=repetition_penalty,
            )

            semantic_lengths = torch.tensor([len(semantic) for semantic in pred_semantic])

            idx_map = torch.argsort(semantic_lengths)

            # 将排序后的索引进行双端交错重排，平衡各 Batch 间的序列长度，避免因长短不一导致的计算负载不均
            n = len(idx_map)
            sorted_indices = torch.arange(n)
            interleave_idx = torch.zeros(n, dtype=torch.long)
            interleave_idx[0::2] = sorted_indices[:(n + 1) // 2]
            interleave_idx[1::2] = sorted_indices[(n + 1) // 2:].flip(0)

            idx_map = idx_map[interleave_idx]

            pred_semantic = [pred_semantic[i] for i in idx_map.tolist()]
            semantic_orig_idx = semantic_orig_idx[idx_map]
            semantic_lengths = semantic_lengths[idx_map]

            logging.info("Running SoVITS batched inference (Semantic-to-Waveform)...")

            generated_audios = []
            generated_subtitles = []
            num_samples = len(pred_semantic)

            for i in tqdm(range(0, num_samples, sovits_batch_size)):
                batch_end = min(i + sovits_batch_size, num_samples)
                
                semantic_list = pred_semantic[i:batch_end]
                curr_orig_indices = semantic_orig_idx[i:batch_end]
                curr_lengths = semantic_lengths[i:batch_end]

                ge_list = []
                phones2_list = []
                for idx, length in enumerate(curr_lengths):
                    orig_idx = curr_orig_indices[idx]
                    ge_list.append(all_ge[orig_idx].expand(-1, length))
                    phones2_list.append(torch.LongTensor(all_phones2[orig_idx]).to(self.tts_config.device))
                
                curr_ge = torch.cat(ge_list, dim=1).unsqueeze(0)
                curr_semantic = torch.cat(semantic_list).unsqueeze(0).unsqueeze(0)
                curr_phones2 = torch.cat(phones2_list).unsqueeze(0)

                curr_word2ph = {
                    "word": [w for idx in curr_orig_indices for w in all_word2ph[idx]["word"]],
                    "ph": [p for idx in curr_orig_indices for p in all_word2ph[idx]["ph"]]
                }

                # ge [B, D, T]
                # semantic [n_q, B, N]
                
                encoded_text, text_mask = vq_model.enc_p.text_encode(curr_phones2)

                audio_batch, attn = vq_model.decode(
                    curr_semantic, encoded_text, text_mask, curr_ge, noise_scale=noise_scale, speed=speed, cuda_graph=False
                )

                audio_batch = audio_batch[0, 0, :]

                if return_subtitles:
                    assign = self.viterbi_monotonic(attn)
                    subtitles = self.get_subtitles(curr_word2ph, assign, speed)

                    if subtitles[-1]['text'] not in self.punctuation:
                        subtitles.append({
                            "text": curr_word2ph['word'][-1],
                            "start_s": subtitles[-1]['end_s'],
                            "end_s": subtitles[-1]['end_s']
                        })

                max_audio = torch.abs(audio_batch).max()
                if max_audio > 1.0:
                    audio_batch = audio_batch / max_audio

                if return_subtitles:
                    last_i = 0
                    for j in range(len(semantic_list)):
                        best_i = self.find_subtitles_by_text(subtitles, all_norm_text[curr_orig_indices[j]], last_i)
                        subtitle = subtitles[last_i:best_i]
                        last_i = best_i
                        
                        last_actual_len = int(subtitle[0]["start_s"] * self.samplerate)
                        actual_len = int(subtitle[-1]["end_s"] * self.samplerate)
                        audio = audio_batch[last_actual_len:actual_len]

                        head_offset = self.find_quietest_offsets(audio)
                        tail_offset = self.find_threshold_offsets(audio)
                        audio = self.fade(audio[head_offset:-tail_offset]).cpu().numpy()

                        subtitle[0]["start_s"] += head_offset / self.samplerate
                        subtitle[-1]["end_s"] -= tail_offset / self.samplerate

                        audio = audio.astype(np.float32)
                        generated_audios.append(audio)
                        generated_subtitles.append(subtitle)
                else:
                    last_actual_len = 0
                    for j in range(len(semantic_list)):
                        actual_len = last_actual_len + curr_lengths[j] * 2 * vq_model.samples_per_frame / speed
                        audio = audio_batch[int(last_actual_len):int(actual_len)]
                        last_actual_len = actual_len

                        head_offset = self.find_quietest_offsets(audio)
                        tail_offset = self.find_threshold_offsets(audio)
                        audio = self.fade(audio[head_offset:-tail_offset]).cpu().numpy()

                        audio = audio.astype(np.float32)
                        generated_audios.append(audio)

            logging.info(f"Inference complete. Generated {len(generated_audios)} audio clips.")

            ordered_audios = [None] * len(generated_audios)
            ordered_subtitles = [None] * len(generated_audios)
            for current_pos, original_pos in enumerate(semantic_orig_idx.tolist()):
                ordered_audios[original_pos] = generated_audios[current_pos]
                if return_subtitles:
                    ordered_subtitles[original_pos] = generated_subtitles[current_pos]

            final_ordered_audios = [[] for _ in range(n_orig)]
            final_ordered_subtitles = [[] for _ in range(n_orig)]
            ordered_norm_texts = ["" for _ in range(n_orig)]
            for i, (audio_data, subtitle) in enumerate(zip(ordered_audios, ordered_subtitles)):
                orig_idx = segment_to_original_map[i]
                final_ordered_audios[orig_idx].append(audio_data)

                if texts[i][-1] in cut_mute_scale_map:
                    cut_mute_scale = cut_mute_scale_map[texts[i][-1]]
                else:
                    cut_mute_scale = 1.0
                silence = np.zeros((int(cut_mute * cut_mute_scale * self.samplerate),))
                final_ordered_audios[orig_idx].append(silence)

                if return_subtitles:
                    subtitle[-1]["end_s"] += cut_mute * cut_mute_scale
                    final_ordered_subtitles[orig_idx].append(subtitle)
                
                ordered_norm_texts[orig_idx] += all_norm_text[i]
            
            result = []
            for audio_list, subtitles_list, norm_text, orig_text in zip(final_ordered_audios, final_ordered_subtitles, ordered_norm_texts, orig_texts):
                audio = np.concatenate(audio_list)
                audio_len_s = len(audio) / self.samplerate
                
                if return_subtitles:
                    subtitle = self.cat_subtitles(*subtitles_list)
                    subtitle = sub2text_index(subtitle, norm_text, orig_text)
                    result.append(AudioClip(self.audio_queue, audio, self.samplerate, audio_len_s, subtitle, orig_text))
                else:
                    result.append(AudioClip(self.audio_queue, audio, self.samplerate, audio_len_s, None, orig_text))
            
            return tuple(result)
        
        finally:
            self.empty_cache()
    
    @torch.inference_mode()
    def infer_vc(
        self,
        spk_audio_path: str | dict,
        prompt_audio_path: str,
        prompt_audio_text: str,
        prompt_audio_language: str,
        noise_scale: float = 0.5,
        speed: float = 1.0,
        sovits_model: str = None,
    ):
        """
        Performs Voice Conversion (VC) to change the timbre of the input audio to the target speaker.

        Args:
            spk_audio_path (str | dict): Path(s) to the target speaker's reference audio file(s).
                - If a `str`, it's a single audio file path for the target speaker.
                - If a `dict`, it enables multi-speaker fusion. The format is `{"audio_file_path.wav": weight}`,
            prompt_audio_path (str): Path to the prompt audio file (reference audio for tone/style).
            prompt_audio_text (str): The transcription (text content) of the prompt audio.
            prompt_audio_language (str): The language of the prompt audio (e.g., "en", "zh", "ja", "yue", "ko").
            noise_scale (float, optional): Controls the standard deviation of the acoustic distribution in the SoVITS decoder. A certain amount of noise can enhance audio naturalness.
            speed (float, optional): Speed factor for the generated audio. 1.0 is normal speed, >1.0 is faster, <1.0 is slower.
            sovits_model (str, optional): The SoVITS model to use for the inference.

        Returns:
            AudioClip: An object encapsulating the generation results, which includes:
                - audio_data (np.ndarray, float32): The generated raw audio waveform data.
                - samplerate (int): The sample rate of the generated audio.
                - audio_len_s (float): The duration of the generated audio in seconds.
                - subtitles (list): Subtitle data corresponding to the generated audio.
        """

        try:
            if prompt_audio_text[-1] not in self.punctuation:
                prompt_audio_text += "."

            logging.info(f"Starting VC inference. Prompt audio: {prompt_audio_path}")

            prompt_audio_language = self.dict_language[prompt_audio_language]

            if sovits_model is None:
                if len(self.sovits_models) > 0:
                    sovits_model = list(self.sovits_models.keys())[0]
                else:
                    sovits_model = self.default_sovits_path
            
            logging.info(f"Using SoVITS model: {sovits_model}")

            sovits, ge = self.prepare_sovits_resources(sovits_model, spk_audio_path)
            vq_model = sovits.vq_model

            logging.info("Extracting semantic features from prompt audio...")
            if self.cnhubert_model is None:
                self.cnhubert_model = CNHubert(self.cnhubert_path, self.tts_config)

            prompt = self.get_prompt(self.cnhubert_model, sovits, prompt_audio_path)

            if not self.always_load_cnhubert:
                self.cnhubert_model = None

            logging.info("Processing text to phones and BERT features...")
            phones, word2ph, _, norm_text = get_phones_and_bert(prompt_audio_text, prompt_audio_language, self.tts_config)

            phones_tensor = torch.LongTensor(phones).to(self.tts_config.device).unsqueeze(0)
            encoded_text, text_mask = vq_model.enc_p.text_encode(phones_tensor)

            logging.info("Running SoVITS inference (Semantic-to-Waveform)...")
            audio, attn = vq_model.decode(
                prompt.unsqueeze(0), encoded_text, text_mask, ge, noise_scale=noise_scale, speed=speed
            )

            audio = audio[0, 0, :].cpu().numpy()
            assign = self.viterbi_monotonic(attn)
            subtitles = self.get_subtitles(word2ph, assign, speed)
            
            if subtitles[-1]['text'] not in self.punctuation:
                subtitles.append({
                    "text": word2ph['word'][-1],
                    "start_s": subtitles[-1]['end_s'],
                    "end_s": subtitles[-1]['end_s']
                })
            subtitles[-1]['end_s'] += 0.2

            subtitles = sub2text_index(subtitles, norm_text, prompt_audio_text)

            max_audio = np.abs(audio).max()
            if max_audio > 1:
                audio = audio / max_audio
            audio = np.concatenate([audio, np.zeros((int(0.2*self.samplerate),), dtype=audio.dtype)])
            
            audio = audio.astype(np.float32)
            
            audio_len_s = len(audio) / self.samplerate

            logging.info(f"VC Inference complete. Generated {audio_len_s:.2f}s of audio.")

            return AudioClip(self.audio_queue, audio, self.samplerate, audio_len_s, subtitles, prompt_audio_text)
        
        finally:
            self.empty_cache()
    
    def prepare_gpt_resources(self, gpt_model, prompt_audio_path, prompt_audio_language, prompt_audio_text):
        if gpt_model not in self.gpt_models:
            self.load_gpt_model(gpt_model)

        if prompt_audio_path not in self.prompt_audio_cache:
            self.cache_prompt_audio(prompt_audio_paths=prompt_audio_path, prompt_audio_texts=prompt_audio_text, prompt_audio_languages=prompt_audio_language)

        prompt = self.prompt_audio_cache[prompt_audio_path]["prompt"]
        phones1 = self.prompt_audio_cache[prompt_audio_path]["phones1"]
        bert1 = self.prompt_audio_cache[prompt_audio_path]["bert1"]

        gpt = self.gpt_models[gpt_model]

        return gpt, prompt, phones1, bert1
    
    def prepare_sovits_resources(self, sovits_model, spk_audio_path):
        if sovits_model not in self.sovits_models:
            self.load_sovits_model(sovits_model)

        if isinstance(spk_audio_path, dict):
            weight_sum = sum(spk_audio_path.values())

            ge = None
            for audio_path, weight in spk_audio_path.items():
                if audio_path not in self.spk_audio_cache:
                    self.cache_spk_audio(audio_path)

                if ge is None:
                    ge = self.spk_audio_cache[audio_path]["ge"] * (weight / weight_sum)
                else:
                    ge += self.spk_audio_cache[audio_path]["ge"] * (weight / weight_sum)
        else:
            if spk_audio_path not in self.spk_audio_cache:
                self.cache_spk_audio(spk_audio_path)

            ge = self.spk_audio_cache[spk_audio_path]["ge"]

        sovits = self.sovits_models[sovits_model]

        return sovits, ge
    
    def verify_speaker(self, speaker1_audio: str, speaker2_audio: str):
        """
        Verifies the similarity between two speaker audio files.

        Args:
            speaker1_audio (str): Path to the first speaker's audio file.
            speaker2_audio (str): Path to the second speaker's audio file.

        Returns:
            numpy.ndarray: A similarity score between the two speakers' embeddings.
        """

        try:
            if not self.sovits_models:
                logging.error('No SoVITS models are currently loaded! Cannot verify speaker.')
                return

            model = self.sovits_models[next(iter(self.sovits_models))]

            if self.sv_model is None:
                self.sv_model = ERes2Net(self.sv_path, self.tts_config)

            if speaker1_audio in self.spk_audio_cache:
                sv_emb1 = self.spk_audio_cache[speaker1_audio]["sv_emb"]
            else:
                _, audio_tensor = self.get_spepc(model.hps, speaker1_audio)
                sv_emb1 = self.sv_model.compute_embedding3(audio_tensor)
            
            if speaker2_audio in self.spk_audio_cache:
                sv_emb2 = self.spk_audio_cache[speaker2_audio]["sv_emb"]
            else:
                _, audio_tensor = self.get_spepc(model.hps, speaker2_audio)
                sv_emb2 = self.sv_model.compute_embedding3(audio_tensor)
            
            similarity = torch.cosine_similarity(sv_emb1, sv_emb2, dim=-1, eps=1e-6)

            if not self.always_load_sv:
                self.sv_model = None
            
            return similarity.item()

        finally:
            self.empty_cache()
    
    def init_language_module(self, *languages: str):
        """
        Pre-loads the necessary language processing modules.

        Args:
            *languages (str): Variable number of language codes (e.g., "en", "zh", "ja", "yue", "ko").
        """
        if isinstance(languages, str): languages = [languages]
        for language in languages:
            if language in self.dict_language:
                lang = self.dict_language[language]
                lang = lang.replace("all_", "").replace("auto_", "")
                text_to_phonemes(" ", lang)
                logging.info(f'Loaded language module: {language}')
            else:
                logging.warning(f'Language "{language}" not found.')

    def load_gpt_model(self, *model_paths: str):
        """
        Loads GPT model weights from the specified paths into memory.

        Args:
            *model_paths (str): Variable number of paths to GPT model checkpoints.
        """
        if not model_paths:
            model_paths = (Path(self.models_dir) / "s1v3.ckpt",)
        for model_path in model_paths:
            self.gpt_models[model_path] = get_gpt_weights(model_path, self.tts_config)
            logging.info(f'Loaded GPT model: {model_path}')
        
    def load_sovits_model(self, *model_paths: str):
        """
        Loads SoVITS model weights from the specified paths into memory.

        Args:
            *model_paths (str): Variable number of paths to SoVITS model checkpoints.
        """
        if not model_paths:
            model_paths = (Path(self.models_dir) / "v2Pro\s2Gv2ProPlus.pth",)
        for model_path in model_paths:
            self.sovits_models[model_path] = get_sovits_weights(model_path, self.tts_config)
            logging.info(f'Loaded SoVITS model: {model_path}')
    
    def unload_gpt_model(self, *model_paths: str):
        """
        Unloads GPT models from memory to free up resources.

        Args:
            *model_paths (str): Variable number of paths to GPT models to unload.
        """
        try:
            for model_path in model_paths:
                if model_path in self.gpt_models:
                    del self.gpt_models[model_path]
                    logging.info(f'Unloaded GPT model: {model_path}')
                else:
                    logging.warning(f'GPT model {model_path} not found.')
        finally:
            self.empty_cache()
    
    def unload_sovits_model(self, *model_paths: str):
        """
        Unloads SoVITS models from memory to free up resources.

        Args:
            *model_paths (str): Variable number of paths to SoVITS models to unload.
        """
        try:
            for model_path in model_paths:
                if model_path in self.sovits_models:
                    del self.sovits_models[model_path]
                    logging.info(f'Unloaded SoVITS model: {model_path}')
                else:
                    logging.warning(f'SoVITS model {model_path} not found.')
        finally:
            self.empty_cache()
    
    def get_gpt_list(self):
        """
        Retrieves a list of currently loaded GPT models.

        Returns:
            list[str]: A list of file paths for the loaded GPT models.
        """
        return list(self.gpt_models.keys())

    def get_sovits_list(self):
        """
        Retrieves a list of currently loaded SoVITS models.

        Returns:
            list[str]: A list of file paths for the loaded SoVITS models.
        """
        return list(self.sovits_models.keys())
    
    @torch.inference_mode()
    def cache_spk_audio(self, *spk_audio_paths: str):
        """
        Processes and caches speaker audio embeddings for voice cloning.

        Args:
            *spk_audio_paths (str): Variable number of paths to speaker audio files.
        """
        try:
            if not self.sovits_models:
                logging.error('No SoVITS models are currently loaded! Cannot cache speaker audio.')
                return

            model = self.sovits_models[next(iter(self.sovits_models))]

            if self.sv_model is None:
                self.sv_model = ERes2Net(self.sv_path, self.tts_config)

            for spk_audio_path in spk_audio_paths:
                refers, audio_tensor = self.get_spepc(model.hps, spk_audio_path)
                sv_emb = self.sv_model.compute_embedding3(audio_tensor)
                ge = model.vq_model.get_ge(refers, sv_emb)
                self.spk_audio_cache[spk_audio_path] = {
                    "ge": ge, 
                    "sv_emb": sv_emb
                }
                logging.info(f'Cached speaker audio: {spk_audio_path}')
            
            if not self.always_load_sv:
                self.sv_model = None

        finally:
            self.empty_cache()
    
    @torch.inference_mode()
    def cache_prompt_audio(self, prompt_audio_paths: str|list[str], prompt_audio_texts: str|list[str], prompt_audio_languages: str|list[str]):
        """
        Pre-processes and caches prompt audio data for faster inference.

        Args:
            prompt_audio_paths (str | list[str]): Path(s) to the prompt audio file(s).
            prompt_audio_texts (str | list[str]): Transcription(s) of the prompt audio. 
                If a single string is provided with multiple paths, it will be applied to all.
            prompt_audio_languages (str | list[str]): Language(s) of the prompt text 
                (e.g., 'en', 'zh', 'ja'). If a single string is provided, it will be applied to all.
        """
        try:
            if not self.sovits_models:
                logging.error('No SoVITS models are currently loaded! Cannot cache prompt audio.')
                return

            model = self.sovits_models[next(iter(self.sovits_models))]
            
            if self.cnhubert_model is None:
                self.cnhubert_model = CNHubert(self.cnhubert_path, self.tts_config)

            if isinstance(prompt_audio_paths, str):
                prompt_audio_paths = [prompt_audio_paths]
            if isinstance(prompt_audio_texts, str):
                prompt_audio_texts = [prompt_audio_texts]*len(prompt_audio_paths)
            if isinstance(prompt_audio_languages, str):
                prompt_audio_languages = [prompt_audio_languages]*len(prompt_audio_paths)

            for i, prompt_audio_path in enumerate(prompt_audio_paths):
                prompt_audio_language, prompt_audio_text = prompt_audio_languages[i], prompt_audio_texts[i]
                prompt = self.get_prompt(self.cnhubert_model, model, prompt_audio_path)
                phones1, _, bert1, _ = get_phones_and_bert(prompt_audio_text, prompt_audio_language, self.tts_config)
                self.prompt_audio_cache[prompt_audio_path] = {
                    "prompt": prompt,
                    "phones1": phones1,
                    "bert1": bert1,
                }
                logging.info(f'Cached prompt audio: {prompt_audio_path}')
            
            if not self.always_load_cnhubert:
                self.cnhubert_model = None
        
        finally:
            self.empty_cache()
    
    def del_spk_audio(self, *spk_audio_list: str):
        """
        Removes speaker audio embeddings from the cache.

        Args:
            *spk_audio_list (str): Variable number of paths to speaker audio files to remove from cache.
        """
        for spk_audio in spk_audio_list:
            if spk_audio in self.spk_audio_cache:
                del self.spk_audio_cache[spk_audio]
                logging.info(f'Deleted speaker audio from cache: {spk_audio}')
            else:
                logging.warning(f'Speaker audio {spk_audio} not found in cache.')
    
    def del_prompt_audio(self, *prompt_audio_list: str):
        """
        Removes prompt audio data from the cache.

        Args:
            *prompt_audio_list (str): Variable number of paths to prompt audio files to remove from cache.
        """
        for prompt_audio in prompt_audio_list:
            if prompt_audio in self.prompt_audio_cache:
                del self.prompt_audio_cache[prompt_audio]
                logging.info(f'Deleted prompt audio from cache: {prompt_audio}')
            else:
                logging.warning(f'Prompt audio {prompt_audio} not found in cache.')
    
    def get_spk_audio_list(self):
        """
        Retrieves a list of cached speaker audio files.

        Returns:
            list[str]: A list of file paths for the cached speaker audio.
        """
        return list(self.spk_audio_cache.keys())

    def get_prompt_audio_list(self):
        """
        Retrieves a list of cached prompt audio files.

        Returns:
            list[str]: A list of file paths for the cached prompt audio.
        """
        return list(self.prompt_audio_cache.keys())
    
    def get_prompt(self, cnhubert_model: CNHubert, sovits_model: Sovits, audio_path: str):
        wav, sr = torchaudio.load(audio_path)
        wav = wav.to(self.tts_config.device)

        wav16k = self.resample(wav, sr, 16000).mean(dim=0)
        wav16k = wav16k.half() if self.tts_config.is_half else wav16k

        tail_offset = self.find_threshold_offsets(wav16k)
        wav16k = wav16k[:-tail_offset]

        silence = torch.zeros(int(16000 * 0.3), device=wav16k.device, dtype=wav16k.dtype)
        wav16k = torch.cat([wav16k, silence])

        ssl_content = cnhubert_model.model(wav16k.unsqueeze(0))["last_hidden_state"].transpose(1, 2)
        codes = sovits_model.vq_model.extract_latent(ssl_content)
        prompt_semantic = codes[0, 0]
        prompt = prompt_semantic.unsqueeze(0).to(self.tts_config.device)
        return prompt
    
    def resample(self, audio_tensor, sr0, sr1):
        key = "%s-%s" % (sr0, sr1)
        if key not in self.resample_transform_dict:
            self.resample_transform_dict[key] = torchaudio.transforms.Resample(sr0, sr1).to(self.tts_config.device)
        return self.resample_transform_dict[key](audio_tensor)
    
    def get_spepc(self, hps, filename):
        sr1 = int(hps.data.sampling_rate)
        audio, sr0 = torchaudio.load(filename)

        audio = audio.to(self.tts_config.device)
        if audio.shape[0] == 2:
            audio = audio.mean(0).unsqueeze(0)
        if sr0 != sr1:
            audio = self.resample(audio, sr0, sr1)

        maxx = audio.abs().max()
        if maxx > 1:
            audio /= min(2, maxx)

        spec = spectrogram_torch(
            audio,
            hps.data.filter_length,
            hps.data.sampling_rate,
            hps.data.hop_length,
            hps.data.win_length,
            center=False,
        )

        spec = spec.to(self.tts_config.dtype)
        audio = self.resample(audio, sr1, 16000)
        if self.tts_config.is_half: audio = audio.half()

        return spec, audio
    
    def sola_algorithm(self, f1_overlap, f2, overlap_len, search_len: int = 320):
        query = f1_overlap
        key = f2[:, :, :overlap_len + search_len]

        corr = F.conv1d(key, query) 
        ones_kernel = torch.ones_like(query)
        energy = F.conv1d(key**2, ones_kernel) + 1e-8
        norm_corr = corr / torch.sqrt(energy)
        offset = norm_corr.argmax(dim=-1)
        
        f2_aligned = f2[:, :, offset.item():]
        alpha = torch.linspace(0, 1, overlap_len, device=self.tts_config.device, dtype=self.tts_config.dtype).view(1, 1, -1)
        f2_overlap = f2_aligned[:, :, :overlap_len]
        f_faded = f1_overlap * (1 - alpha) + f2_overlap * alpha
        f2_real = torch.cat([f_faded, f2_aligned[:, :, overlap_len:]], dim=-1)
        return f2_real, offset
    
    def find_quietest_offsets(self, audio, frame_length=512, hop_length=256, search_len=6400):
        search_audio = audio[:search_len]
        frames = search_audio.unfold(0, frame_length, hop_length)
        rms_values = torch.sqrt(torch.mean(frames**2, dim=1))
        best_frame_idx = torch.argmin(rms_values).item()
        head_offset = best_frame_idx * hop_length
        
        return head_offset

    def find_threshold_offsets(self, audio, threshold=0.01, frame_length=512, hop_length=256, search_len=6400):
        threshold = threshold * audio.max()

        search_audio_tail = audio[-search_len:]
        frames_tail = search_audio_tail.unfold(0, frame_length, hop_length)
        rms_tail = torch.sqrt(torch.mean(frames_tail**2, dim=1))
        
        tail_mask = rms_tail > threshold
        tail_indices = torch.nonzero(tail_mask)
        
        if tail_indices.numel() > 0:
            tail_frame_idx = tail_indices[-1].item()
            tail_offset = search_len - (tail_frame_idx * hop_length)
        else:
            tail_offset = 1
            
        return tail_offset
    
    def fade(self, audio, fade_len=3200):
        audio = audio.clone()
        fade_in_vec = torch.linspace(0, 1, fade_len, device=audio.device)
        fade_out_vec = torch.linspace(1, 0, fade_len, device=audio.device)
        audio[:fade_len] *= fade_in_vec
        audio[-fade_len:] *= fade_out_vec
        
        return audio
    
    def get_subtitles(self, word2ph, assign, speed, last_end_s = 0):
        frame_time = (1 / self.sovits_hz) / speed

        ph_end_s = []
        cur_ph = int(assign[0])
        for f in range(1, assign.shape[-1]):
            ph = int(assign[f])
            if ph != cur_ph:
                ph_end_s.append(f * frame_time)
                cur_ph = ph
        ph_end_s.append(assign.shape[-1] * frame_time)

        idx = -1
        if assign[0] == -1:
            end_s = last_end_s + ph_end_s.pop(0)
        else:
            end_s = last_end_s
        subtitles = []
        for i in range(len(word2ph["word"])):
            word, ph = word2ph["word"][i], word2ph["ph"][i]

            idx += ph
            if idx >= len(ph_end_s): break

            start_s = end_s
            end_s = ph_end_s[idx] + last_end_s

            subtitles.append({
                "text": word,
                "start_s": start_s,
                "end_s": end_s
            })
        
        if end_s - last_end_s != ph_end_s[-1]:
            start_s = end_s
            end_s = ph_end_s[-1] + last_end_s

            subtitles.append({
                "text": word,
                "start_s": start_s,
                "end_s": end_s
            })
        
        return subtitles

    def find_subtitles_by_text(self, subtitles, text, last_i):
        text = text.replace(" ", "")
        m = 0
        for i in range(last_i, len(subtitles)):
            subtitle = subtitles[i]
            if subtitle["text"] == text[m:m+len(subtitle["text"])]:
                m += len(subtitle["text"])
            else:
                break
        else:
            i = len(subtitles)
        
        return i
    
    def cat_subtitles(self, *subtitles_list):
        last_end_s = 0
        result = []
        for subtitles in subtitles_list:
            m = subtitles[0]["start_s"] - last_end_s
            for subtitle in subtitles:
                subtitle["start_s"] -= m
                subtitle["end_s"] -= m
                result.append(subtitle)
            last_end_s = subtitles[-1]["end_s"]

        return result

    def viterbi_monotonic(self, attn: torch.Tensor):
        B, T, N = attn.shape
        device = attn.device

        max_idx = torch.argmax(attn, dim=-1)
        mask = (max_idx != N - 1)
        
        masked_attn = attn * mask.unsqueeze(-1).float()
        sum_attn = torch.sum(masked_attn, dim=0)
        count = torch.sum(mask, dim=0).unsqueeze(-1)

        default_distribution = torch.full((T, N), 1.0 / N, device=device)
        default_distribution[:, N-1] = 0.9 / N
        default_distribution[:, 1] = 1.1 / N

        default_distribution /= default_distribution.sum(dim=-1, keepdim=True)

        normal_attn = torch.where(
            count > 0,
            sum_attn / (count + 1e-9),
            default_distribution
        )


        normal_attn_argmax = torch.argmax(normal_attn, dim=-1)

        is_zero = (normal_attn_argmax == 0)

        first_zero_idx = torch.where(is_zero)[0][0] if is_zero.any() else torch.tensor(0, device=device)

        dp = torch.zeros((T, N), device=device)
        ptr = torch.zeros((T, N), dtype=torch.long, device=device)
        
        dp[0] = normal_attn[0]
        
        for t in range(1, T):
            prev_dp = dp[t-1]
            prev_dp_shifted = torch.cat([torch.tensor([-float('inf')], device=device), prev_dp[:-1]])
            
            stacked = torch.stack([prev_dp, prev_dp_shifted], dim=0)
            max_vals, relative_indices = torch.max(stacked, dim=0)
            
            dp[t] = normal_attn[t] + max_vals
            ptr[t] = torch.arange(N, device=device) - relative_indices

        assign_path = torch.zeros(T, dtype=torch.long, device=device)
        assign_path[-1] = torch.argmax(dp[-1])
        
        for t in range(T - 2, -1, -1):
            assign_path[t] = ptr[t+1, assign_path[t+1]]

        assign_path[:first_zero_idx] = -1

        return assign_path
    
    def empty_cache(self):
        try:
            gc.collect()
            if "cuda" in str(self.tts_config.device):
                torch.cuda.empty_cache()
            elif str(self.tts_config.device) == "mps":
                torch.mps.empty_cache()
        except:
            pass