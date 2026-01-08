import gc
import os
import sys
import torch
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(filename)s - %(levelname)s: %(message)s'
)
import librosa
import torchaudio
import numpy as np

script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)
sys.path.append(script_dir)
sys.path.append("%s/GPT_SoVITS" % (script_dir))

from Loader import get_gpt_weights, get_sovits_weights, Gpt, Sovits
from download import check_pretrained_models, download_model
from TextPreprocessor import get_phones_and_bert, cut_text
from GPT_SoVITS.feature_extractor import cnhubert, cnroberta
from GPT_SoVITS.eres2net.sv import SV
from GPT_SoVITS.SoVITS.module.mel_processing import spectrogram_torch
from config import tts_config


class TTS:
    def __init__(
        self,
        gpt_cache: list[int] = [500],
        sovits_cache: list[int] = [50, 300],
        device: str = None,
        is_half: bool = None,
        use_g2pw: bool = False,
        use_bert: bool = False,
    ):
        if not device is None:
            tts_config.device = device
        if not is_half is None:
            tts_config.is_half = is_half
            tts_config.dtype = torch.float16 if is_half else torch.float32

        tts_config.gpt_cache = gpt_cache
        tts_config.sovits_cache = sovits_cache

        self.gpt_models: dict[str, Gpt] = {}
        self.sovits_models: dict[str, Sovits] = {}
        self.resample_transform_dict = {}
        self.spk_audio_cache = {}
        self.prompt_audio_cache = {}

        check_pretrained_models()
        if use_bert and not os.path.exists("GPT_SoVITS\pretrained_models\chinese-roberta-wwm-ext-large"):
            download_model(
                url="/GPTSoVITS-RT/resolve/master/chinese-roberta.zip",
                zip_filename="GPT_SoVITS\pretrained_models\chinese-roberta-wwm-ext-large.zip"
            )
        if use_g2pw and not os.path.exists("GPT_SoVITS\pretrained_models\G2PW"):
            download_model(
                url="/GPTSoVITS-RT/resolve/master/G2PW.zip",
                zip_filename="GPT_SoVITS\pretrained_models\G2PW.zip"
            )

        self.cnhubert_path = "GPT_SoVITS/pretrained_models/chinese-hubert-base"
        self.cnroberta_path = "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large"
        self.sv_path = "GPT_SoVITS/pretrained_models/sv/pretrained_eres2netv2w24s4ep4.ckpt"
        self.default_gpt_path = "GPT_SoVITS/pretrained_models/s1v3.ckpt"
        self.default_sovits_path = "GPT_SoVITS/pretrained_models/v2Pro/s2Gv2ProPlus.pth"

        if use_bert:
            tts_config.cnroberta = cnroberta.CNRoberta(self.cnroberta_path)

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

        if use_g2pw:
            tts_config.language_module_map["zh"] = "chinese2"
        
        logging.info(f"Device: {tts_config.device}")
        logging.info(f"Half: {tts_config.is_half}, dtype: {tts_config.dtype}")

    
    @torch.inference_mode()
    def infer(
        self,
        spk_audio_path: str,
        prompt_audio_path: str,
        prompt_audio_text: str,
        prompt_audio_language: str,
        text: str,
        text_language: str,
        top_k: int = 15,
        top_p: float = 1.0,
        temperature: float = 1.0,
        noise_scale: float = 0.5,
        speed: float = 1.0,
        gpt_cache: int = -1,
        gpt_model: str = None,
        sovits_model: str = None,
    ):
        logging.info(f"Starting inference for text: '{text[:20]}...'")
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

        if gpt_model not in self.gpt_models:
            self.load_gpt_model(gpt_model)
        if sovits_model not in self.sovits_models:
            self.load_sovits_model(sovits_model)
        
        if spk_audio_path not in self.spk_audio_cache:
            self.cache_spk_audio(spk_audio_path)
        if prompt_audio_path not in self.prompt_audio_cache:
            self.cache_prompt_audio({"audio":prompt_audio_path, "language":prompt_audio_language, "text":prompt_audio_text})
        
        ge = self.spk_audio_cache[spk_audio_path]["ge"]
        prompt = self.prompt_audio_cache[prompt_audio_path]["prompt"]
        phones1 = self.prompt_audio_cache[prompt_audio_path]["phones1"]
        bert1 = self.prompt_audio_cache[prompt_audio_path]["bert1"]

        gpt = self.gpt_models[gpt_model]
        sovits = self.sovits_models[sovits_model]
        t2s_model = gpt.t2s_model
        vq_model = sovits.vq_model

        logging.debug("Processing text to phones and BERT features...")
        phones2, word2ph, bert2, _ = get_phones_and_bert(text, text_language)
        all_phoneme_ids = torch.LongTensor(phones1 + phones2).to(tts_config.device).unsqueeze(0)
        bert = torch.cat([bert1, bert2], dim=1)
        bert = bert.to(tts_config.device).unsqueeze(0)

        logging.debug("Running GPT inference (Text-to-Semantic)...")
        pred_semantic = t2s_model.infer(
            all_phoneme_ids,
            prompt,
            bert,
            max_kv_cache=gpt_cache,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
        )

        logging.debug("Running SoVITS inference (Semantic-to-Waveform)...")
        phones2_tensor = torch.LongTensor(phones2).to(tts_config.device).unsqueeze(0)
        phones2_lengths = torch.LongTensor([phones2_tensor.size(-1)]).to(tts_config.device)
        encoded_text, text_mask = vq_model.enc_p.text_encode(phones2_tensor, phones2_lengths)

        audio, attn = vq_model.decode(
            pred_semantic, encoded_text, text_mask, ge, noise_scale=noise_scale, speed=speed
        )

        audio = audio[0, 0, :].cpu().numpy()
        attn = attn.cpu().numpy()
        assign, _ = self.viterbi_monotonic(attn)
        subtitles = self.get_subtitles(word2ph, assign, speed)

        max_audio = np.abs(audio).max()
        if max_audio > 1:
            audio = audio / max_audio
        audio = audio.astype(np.float32)

        samplerate = vq_model.samples_per_frame * vq_model.hz
        
        audio_len_s = float(len(audio)) / float(samplerate)

        results = {
            "audio_data": audio,
            "samplerate": samplerate,
            "start_s": 0.0,
            "end_s": audio_len_s,
            "subtitles": subtitles,
        }

        logging.info(f"Inference complete. Generated {audio_len_s:.2f}s of audio.")
        return results

    @torch.inference_mode()
    def infer_stream(
        self,
        spk_audio_path: str,
        prompt_audio_path: str,
        prompt_audio_text: str,
        prompt_audio_language: str,
        text: str,
        text_language: str,
        cut_punds: dict = {",", ".", ";", "?", "!", "、", "，", "。", "？", "！", ";", "："},
        cut_minlen: int = 5,
        cut_mute: int = 0.2,
        stream_mode: str = "token" or "sentence",
        stream_chunk: int = 25,
        overlap_len: int = 10,
        top_k: int = 15,
        top_p: float = 1.0,
        temperature: float = 1.0,
        noise_scale: float = 0.5,
        gpt_cache: int = -1,
        gpt_model: str = None,
        sovits_model: str = None,
    ):
        logging.info(f"Starting Stream inference for text: '{text[:20]}...'")
        if stream_mode == "sentence": stream_chunk = 10000
        
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

        if gpt_model not in self.gpt_models:
            self.load_gpt_model(gpt_model)
        if sovits_model not in self.sovits_models:
            self.load_sovits_model(sovits_model)
        
        if spk_audio_path not in self.spk_audio_cache:
            self.cache_spk_audio(spk_audio_path)
        if prompt_audio_path not in self.prompt_audio_cache:
            self.cache_prompt_audio({"audio":prompt_audio_path, "language":prompt_audio_language, "text":prompt_audio_text})
        
        ge = self.spk_audio_cache[spk_audio_path]["ge"]
        prompt = self.prompt_audio_cache[prompt_audio_path]["prompt"]
        phones1 = self.prompt_audio_cache[prompt_audio_path]["phones1"]
        bert1 = self.prompt_audio_cache[prompt_audio_path]["bert1"]

        gpt = self.gpt_models[gpt_model]
        sovits = self.sovits_models[sovits_model]
        t2s_model = gpt.t2s_model
        vq_model = sovits.vq_model
        overlap_samples = overlap_len * vq_model.samples_per_frame

        audio_len_s = 0
        last_end_s = 0
        first_cut = True

        text_cuts = cut_text(text, cut_punds, cut_minlen)
        for i, text_cut in enumerate(text_cuts):
            logging.info(f"Processing segment {i+1}/{len(text_cuts)}: '{text_cut[:20]}...'")
            phones2, word2ph, bert2, _ = get_phones_and_bert(text_cut, text_language)
            all_phoneme_ids = torch.LongTensor(phones1 + phones2).to(tts_config.device).unsqueeze(0)
            bert = torch.cat([bert1, bert2], dim=1)
            bert = bert.to(tts_config.device).unsqueeze(0)

            phones2_tensor = torch.LongTensor(phones2).to(tts_config.device).unsqueeze(0)
            phones2_lengths = torch.LongTensor([phones2_tensor.size(-1)]).to(tts_config.device)
            encoded_text, text_mask = vq_model.enc_p.text_encode(phones2_tensor, phones2_lengths)

            samplerate = vq_model.samples_per_frame * vq_model.hz
            
            generator = t2s_model.infer_stream(
                all_phoneme_ids,
                prompt,
                bert,
                max_kv_cache=gpt_cache,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
                stream_chunk=stream_chunk,
            )

            last_attn = None
            last_subtitles_end = 0
            last_overlap_audio = None
            chunk_idx = 0
            for pred_semantic, is_final in generator:
                audio, attn = vq_model.decode(
                    pred_semantic,
                    encoded_text,
                    text_mask,
                    ge,
                    noise_scale=noise_scale,
                    stream_mode=True,
                    valid_start_idx=max(0, chunk_idx * stream_chunk * 2 - overlap_len),
                    overlap_len=overlap_len,
                )

                if not last_overlap_audio is None:
                    alpha = torch.linspace(0, 1, overlap_samples, device=tts_config.device, dtype=tts_config.dtype).view(1, 1, -1)
                    overlap_audio = audio[:, :, :overlap_samples]
                    overlap_audio_faded = last_overlap_audio * (1 - alpha) + overlap_audio * alpha
                    audio = torch.cat([overlap_audio_faded, audio[:, :, overlap_samples:]], dim=-1)
                last_overlap_audio = audio[:, :, -overlap_samples:]

                if not is_final:
                    audio = audio[:, :, :-overlap_samples]
                    attn = attn[:, :-overlap_len, :]
                
                audio = audio[0, 0, :].cpu().numpy()
                attn = attn.cpu().numpy()

                if not last_attn is None:
                    attn[:, :last_attn.shape[1], :] = last_attn

                assign, last_attn = self.viterbi_monotonic(attn, last_N_back=False)
                subtitles = self.get_subtitles(word2ph, assign, last_end_s=last_end_s)
                new_subtitles = subtitles[last_subtitles_end:]
                last_subtitles_end = len(subtitles)-1
                if not is_final and new_subtitles:
                    new_subtitles[-1]['end_s'] = None
                
                if not first_cut and chunk_idx == 0:
                    audio = np.concatenate([np.zeros((int(cut_mute*samplerate),), dtype=audio.dtype), audio])
                audio = audio.astype(np.float32)

                audio_len_s += float(len(audio)) / float(samplerate)

                results = {
                    "audio_data": audio,
                    "samplerate": samplerate,
                    "new_subtitles": new_subtitles,
                }

                yield results
                chunk_idx += 1
            
            if new_subtitles:
                last_end_s = new_subtitles[-1]['end_s'] + cut_mute
            first_cut = False
            vq_model.enc_p.y_overlap = None
        
        logging.info(f"Stream inference complete. Generated {audio_len_s:.2f}s of audio.")
    
    @torch.inference_mode()
    def infer_vc(
        self,
        spk_audio_path: str,
        prompt_audio_path: str,
        prompt_audio_text: str,
        prompt_audio_language: str,
        noise_scale: float = 0.5,
        speed: float = 1.0,
        sovits_model: str = None,
    ):
        logging.info(f"Starting VC inference. Prompt audio: {prompt_audio_path}")
        prompt_audio_language = self.dict_language[prompt_audio_language]

        if sovits_model is None:
            if len(self.sovits_models) > 0:
                sovits_model = list(self.sovits_models.keys())[0]
            else:
                sovits_model = self.default_sovits_path
        
        logging.info(f"Using SoVITS model: {sovits_model}")

        if sovits_model not in self.sovits_models:
            self.load_sovits_model(sovits_model)
        
        if spk_audio_path not in self.spk_audio_cache:
            self.cache_spk_audio(spk_audio_path)
        
        ge = self.spk_audio_cache[spk_audio_path]["ge"]
        sovits = self.sovits_models[sovits_model]
        vq_model = sovits.vq_model

        logging.debug("Extracting semantic features from prompt audio...")
        ssl_model = cnhubert.CNHubert(self.cnhubert_path)
        ssl_model.eval().to(tts_config.device)
        ssl_model = ssl_model.half() if tts_config.is_half else ssl_model
        prompt = self.get_prompt(ssl_model, sovits, prompt_audio_path)
        del ssl_model
        self.empty_cache()

        logging.debug("Processing text to phones and BERT features...")
        phones, word2ph, _, _ = get_phones_and_bert(prompt_audio_text, prompt_audio_language)

        phones_tensor = torch.LongTensor(phones).to(tts_config.device).unsqueeze(0)
        phones_lengths = torch.LongTensor([phones_tensor.size(-1)]).to(tts_config.device)
        encoded_text, text_mask = vq_model.enc_p.text_encode(phones_tensor, phones_lengths)

        logging.debug("Running SoVITS inference (Semantic-to-Waveform)...")
        audio, attn = vq_model.decode(
            prompt.unsqueeze(0), encoded_text, text_mask, ge, noise_scale=noise_scale, speed=speed
        )

        audio = audio[0, 0, :].cpu().numpy()
        attn = attn.cpu().numpy()
        assign, _ = self.viterbi_monotonic(attn)
        subtitles = self.get_subtitles(word2ph, assign, speed)

        max_audio = np.abs(audio).max()
        if max_audio > 1:
            audio = audio / max_audio
        audio = audio.astype(np.float32)

        samplerate = vq_model.samples_per_frame * vq_model.hz
        
        audio_len_s = float(len(audio)) / float(samplerate)

        results = {
            "audio_data": audio,
            "samplerate": samplerate,
            "start_s": 0.0,
            "end_s": audio_len_s,
            "subtitles": subtitles,
        }

        logging.info(f"VC Inference complete. Generated {audio_len_s:.2f}s of audio.")
        return results
    
    def init_language_module(self, languages: str|list[str]):
        if isinstance(languages, str): languages = [languages]
        for language in languages:
            if language in self.dict_language:
                lang_code = self.dict_language[language]
                if lang_code in tts_config.language_module_map:
                    language_module = tts_config.language_module_map[lang_code]
                    __import__("text." + language_module, fromlist=[language_module])
                    logging.info(f'Loaded language module: {language}')
                else:
                    logging.error(f'Failed to import module for language "{language}"')
            else:
                logging.warning(f'Language "{language}" not found.')

    def load_gpt_model(self, model_paths: str|list[str] = "GPT_SoVITS/pretrained_models/s1v3.ckpt"):
        if isinstance(model_paths, str): model_paths = [model_paths]
        for model_path in model_paths:
            self.gpt_models[model_path] = get_gpt_weights(model_path)
            logging.info(f'Loaded GPT model: {model_path}')
    
    def load_sovits_model(self, model_paths: str|list[str] = "GPT_SoVITS/pretrained_models/v2Pro/s2Gv2ProPlus.pth"):
        if isinstance(model_paths, str): model_paths = [model_paths]
        for model_path in model_paths:
            self.sovits_models[model_path] = get_sovits_weights(model_path)
            logging.info(f'Loaded SoVITS model: {model_path}')
    
    def unload_gpt_model(self, model_paths: str|list[str]):
        if isinstance(model_paths, str): model_paths = [model_paths]
        for model_path in model_paths:
            if model_path in self.gpt_models:
                del self.gpt_models[model_path]
                logging.info(f'Unloaded GPT model: {model_path}')
            else:
                logging.warning(f'GPT model {model_path} not found.')
        self.empty_cache()
    
    def unload_sovits_model(self, model_paths: str|list[str]):
        if isinstance(model_paths, str): model_paths = [model_paths]
        for model_path in model_paths:
            if model_path in self.sovits_models:
                del self.sovits_models[model_path]
                logging.info(f'Unloaded SoVITS model: {model_path}')
            else:
                logging.warning(f'SoVITS model {model_path} not found.')
        self.empty_cache()
    
    def get_gpt_list(self):
        return list(self.gpt_models.keys())

    def get_sovits_list(self):
        return list(self.sovits_models.keys())
    
    @torch.inference_mode()
    def cache_spk_audio(self, spk_audio_paths: str|list[str]):
        if isinstance(spk_audio_paths, str): spk_audio_paths = [spk_audio_paths]

        if not self.sovits_models:
            logging.error('No SoVITS models are currently loaded! Cannot cache speaker audio.')
            return

        model = self.sovits_models[next(iter(self.sovits_models))]

        sv_cn_model = SV(self.sv_path)

        for spk_audio_path in spk_audio_paths:
            refers, audio_tensor = self.get_spepc(model.hps, spk_audio_path)
            sv_emb = sv_cn_model.compute_embedding3(audio_tensor)
            sv_emb = sv_emb.half() if tts_config.is_half else sv_emb
            ge = model.vq_model.get_ge(refers, sv_emb)
            self.spk_audio_cache[spk_audio_path] = {"ge": ge}
            logging.info(f'Cached speaker audio: {spk_audio_path}')
        
        del sv_cn_model
        self.empty_cache()
    
    @torch.inference_mode()
    def cache_prompt_audio(self, prompt_audio_list: dict|list[dict]):
        if isinstance(prompt_audio_list, dict): prompt_audio_list = [prompt_audio_list]

        if not self.sovits_models:
            logging.error('No SoVITS models are currently loaded! Cannot cache prompt audio.')
            return

        model = self.sovits_models[next(iter(self.sovits_models))]

        ssl_model = cnhubert.CNHubert(self.cnhubert_path)
        ssl_model.eval().to(tts_config.device)
        ssl_model = ssl_model.half() if tts_config.is_half else ssl_model

        for prompt_audio in prompt_audio_list:
            prompt_audio_path, prompt_audio_language, prompt_audio_text = prompt_audio["audio"], prompt_audio["language"], prompt_audio["text"]
            prompt = self.get_prompt(ssl_model, model, prompt_audio_path)
            phones1, _, bert1, _ = get_phones_and_bert(prompt_audio_text, prompt_audio_language)
            self.prompt_audio_cache[prompt_audio_path] = {
                "prompt": prompt,
                "phones1": phones1,
                "bert1": bert1,
            }
            logging.info(f'Cached prompt audio: {prompt_audio_path}')
        
        del ssl_model
        self.empty_cache()
    
    def del_spk_audio(self, spk_audio_list: str|list[str]):
        if isinstance(spk_audio_list, str): spk_audio_list = [spk_audio_list]
        for spk_audio in spk_audio_list:
            if spk_audio in self.spk_audio_cache:
                del self.spk_audio_cache[spk_audio]
                logging.info(f'Deleted speaker audio from cache: {spk_audio}')
            else:
                logging.warning(f'Speaker audio {spk_audio} not found in cache.')
    
    def del_prompt_audio(self, prompt_audio_list: str|list[str]):
        if isinstance(prompt_audio_list, str): prompt_audio_list = [prompt_audio_list]
        for prompt_audio in prompt_audio_list:
            if prompt_audio in self.prompt_audio_cache:
                del self.prompt_audio_cache[prompt_audio]
                logging.info(f'Deleted prompt audio from cache: {prompt_audio}')
            else:
                logging.warning(f'Prompt audio {prompt_audio} not found in cache.')
    
    def get_spk_audio_list(self):
        return list(self.spk_audio_cache.keys())

    def get_prompt_audio_list(self):
        return list(self.prompt_audio_cache.keys())
    
    def resample(self, audio_tensor, sr0, sr1):
        key = "%s-%s" % (sr0, sr1)
        if key not in self.resample_transform_dict:
            self.resample_transform_dict[key] = torchaudio.transforms.Resample(sr0, sr1).to(tts_config.device)
        return self.resample_transform_dict[key](audio_tensor)
    
    def get_prompt(self, ssl_model: cnhubert.CNHubert, sovits_model: Sovits, audio_path: str):
        wav16k, sr_r = librosa.load(audio_path, sr=16000)
        wav16k = torch.from_numpy(wav16k).to(tts_config.device)
        wav16k = wav16k.half() if tts_config.is_half else wav16k

        ssl_content = ssl_model.model(wav16k.unsqueeze(0))["last_hidden_state"].transpose(1, 2)

        codes = sovits_model.vq_model.extract_latent(ssl_content)
        prompt_semantic = codes[0, 0]
        prompt = prompt_semantic.unsqueeze(0).to(tts_config.device)
        return prompt
    
    def get_spepc(self, hps, filename):
        sr1 = int(hps.data.sampling_rate)
        audio, sr0 = torchaudio.load(filename)

        audio = audio.to(tts_config.device)
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
        spec = spec.to(tts_config.dtype)
        audio = self.resample(audio, sr1, 16000)
        return spec, audio
    
    def get_subtitles(self, word2ph, assign, speed = 1, last_end_s = 0):
        frame_time = 0.02 / max(speed, 1e-6) # 50HZ -> 0.02s

        ph_end_s = []
        cur_ph = int(assign[0])
        for f in range(1, assign.shape[-1]):
            ph = int(assign[f])
            if ph != cur_ph:
                ph_end_s.append(f * frame_time)
                cur_ph = ph
        ph_end_s.append(assign.shape[-1] * frame_time)

        idx = -1
        end_s = last_end_s
        subtitles = []
        for i in range(len(word2ph["word"])):
            word, ph = word2ph["word"][i], word2ph["ph"][i]
            start_s = end_s
            if idx+ph >= len(ph_end_s): break
            end_s = ph_end_s[idx+ph] + last_end_s
            idx += ph

            if word == '':
                if subtitles:
                    subtitles[-1]["end_s"] = end_s
            else:
                subtitles.append({
                    "text": word,
                    "start_s": start_s,
                    "end_s": end_s
                })
        
        return subtitles

    def viterbi_monotonic(self, attn: np.ndarray, last_N_back: bool = True):
        B, T, N = attn.shape
        
        eps = 1e-8
        cost = -np.log(attn + eps)
        
        dp = np.empty((B, T, N), dtype=cost.dtype)
        prev = np.zeros((B, T, N), dtype=np.uint8)
        
        dp[:, 0, 0] = cost[:, 0, 0]
        if N > 1:
            dp[:, 0, 1:] = float("inf")
        
        for t_i in range(1, T):
            dp[:, t_i, 0] = dp[:, t_i - 1, 0] + cost[:, t_i, 0]
            prev[:, t_i, 0] = 0
            
            if N > 1:
                stay = dp[:, t_i - 1, 1:] + cost[:, t_i, 1:]
                move = dp[:, t_i - 1, :-1] + cost[:, t_i, 1:]
                better_move = move < stay
                dp[:, t_i, 1:] = np.where(better_move, move, stay)
                prev[:, t_i, 1:] = better_move.astype(np.uint8)
        
        assign_paths = np.empty((B, T), dtype=np.int64)
        
        for b in range(B):
            if last_N_back:
                j = N - 1
            else:
                j = np.argmin(dp[b, T - 1, :])
            for t_i in range(T - 1, -1, -1):
                assign_paths[b, t_i] = j
                if t_i > 0 and prev[b, t_i, j] == 1:
                    j = max(j - 1, 0)

        # 使用熵来评估路径分配的均匀性
        state_counts = np.zeros((B, N), dtype=np.float32)
        for b in range(B):
            state_counts[b] = np.bincount(assign_paths[b], minlength=N)
        
        state_probs = state_counts / T
        state_probs = state_probs + 1e-10
        entropy = -(state_probs * np.log(state_probs)).sum(axis=1)
        max_entropy = np.log(N)
        uniformity_scores = entropy / max_entropy

        best_idx = np.argmax(uniformity_scores)
        best_path = assign_paths[best_idx]
        best_attn = attn[best_idx:best_idx+1]
        return best_path, best_attn
    
    def empty_cache(self):
        try:
            gc.collect()
            if "cuda" in str(tts_config.device):
                torch.cuda.empty_cache()
            elif str(tts_config.device) == "mps":
                torch.mps.empty_cache()
        except:
            pass