import torch
import hashlib
from io import BytesIO

from .config import Config
from .GPT_SoVITS.SoVITS.models import SynthesizerTrn
from .GPT_SoVITS.GPT.t2s_model import Text2SemanticDecoder
from .GPT_SoVITS import utils

import sys
sys.modules['utils'] = utils


class Sovits:
    def __init__(self, vq_model, hps):
        self.vq_model: SynthesizerTrn = vq_model
        self.hps = hps

def get_hash_from_file(sovits_path):
    with open(sovits_path, "rb") as f:
        data = f.read(8192)
    hash_md5 = hashlib.md5()
    hash_md5.update(data)
    return hash_md5.hexdigest()

def load_sovits_new(sovits_path):
    hash = get_hash_from_file(sovits_path)

    f = open(sovits_path, "rb")
    meta = f.read(2)
    
    assert (hash in ["c7e9fce2223f3db685cdfa1e6368728a", "66b313e39455b57ab1b0bc0b239c9d0a"] or meta in [b"05", b"06"]), "sovits模型不是v2Pro版本，请检查模型文件。"

    if meta != b"PK":
        data = b"PK" + f.read()
        bio = BytesIO()
        bio.write(data)
        bio.seek(0)
        return torch.load(bio, map_location="cpu", weights_only=False)
    return torch.load(sovits_path, map_location="cpu", weights_only=False)

def get_sovits_weights(sovits_path, tts_config: Config):
    dict_s2 = load_sovits_new(sovits_path)
    
    hps = utils.DictToAttrRecursive(dict_s2["config"])
    hps.model.semantic_frame_rate = "25hz"
    
    vq_model = SynthesizerTrn(
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        **vars(hps.model),
    )

    if tts_config.is_half:
        vq_model = vq_model.half().to(tts_config.device)
    else:
        vq_model = vq_model.to(tts_config.device)
    
    vq_model.load_state_dict(dict_s2["weight"], strict=False)

    vq_model.eval()
    vq_model.dec.remove_weight_norm()
    vq_model.warmup(tts_config.dtype, tts_config.device, tts_config.sovits_cache)

    sovits = Sovits(vq_model, hps)

    return sovits


class Gpt:
    def __init__(self, t2s_model):
        self.t2s_model: Text2SemanticDecoder = t2s_model

def get_gpt_weights(gpt_path, tts_config: Config):
    dict_s1 = torch.load(gpt_path, map_location="cpu", weights_only=False)
    config = dict_s1["config"]
    
    w_key_map = [
        ['self_attn.in_proj_weight', 'qkv.weight'],
        ['self_attn.in_proj_bias', 'qkv.bias'],
        ['self_attn.out_proj.weight', 'out_proj.weight'],
        ['self_attn.out_proj.bias', 'out_proj.bias'],
        ['linear1.weight', 'mlp.0.weight'],
        ['linear1.bias', 'mlp.0.bias'],
        ['linear2.weight', 'mlp.2.weight'],
        ['linear2.bias', 'mlp.2.bias'],
        ['norm1.weight', 'norm1.weight'],
        ['norm1.bias', 'norm1.bias'],
        ['norm2.weight', 'norm2.weight'],
        ['norm2.bias', 'norm2.bias']
    ]

    for i in range(config["model"]["n_layer"]):
        original_l_key = f'model.h.layers.{i}.'
        new_l_key = f't2s_transformer.blocks.{i}.'
        for original_w_key, new_w_key in w_key_map:
            dict_s1["weight"][new_l_key+new_w_key] = dict_s1["weight"].pop(original_l_key+original_w_key)
    
    dict_s1["weight"] = {
        k.replace("model.", "", 1) if k.startswith("model.") else k: v 
        for k, v in dict_s1["weight"].items()
    }

    if tts_config.use_flash_attn:
        from .GPT_SoVITS.GPT.t2s_model_flash_attn import Text2SemanticDecoder as Text2SemanticDecoder_flash_attn
        t2s_model = Text2SemanticDecoder_flash_attn(config)
    else:
        t2s_model = Text2SemanticDecoder(config)

    if tts_config.is_half:
        t2s_model = t2s_model.half().to(tts_config.device)
    else:
        t2s_model = t2s_model.float().to(tts_config.device)
    
    t2s_model.load_state_dict(dict_s1["weight"])

    t2s_model.eval()
    t2s_model.warmup(tts_config.dtype, tts_config.device, tts_config.gpt_cache)

    gpt = Gpt(t2s_model)
    return gpt