import torch
from config import tts_config
from io import BytesIO
from GPT_SoVITS.SoVITS.models import SynthesizerTrn
from GPT_SoVITS.GPT.t2s_model import Text2SemanticDecoder
from GPT_SoVITS.utils import DictToAttrRecursive


class Sovits:
    def __init__(self, vq_model, hps):
        self.vq_model: SynthesizerTrn = vq_model
        self.hps = hps

def load_sovits_new(sovits_path):
    f = open(sovits_path, "rb")
    meta = f.read(2)
    if meta != b"PK":
        data = b"PK" + f.read()
        bio = BytesIO()
        bio.write(data)
        bio.seek(0)
        return torch.load(bio, map_location="cpu", weights_only=False)
    return torch.load(sovits_path, map_location="cpu", weights_only=False)

def get_sovits_weights(sovits_path):
    dict_s2 = load_sovits_new(sovits_path)
    hps = dict_s2["config"]
    hps = DictToAttrRecursive(hps)
    hps.model.semantic_frame_rate = "25hz"

    model_params_dict = vars(hps.model)

    vq_model = SynthesizerTrn(
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        **model_params_dict,
    )

    if tts_config.is_half == True:
        vq_model = vq_model.half().to(tts_config.device)
    else:
        vq_model = vq_model.to(tts_config.device)

    vq_model.eval()
    vq_model.load_state_dict(dict_s2["weight"], strict=False)
    vq_model.dec.remove_weight_norm()
    vq_model.warmup(tts_config.dtype, tts_config.device, tts_config.sovits_cache)

    sovits = Sovits(vq_model, hps)

    return sovits


class Gpt:
    def __init__(self, t2s_model):
        self.t2s_model: Text2SemanticDecoder = t2s_model

def get_gpt_weights(gpt_path):
    dict_s1 = torch.load(gpt_path, map_location="cpu", weights_only=False)
    config = dict_s1["config"]
    
    w_key_map = [
        ['self_attn.in_proj_weight', 'qkv_w'],
        ['self_attn.in_proj_bias', 'qkv_b'],
        ['self_attn.out_proj.weight', 'out_w'],
        ['self_attn.out_proj.bias', 'out_b'],
        ['linear1.weight', 'mlp_w1'],
        ['linear1.bias', 'mlp_b1'],
        ['linear2.weight', 'mlp_w2'],
        ['linear2.bias', 'mlp_b2'],
        ['norm1.weight', 'norm_w1'],
        ['norm1.bias', 'norm_b1'],
        ['norm2.weight', 'norm_w2'],
        ['norm2.bias', 'norm_b2']
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

    t2s_model = Text2SemanticDecoder(dict_s1)
    t2s_model.load_state_dict(dict_s1["weight"])
    if tts_config.is_half == True:
        t2s_model = t2s_model.half()
    else:
        t2s_model = t2s_model.float()
    t2s_model = t2s_model.to(tts_config.device)
    t2s_model.eval()
    t2s_model.warmup(tts_config.dtype, tts_config.device, tts_config.gpt_cache)

    gpt = Gpt(t2s_model)
    return gpt