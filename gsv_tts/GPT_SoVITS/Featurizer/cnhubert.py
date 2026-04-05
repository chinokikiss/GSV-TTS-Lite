from transformers import logging as tf_logging

tf_logging.set_verbosity_error()

import logging

logging.getLogger("numba").setLevel(logging.WARNING)

from transformers import (
    Wav2Vec2FeatureExtractor,
    HubertModel,
)

import torch.nn as nn
import torch
from ...Config import Config


class CNHubert(nn.Module):
    def __init__(self, base_path, tts_config: Config):
        super().__init__()
        # 明确指定 dtype，确保 MPS 设备使用 float32
        self.model = HubertModel.from_pretrained(base_path, local_files_only=True, torch_dtype=tts_config.dtype)
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(base_path, local_files_only=True)
        self.eval()
        self = self.to(tts_config.device, tts_config.dtype)

    def forward(self, x):
        input_values = self.feature_extractor(x, return_tensors="pt", sampling_rate=16000).input_values.to(x.device)
        feats = self.model(input_values)["last_hidden_state"]
        return feats