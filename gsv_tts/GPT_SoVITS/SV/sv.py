import torch
from ...Config import Config
from .ERes2NetV2 import ERes2NetV2
import torchaudio.compliance.kaldi as Kaldi


class ERes2Net:
    def __init__(self, sv_path, tts_config: Config):
        pretrained_state = torch.load(sv_path, map_location="cpu", weights_only=False)
        self.embedding_model = ERes2NetV2(baseWidth=24, scale=4, expansion=4)
        self.embedding_model.load_state_dict(pretrained_state)
        self.embedding_model.eval()
        self.embedding_model = self.embedding_model.to(tts_config.device, tts_config.dtype)
        self.dtype = tts_config.dtype

    def compute_embedding3(self, wav):
        with torch.no_grad():
            feat = torch.stack(
                [Kaldi.fbank(wav0.unsqueeze(0).float(), num_mel_bins=80, sample_frequency=16000, dither=0) for wav0 in wav]
            ).to(self.dtype)
            sv_emb = self.embedding_model.forward3(feat)
        return sv_emb
