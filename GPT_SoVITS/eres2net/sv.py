import torch
from config import tts_config
from eres2net.ERes2NetV2 import ERes2NetV2
import eres2net.kaldi as Kaldi


class SV:
    def __init__(self, sv_path):
        pretrained_state = torch.load(sv_path, map_location="cpu", weights_only=False)
        embedding_model = ERes2NetV2(baseWidth=24, scale=4, expansion=4)
        embedding_model.load_state_dict(pretrained_state)
        embedding_model.eval()
        self.embedding_model = embedding_model
        self.embedding_model = self.embedding_model.to(tts_config.device)

    def compute_embedding3(self, wav):
        with torch.no_grad():
            feat = torch.stack(
                [Kaldi.fbank(wav0.unsqueeze(0), num_mel_bins=80, sample_frequency=16000, dither=0) for wav0 in wav]
            )
            sv_emb = self.embedding_model.forward3(feat)
        return sv_emb
