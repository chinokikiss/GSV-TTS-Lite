import torch
import torch.nn as nn
from config import tts_config
from transformers import AutoModelForMaskedLM, AutoTokenizer

class CNRoberta(nn.Module):
    def __init__(self, base_path: str = None):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(base_path)
        self.bert_model = AutoModelForMaskedLM.from_pretrained(base_path)
        self.bert_model = self.bert_model.eval()
        self.bert_model = self.bert_model.to(tts_config.device)
        if tts_config.is_half: self.bert_model = self.bert_model.half()
    
    def forward(self, text: str, word2ph: list):
        with torch.no_grad():
            inputs = self.tokenizer(text, return_tensors="pt")
            for i in inputs:
                inputs[i] = inputs[i].to(tts_config.device)
            res = self.bert_model(**inputs, output_hidden_states=True)
            res = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()[1:-1]
        assert len(word2ph) == len(text)
        phone_level_feature = []
        for i in range(len(word2ph)):
            repeat_feature = res[i].repeat(word2ph[i], 1)
            phone_level_feature.append(repeat_feature)
        phone_level_feature = torch.cat(phone_level_feature, dim=0)
        return phone_level_feature.T.to(dtype=tts_config.dtype, device=tts_config.device)