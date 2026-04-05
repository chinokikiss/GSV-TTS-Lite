import torch
import torch.nn as nn
from ...Config import Config
from transformers import AutoModelForMaskedLM, AutoTokenizer
from typing import List

class CNRoberta(nn.Module):
    def __init__(self, base_path, tts_config: Config):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(base_path)
        self.bert_model = AutoModelForMaskedLM.from_pretrained(base_path)
        self.bert_model.eval()
        self.bert_model.to(tts_config.device, tts_config.dtype)
    
    def forward(self, texts: List[str], word2ph_list: List[List[int]]):
        with torch.no_grad():
            inputs = self.tokenizer(
                texts, 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=512
            ).to(self.bert_model.device)
            
            res = self.bert_model(**inputs, output_hidden_states=True)
            hidden_states = res["hidden_states"][-3]
            
            batch_phone_features = []
            for i in range(len(texts)):
                mask = inputs['attention_mask'][i] == 1
                char_features = hidden_states[i][mask]
                char_features = char_features[1:-1, :]
                
                repeats = torch.tensor(word2ph_list[i], device=char_features.device)
                phone_feature = torch.repeat_interleave(char_features, repeats, dim=0)
                
                batch_phone_features.append(phone_feature)

            return batch_phone_features