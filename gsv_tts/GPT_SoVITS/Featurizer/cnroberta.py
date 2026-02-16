import torch
import torch.nn as nn
from ...config import Config
from transformers import AutoModelForMaskedLM, AutoTokenizer


class CNRoberta(nn.Module):
    def __init__(self, base_path, tts_config: Config):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(base_path)
        self.bert_model = AutoModelForMaskedLM.from_pretrained(base_path)
        self.bert_model.eval()
        self.bert_model.to(tts_config.device)
        if tts_config.is_half: 
            self.bert_model = self.bert_model.half()
    
    def forward(self, texts: list[str], word2ph_list: list[list[int]]):
        with torch.no_grad():
            sep = self.tokenizer.sep_token
            combined_text = sep.join(texts)
            
            inputs = self.tokenizer(combined_text, return_tensors="pt").to(self.bert_model.device)
            
            res = self.bert_model(**inputs, output_hidden_states=True)
            hidden_states = res["hidden_states"][-3] 
            input_ids = inputs["input_ids"][0]

            special_tokens = {self.tokenizer.cls_token_id, self.tokenizer.sep_token_id, self.tokenizer.pad_token_id}
            mask = [i for i, tid in enumerate(input_ids.tolist()) if tid not in special_tokens]
            all_char_features = hidden_states[0, mask, :]

            batch_phone_features = []
            current_pos = 0
            for i in range(len(texts)):
                num_chars = len(word2ph_list[i])
                char_feature = all_char_features[current_pos : current_pos + num_chars]
                current_pos += num_chars
                
                repeats = torch.tensor(word2ph_list[i], device=char_feature.device)
                phone_feature = torch.repeat_interleave(char_feature, repeats, dim=0)
                
                batch_phone_features.append(phone_feature)

            return batch_phone_features