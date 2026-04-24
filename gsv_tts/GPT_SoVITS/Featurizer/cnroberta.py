import os
import torch
import torch.nn as nn
import numpy as np
from ...Config import Config
from transformers import AutoModelForMaskedLM, AutoTokenizer
from typing import List
from pathlib import Path

try:
    import onnxruntime as ort
    HAS_ONNXRUNTIME = True
except ImportError:
    HAS_ONNXRUNTIME = False


class CNRobertaONNX(nn.Module):
    """
    CNRoberta INT8 Dynamic ONNX 推理实现
    专为 CPU 优化，比 PyTorch 快 1.6x，内存占用更低
    """
    
    def __init__(self, base_path, tts_config: Config):
        super().__init__()
        self.tts_config = tts_config
        self.base_path = Path(base_path) if isinstance(base_path, str) else base_path
        
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.base_path))
        self.session = None
        self._create_session()
    
    def _create_session(self):
        """创建 ONNX Runtime Session (CPU 优化)"""
        onnx_path = self.base_path / "cnroberta_int8_dynamic.onnx"
        
        if not onnx_path.exists():
            raise FileNotFoundError(f"INT8 ONNX 模型不存在: {onnx_path}")
        
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.enable_mem_pattern = True
        sess_options.enable_mem_reuse = True
        
        # CPU 线程优化
        import psutil
        cpu_count = psutil.cpu_count(logical=False) or 4
        sess_options.intra_op_num_threads = max(1, cpu_count)
        sess_options.inter_op_num_threads = max(1, cpu_count)
        
        self.session = ort.InferenceSession(
            str(onnx_path),
            sess_options=sess_options,
            providers=["CPUExecutionProvider"],
        )
        
        self.output_names = [out.name for out in self.session.get_outputs()]
    
    def forward(self, word2ph_list: List):
        with torch.no_grad():
            texts = ["".join(word2ph["word"]) for word2ph in word2ph_list]
            
            inputs = self.tokenizer(
                texts, 
                return_tensors="np", 
                padding=True, 
                truncation=True, 
                max_length=512
            )
            
            feed_dict = {
                "input_ids": inputs["input_ids"].astype(np.int64),
                "attention_mask": inputs["attention_mask"].astype(np.int64),
            }
            
            outputs = self.session.run(self.output_names, feed_dict)
            hidden_states_np = outputs[0]
            
            # ONNX INT8 输出为 float32，转换为 torch tensor
            hidden_states = torch.from_numpy(hidden_states_np)
            
            batch_phone_features = []
            for i in range(len(texts)):
                mask = inputs["attention_mask"][i] == 1
                char_features = hidden_states[i][mask]
                char_features = char_features[1:-1, :]
                
                repeats = torch.tensor(word2ph_list[i]["ph"])
                phone_feature = torch.repeat_interleave(char_features, repeats, dim=0)
                
                batch_phone_features.append(phone_feature)

            return batch_phone_features
    
    def cleanup(self):
        """释放 ONNX Session 资源"""
        if self.session is not None:
            del self.session
            self.session = None


class CNRoberta(nn.Module):
    """
    CNRoberta 兼容接口
    - GPU: 使用 PyTorch 模型 (CUDA 加速)
    - CPU: 使用 INT8 Dynamic ONNX (更快、更省内存)
    """
    
    def __init__(self, base_path, tts_config: Config):
        super().__init__()
        self.base_path = base_path
        self.tts_config = tts_config
        self.use_onnx = False
        
        # 判断是否使用 ONNX (仅 CPU 场景)
        if tts_config.device.type == "cpu" and HAS_ONNXRUNTIME:
            onnx_path = Path(base_path) / "cnroberta_int8_dynamic.onnx" if isinstance(base_path, str) else base_path / "cnroberta_int8_dynamic.onnx"
            if onnx_path.exists():
                self.use_onnx = True
        
        if self.use_onnx:
            self.model = CNRobertaONNX(base_path, tts_config)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(base_path)
            self.bert_model = AutoModelForMaskedLM.from_pretrained(base_path)
            self.bert_model.eval()
            self.bert_model.to(tts_config.device, tts_config.dtype)
            self.model = None
    
    def forward(self, word2ph_list: List):
        if self.use_onnx:
            return self.model(word2ph_list)
        else:
            return self._forward_pytorch(word2ph_list)
    
    def _forward_pytorch(self, word2ph_list: List):
        with torch.no_grad():
            texts = ["".join(word2ph["word"]) for word2ph in word2ph_list]

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
                
                repeats = torch.tensor(word2ph_list[i]["ph"], device=char_features.device)
                phone_feature = torch.repeat_interleave(char_features, repeats, dim=0)
                
                batch_phone_features.append(phone_feature)

            return batch_phone_features
