from typing import List

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

from .utils import sample
from .embedding import SinePositionalEmbedding, TokenEmbedding


class T2SBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads, mlp_ratio=4):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.head_dim = hidden_dim // num_heads
        
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.qkv = nn.Linear(hidden_dim, hidden_dim * 3)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * mlp_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim * mlp_ratio, hidden_dim)
        )
    
    def process_prompt(
        self,
        x: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        attn_mask: torch.Tensor
    ):
        B, L, _ = x.shape

        residual = x
        
        qkv = self.qkv(x).view(B, L, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(2)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        k_cache[:, :, :L] = k
        v_cache[:, :, :L] = v

        x = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)

        x = x.transpose(1, 2).view(B, L, self.hidden_dim)
        x = self.out_proj(x)
        
        x = residual + x
        x = self.norm1(x)
        
        residual = x
        x = self.mlp(x)
        x = residual + x
        x = self.norm2(x)
        
        return x

    def decode_next_token(
        self,
        x: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        attn_mask: torch.Tensor,
        kv_cache_len: torch.Tensor,
        batch_indices: torch.Tensor
    ):
        B, L, _ = x.shape

        residual = x

        qkv = self.qkv(x).view(B, L, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(2)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        k_cache[batch_indices, :, kv_cache_len] = k.squeeze(2)
        v_cache[batch_indices, :, kv_cache_len] = v.squeeze(2)

        # kv_cache shape [batch_size, num_heads, kv_len, head_dim/num_heads]

        x = F.scaled_dot_product_attention(q, k_cache, v_cache, attn_mask=attn_mask)
        
        x = x.transpose(1, 2).view(B, L, self.hidden_dim)
        x = self.out_proj(x)
        
        x = residual + x
        x = self.norm1(x)
        
        residual = x
        x = self.mlp(x)
        x = residual + x
        x = self.norm2(x)
        
        return x


class T2STransformer(nn.Module):
    def __init__(self, num_blocks: int, blocks: List[T2SBlock]):
        super().__init__()
        self.num_blocks: int = num_blocks
        self.blocks = nn.ModuleList(blocks)

    def process_prompt(
        self,
        x: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        kv_cache_len: torch.Tensor,
        attn_mask: torch.Tensor
    ):
        for i in range(self.num_blocks):
            x = self.blocks[i].process_prompt(
                x, k_cache[i], v_cache[i], attn_mask
            )
        kv_cache_len.fill_(x.shape[1])
        return x

    def decode_next_token(
        self,
        x: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        kv_cache_len: torch.Tensor,
        attn_mask: torch.Tensor,
        batch_indices: torch.Tensor
    ):
        for i in range(self.num_blocks):
            x = self.blocks[i].decode_next_token(
                x, k_cache[i], v_cache[i], attn_mask, kv_cache_len, batch_indices
            )
        kv_cache_len += 1
        return x


class Bucket:
    cuda_graph: torch.cuda.CUDAGraph = None
    graph_xy_pos: torch.Tensor = None
    graph_xy_dec: torch.Tensor = None
    kv_cache_len: torch.Tensor = None
    k_cache: torch.Tensor = None
    v_cache: torch.Tensor = None
    decode_attn_mask: torch.Tensor = None
    max_kv_cache: int = None
    batch_size: int = None
    batch_indices: int = None

class Text2SemanticDecoder(nn.Module):
    def __init__(self, config):
        super(Text2SemanticDecoder, self).__init__()
        self.model_dim = config["model"]["hidden_dim"]
        self.embedding_dim = config["model"]["embedding_dim"]
        self.num_head = config["model"]["head"]
        self.num_layers = config["model"]["n_layer"]
        self.vocab_size = config["model"]["vocab_size"]
        self.phoneme_vocab_size = config["model"]["phoneme_vocab_size"]
        self.p_dropout = config["model"]["dropout"]
        self.EOS = config["model"]["EOS"]

        self.bert_proj = nn.Linear(1024, self.embedding_dim)
        self.ar_text_embedding = TokenEmbedding(
            self.embedding_dim,
            self.phoneme_vocab_size,
            self.p_dropout,
        )
        self.ar_text_position = SinePositionalEmbedding(
            self.embedding_dim,
            dropout=0.1,
            scale=False,
            alpha=True,
        )
        self.ar_audio_embedding = TokenEmbedding(
            self.embedding_dim,
            self.vocab_size,
            self.p_dropout,
        )
        self.ar_audio_position = SinePositionalEmbedding(
            self.embedding_dim,
            dropout=0.1,
            scale=False,
            alpha=True,
        )

        self.ar_predict_layer = nn.Linear(self.model_dim, self.vocab_size, bias=False)

        blocks = []
        for i in range(self.num_layers):
            block = T2SBlock(
                self.model_dim,
                self.num_head,
            )
            blocks.append(block)

        self.t2s_transformer = T2STransformer(self.num_layers, blocks)

        self.cuda_graph_buckets = {}
    
    @torch.inference_mode()
    def warmup(self, dtype, device, gpt_cache):
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())

        with torch.cuda.stream(s):
            for batch_size, max_kv_cache in gpt_cache:
                self.cuda_graph_buckets[(batch_size, max_kv_cache)] = Bucket()
                bucket: Bucket = self.cuda_graph_buckets[(batch_size, max_kv_cache)]

                bucket.max_kv_cache = max_kv_cache
                bucket.batch_size = batch_size

                bucket.kv_cache_len = torch.zeros((batch_size,), dtype=torch.int64, device=device)
                bucket.k_cache = torch.zeros((self.num_layers, batch_size, self.num_head, max_kv_cache, int(self.model_dim/self.num_head)), dtype=dtype, device=device)
                bucket.v_cache = torch.zeros((self.num_layers, batch_size, self.num_head, max_kv_cache, int(self.model_dim/self.num_head)), dtype=dtype, device=device)
                bucket.graph_xy_pos = torch.zeros((batch_size, 1, self.model_dim), dtype=dtype, device=device)
                bucket.decode_attn_mask = torch.zeros((batch_size, self.num_head, 1, max_kv_cache), dtype=torch.bool, device=device)
                bucket.batch_indices = torch.arange(batch_size, dtype=torch.int64, device=device)

                for _ in range(3):
                    self.t2s_transformer.decode_next_token(
                        bucket.graph_xy_pos, bucket.k_cache, bucket.v_cache, bucket.kv_cache_len, bucket.decode_attn_mask, bucket.batch_indices
                    )

                bucket.kv_cache_len.fill_(0) 

                torch.cuda.current_stream().synchronize()

                bucket.cuda_graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(bucket.cuda_graph):
                    bucket.graph_xy_dec = self.t2s_transformer.decode_next_token(
                        bucket.graph_xy_pos, bucket.k_cache, bucket.v_cache, bucket.kv_cache_len, bucket.decode_attn_mask, bucket.batch_indices
                    )
        
        torch.cuda.current_stream().wait_stream(s)
    
    def process_batch_data(self, x, y, bert_feature, x_lens, y_lens):
        device = x.device
        B = x.shape[0]

        xy_lens = x_lens + y_lens

        xy_len = xy_lens.max()
        x_len = x_lens.max()
        y_len = y_lens.max()

        xy_indices = torch.arange(xy_len, device=device).unsqueeze(0)
        x_mask1 = xy_indices < x_lens
        indices = torch.arange(x_len, device=device)
        x_mask2 = indices.unsqueeze(0) < x_lens

        y_mask1 = (x_lens <= xy_indices) & (xy_indices < xy_lens)
        indices = torch.arange(y_len, device=device).unsqueeze(0)
        y_mask2 = indices < y_lens


        x = self.ar_text_embedding(x)
        x = x + self.bert_proj(bert_feature)
        x = self.ar_text_position(x)

        y_emb = self.ar_audio_embedding(y)
        y_pos = self.ar_audio_position(y_emb)

        xy_pos = torch.zeros((B, xy_len, self.model_dim), dtype=y_pos.dtype, device=device)
        xy_pos[x_mask1] = x[x_mask2]
        xy_pos[y_mask1] = y_pos[y_mask2]


        # 音素可以关注自身(双向),但不能关注音频 音频可以关注自身(因果),也能关注音素(双向)
        prompt_attn_mask = torch.zeros((B, xy_len, xy_len), dtype=torch.bool, device=device)

        x_attn_mask = x_mask1.unsqueeze(1).expand(-1, x_len, -1).clone()
        prompt_attn_mask[x_mask1] = x_attn_mask[x_mask2]

        y_attn_mask = x_mask1.unsqueeze(1).expand(-1, y_len, -1).clone()
        tril_mask = torch.tril(torch.ones(B, y_len, xy_len, dtype=torch.bool, device=device))
        mask = xy_indices < (xy_len - x_lens)
        mask = mask.unsqueeze(1).expand(-1, y_len, -1)
        y_attn_mask[~y_attn_mask] = tril_mask[mask]
        prompt_attn_mask[y_mask1] = y_attn_mask[y_mask2]

        prompt_attn_mask = prompt_attn_mask.unsqueeze(1).expand(-1, self.num_head, -1, -1)

        return xy_pos, prompt_attn_mask

    def process_single_data(self, x, y, bert_feature):
        x_len = x.shape[1]
        x = self.ar_text_embedding(x)
        x = x + self.bert_proj(bert_feature)
        x = self.ar_text_position(x)

        y_len = y.shape[1]
        y_emb = self.ar_audio_embedding(y)
        y_pos = self.ar_audio_position(y_emb)

        xy_pos = torch.concat([x, y_pos], dim=1)

        B, device = x.shape[0], x.device

        # 音素可以关注自身(双向),但不能关注音频 音频可以关注自身(因果),也能关注音素(双向)
        xy_len = x_len + y_len
        x_attn_mask = F.pad(
            torch.ones((x_len, x_len), dtype=torch.bool),
            (0, y_len),
            value=False,
        )
        y_attn_mask = F.pad(
            torch.tril(torch.ones(y_len, y_len, dtype=torch.bool)),
            (x_len, 0),
            value=True,
        )
        prompt_attn_mask = (
            torch.concat([x_attn_mask, y_attn_mask], dim=0)
            .unsqueeze(0)
            .expand(B * self.num_head, -1, -1)
            .view(B, self.num_head, xy_len, xy_len)
            .to(device=device, dtype=torch.bool)
        )
        
        return xy_pos, prompt_attn_mask

    @torch.inference_mode()
    def infer(
        self,
        x: torch.LongTensor,
        y: torch.LongTensor,
        bert_feature: torch.LongTensor,
        max_kv_cache: int = -1,
        top_k: int = 15,
        top_p: int = 1.0,
        temperature: float = 1.0,
        repetition_penalty: float = 1.35,
    ):
        xy_pos, prompt_attn_mask = self.process_single_data(x, y, bert_feature)

        if (1, max_kv_cache) not in self.cuda_graph_buckets:
            available = next((k for k in self.cuda_graph_buckets if k[0] == 1), None)
            bucket: Bucket = self.cuda_graph_buckets[available]
            max_kv_cache = bucket.max_kv_cache
        else:
            bucket: Bucket = self.cuda_graph_buckets[(1, max_kv_cache)]

        bucket.kv_cache_len.fill_(0)
        bucket.k_cache.fill_(0)
        bucket.v_cache.fill_(0)

        pe_cache = self.ar_audio_position.alpha * self.ar_audio_position.pe
        pe_cache = pe_cache.transpose(0, 1)

        pre_tokens = y

        xy_dec = self.t2s_transformer.process_prompt(xy_pos, bucket.k_cache, bucket.v_cache, bucket.kv_cache_len, prompt_attn_mask)
        logits = self.ar_predict_layer(xy_dec[:, -1])
        samples = sample(logits[:, :-1], pre_tokens, top_k=top_k, top_p=top_p, repetition_penalty=repetition_penalty, temperature=temperature)[0]
        pre_tokens = torch.concat([pre_tokens, samples], dim=1)
        y_emb = self.ar_audio_embedding(samples)
        xy_pos = y_emb * self.ar_audio_position.x_scale + pe_cache[bucket.kv_cache_len]

        bucket.decode_attn_mask.fill_(False)
        bucket.decode_attn_mask[:, :, :, :bucket.kv_cache_len] = True
        
        for idx in tqdm(range(1, bucket.max_kv_cache - bucket.kv_cache_len + 1)):
            bucket.decode_attn_mask[:, :, :, bucket.kv_cache_len] = True
            bucket.graph_xy_pos.copy_(xy_pos)
            bucket.cuda_graph.replay()
            xy_dec = bucket.graph_xy_dec.clone()

            # xy_dec = self.t2s_transformer.decode_next_token(xy_pos, bucket.k_cache, bucket.v_cache, bucket.kv_cache_len)

            logits = self.ar_predict_layer(xy_dec[:, -1])

            samples = sample(logits, pre_tokens, top_k=top_k, top_p=top_p, repetition_penalty=repetition_penalty, temperature=temperature)[0]
            
            if samples[0, 0] == self.EOS:
                break

            pre_tokens = torch.concat([pre_tokens, samples], dim=1)

            y_emb = self.ar_audio_embedding(samples)
            xy_pos = y_emb * self.ar_audio_position.x_scale + pe_cache[bucket.kv_cache_len]

        return pre_tokens[:, -idx:].unsqueeze(0)
        
    @torch.inference_mode()
    def infer_stream(
        self,
        x: torch.LongTensor,
        y: torch.LongTensor,
        bert_feature: torch.LongTensor,
        max_kv_cache: int = -1,
        top_k: int = 15,
        top_p: int = 1.0,
        temperature: float = 1.0,
        repetition_penalty: float = 1.35,
        stream_chunk: int = 25,
        boost_first_chunk: bool = True,
        debug: bool = True,
    ):
        xy_pos, prompt_attn_mask = self.process_single_data(x, y, bert_feature)

        if (1, max_kv_cache) not in self.cuda_graph_buckets:
            available = next((k for k in self.cuda_graph_buckets if k[0] == 1), None)
            bucket: Bucket = self.cuda_graph_buckets[available]
            max_kv_cache = bucket.max_kv_cache
        else:
            bucket: Bucket = self.cuda_graph_buckets[(1, max_kv_cache)]

        bucket.kv_cache_len.fill_(0)
        bucket.k_cache.fill_(0)
        bucket.v_cache.fill_(0)

        pe_cache = self.ar_audio_position.alpha * self.ar_audio_position.pe
        pe_cache = pe_cache.transpose(0, 1)

        pre_tokens = y

        xy_dec = self.t2s_transformer.process_prompt(xy_pos, bucket.k_cache, bucket.v_cache, bucket.kv_cache_len, prompt_attn_mask)
        logits = self.ar_predict_layer(xy_dec[:, -1])
        samples = sample(logits[:, :-1], pre_tokens, top_k=top_k, top_p=top_p, repetition_penalty=repetition_penalty, temperature=temperature)[0]
        pre_tokens = torch.concat([pre_tokens, samples], dim=1)
        y_emb = self.ar_audio_embedding(samples)
        xy_pos = y_emb * self.ar_audio_position.x_scale + pe_cache[bucket.kv_cache_len]

        bucket.decode_attn_mask.fill_(False)
        bucket.decode_attn_mask[:, :, :, :bucket.kv_cache_len] = True
        
        first_chunk = True
        pre_chunk = None
        for idx in tqdm(range(1, bucket.max_kv_cache - bucket.kv_cache_len + 1), disable=not debug):
            bucket.decode_attn_mask[:, :, :, bucket.kv_cache_len] = True
            bucket.graph_xy_pos.copy_(xy_pos)
            bucket.cuda_graph.replay()
            xy_dec = bucket.graph_xy_dec.clone()

            # xy_dec = self.t2s_transformer.decode_next_token(xy_pos, bucket.k_cache, bucket.v_cache, bucket.kv_cache_len)

            logits = self.ar_predict_layer(xy_dec[:, -1])

            samples = sample(logits, pre_tokens, top_k=top_k, top_p=top_p, repetition_penalty=repetition_penalty, temperature=temperature)[0]
            
            if samples[0, 0] == self.EOS:
                break

            pre_tokens = torch.concat([pre_tokens, samples], dim=1)

            if idx % stream_chunk == 0:
                if not pre_chunk is None:
                    yield pre_chunk, False
                pre_chunk = pre_tokens[:, -idx:].unsqueeze(0)

                if boost_first_chunk:
                    if first_chunk:
                        first_chunk = False
                        yield pre_chunk, False
                        pre_chunk = None

            y_emb = self.ar_audio_embedding(samples)
            xy_pos = y_emb * self.ar_audio_position.x_scale + pe_cache[bucket.kv_cache_len]

        yield pre_tokens[:, -idx:].unsqueeze(0), True
    
    @torch.inference_mode()
    def infer_batched(
        self,
        x: List[torch.LongTensor],
        y: List[torch.LongTensor],
        bert_feature: List[torch.LongTensor],
        batch_size: int = -1,
        max_kv_cache: int = -1,
        top_k: int = 15,
        top_p: int = 1.0,
        temperature: float = 1.0,
        repetition_penalty: float = 1.35,
    ):
        B, device = len(x), x[0].device
        
        if (batch_size, max_kv_cache) not in self.cuda_graph_buckets:
            max_b = 0
            for k in self.cuda_graph_buckets:
                if k[0] <= B and k[0] > max_b:
                    max_b = k[0]
                    best = k
                    
            bucket: Bucket = self.cuda_graph_buckets[best]
            batch_size = bucket.batch_size
            max_kv_cache = bucket.max_kv_cache
        else:
            bucket: Bucket = self.cuda_graph_buckets[(batch_size, max_kv_cache)]

        bucket.kv_cache_len.fill_(0)
        bucket.k_cache.fill_(0)
        bucket.v_cache.fill_(0)

        batch_indices = torch.arange(batch_size, dtype=torch.int64, device=device)
        
        batch_x = pad_sequence(x[:batch_size], batch_first=True, padding_value=0)
        batch_y = pad_sequence(y[:batch_size], batch_first=True, padding_value=0)
        batch_bert_feature = pad_sequence(bert_feature[:batch_size], batch_first=True, padding_value=0)
        x_lens = torch.tensor([i.shape[0] for i in x[:batch_size]], device=device)
        y_lens = torch.tensor([i.shape[0] for i in y[:batch_size]], device=device)
        xy_lens = x_lens + y_lens

        xy_pos, prompt_attn_mask = self.process_batch_data(
            batch_x,
            batch_y,
            batch_bert_feature,
            x_lens.unsqueeze(1),
            y_lens.unsqueeze(1),
        )

        current_batch = batch_size

        pe_cache = self.ar_audio_position.alpha * self.ar_audio_position.pe
        pe_cache = pe_cache.transpose(0, 1)

        pre_tokens = torch.zeros((batch_size, max_kv_cache), dtype=torch.int64, device=device)
        pre_tokens[:, :y_lens.max()] = batch_y


        xy_dec = self.t2s_transformer.process_prompt(xy_pos, bucket.k_cache, bucket.v_cache, bucket.kv_cache_len, prompt_attn_mask)
        logits = self.ar_predict_layer(xy_dec[:, -1])

        bucket.kv_cache_len.copy_(xy_lens)

        samples = sample(logits[:, :-1], pre_tokens, top_k=top_k, top_p=top_p, repetition_penalty=repetition_penalty, temperature=temperature)[0]
        pre_tokens[batch_indices, bucket.kv_cache_len] = samples.squeeze()
        y_emb = self.ar_audio_embedding(samples)
        xy_pos = y_emb * self.ar_audio_position.x_scale + pe_cache[bucket.kv_cache_len]
        

        bucket.decode_attn_mask.fill_(False)
        indices = torch.arange(max_kv_cache, device=device)
        mask = indices[None, :] < bucket.kv_cache_len[:, None]
        bucket.decode_attn_mask.copy_(mask.view(batch_size, 1, 1, max_kv_cache))

        stop = False
        pred_semantic = []
        semantic_orig_idx = []
        batch_orig_idx = torch.linspace(0, batch_size-1, batch_size, dtype=torch.int64, device=device)
        decode_steps = torch.zeros(batch_size, dtype=torch.int64, device=device)
        ignore_batch = torch.zeros(batch_size, dtype=torch.bool, device=device)
        for idx in tqdm(range(1500)):
            decode_steps += 1

            bucket.decode_attn_mask[batch_indices, :, :, bucket.kv_cache_len] = True
            bucket.graph_xy_pos.copy_(xy_pos)
            bucket.cuda_graph.replay()
            xy_dec = bucket.graph_xy_dec.clone()

            # xy_dec = self.t2s_transformer.decode_next_token(xy_pos, bucket.k_cache, bucket.v_cache, bucket.kv_cache_len)

            logits = self.ar_predict_layer(xy_dec[:, -1])

            samples = sample(logits, pre_tokens, top_k=top_k, top_p=top_p, repetition_penalty=repetition_penalty, temperature=temperature)[0]
            
            eos_in_current_step = (samples[:, 0] == self.EOS) | (bucket.kv_cache_len == max_kv_cache)
            finished = (~ignore_batch) & eos_in_current_step

            if finished.any():
                finished_indices = torch.where(finished)[0]
                for i in finished_indices.tolist():
                    pred_semantic.append(pre_tokens[i, bucket.kv_cache_len[i]-decode_steps[i] : bucket.kv_cache_len[i]].clone())
                    semantic_orig_idx.append(batch_orig_idx[i].clone())
                    decode_steps[i] = 0
                    
                    if current_batch == B:
                        bucket.kv_cache_len[i] = 0
                        ignore_batch[i] = True
                        if ignore_batch.all():
                            stop = True
                            break
                    else:
                        single_x = x[current_batch]
                        single_y = y[current_batch]
                        single_bert_feature = bert_feature[current_batch]

                        xy_pos, prompt_attn_mask = self.process_single_data(
                            single_x.unsqueeze(0),
                            single_y.unsqueeze(0),
                            single_bert_feature.unsqueeze(0),
                        )

                        xy_dec = self.t2s_transformer.process_prompt(xy_pos, bucket.k_cache[:, i:i+1], bucket.v_cache[:, i:i+1], bucket.kv_cache_len[i:i+1], prompt_attn_mask)
                        logits = self.ar_predict_layer(xy_dec[:, -1])

                        bucket.kv_cache_len[i].copy_(single_x.shape[0] + single_y.shape[0])
                        pre_tokens[i].fill_(0)
                        pre_tokens[i, :single_y.shape[0]] = single_y

                        new_samples = sample(logits, pre_tokens[i:i+1], top_k=top_k, top_p=top_p, repetition_penalty=repetition_penalty, temperature=temperature)[0]
                        samples[i:i+1] = new_samples

                        bucket.decode_attn_mask[i:i+1].fill_(False)
                        indices = torch.arange(max_kv_cache, device=device)
                        mask = indices[None, :] < bucket.kv_cache_len[i:i+1, None]
                        bucket.decode_attn_mask[i:i+1].copy_(mask.view(1, 1, 1, max_kv_cache))

                        batch_orig_idx[i] = current_batch
                        current_batch += 1
                
                if stop:
                    break
            
            pre_tokens[batch_indices, bucket.kv_cache_len] = samples.squeeze()

            y_emb = self.ar_audio_embedding(samples)
            xy_pos = y_emb * self.ar_audio_position.x_scale + pe_cache[bucket.kv_cache_len]

        semantic_orig_idx = torch.tensor(semantic_orig_idx, device=device)
        return pred_semantic, semantic_orig_idx