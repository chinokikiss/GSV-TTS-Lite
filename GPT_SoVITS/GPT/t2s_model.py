from typing import List

import torch
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm

from GPT.utils import sample
from GPT.embedding import SinePositionalEmbedding, TokenEmbedding


class T2SBlock(nn.Module):
    def __init__(
        self,
        num_heads,
        hidden_dim: int,
        mlp_w1,
        mlp_b1,
        mlp_w2,
        mlp_b2,
        qkv_w,
        qkv_b,
        out_w,
        out_b,
        norm_w1,
        norm_b1,
        norm_eps1,
        norm_w2,
        norm_b2,
        norm_eps2,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim: int = hidden_dim
        self.head_dim = hidden_dim // num_heads

        self.mlp_w1 = nn.Parameter(mlp_w1)
        self.mlp_b1 = nn.Parameter(mlp_b1)
        self.mlp_w2 = nn.Parameter(mlp_w2)
        self.mlp_b2 = nn.Parameter(mlp_b2)
        
        self.qkv_w = nn.Parameter(qkv_w)
        self.qkv_b = nn.Parameter(qkv_b)
        self.out_w = nn.Parameter(out_w)
        self.out_b = nn.Parameter(out_b)
        
        self.norm_w1 = nn.Parameter(norm_w1)
        self.norm_b1 = nn.Parameter(norm_b1)
        self.norm_eps1 = norm_eps1
        
        self.norm_w2 = nn.Parameter(norm_w2)
        self.norm_b2 = nn.Parameter(norm_b2)
        self.norm_eps2 = norm_eps2
    
    def process_prompt(
        self,
        x: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        attn_mask: torch.Tensor
    ):
        q, k, v = F.linear(x, self.qkv_w, self.qkv_b).chunk(3, dim=-1)

        batch_size, kv_cache_len, q_len = x.shape[0], x.shape[1], q.shape[1]

        q = q.view(batch_size, q_len, self.num_heads, -1).transpose(1, 2)
        k = k.view(batch_size, k.shape[1], self.num_heads, -1).transpose(1, 2)
        v = v.view(batch_size, v.shape[1], self.num_heads, -1).transpose(1, 2)

        k_cache[:, :, :kv_cache_len] = k
        v_cache[:, :, :kv_cache_len] = v

        attn = F.scaled_dot_product_attention(q, k, v, attn_mask)

        attn = attn.transpose(1, 2).reshape(batch_size, q_len, -1)
        attn = F.linear(attn, self.out_w, self.out_b)

        x = x + attn
        x = F.layer_norm(x, [self.hidden_dim], self.norm_w1, self.norm_b1, self.norm_eps1)

        mlp = F.relu(F.linear(x, self.mlp_w1, self.mlp_b1))
        mlp = F.linear(mlp, self.mlp_w2, self.mlp_b2)

        x = x + mlp
        x = F.layer_norm(x, [self.hidden_dim], self.norm_w2, self.norm_b2, self.norm_eps2)
        
        return x

    def decode_next_token(
        self,
        x: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        attn_mask: torch.Tensor,
        kv_cache_len: torch.Tensor
    ):
        q, k, v = F.linear(x, self.qkv_w, self.qkv_b).chunk(3, dim=-1)

        batch_size, q_len = x.shape[0], q.shape[1]
        
        q = q.view(batch_size, q_len, self.num_heads, -1).transpose(1, 2)
        k = k.view(batch_size, k.shape[1], self.num_heads, -1).transpose(1, 2)
        v = v.view(batch_size, v.shape[1], self.num_heads, -1).transpose(1, 2)

        k_cache.index_copy_(2, kv_cache_len, k)
        v_cache.index_copy_(2, kv_cache_len, v)

        # kv_cache shape [batch_size, num_heads, kv_len, head_dim/num_heads]

        attn = F.scaled_dot_product_attention(q, k_cache, v_cache, attn_mask)

        attn = attn.transpose(1, 2).reshape(batch_size, q_len, -1)
        attn = F.linear(attn, self.out_w, self.out_b)

        x = x + attn
        x = F.layer_norm(x, [self.hidden_dim], self.norm_w1, self.norm_b1, self.norm_eps1)
        
        mlp = F.relu(F.linear(x, self.mlp_w1, self.mlp_b1))
        mlp = F.linear(mlp, self.mlp_w2, self.mlp_b2)

        x = x + mlp
        x = F.layer_norm(x, [self.hidden_dim], self.norm_w2, self.norm_b2, self.norm_eps2)
        
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
        attn_mask: torch.Tensor
    ):
        attn_mask.index_fill_(3, kv_cache_len, True)
        for i in range(self.num_blocks):
            x = self.blocks[i].decode_next_token(
                x, k_cache[i], v_cache[i], attn_mask, kv_cache_len
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

class Text2SemanticDecoder(nn.Module):
    def __init__(self, dict_s1):
        super(Text2SemanticDecoder, self).__init__()
        config = dict_s1["config"]
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

        state_dict = dict_s1["weight"]
        blocks = []
        for i in range(self.num_layers):
            prefix = f"t2s_transformer.blocks.{i}."       
            qkv_w = state_dict[prefix + "qkv_w"]
            qkv_b = state_dict[prefix + "qkv_b"]
            out_w = state_dict[prefix + "out_w"]
            out_b = state_dict[prefix + "out_b"]
            mlp_w1 = state_dict[prefix + "mlp_w1"]
            mlp_b1 = state_dict[prefix + "mlp_b1"]
            mlp_w2 = state_dict[prefix + "mlp_w2"]
            mlp_b2 = state_dict[prefix + "mlp_b2"]
            norm_w1 = state_dict[prefix + "norm_w1"]
            norm_b1 = state_dict[prefix + "norm_b1"]
            norm_eps1 = 1e-05
            norm_w2 = state_dict[prefix + "norm_w2"]
            norm_b2 = state_dict[prefix + "norm_b2"]
            norm_eps2 = 1e-05

            block = T2SBlock(
                self.num_head,
                self.model_dim,
                mlp_w1,
                mlp_b1,
                mlp_w2,
                mlp_b2,
                qkv_w,
                qkv_b,
                out_w,
                out_b,
                norm_w1,
                norm_b1,
                norm_eps1,
                norm_w2,
                norm_b2,
                norm_eps2,
            )
            blocks.append(block)

        self.t2s_transformer = T2STransformer(self.num_layers, blocks)

        self.cuda_graph_buckets = {}
    
    @torch.inference_mode()
    def warmup(self, dtype, device, max_kv_caches):
        batch_size = 1

        for max_kv_cache in max_kv_caches:
            self.cuda_graph_buckets[max_kv_cache] = Bucket()
            bucket: Bucket = self.cuda_graph_buckets[max_kv_cache]

            bucket.max_kv_cache = max_kv_cache
            bucket.kv_cache_len = torch.tensor([0], device=device)
            bucket.k_cache = torch.zeros((self.num_layers, batch_size, self.num_head, max_kv_cache, int(self.model_dim/self.num_head)), dtype=dtype, device=device)
            bucket.v_cache = torch.zeros((self.num_layers, batch_size, self.num_head, max_kv_cache, int(self.model_dim/self.num_head)), dtype=dtype, device=device)
            bucket.graph_xy_pos = torch.zeros((batch_size, 1, self.model_dim), dtype=dtype, device=device)
            bucket.decode_attn_mask = torch.zeros((batch_size, self.num_head, 1, max_kv_cache), dtype=torch.bool, device=device)

            self.t2s_transformer.decode_next_token(bucket.graph_xy_pos, bucket.k_cache, bucket.v_cache, bucket.kv_cache_len, bucket.decode_attn_mask)
            
            bucket.cuda_graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(bucket.cuda_graph):
                bucket.graph_xy_dec = self.t2s_transformer.decode_next_token(bucket.graph_xy_pos, bucket.k_cache, bucket.v_cache, bucket.kv_cache_len, bucket.decode_attn_mask)

    @torch.inference_mode()
    def infer(
        self,
        x: torch.LongTensor,
        prompts: torch.LongTensor,
        bert_feature: torch.LongTensor,
        max_kv_cache: int = -1,
        top_k: int = -100,
        top_p: int = 100,
        temperature: float = 1.0,
        repetition_penalty: float = 1.35,
    ):
        x = self.ar_text_embedding(x)
        x = x + self.bert_proj(bert_feature.transpose(1, 2))
        x = self.ar_text_position(x)

        y = prompts

        x_len = x.shape[1]

        y_emb = self.ar_audio_embedding(y)
        y_len = y_emb.shape[1]
        y_pos = self.ar_audio_position(y_emb)
        xy_pos = torch.concat([x, y_pos], dim=1)

        bsz, device = x.shape[0], x.device

        # 音素可以关注自身(双向),但不能关注音频 音频可以关注自身(因果),也能关注音素(双向)
        src_len = x_len + y_len
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
            .expand(bsz * self.num_head, -1, -1)
            .view(bsz, self.num_head, src_len, src_len)
            .to(device=device, dtype=torch.bool)
        )

        if max_kv_cache not in self.cuda_graph_buckets:
            max_kv_cache = list(self.cuda_graph_buckets.keys())[0]
        bucket: Bucket = self.cuda_graph_buckets[max_kv_cache]

        bucket.kv_cache_len.fill_(0)
        bucket.k_cache.fill_(0)
        bucket.v_cache.fill_(0)

        pe_cache = self.ar_audio_position.pe[:, y_len:bucket.max_kv_cache].to(dtype=y_emb.dtype, device=device)
        pe_cache = self.ar_audio_position.alpha * pe_cache

        xy_dec = self.t2s_transformer.process_prompt(xy_pos, bucket.k_cache, bucket.v_cache, bucket.kv_cache_len, prompt_attn_mask)
        logits = self.ar_predict_layer(xy_dec[:, -1])
        samples = sample(logits, y, top_k=top_k, top_p=top_p, repetition_penalty=repetition_penalty, temperature=temperature)[0]
        y = torch.concat([y, samples], dim=1)
        y_emb = self.ar_audio_embedding(y[:, -1:])
        xy_pos = y_emb * self.ar_audio_position.x_scale + pe_cache[:, 0:1]

        bucket.decode_attn_mask.fill_(False)
        bucket.decode_attn_mask[:, :, :, :bucket.kv_cache_len] = True
        
        for idx in tqdm(range(bucket.max_kv_cache - bucket.kv_cache_len)):
            idx = idx + 1

            bucket.graph_xy_pos.copy_(xy_pos)
            bucket.cuda_graph.replay()
            xy_dec = bucket.graph_xy_dec.clone()

            # xy_dec = self.t2s_transformer.decode_next_token(xy_pos, bucket.k_cache, bucket.v_cache, bucket.kv_cache_len, bucket.decode_attn_mask)

            logits = self.ar_predict_layer(xy_dec[:, -1])

            samples = sample(logits, y, top_k=top_k, top_p=top_p, repetition_penalty=repetition_penalty, temperature=temperature)[0]

            if torch.argmax(logits, dim=-1)[0] == self.EOS or samples[0, 0] == self.EOS:
                break

            y = torch.concat([y, samples], dim=1)

            y_emb = self.ar_audio_embedding(y[:, -1:])
            xy_pos = y_emb * self.ar_audio_position.x_scale + pe_cache[:, idx:idx+1]

        return y[:, -idx:].unsqueeze(0)
        
    @torch.inference_mode()
    def infer_stream(
        self,
        x: torch.LongTensor,
        prompts: torch.LongTensor,
        bert_feature: torch.LongTensor,
        max_kv_cache: int = -1,
        top_k: int = -100,
        top_p: int = 100,
        temperature: float = 1.0,
        repetition_penalty: float = 1.35,
        stream_chunk: int = 25,
    ):
        x = self.ar_text_embedding(x)
        x = x + self.bert_proj(bert_feature.transpose(1, 2))
        x = self.ar_text_position(x)

        y = prompts

        x_len = x.shape[1]

        y_emb = self.ar_audio_embedding(y)
        y_len = y_emb.shape[1]
        y_pos = self.ar_audio_position(y_emb)
        xy_pos = torch.concat([x, y_pos], dim=1)

        bsz, device = x.shape[0], x.device

        # 音素可以关注自身(双向),但不能关注音频 音频可以关注自身(因果),也能关注音素(双向)
        src_len = x_len + y_len
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
            .expand(bsz * self.num_head, -1, -1)
            .view(bsz, self.num_head, src_len, src_len)
            .to(device=device, dtype=torch.bool)
        )

        if max_kv_cache not in self.cuda_graph_buckets:
            max_kv_cache = list(self.cuda_graph_buckets.keys())[0]
        bucket: Bucket = self.cuda_graph_buckets[max_kv_cache]

        bucket.kv_cache_len.fill_(0)
        bucket.k_cache.fill_(0)
        bucket.v_cache.fill_(0)

        pe_cache = self.ar_audio_position.pe[:, y_len:bucket.max_kv_cache].to(dtype=y_emb.dtype, device=device)
        pe_cache = self.ar_audio_position.alpha * pe_cache

        xy_dec = self.t2s_transformer.process_prompt(xy_pos, bucket.k_cache, bucket.v_cache, bucket.kv_cache_len, prompt_attn_mask)
        logits = self.ar_predict_layer(xy_dec[:, -1])
        samples = sample(logits, y, top_k=top_k, top_p=top_p, repetition_penalty=repetition_penalty, temperature=temperature)[0]
        y = torch.concat([y, samples], dim=1)
        y_emb = self.ar_audio_embedding(y[:, -1:])
        xy_pos = y_emb * self.ar_audio_position.x_scale + pe_cache[:, 0:1]

        bucket.decode_attn_mask.fill_(False)
        bucket.decode_attn_mask[:, :, :, :bucket.kv_cache_len] = True
        
        first_chunk = True
        pre_chunk = None
        for idx in tqdm(range(bucket.max_kv_cache - bucket.kv_cache_len)):
            idx = idx + 1

            bucket.graph_xy_pos.copy_(xy_pos)
            bucket.cuda_graph.replay()
            xy_dec = bucket.graph_xy_dec.clone()

            # xy_dec = self.t2s_transformer.decode_next_token(xy_pos, bucket.k_cache, bucket.v_cache, bucket.kv_cache_len, bucket.decode_attn_mask)

            logits = self.ar_predict_layer(xy_dec[:, -1])

            samples = sample(logits, y, top_k=top_k, top_p=top_p, repetition_penalty=repetition_penalty, temperature=temperature)[0]

            if torch.argmax(logits, dim=-1)[0] == self.EOS or samples[0, 0] == self.EOS:
                break

            y = torch.concat([y, samples], dim=1)

            if idx % stream_chunk == 0:
                if not pre_chunk is None:
                    yield pre_chunk, False
                pre_chunk = y[:, -idx:].unsqueeze(0)

                if first_chunk:
                    first_chunk = False
                    yield pre_chunk, False
                    pre_chunk = None

            y_emb = self.ar_audio_embedding(y[:, -1:])
            xy_pos = y_emb * self.ar_audio_position.x_scale + pe_cache[:, idx:idx+1]

        yield y[:, -idx:].unsqueeze(0), True