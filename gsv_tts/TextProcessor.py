import re
import torch
import pysbd
import bisect

from .Config import Config
from .LangSegment import LangSegment
from .GPT_SoVITS.G2P import phonemes_to_ids, text_to_phonemes

seg = pysbd.Segmenter()


def get_semantic_length(text, en_weight=1.75):
    cjk_count = len(re.findall(r'[\u3040-\u309f\u30a0-\u30ff\u4e00-\u9fa5\uff66-\uff9f]', text)) # 中日统一
    en_count = len(re.findall(r'[a-zA-Z0-9]+', text))
    return cjk_count + (en_count * en_weight)

def cut_text(text, cut_minlen=10):
    sentences = seg.segment(text)

    for i in range(1, len(sentences)):
        if sentences[i][0] in ['!', '！', '?', '？', '.']:
            sentences[i-1] += sentences[i][0]
            sentences[i] = sentences[i][1:]
    
    text_cuts = []
    punds_pattern = r'([，,；;：:、~・…]+)'

    clauses = []
    for sentence in sentences:
        parts = re.split(punds_pattern, sentence)
        
        for i in range(0, len(parts)-1, 2):
            clause = parts[i] + parts[i+1]
            clauses.append(clause)
        
        if len(parts) % 2 != 0 and parts[-1]:
            clauses.append(parts[-1])

    current_segment = ""
    for s in clauses:
        current_segment += s
        if get_semantic_length(current_segment) >= cut_minlen:
            text_cuts.append(current_segment)
            current_segment = ""
    
    if current_segment:
        if text_cuts:
            text_cuts[-1] += current_segment
        else:
            text_cuts.append(current_segment)

    # 确保至少返回一个非空文本段
    if not text_cuts and text:
        text_cuts.append(text)

    return text_cuts


def get_phones_and_bert(texts, tts_config: Config):
    is_batch = True
    if isinstance(texts, str):
        texts = [texts]
        is_batch = False

    batch_phones = []
    batch_word2ph = []
    batch_bert = []
    batch_norm_text = []

    bert_tasks = {"pos":[], "norm_text":[], "word2ph":[]}

    for text in texts:
        segments = LangSegment.getTexts(text)

        phones_list = []
        norm_text_list = []
        word2ph = {"word":[], "ph":[]}
        batch_bert.append([])

        for segment in segments:
            phones_raw, _word2ph, norm_text = text_to_phonemes(segment['text'], segment['lang'])
            phones = phonemes_to_ids(phones_raw)

            word2ph["word"] += _word2ph["word"]
            word2ph["ph"] += _word2ph["ph"]
            if tts_config.cnroberta and segment['lang'] == "zh":
                bert_tasks["pos"].append((len(batch_bert) - 1, len(batch_bert[-1])))
                bert_tasks["norm_text"].append(norm_text)
                bert_tasks["word2ph"].append(_word2ph["ph"])
                batch_bert[-1].append(None)
            else:
                batch_bert[-1].append(torch.zeros((len(phones), 1024), dtype=tts_config.dtype, device=tts_config.device))
                
            phones_list.append(phones)
            norm_text_list.append(norm_text)

        phones = sum(phones_list, [])
        norm_text = "".join(norm_text_list)

        batch_phones.append(phones)
        batch_word2ph.append(word2ph)
        batch_norm_text.append(norm_text)
    
    if bert_tasks["norm_text"] and bert_tasks["word2ph"]:
        berts = tts_config.cnroberta(bert_tasks["norm_text"], bert_tasks["word2ph"])
        for (i, j), bert in zip(bert_tasks["pos"], berts):
            batch_bert[i][j] = bert
    
    # 处理空的bert_tensors列表
    processed_batch_bert = []
    for bert_tensors in batch_bert:
        if len(bert_tensors) > 0:
            processed_batch_bert.append(torch.cat(bert_tensors))
        else:
            # 如果为空，添加一个默认的零张量
            processed_batch_bert.append(torch.zeros((0, 1024), dtype=tts_config.dtype, device=tts_config.device))
    batch_bert = processed_batch_bert

    if is_batch:
        return batch_phones, batch_word2ph, batch_bert, batch_norm_text
    else:
        return batch_phones[0], batch_word2ph[0], batch_bert[0], batch_norm_text[0]


def split_text(text):
    pattern = re.compile(r'[a-zA-Z]+|.', flags=re.DOTALL)
    return pattern.findall(text)

def LIS_mapping(norm_split_orig_idx):
    dp = []
    trace = [[] for _ in range(len(norm_split_orig_idx))]
    
    for i, candidates in enumerate(norm_split_orig_idx):
        current_updates = []
        
        for val in candidates:
            idx = bisect.bisect_left(dp, val)
            current_updates.append((idx, val))
            trace[i].append((val, idx + 1))
        
        for idx, val in current_updates:
            if idx < len(dp):
                dp[idx] = min(dp[idx], val)
            else:
                dp.append(val)

    max_len = len(dp)
    if max_len == 0:
        return [-1] * len(norm_split_orig_idx)

    result = [-1] * len(norm_split_orig_idx)
    
    current_len = max_len
    last_val = float('inf')
    
    for i in range(len(norm_split_orig_idx) - 1, -1, -1):
        candidates_with_len = [item for item in trace[i] if item[1] == current_len]
        candidates_with_len.sort(key=lambda x: x[0], reverse=True)
        for val, length in candidates_with_len:
            if val < last_val:
                result[i] = val
                last_val = val
                current_len -= 1
                break

    return result

def linear_interpolate(indices):
    n = len(indices)
    result = list(indices)
    valid_points = [(i, val) for i, val in enumerate(result) if val != -1]
    
    if not valid_points: return result 

    first_idx, first_val = valid_points[0]
    
    if first_idx > 0:
        start_val = 0
        val_diff = first_val - start_val
        steps = first_idx
        for i in range(first_idx):
            interpolated_val = start_val + (val_diff / steps) * i 
            result[i] = int(round(interpolated_val))

    for k in range(len(valid_points) - 1):
        idx_start, val_start = valid_points[k]
        idx_end, val_end = valid_points[k+1]
        steps = idx_end - idx_start
        val_diff = val_end - val_start
        for i in range(1, steps):
            interpolated_val = val_start + (val_diff / steps) * i
            result[idx_start + i] = int(round(interpolated_val))

    last_idx, last_val = valid_points[-1]
    for i in range(last_idx + 1, n):
        result[i] = last_val + (i - last_idx)

    return result

def sub2text_index(subtitles, norm_text: str, orig_text: str):
    idx = 0
    sub_norm_idx = []
    for subtitle in subtitles:
        text = subtitle['text']
        idx = norm_text.find(text, idx)
        sub_norm_idx.append({"start":idx, "end":idx+len(text)-1})

    orig_split_text = split_text(orig_text)
    norm_split_text = split_text(norm_text)
    
    norm_split_orig_idx = []
    for t1 in norm_split_text:
        indices = [i for i, t2 in enumerate(orig_split_text) if t2 == t1]
        norm_split_orig_idx.append(indices)

    norm_split_orig_idx = LIS_mapping(norm_split_orig_idx)

    norm_orig_idx = []
    for i, idx in enumerate(norm_split_orig_idx):
        if idx == -1:
            norm_orig_idx += [-1]*len(norm_split_text[i])
        else:
            last_i = sum([len(t) for t in orig_split_text[:idx]])
            norm_orig_idx += list(range(last_i, last_i+len(norm_split_text[i])))
    
    norm_orig_idx = linear_interpolate(norm_orig_idx)
    
    for i, norm_idx in enumerate(sub_norm_idx):
        orig_idx_start, orig_idx_end = norm_orig_idx[norm_idx["start"]], norm_orig_idx[norm_idx["end"]]
        subtitles[i]["orig_idx_start"] = orig_idx_start
        subtitles[i]["orig_idx_end"] = orig_idx_end + 1
    
    return subtitles