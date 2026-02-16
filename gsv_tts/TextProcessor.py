import re
import torch
import bisect

from .config import Config, global_config
from .GPT_SoVITS.G2P import phonemes_to_ids, text_to_phonemes


def cut_text(text, punds, cut_minlen):
    text = text.strip("\n")
    mergeitems = []
    items = []
    logical_count = 0
    
    tokens = re.findall(r'[a-zA-Z0-9]+|.', text)

    for i, token in enumerate(tokens):
        items.append(token)
        
        if token not in punds and not token.isspace():
            logical_count += 1
            
        if token in punds:
            is_decimal = (token == "." and i > 0 and i < len(tokens) - 1 and tokens[i-1].isdigit() and tokens[i+1].isdigit())
            
            if not is_decimal and logical_count >= cut_minlen:
                mergeitems.append("".join(items))
                items = []
                logical_count = 0
        
    if items:
        mergeitems.append("".join(items))

    return [item for item in mergeitems if any(c not in punds and not c.isspace() for c in item)]


def process_text(text, language):
    language = language.replace("all_", "").replace("auto_", "")
    phones_raw, word2ph, norm_text = text_to_phonemes(text, language)
    phones = phonemes_to_ids(phones_raw)
    return phones, word2ph, norm_text


def get_phones_and_bert(texts, languages, tts_config: Config):
    is_batch = True
    if isinstance(texts, str) and isinstance(languages, str):
        texts = [texts]
        languages = [languages]
        is_batch = False

    batch_phones = []
    batch_word2ph = []
    batch_bert = []
    batch_norm_text = []

    bert_tasks = {"pos":[], "norm_text":[], "word2ph":[]}

    for text, language in zip(texts, languages):
        text = re.sub(r' {2,}', ' ', text)
        textlist = []
        langlist = []
        if language == "all_zh":
            for tmp in global_config.lang_segmenter.getTexts(text,"zh"):
                langlist.append(tmp["lang"])
                textlist.append(tmp["text"])
        elif language == "all_yue":
            for tmp in global_config.lang_segmenter.getTexts(text,"zh"):
                if tmp["lang"] == "zh":
                    tmp["lang"] = "yue"
                langlist.append(tmp["lang"])
                textlist.append(tmp["text"])
        elif language == "all_ja":
            for tmp in global_config.lang_segmenter.getTexts(text,"ja"):
                langlist.append(tmp["lang"])
                textlist.append(tmp["text"])
        elif language == "all_ko":
            for tmp in global_config.lang_segmenter.getTexts(text,"ko"):
                langlist.append(tmp["lang"])
                textlist.append(tmp["text"])
        elif language == "en":
            langlist.append("en")
            textlist.append(text)
        elif language == "auto":
            for tmp in global_config.lang_segmenter.getTexts(text):
                langlist.append(tmp["lang"])
                textlist.append(tmp["text"])
        elif language == "auto_yue":
            for tmp in global_config.lang_segmenter.getTexts(text):
                if tmp["lang"] == "zh":
                    tmp["lang"] = "yue"
                langlist.append(tmp["lang"])
                textlist.append(tmp["text"])
        else:
            for tmp in global_config.lang_segmenter.getTexts(text):
                if langlist:
                    if (tmp["lang"] == "en" and langlist[-1] == "en") or (tmp["lang"] != "en" and langlist[-1] != "en"):
                        textlist[-1] += tmp["text"]
                        continue
                if tmp["lang"] == "en":
                    langlist.append(tmp["lang"])
                else:
                    langlist.append(language)
                textlist.append(tmp["text"])

        phones_list = []
        norm_text_list = []
        word2ph = {"word":[], "ph":[]}
        batch_bert.append([])

        for i in range(len(textlist)):
            lang = langlist[i]
            phones, _word2ph, norm_text = process_text(textlist[i], lang)
            word2ph["word"] += _word2ph["word"]
            word2ph["ph"] += _word2ph["ph"]
            if tts_config.cnroberta and "zh" in lang:
                bert_tasks["pos"].append((len(batch_bert) - 1, len(batch_bert[-1])))
                bert_tasks["norm_text"].append(norm_text)
                bert_tasks["word2ph"].append(word2ph["ph"])
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
    
    batch_bert = [torch.cat(bert_tensors) for bert_tensors in batch_bert]

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

def sub2text_index(subtitles, norm_text: str, orig_text: str, start_idx: int = 0, end_idx: int = None):
    idx = 0
    sub_norm_idx = []
    for subtitle in subtitles:
        text = subtitle['text']
        idx = norm_text.find(text, idx)
        sub_norm_idx.append({"start":idx, "end":idx+len(text)-1})

    orig_split_text = split_text(orig_text)

    if end_idx is None: end_idx = len(orig_text)-1
    last_l = 0
    start_split_idx = 0
    end_split_idx = 0
    for i in orig_split_text:
        last_l += len(i)
        if last_l <= start_idx:
            start_split_idx += 1
        if last_l <= end_idx:
            end_split_idx += 1
    
    norm_split_text = split_text(norm_text)
    norm_split_orig_idx = []
    for t1 in norm_split_text:
        indices = [i for i, t2 in enumerate(orig_split_text) if t2 == t1 and i >= start_split_idx and i <= end_split_idx]
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