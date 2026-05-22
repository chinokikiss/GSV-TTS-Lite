import re

from . import Symbols
from . import Pause
from ...Config import global_config


symbol_to_id = {s: i for i, s in enumerate(Symbols.symbols)}


def phonemes_to_ids(phones_raw):
    phones = [symbol_to_id[symbol] for symbol in phones_raw]
    return phones


def text_to_phonemes(text, language):

    text = re.sub(r'\.{3,}|。{3,}', '…', text)

    if language == "zh":
        from .Chinese import ChineseG2P

        if global_config.chinese_g2p is None:
            global_config.chinese_g2p = ChineseG2P(global_config.models_dir)

        norm_text = global_config.chinese_g2p.text_normalize(text)
        phones, word2ph = global_config.chinese_g2p.g2p(norm_text)

    elif language == "ja":
        from .Japanese import JapaneseG2P

        if global_config.japanese_g2p is None:
            global_config.japanese_g2p = JapaneseG2P()

        phones, word2ph = global_config.japanese_g2p.g2p(text)
        norm_text = "".join(word2ph["word"])
    
    else:
        from .English import EnglishG2P

        if global_config.english_g2p is None:
            global_config.english_g2p = EnglishG2P(global_config.models_dir)

        norm_text = global_config.english_g2p.text_normalize(text)
        phones, word2ph = global_config.english_g2p.g2p(norm_text)
    
    assert len(phones) == sum(word2ph["ph"]), f"length mismatch: The length of phones is {len(phones)}, while the total of word2ph is {sum(word2ph['ph'])}"
    
    # 替换停顿符
    for i, ph in enumerate(phones):
        if ph in Pause.pause_map.keys():
            phones[i] = Pause.pause_map[ph]
    
    phones = ["UNK" if ph not in Symbols.symbols else ph for ph in phones]

    dup_indices = []
    for i in range(1, len(phones)):
        if phones[i] == phones[i-1] and phones[i] in Symbols.punctuation:
            dup_indices.append(i)

    # 过滤UNK / 去重
    ph_idx = len(phones)
    for w_idx in range(len(word2ph["ph"]) - 1, -1, -1):
        del_count = 0

        for _ in range(word2ph["ph"][w_idx]):
            ph_idx -= 1
            if phones[ph_idx] == "UNK" or ph_idx in dup_indices:
                del_count += 1
                phones.pop(ph_idx)
        
        word2ph["ph"][w_idx] -= del_count
        if word2ph["ph"][w_idx] == 0:
            word2ph["ph"].pop(w_idx)
            word2ph["word"].pop(w_idx)
    
    return phones, word2ph, norm_text