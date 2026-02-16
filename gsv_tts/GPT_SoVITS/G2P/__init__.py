from . import Symbols
from ...config import global_config


symbol_to_id = {s: i for i, s in enumerate(Symbols.symbols)}


def phonemes_to_ids(phones_raw):
    phones = [symbol_to_id[symbol] for symbol in phones_raw]
    return phones


def text_to_phonemes(text, language):
    if language == "zh":
        from .Chinese import ChineseG2P

        if global_config.chinese_g2p is None:
            global_config.chinese_g2p = ChineseG2P(global_config.models_dir)
        norm_text = global_config.chinese_g2p.text_normalize(text)
        phones, word2ph = global_config.chinese_g2p.g2p(norm_text)

    elif language == "ja":
        from .Japanese import JapaneseG2P

        if global_config.japanese_g2p is None:
            global_config.japanese_g2p = JapaneseG2P(global_config.models_dir)
        norm_text = global_config.japanese_g2p.text_normalize(text)
        phones, word2ph = global_config.japanese_g2p.g2p(norm_text)
    
    elif language == "ko":
        from .Korean import KoreanG2P
        
        if global_config.korean_g2p is None:
            global_config.korean_g2p = KoreanG2P()
        norm_text = text
        phones, word2ph = global_config.korean_g2p.g2p(norm_text)
    
    elif language == "yue":
        from .Chinese import CantoneseG2P

        if global_config.cantonese_g2p is None:
            global_config.cantonese_g2p = CantoneseG2P()
        norm_text = global_config.cantonese_g2p.text_normalize(text)
        phones, word2ph = global_config.cantonese_g2p.g2p(norm_text)
    
    else:
        from .English import EnglishG2P

        if global_config.english_g2p is None:
            global_config.english_g2p = EnglishG2P(global_config.models_dir)
        norm_text = global_config.english_g2p.text_normalize(text)
        phones, word2ph = global_config.english_g2p.g2p(norm_text)
    
    assert len(phones) == sum(word2ph["ph"]), f"length mismatch: The length of phones is {len(phones)}, while the total of word2ph is {sum(word2ph['ph'])}"

    phones = ["UNK" if ph not in Symbols.symbols else ph for ph in phones]
    return phones, word2ph, norm_text