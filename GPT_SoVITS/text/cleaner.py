from config import tts_config
from text import symbols2 as symbols_v2


def clean_text(text, language):
    symbols = symbols_v2.symbols

    if language not in tts_config.language_module_map:
        language = "en"
        text = " "
        
    language_module = __import__("text." + tts_config.language_module_map[language], fromlist=[tts_config.language_module_map[language]])
    
    if hasattr(language_module, "text_normalize"):
        norm_text = language_module.text_normalize(text)
    else:
        norm_text = text

    phones, word2ph = language_module.g2p(norm_text)

    phones = ["UNK" if ph not in symbols else ph for ph in phones]
    return phones, word2ph, norm_text


if __name__ == "__main__":
    print(clean_text("你好%啊啊啊额、还是到付红四方。", "zh"))
