# modified from https://github.com/CjangCjengh/vits/blob/main/text/japanese.py
import re
import os
from text.alignment import phoneme_word2ph_alignment
import pyopenjtalk

from config import tts_config

USERDIC_CSV_PATH = os.path.join(tts_config.models_dir,r"text\ja\userdict.csv")
USERDIC_BIN_PATH = os.path.join(tts_config.models_dir,r"text\ja\user.dict")

if os.path.exists(USERDIC_CSV_PATH) and not os.path.exists(USERDIC_BIN_PATH):
    pyopenjtalk.mecab_dict_index(USERDIC_CSV_PATH, USERDIC_BIN_PATH)

if os.path.exists(USERDIC_BIN_PATH):
    pyopenjtalk.update_global_jtalk_with_user_dict(USERDIC_BIN_PATH)

from text.symbols2 import punctuation

# Regular expression matching Japanese without punctuation marks:
_japanese_characters = re.compile(
    r"[A-Za-z\d\u3005\u3040-\u30ff\u4e00-\u9fff\uff11-\uff19\uff21-\uff3a\uff41-\uff5a\uff66-\uff9d]"
)

# Regular expression matching non-Japanese characters or punctuation marks:
_japanese_marks = re.compile(
    r"[^A-Za-z\d\u3005\u3040-\u30ff\u4e00-\u9fff\uff11-\uff19\uff21-\uff3a\uff41-\uff5a\uff66-\uff9d]"
)

# List of (symbol, Japanese) pairs for marks:
_symbols_to_japanese = [(re.compile("%s" % x[0]), x[1]) for x in [("％", "パーセント")]]


def post_replace_ph(ph):
    rep_map = {
        "：": ",",
        "；": ",",
        "，": ",",
        "。": ".",
        "！": "!",
        "？": "?",
        "\n": ".",
        "·": ",",
        "、": ",",
        "...": "…",
    }

    if ph in rep_map.keys():
        ph = rep_map[ph]
    return ph


def replace_consecutive_punctuation(text):
    punctuations = "".join(re.escape(p) for p in punctuation)
    pattern = f"([{punctuations}])([{punctuations}])+"
    result = re.sub(pattern, r"\1", text)
    return result


def symbols_to_japanese(text):
    for regex, replacement in _symbols_to_japanese:
        text = re.sub(regex, replacement, text)
    return text


def preprocess_jap(text, with_prosody=False):
    """Reference https://r9y9.github.io/ttslearn/latest/notebooks/ch10_Recipe-Tacotron.html"""
    text = symbols_to_japanese(text)
    # English words to lower case, should have no influence on japanese words.
    text = text.lower()
    sentences = re.split(_japanese_marks, text)
    marks = re.findall(_japanese_marks, text)
    text = []
    word2ph = {"word":[], "ph":[]}
    for i, sentence in enumerate(sentences):
        if re.match(_japanese_characters, sentence):
            if with_prosody:
                text += pyopenjtalk_g2p_prosody(sentence, word2ph)[1:-1]
            else:
                p = pyopenjtalk.g2p(sentence)
                text += p.split(" ")

        if i < len(marks):
            if marks[i] == " ":  # 防止意外的UNK
                continue
            text += [marks[i].replace(" ", "")]
            word2ph["word"].append(marks[i])
            word2ph["ph"].append(1)
    return text, word2ph


def text_normalize(text):
    # todo: jap text normalize

    # 避免重复标点引起的参考泄露
    text = replace_consecutive_punctuation(text)
    return text


def char_phoneme_mapping(text):
    result = []
    
    for char in text:
        char_phoneme = []
        if char != 'ー':
            labels = pyopenjtalk.make_label(pyopenjtalk.run_frontend(char))
            for label in labels:
                match = re.search(r"\-(.*?)\+", label)
                if match:
                    phoneme = match.group(1)
                    if phoneme not in ['sil', 'pau']:
                        char_phoneme.append(phoneme)
        result.append([char, char_phoneme])
    
    return result


# Copied from espnet https://github.com/espnet/espnet/blob/master/espnet2/text/phoneme_tokenizer.py
def pyopenjtalk_g2p_prosody(text, word2ph, drop_unvoiced_vowels=True):
    """Extract phoneme + prosoody symbol sequence from input full-context labels.

    The algorithm is based on `Prosodic features control by symbols as input of
    sequence-to-sequence acoustic modeling for neural TTS`_ with some r9y9's tweaks.

    Args:
        text (str): Input text.
        drop_unvoiced_vowels (bool): whether to drop unvoiced vowels.

    Returns:
        List[str]: List of phoneme + prosody symbols.

    Examples:
        >>> from espnet2.text.phoneme_tokenizer import pyopenjtalk_g2p_prosody
        >>> pyopenjtalk_g2p_prosody("こんにちは。")
        ['^', 'k', 'o', '[', 'N', 'n', 'i', 'ch', 'i', 'w', 'a', '$']

    .. _`Prosodic features control by symbols as input of sequence-to-sequence acoustic
        modeling for neural TTS`: https://doi.org/10.1587/transinf.2020EDP7104

    """
    labels = pyopenjtalk.make_label(pyopenjtalk.run_frontend(text))
    N = len(labels)

    phonemes = []
    other_phonemes = []

    phones = []
    for n in range(N):
        p4, has_other = None, False
        lab_curr = labels[n]

        # current phoneme
        p3 = re.search(r"\-(.*?)\+", lab_curr).group(1)
        # deal unvoiced vowels as normal vowels
        if drop_unvoiced_vowels and p3 in "AEIOU":
            p3 = p3.lower()

        # deal with sil at the beginning and the end of text
        if p3 == "sil":
            assert n == 0 or n == N - 1
            if n == 0:
                phones.append("^")
            elif n == N - 1:
                # check question form or not
                e3 = _numeric_feature_by_regex(r"!(\d+)_", lab_curr)
                if e3 == 0:
                    phones.append("$")
                elif e3 == 1:
                    phones.append("?")
            continue
        elif p3 == "pau":
            phones.append("_")
            continue
        else:
            p4 = p3
            phones.append(p3)

        # accent type and position info (forward or backward)
        a1 = _numeric_feature_by_regex(r"/A:([0-9\-]+)\+", lab_curr)
        a2 = _numeric_feature_by_regex(r"\+(\d+)\+", lab_curr)
        a3 = _numeric_feature_by_regex(r"\+(\d+)/", lab_curr)

        # number of mora in accent phrase
        f1 = _numeric_feature_by_regex(r"/F:(\d+)_", lab_curr)

        a2_next = _numeric_feature_by_regex(r"\+(\d+)\+", labels[n + 1])
        # accent phrase border
        if a3 == 1 and a2_next == 1 and p3 in "aeiouAEIOUNcl":
            has_other = True
            phones.append("#")
        # pitch falling
        elif a1 == 0 and a2_next == a2 + 1 and a2 != f1:
            has_other = True
            phones.append("]")
        # pitch rising
        elif a2 == 1 and a2_next == 2:
            has_other = True
            phones.append("[")
        
        if p4:
            phonemes.append(p4)
        if has_other:
            other_phonemes.append(1)
        else:
            other_phonemes.append(0)
    
    char_phonemes = char_phoneme_mapping(text)
    phoneme_word2ph_alignment(phonemes, other_phonemes, char_phonemes, word2ph)

    return phones


# Copied from espnet https://github.com/espnet/espnet/blob/master/espnet2/text/phoneme_tokenizer.py
def _numeric_feature_by_regex(regex, s):
    match = re.search(regex, s)
    if match is None:
        return -50
    return int(match.group(1))


def g2p(norm_text, with_prosody=True):
    phones, word2ph = preprocess_jap(norm_text, with_prosody)
    phones = [post_replace_ph(i) for i in phones]
    # todo: implement tones and word2ph
    return phones, word2ph


if __name__ == "__main__":
    phones = g2p("Hello.こんにちは！今日もNiCe天気ですね！tokyotowerに行きましょう！")
    print(phones)
