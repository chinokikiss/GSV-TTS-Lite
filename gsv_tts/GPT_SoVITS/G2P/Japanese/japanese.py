# modified from https://github.com/CjangCjengh/vits/blob/main/text/japanese.py
import re
import os
import pyopenjtalk
from ..Symbols import punctuation
from pathlib import Path


class JapaneseG2P:
    def __init__(self, models_dir):
        self.USERDIC_CSV_PATH = str(Path(models_dir) / "g2p" / "ja" / "userdict.csv")
        self.USERDIC_BIN_PATH = str(Path(models_dir) / "g2p" / "ja" / "user.dict")

        if os.path.exists(self.USERDIC_CSV_PATH) and not os.path.exists(self.USERDIC_BIN_PATH):
            pyopenjtalk.mecab_dict_index(self.USERDIC_CSV_PATH, self.USERDIC_BIN_PATH)

        if os.path.exists(self.USERDIC_BIN_PATH):
            pyopenjtalk.update_global_jtalk_with_user_dict(self.USERDIC_BIN_PATH)

        # Regular expression matching Japanese without punctuation marks:
        self._japanese_characters = re.compile(
            r"[A-Za-z\d\u3005\u3040-\u30ff\u4e00-\u9fff\uff11-\uff19\uff21-\uff3a\uff41-\uff5a\uff66-\uff9d]"
        )

        # Regular expression matching non-Japanese characters or punctuation marks:
        self._japanese_marks = re.compile(
            r"[^A-Za-z\d\u3005\u3040-\u30ff\u4e00-\u9fff\uff11-\uff19\uff21-\uff3a\uff41-\uff5a\uff66-\uff9d]"
        )

        # List of (symbol, Japanese) pairs for marks:
        self._symbols_to_japanese = [(re.compile("%s" % x[0]), x[1]) for x in [("％", "パーセント")]]

        self.rep_map = {
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
    
    def post_replace_ph(self, ph):
        if ph in self.rep_map.keys():
            ph = self.rep_map[ph]
        return ph

    def symbols_to_japanese(self, text):
        for regex, replacement in self._symbols_to_japanese:
            text = re.sub(regex, replacement, text)
        return text
    
    # Copied from espnet https://github.com/espnet/espnet/blob/master/espnet2/text/phoneme_tokenizer.py
    def _numeric_feature_by_regex(self, regex, s):
        match = re.search(regex, s)
        if match is None:
            return -50
        return int(match.group(1))
    
    def alignment(self, features, other_phonemes, word2ph):
        last = 0
        for node in features:
            surface = node['string']
            pron = node['pron']
            
            if pron == 'IDLE': 
                continue

            phones = pyopenjtalk.g2p(surface).split()
            
            num_others = sum(other_phonemes[last : last + len(phones)])
            total_ph_count = len(phones) + num_others
            
            num_chars = len(surface)
            if num_chars <= 1:
                word2ph["word"].append(surface)
                word2ph["ph"].append(total_ph_count)
            else:
                # 由于在日语中，一个字对应的音素长度是不固定的，所以这里直接按字符数平分
                avg_ph = total_ph_count // num_chars
                remainder = total_ph_count % num_chars
                for i in range(num_chars):
                    word2ph["word"].append(surface[i])
                    if i < remainder:
                        word2ph["ph"].append(avg_ph + 1)
                    else:
                        word2ph["ph"].append(avg_ph)
            
            last += len(phones)

        return word2ph

    # Copied from espnet https://github.com/espnet/espnet/blob/master/espnet2/text/phoneme_tokenizer.py
    def pyopenjtalk_g2p_prosody(self, text, word2ph, drop_unvoiced_vowels=True):
        features = pyopenjtalk.run_frontend(text)
        labels = pyopenjtalk.make_label(features)
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
                    e3 = self._numeric_feature_by_regex(r"!(\d+)_", lab_curr)
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
            a1 = self._numeric_feature_by_regex(r"/A:([0-9\-]+)\+", lab_curr)
            a2 = self._numeric_feature_by_regex(r"\+(\d+)\+", lab_curr)
            a3 = self._numeric_feature_by_regex(r"\+(\d+)/", lab_curr)

            # number of mora in accent phrase
            f1 = self._numeric_feature_by_regex(r"/F:(\d+)_", lab_curr)

            a2_next = self._numeric_feature_by_regex(r"\+(\d+)\+", labels[n + 1])
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
        
        word2ph = self.alignment(features, other_phonemes, word2ph)

        return phones, word2ph
    
    def preprocess_jap(self, text, with_prosody=False):
        """Reference https://r9y9.github.io/ttslearn/latest/notebooks/ch10_Recipe-Tacotron.html"""
        text = self.symbols_to_japanese(text)
        # English words to lower case, should have no influence on japanese words.
        text = text.lower()
        sentences = re.split(self._japanese_marks, text)
        marks = re.findall(self._japanese_marks, text)
        text = []
        word2ph = {"word":[], "ph":[]}
        for i, sentence in enumerate(sentences):
            if re.match(self._japanese_characters, sentence):
                if with_prosody:
                    ph, word2ph = self.pyopenjtalk_g2p_prosody(sentence, word2ph)
                    text += ph[1:-1]
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

    def text_normalize(self, text):
        punctuations = "".join(re.escape(p) for p in punctuation)
        pattern = f"([{punctuations}])([{punctuations}])+"
        result = re.sub(pattern, r"\1", text)
        return result

    def g2p(self, norm_text, with_prosody=True):
        phones, word2ph = self.preprocess_jap(norm_text, with_prosody)
        phones = [self.post_replace_ph(i) for i in phones]
        # todo: implement tones and word2ph
        return phones, word2ph