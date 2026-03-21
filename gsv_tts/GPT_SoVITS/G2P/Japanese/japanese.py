# modified from https://github.com/CjangCjengh/vits/blob/main/text/japanese.py
import re
import pyopenjtalk


class JapaneseG2P:
    def __init__(self):
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
    
    def pyopenjtalk_g2p_prosody(self, text, word2ph, drop_unvoiced_vowels=True):
        features = pyopenjtalk.run_frontend(text)
        labels = pyopenjtalk.make_label(features)
        N = len(labels)

        phones = []
        # 记录每个 node 对应的音素总数
        node_phone_counts = [0] * len(features)
        
        node_idx = 0
        
        for n in range(N):
            lab_curr = labels[n]
            p3 = re.search(r"\-(.*?)\+", lab_curr).group(1)
            
            if drop_unvoiced_vowels and p3 in "AEIOU":
                p3 = p3.lower()

            # 处理特殊符号
            res_p = None
            if p3 == "sil":
                if n == 0: res_p = "^"
                elif n == N - 1:
                    e3 = self._numeric_feature_by_regex(r"!(\d+)_", lab_curr)
                    res_p = "$" if e3 == 0 else "?"
            elif p3 == "pau":
                res_p = "_"
            else:
                res_p = p3

            if res_p:
                phones.append(res_p)
                if p3 not in ["sil", "pau"]:
                    node_phone_counts[node_idx] += 1

            # 韵律符号处理
            has_other = False
            if p3 not in ["sil", "pau"]:
                a1 = self._numeric_feature_by_regex(r"/A:([0-9\-]+)\+", lab_curr)
                a2 = self._numeric_feature_by_regex(r"\+(\d+)\+", lab_curr)
                a3 = self._numeric_feature_by_regex(r"\+(\d+)/", lab_curr)
                f1 = self._numeric_feature_by_regex(r"/F:(\d+)_", lab_curr)
                a2_next = self._numeric_feature_by_regex(r"\+(\d+)\+", labels[n + 1]) if n+1 < N else -1

                if a3 == 1 and a2_next == 1 and p3 in "aeiouAEIOUNcl":
                    phones.append("#")
                    has_other = True
                elif a1 == 0 and a2_next == a2 + 1 and a2 != f1:
                    phones.append("]")
                    has_other = True
                elif a2 == 1 and a2_next == 2:
                    phones.append("[")
                    has_other = True
                
                if has_other:
                    node_phone_counts[node_idx] += 1

            if n < N - 1:
                # 判断是否即将进入下一个词
                # 使用正则表达式提取当前词在 label 中的位置信息
                # 如果下一个 label 的词开始标识变化了，则 node_idx += 1
                curr_word_id = self._numeric_feature_by_regex(r"/C:(\d+)_", lab_curr)
                next_word_id = self._numeric_feature_by_regex(r"/C:(\d+)_", labels[n+1])
                if curr_word_id != next_word_id and p3 not in ["sil", "pau"]:
                    if node_idx < len(features) - 1:
                        node_idx += 1

        for i, node in enumerate(features):
            surface = node['string']
            if node['pron'] == 'IDLE': continue
            
            total_ph_count = node_phone_counts[i]
            num_chars = len(surface)
            
            if num_chars <= 1:
                word2ph["word"].append(surface)
                word2ph["ph"].append(total_ph_count)
            else:
                # 由于在日语中，一个字对应的音素长度是不固定的，所以这里直接按字符数平分
                avg_ph = total_ph_count // num_chars
                remainder = total_ph_count % num_chars
                for j in range(num_chars):
                    word2ph["word"].append(surface[j])
                    word2ph["ph"].append(avg_ph + 1 if j < remainder else avg_ph)

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

    def g2p(self, norm_text, with_prosody=True):
        phones, word2ph = self.preprocess_jap(norm_text, with_prosody)
        return phones, word2ph