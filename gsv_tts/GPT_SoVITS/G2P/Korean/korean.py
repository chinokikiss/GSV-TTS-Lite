import re
from jamo import h2j, j2hcj
from kiwipiepy import Kiwi
from ..Symbols import symbols


class KoreanG2P:
    def __init__(self):
        # List of (hangul, hangul divided) pairs:
        self._hangul_divided = [
            (re.compile("%s" % x[0]), x[1])
            for x in [
                # ('ㄳ', 'ㄱㅅ'),   # g2pk2, A Syllable-ending Rule
                # ('ㄵ', 'ㄴㅈ'),
                # ('ㄶ', 'ㄴㅎ'),
                # ('ㄺ', 'ㄹㄱ'),
                # ('ㄻ', 'ㄹㅁ'),
                # ('ㄼ', 'ㄹㅂ'),
                # ('ㄽ', 'ㄹㅅ'),
                # ('ㄾ', 'ㄹㅌ'),
                # ('ㄿ', 'ㄹㅍ'),
                # ('ㅀ', 'ㄹㅎ'),
                # ('ㅄ', 'ㅂㅅ'),
                ("ㅘ", "ㅗㅏ"),
                ("ㅙ", "ㅗㅐ"),
                ("ㅚ", "ㅗㅣ"),
                ("ㅝ", "ㅜㅓ"),
                ("ㅞ", "ㅜㅔ"),
                ("ㅟ", "ㅜㅣ"),
                ("ㅢ", "ㅡㅣ"),
                ("ㅑ", "ㅣㅏ"),
                ("ㅒ", "ㅣㅐ"),
                ("ㅕ", "ㅣㅓ"),
                ("ㅖ", "ㅣㅔ"),
                ("ㅛ", "ㅣㅗ"),
                ("ㅠ", "ㅣㅜ"),
            ]
        ]

        # List of (Latin alphabet, hangul) pairs:
        self._latin_to_hangul = [
            (re.compile("%s" % x[0], re.IGNORECASE), x[1])
            for x in [
                ("a", "에이"),
                ("b", "비"),
                ("c", "시"),
                ("d", "디"),
                ("e", "이"),
                ("f", "에프"),
                ("g", "지"),
                ("h", "에이치"),
                ("i", "아이"),
                ("j", "제이"),
                ("k", "케이"),
                ("l", "엘"),
                ("m", "엠"),
                ("n", "엔"),
                ("o", "오"),
                ("p", "피"),
                ("q", "큐"),
                ("r", "아르"),
                ("s", "에스"),
                ("t", "티"),
                ("u", "유"),
                ("v", "브이"),
                ("w", "더블유"),
                ("x", "엑스"),
                ("y", "와이"),
                ("z", "제트"),
            ]
        ]

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
            " ": "空",
        }

        self.kiwi = Kiwi()

    def fix_g2pk2_error(self, text):
        new_text = ""
        i = 0
        while i < len(text) - 4:
            if (text[i : i + 3] == "ㅇㅡㄹ" or text[i : i + 3] == "ㄹㅡㄹ") and text[i + 3] == " " and text[i + 4] == "ㄹ":
                new_text += text[i : i + 3] + " " + "ㄴ"
                i += 5
            else:
                new_text += text[i]
                i += 1

        new_text += text[i:]
        return new_text

    def latin_to_hangul(self, text):
        for regex, replacement in self._latin_to_hangul:
            text = re.sub(regex, replacement, text)
        return text

    def divide_hangul(self, text):
        text = j2hcj(h2j(text))
        for regex, replacement in self._hangul_divided:
            text = re.sub(regex, replacement, text)
        return text

    def post_replace_ph(self, ph):
        if ph in self.rep_map.keys():
            ph = self.rep_map[ph]
        if ph in symbols:
            return ph
        if ph not in symbols:
            ph = "停"
        return ph

    def g2p(self, text):
        norm_text = self.latin_to_hangul(text)

        tokens = self.kiwi.tokenize(norm_text)
        phones_text = ""
        for token in tokens:
            phones_text += token.form
        
        phonemes = []
        word2ph = {"word": list(norm_text), "ph": []}
        
        for p_char in phones_text:
            divided = self.divide_hangul(p_char)

            p_list = [self.post_replace_ph(p) for p in divided]

            phonemes.extend(p_list)
            word2ph["ph"].append(len(p_list))
            
        if len(phonemes) > 0 and phonemes[-1].isalnum():
            phonemes.append(".")
            if word2ph["ph"]:
                word2ph["ph"][-1] += 1

        return phonemes, word2ph