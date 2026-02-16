import re
import os

import cn2an
from pypinyin import lazy_pinyin, Style

from ..Symbols import punctuation
from .tone_sandhi import ToneSandhi
from .Normalization.text_normlization import TextNormalizer

import jieba_fast
import logging

jieba_fast.setLogLevel(logging.CRITICAL)
import jieba_fast.posseg as psg
from pathlib import Path


class ChineseG2P:
    def __init__(self, models_dir):
        self.normalizer = lambda x: cn2an.transform(x, "an2cn")

        self.pinyin_to_symbol_map = {
            line.split("\t")[0]: line.strip().split("\t")[1]
            for line in open(Path(models_dir) / r"text\zh\opencpop-strict.txt").readlines()
        }

        self.tone_modifier = ToneSandhi()

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
            "$": ".",
            "/": ",",
            "—": "-",
            "~": "…",
            "～": "…",
        }

        self.must_erhua = {"小院儿", "胡同儿", "范儿", "老汉儿", "撒欢儿", "寻老礼儿", "妥妥儿", "媳妇儿"}
        
        self.not_erhua = {
            "虐儿",
            "为儿",
            "护儿",
            "瞒儿",
            "救儿",
            "替儿",
            "有儿",
            "一儿",
            "我儿",
            "俺儿",
            "妻儿",
            "拐儿",
            "聋儿",
            "乞儿",
            "患儿",
            "幼儿",
            "孤儿",
            "婴儿",
            "婴幼儿",
            "连体儿",
            "脑瘫儿",
            "流浪儿",
            "体弱儿",
            "混血儿",
            "蜜雪儿",
            "舫儿",
            "祖儿",
            "美儿",
            "应采儿",
            "可儿",
            "侄儿",
            "孙儿",
            "侄孙儿",
            "女儿",
            "男儿",
            "红孩儿",
            "花儿",
            "虫儿",
            "马儿",
            "鸟儿",
            "猪儿",
            "猫儿",
            "狗儿",
            "少儿",
        }

    def replace_punctuation(self, text):
        text = text.replace("嗯", "恩").replace("呣", "母")
        pattern = re.compile("|".join(re.escape(p) for p in self.rep_map.keys()))

        replaced_text = pattern.sub(lambda x: self.rep_map[x.group()], text)

        replaced_text = re.sub(r"[^\u4e00-\u9fa5" + "".join(punctuation) + r"]+", "", replaced_text)

        return replaced_text

    def _get_initials_finals(self, word):
        initials = []
        finals = []

        orig_initials = lazy_pinyin(word, neutral_tone_with_five=True, style=Style.INITIALS)
        orig_finals = lazy_pinyin(word, neutral_tone_with_five=True, style=Style.FINALS_TONE3)

        for c, v in zip(orig_initials, orig_finals):
            initials.append(c)
            finals.append(v)
        return initials, finals

    def _merge_erhua(self, initials: list[str], finals: list[str], word: str, pos: str) -> list[list[str]]:
        """
        Do erhub.
        """
        # fix er1
        for i, phn in enumerate(finals):
            if i == len(finals) - 1 and word[i] == "儿" and phn == "er1":
                finals[i] = "er2"

        # 发音
        if word not in self.must_erhua and (word in self.not_erhua or pos in {"a", "j", "nr"}):
            return initials, finals

        # "……" 等情况直接返回
        if len(finals) != len(word):
            return initials, finals

        assert len(finals) == len(word)

        # 与前一个字发同音
        new_initials = []
        new_finals = []
        for i, phn in enumerate(finals):
            if (
                i == len(finals) - 1
                and word[i] == "儿"
                and phn in {"er2", "er5"}
                and word[-2:] not in self.not_erhua
                and new_finals
            ):
                phn = "er" + new_finals[-1][-1]

            new_initials.append(initials[i])
            new_finals.append(phn)

        return new_initials, new_finals

    def _g2p(self, segments):
        phones_list = []
        word2ph = {"word":[], "ph":[]}
        for seg in segments:
            # Replace all English words in the sentence
            seg = re.sub("[a-zA-Z]+", "", seg)
            seg_cut = psg.lcut(seg)
            seg_cut = self.tone_modifier.pre_merge_for_modify(seg_cut)
            initials = []
            finals = []

            for word, pos in seg_cut:
                if pos == "eng":
                    continue
                sub_initials, sub_finals = self._get_initials_finals(word)
                sub_finals = self.tone_modifier.modified_tone(word, pos, sub_finals)
                # 儿化
                sub_initials, sub_finals = self._merge_erhua(sub_initials, sub_finals, word, pos)
                initials.append(sub_initials)
                finals.append(sub_finals)

                for word_chr in word:
                    word2ph["word"].append(word_chr)
                # assert len(sub_initials) == len(sub_finals) == len(word)
            initials = sum(initials, [])
            finals = sum(finals, [])

            for c, v in zip(initials, finals):
                raw_pinyin = c + v
                # NOTE: post process for pypinyin outputs
                # we discriminate i, ii and iii
                if c == v:
                    assert c in punctuation
                    phone = [c]
                    word2ph["ph"].append(1)
                else:
                    v_without_tone = v[:-1]
                    tone = v[-1]

                    pinyin = c + v_without_tone
                    assert tone in "12345"

                    if c:
                        # 多音节
                        v_rep_map = {
                            "uei": "ui",
                            "iou": "iu",
                            "uen": "un",
                        }
                        if v_without_tone in v_rep_map.keys():
                            pinyin = c + v_rep_map[v_without_tone]
                    else:
                        # 单音节
                        pinyin_rep_map = {
                            "ing": "ying",
                            "i": "yi",
                            "in": "yin",
                            "u": "wu",
                        }
                        if pinyin in pinyin_rep_map.keys():
                            pinyin = pinyin_rep_map[pinyin]
                        else:
                            single_rep_map = {
                                "v": "yu",
                                "e": "e",
                                "i": "y",
                                "u": "w",
                            }
                            if pinyin[0] in single_rep_map.keys():
                                pinyin = single_rep_map[pinyin[0]] + pinyin[1:]

                    assert pinyin in self.pinyin_to_symbol_map.keys(), (pinyin, seg, raw_pinyin)
                    new_c, new_v = self.pinyin_to_symbol_map[pinyin].split(" ")
                    new_v = new_v + tone
                    phone = [new_c, new_v]
                    word2ph["ph"].append(len(phone))

                phones_list += phone
        return phones_list, word2ph

    def replace_consecutive_punctuation(self, text):
        punctuations = "".join(re.escape(p) for p in punctuation)
        pattern = f"([{punctuations}])([{punctuations}])+"
        result = re.sub(pattern, r"\1", text)
        return result

    def text_normalize(self, text):
        # https://github.com/PaddlePaddle/PaddleSpeech/tree/develop/paddlespeech/t2s/frontend/zh_normalization
        tx = TextNormalizer()
        sentences = tx.normalize(text)
        dest_text = ""
        for sentence in sentences:
            dest_text += self.replace_punctuation(sentence)

        # 避免重复标点引起的参考泄露
        dest_text = self.replace_consecutive_punctuation(dest_text)
        return dest_text
    
    def g2p(self, text):
        pattern = r"(?<=[{0}])\s*".format("".join(punctuation))
        sentences = [i for i in re.split(pattern, text) if i.strip() != ""]
        phones, word2ph = self._g2p(sentences)
        return phones, word2ph