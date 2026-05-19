"""
This file bundles language identification functions.

Modifications (fork): Copyright (c) 2021, Adrien Barbaresi.

Original code: Copyright (c) 2011 Marco Lui <saffsd@gmail.com>.
Based on research by Marco Lui and Tim Baldwin.

See LICENSE file for more info.
https://github.com/adbar/py3langid

Projects:
https://github.com/juntaosun/LangSegment
"""

import re
from collections import defaultdict
import py3langid as langid


class LangSegment():
    _text_cache = None
    _text_lasts = None
    _text_langs = None
    _lang_count = None
    _lang_eos =   None
    
    SYMBOLS_PATTERN = r'(<([a-zA-Z|-]*)>(.*?)<\/*[a-zA-Z|-]*>)'
    
    Langfilters = ["zh", "en", "ja", "ko"]

    PARSE_TAG = re.compile(r'(⑥\$\d+[\d]{6,}⑥)')
    
    @staticmethod
    def _clears():
        LangSegment._text_cache = None
        LangSegment._text_lasts = None
        LangSegment._text_langs = None
        LangSegment._text_waits = None
        LangSegment._lang_count = None
        LangSegment._lang_eos   = None
        pass
    
    @staticmethod
    def _is_english_word(word):
        return bool(re.match(r'^[a-zA-Z]+$', word))

    @staticmethod
    def _is_chinese(word):
        for char in word:
            if '\u4e00' <= char <= '\u9fff':
                return True
        return False
    
    @staticmethod
    def _is_japanese_kana(word):
        pattern = re.compile(r'[\u3040-\u309F\u30A0-\u30FF]+')
        matches = pattern.findall(word)
        return len(matches) > 0
    
    @staticmethod
    def _insert_english_uppercase(word):
        modified_text = re.sub(r'(?<!\b)([A-Z])', r' \1', word)
        modified_text = modified_text.strip('-')
        return modified_text + " "
    
    @staticmethod
    def _saveData(words,language:str,text:str):
        # Language word statistics
        lang_count = LangSegment._lang_count
        if lang_count is None:lang_count = defaultdict(int)
        if not "|" in language:lang_count[language] += int(len(text)//2) if language == "en" else len(text)
        LangSegment._lang_count = lang_count
        # Merge the same language and save the results
        preData = words[-1] if len(words) > 0 else None
        if preData and  (preData["lang"] == language):
            text = preData["text"] + text
            preData["text"] = text
            return preData
        data = {"lang":language,"text": text}
        filters = LangSegment.Langfilters
        if filters is None or len(filters) == 0 or "?" in language or   \
            language in filters or language in filters[0] or \
            filters[0] == "*" or filters[0] in "alls-mixs-autos":
            words.append(data)
        return data

    @staticmethod
    def _addwords(words,language,text):
        if text is None or len(text.strip()) == 0:return True
        if language is None:language = ""
        language = language.lower()
        if language == 'en':text = LangSegment._insert_english_uppercase(text)
        # text = re.sub(r'[(（）)]', ',' , text) # Keep it.
        text_waits = LangSegment._text_waits
        ispre_waits = len(text_waits)>0
        preResult = text_waits.pop() if ispre_waits else None
        if preResult is None:preResult = words[-1] if len(words) > 0 else None
        if preResult and ("|" in preResult["lang"]):   
            pre_lang = preResult["lang"]
            if language in pre_lang:preResult["lang"] = language = language.split("|")[0]
            else:preResult["lang"]=pre_lang.split("|")[0]
            if ispre_waits:preResult = LangSegment._saveData(words,preResult["lang"],preResult["text"])
        pre_lang = preResult["lang"] if preResult else None
        if ("|" in language) and (pre_lang and not pre_lang in language and not "…" in language):language = language.split("|")[0]
        filters = LangSegment.Langfilters
        if "|" in language:LangSegment._text_waits.append({"lang":language,"text": text})
        else:LangSegment._saveData(words,language,text)
        return False
    
    @staticmethod
    def _get_prev_data(words):
        data = words[-1] if words and len(words) > 0 else None
        if data:return (data["lang"] , data["text"])
        return (None,"")
    
    @staticmethod
    def _match_ending(input , index):
        if input is None or len(input) == 0:return False,None
        input = re.sub(r'\s+', '', input)
        if len(input) == 0 or abs(index) > len(input):return False,None
        ending_pattern = re.compile(r'([「」“”‘’"\':：。.！!?．？])')
        return ending_pattern.match(input[index]),input[index]
    
    @staticmethod
    def _cleans_text(cleans_text):
        cleans_text = re.sub(r'([^\w]+)', '', cleans_text)
        return cleans_text
    
    @staticmethod
    def _lang_classify(cleans_text):
        language, *_ = langid.classify(cleans_text)
        return language
    
    @staticmethod
    def _parse_language(words , segment):
        LANG_JA = "ja"
        LANG_ZH = "zh"
        language = LANG_ZH
        regex_pattern = re.compile(r'([^\w\s]+)')
        lines = regex_pattern.split(segment)
        lines_max = len(lines)
        LANG_EOS =LangSegment._lang_eos
        for index, text in enumerate(lines):
            if len(text) == 0:continue
            EOS = index >= (lines_max - 1)
            nextId = index + 1
            nextText = lines[nextId] if not EOS else ""
            nextPunc = len(re.sub(regex_pattern,'',re.sub(r'\n+','',nextText)).strip()) == 0
            textPunc = len(re.sub(regex_pattern,'',re.sub(r'\n+','',text)).strip()) == 0
            if not EOS and (textPunc == True or ( len(nextText.strip()) >= 0 and nextPunc == True)):
                lines[nextId] = f'{text}{nextText}'
                continue
            number_tags = re.compile(r'(⑥\d{6,}⑥)')
            cleans_text = re.sub(number_tags, '' ,text)
            cleans_text = LangSegment._cleans_text(cleans_text)
            language = LangSegment._lang_classify(cleans_text)
            prev_language , prev_text = LangSegment._get_prev_data(words)
            if len(cleans_text) <= 3 and LangSegment._is_chinese(cleans_text):
                if LangSegment._is_japanese_kana(cleans_text):
                    language = LANG_JA
                else:
                    LANG_UNKNOWN = f'{LANG_ZH}|{LANG_JA}'
                    referen = prev_language in LANG_UNKNOWN or LANG_UNKNOWN in prev_language if prev_language else False
                    if referen and len(words) > 0:
                        language = prev_language
                    elif EOS and LANG_EOS and len(cleans_text) <= 1:
                        language = LANG_ZH
                    else:
                        match_end,match_char = LangSegment._match_ending(text, -1)
                        if match_char in "。.?？": language = prev_language if referen and len(words) > 0 else language
                        else:language = f"{LANG_UNKNOWN}|…"
            text,*_ = re.subn(number_tags , LangSegment._restore_number , text )
            LangSegment._addwords(words,language,text)
            pass
        pass
    
    @staticmethod
    def _restore_number(matche):
        value = matche.group(0)
        text_cache = LangSegment._text_cache
        if value in text_cache:
            process , data = text_cache[value]
            tag , match = data
            value = match
        return value
    
    @staticmethod
    def _pattern_symbols(item , text):
        if text is None:return text
        tag , pattern , process = item
        matches = pattern.findall(text)
        if len(matches) == 1 and "".join(matches[0]) == text:
            return text
        for i , match in enumerate(matches):
            key = f"⑥{tag}{i:06d}⑥"
            text = re.sub(pattern , key , text , count=1)
            LangSegment._text_cache[key] = (process , (tag , match))
        return text
    
    @staticmethod
    def _process_symbol(words,data):
        tag , match = data
        language = match[1]
        text = match[2]
        LangSegment._addwords(words,language,text)
        pass
    
    @staticmethod
    def _process_english(words,data):
        tag , match = data
        text = match[0]
        language = "en"
        LangSegment._addwords(words,language,text)
        pass
    
    @staticmethod
    def _process_korean(words,data):
        tag , match = data
        text = match[0]
        language = "ko"
        LangSegment._addwords(words,language,text)
        pass
    
    @staticmethod
    def _process_quotes(words,data):
        tag , match = data
        text = "".join(match)
        childs = LangSegment.PARSE_TAG.findall(text)
        if len(childs) > 0:
            LangSegment._process_tags(words , text , False)
        else:
            cleans_text = LangSegment._cleans_text(match[1])
            if len(cleans_text) <= 3:
                LangSegment._parse_language(words,text)
            else:
                language = LangSegment._lang_classify(cleans_text)
                LangSegment._addwords(words,language,text)
        pass
    
    @staticmethod
    def _process_number(words,data): # "$0" process only
        """
        Numbers alone cannot accurately identify language.
        Because numbers are universal in all languages.
        So it won't be executed here, just for testing.
        """
        tag , match = data
        language = words[0]["lang"] if len(words) > 0 else "zh"
        text = match
        LangSegment._addwords(words,language,text)
        pass
    
    @staticmethod
    def _process_tags(words , text , root_tag):
        text_cache = LangSegment._text_cache
        segments = re.split(LangSegment.PARSE_TAG, text)
        segments_len = len(segments) - 1
        for index , text in enumerate(segments):
            if root_tag:LangSegment._lang_eos = index >= segments_len
            if LangSegment.PARSE_TAG.match(text):
                process , data = text_cache[text]
                if process:process(words , data)
            else:
                LangSegment._parse_language(words , text)
            pass
        return words
    
    @staticmethod
    def _parse_symbols(text):
        TAG_NUM = "00" # "00" => default channels , "$0" => testing channel
        TAG_S1,TAG_P1,TAG_P2,TAG_EN,TAG_KO = "$1" ,"$2" ,"$3" ,"$4" ,"$5"
        process_list = [
            (  TAG_S1  , re.compile(LangSegment.SYMBOLS_PATTERN) , LangSegment._process_symbol  ),      # Symbol Tag
            (  TAG_KO  , re.compile('(([【《（(“‘"\']*(\d+\W*\s*)*[\uac00-\ud7a3]+[\W\s]*)+)')  , LangSegment._process_korean  ),      # Korean words
            (  TAG_NUM , re.compile(r'(\W*\d+\W+\d*\W*\d*)')        , LangSegment._process_number  ),      # Number words, Universal in all languages, Ignore it.
            (  TAG_EN  , re.compile(r'(([【《（(“‘"\']*[a-zA-Z]+[\W\s]*)+)')    , LangSegment._process_english ),                      # English words
            (  TAG_P1  , re.compile(r'(["\'])(.*?)(\1)')         , LangSegment._process_quotes  ),      # Regular quotes
            (  TAG_P2  , re.compile(r'([\n]*[【《（(“‘])([^【《（(“‘’”)）》】]{3,})([’”)）》】][\W\s]*[\n]{,1})')   , LangSegment._process_quotes  ),  # Special quotes, There are left and right.
        ]
        LangSegment._lang_eos = False
        text_cache = LangSegment._text_cache = {}
        for item in process_list:
            text = LangSegment._pattern_symbols(item , text)
        words = LangSegment._process_tags([] , text , True)
        lang_count = LangSegment._lang_count
        if lang_count and len(lang_count) > 0:
            lang_count = dict(sorted(lang_count.items(), key=lambda x: x[1], reverse=True))
            lang_count = list(lang_count.items())
            LangSegment._lang_count = lang_count
        return words
    
    @staticmethod
    def getTexts(text:str):
        if text is None or len(text.strip()) == 0:
            LangSegment._clears()
            return []
        # lasts
        text_langs = LangSegment._text_langs
        if LangSegment._text_lasts == text and text_langs is not None:return text_langs 
        # parse
        LangSegment._text_waits = []
        LangSegment._lang_count = None
        LangSegment._text_lasts = text
        text = LangSegment._parse_symbols(text)
        for wait_item in LangSegment._text_waits:
            lang = wait_item["lang"].split("|")[0]
            if text:
                prev_lang = text[-1]["lang"]
                if prev_lang and prev_lang in wait_item["lang"]:
                    lang = prev_lang
            LangSegment._saveData(text, lang, wait_item["text"])
        LangSegment._text_waits = []
        LangSegment._text_langs = text
        text = LangSegment._post_process_short_cjk(text)
        return text

    @staticmethod
    def _is_cjk_only(s):
        return bool(re.match(r'^[\u4e00-\u9fff\u3000-\u303f\uff00-\uffef]+$', re.sub(r'[^\w]', '', s)))

    @staticmethod
    def _post_process_short_cjk(segments):
        if not segments or len(segments) <= 1:
            return segments
        cjk_pattern = re.compile(r'[\u4e00-\u9fff]')
        for i, seg in enumerate(segments):
            if seg['lang'] != 'zh':
                continue
            cjk_chars = cjk_pattern.findall(seg['text'])
            if len(cjk_chars) > 3:
                continue
            if LangSegment._is_japanese_kana(seg['text']):
                continue
            neighbor_lang = None
            if i > 0 and segments[i - 1]['lang'] in ('ja', 'ko'):
                neighbor_lang = segments[i - 1]['lang']
            elif i < len(segments) - 1 and segments[i + 1]['lang'] in ('ja', 'ko'):
                neighbor_lang = segments[i + 1]['lang']
            if neighbor_lang:
                segments[i] = {'lang': neighbor_lang, 'text': seg['text']}
        return segments