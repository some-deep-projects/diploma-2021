import json
from pathlib import Path
from joblib import Memory
from word2word import Word2word
import requests

import logging

logger = logging.getLogger(Path(__file__).name)

CACHE_DIR = Path(__file__).resolve().parent / "data" / "trans_cache"
memory = Memory(str(CACHE_DIR), verbose=0)


# @lru_cache(maxsize=8192)
@memory.cache
def get_translations_w2w(words, src_lang: str = "en", dst_lang: str = "es"):
    en2es = Word2word(src_lang, dst_lang)
    NOT_TRANSLATED, TRANSLATION_QUERIES = 0, 0
    unique_words = set(words)
    TRANSLATION_QUERIES += len(unique_words)
    lemma2translations = {}
    for word in unique_words:
        try:
            translations = en2es(word)
        except KeyError:
            translations = []
            NOT_TRANSLATED += 1
        lemma2translations[word] = translations
    return lemma2translations, TRANSLATION_QUERIES, NOT_TRANSLATED


YANDEX_API_KEY = 'dict.1.1.20210130T135840Z.8ce09c5bb76e54cc.a66878114ec9769ba494f26029543146bfb43f73'

@memory.cache
def request_yandex(word: str, lang: str = 'en-es'):
    request = 'https://dictionary.yandex.net/api/v1/dicservice.json/lookup?key={}&lang={}&text={}'
    requestf = request.format(YANDEX_API_KEY, lang, word)
    request_text = requests.get(requestf).text
    if "code" in json.loads(request_text):
        raise RuntimeError(f"Error while translating {word} ({lang}): {request_text}")
    return request_text


@memory.cache
def get_translations_yandex(words, src_lang: str = "en", dst_lang: str = "es"):
    # request = 'https://dictionary.yandex.net/api/v1/dicservice.json/lookup?key={}&lang=en-es&text={}'
    NOT_TRANSLATED, TRANSLATION_QUERIES = 0, 0
    unique_words = set(words)
    TRANSLATION_QUERIES += len(unique_words)
    lemma2translations = {}
    for word in unique_words:
        translations = []
        response = json.loads(request_yandex(word, f"{src_lang}-{dst_lang}"))
        try:
            for m in response["def"]:
                for tr in m["tr"]:
                    translations.append(tr['text'])
        except Exception as e:
            print(f"Exception while translating '{word}'")
            print(response)
            print(YANDEX_API_KEY)
            raise e
        if len(translations) == 0:
            NOT_TRANSLATED += 1
        lemma2translations[word] = translations
    return lemma2translations, TRANSLATION_QUERIES, NOT_TRANSLATED

@memory.cache
def get_translations_yandex_syns(words, src_lang: str = "en", dst_lang: str = "es"):
    NOT_TRANSLATED, TRANSLATION_QUERIES = 0, 0
    unique_words = set(words)
    TRANSLATION_QUERIES += len(unique_words)
    lemma2translations = {}
    for word in unique_words:
        translations = []
        response = json.loads(request_yandex(word, f"{src_lang}-{dst_lang}"))
        try:
            for m in response["def"]:
                for tr in m["tr"]:
                    translations.append(tr['text'])
                    if 'syn' in tr:
                        for s in tr['syn']:
                            translations.append(s['text'])
        except Exception as e:
            print(f"Exception while translating '{word}'")
            print(response)
            print(YANDEX_API_KEY)
            raise e
        if len(translations) == 0:
            NOT_TRANSLATED += 1
        lemma2translations[word] = translations
    return lemma2translations, TRANSLATION_QUERIES, NOT_TRANSLATED


TRANSLATIONS_MODES = ["w2w", "yandex", "yandex_syns"]


def get_translations(words, mode, src_lang: str = "en", dst_lang: str = "es"):
    logger.info(f"Getting translations using {mode} from {src_lang} to {dst_lang}")
    if mode == "w2w":
        lemma2translations, aggr_total, aggr_neg = get_translations_w2w(tuple(words), src_lang, dst_lang)
    # elif mode == "babelnet":
    #     lemma2translations, aggr_total, aggr_neg = get_translations_babelnet(tuple(words), src_lang, dst_lang)
    elif mode == "yandex":
        lemma2translations, aggr_total, aggr_neg = get_translations_yandex(tuple(words), src_lang, dst_lang)
    elif mode == "yandex_syns":
        lemma2translations, aggr_total, aggr_neg = get_translations_yandex_syns(tuple(words), src_lang, dst_lang)
    elif "+" in mode:
        modes = mode.split('+')
        assert all(m in TRANSLATIONS_MODES for m in modes), f"Incorrect mode: {mode}"
        aggr_tr, aggr_total, aggr_neg = [], [], []
        for m in modes:
            tr, total, neg = get_translations(words, m, src_lang, dst_lang)
            aggr_tr.append(tr)
            aggr_total.append(total)
            aggr_neg.append(neg)
        lemma2translations = dict()
        for lemma in aggr_tr[0].keys():
            lemma2translations[lemma] = [
                tr for l2tr in aggr_tr for tr in l2tr[lemma]
            ]
        aggr_total = sum(aggr_total)
        aggr_neg = sum(aggr_neg)
        return lemma2translations, aggr_total, aggr_neg
    else:
        raise ValueError(f"Incorrect mode: {mode}")

    for lemma in lemma2translations.keys():
        lemma2translations[lemma] = [
            t.lower() for t in lemma2translations[lemma]
        ]

    return lemma2translations, aggr_total, aggr_neg
