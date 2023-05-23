import nltk
from nltk import FreqDist
import feature_counter as fc


def word_variation(sentence: list[str]) -> float:
    """文章中の単語の豊富さを計算する: (単語の種類の数)/(文章中の単語数)"""
    return fc.count_word_types(sentence) / fc.sentence_length(sentence)


def average_word_length(sentence: list[str]) -> float:
    """文章中の単語の平均文字数を計算する"""
    return sum([len(word) for word in sentence]) / fc.sentence_length(sentence)


def comma_frequency(sentence: list[str]) -> float:
    """文章内で出現するカンマの割合を計算する"""
    return fc.count_comma(sentence) / fc.sentence_length(sentence)


def period_frequency(sentence: list[str]) -> float:
    """文章内で出現するピリオドの割合を計算する"""
    return fc.count_period(sentence) / fc.sentence_length(sentence)


def attention_mark_frequency(sentence: list[str]) -> float:
    """文章内で出現する感嘆符の割合を計算する"""
    return fc.count_attention_mark(sentence) / fc.sentence_length(sentence)


def question_mark_frequency(sentence: list[str]) -> float:
    """文章内で出現する疑問符の割合を計算する"""
    return fc.count_question_mark(sentence) / fc.sentence_length(sentence)


def double_quotation_frequency(sentence: list[str]) -> float:
    """文章内で出現する二重引用符の割合を計算する"""
    return fc.count_double_quotation(sentence) / fc.sentence_length(sentence)


def single_quotation_frequency(sentence: list[str]) -> float:
    """文章内で出現する一重引用符の割合を計算する"""
    return fc.count_single_quotation(sentence) / fc.sentence_length(sentence)


def semicolon_frequency(sentence: list[str]) -> float:
    """文章内で出現するセミコロンの割合を計算する"""
    return fc.count_semicolon(sentence) / fc.sentence_length(sentence)


def colon_frequency(sentence: list[str]) -> float:
    """文章内で出現するコロンの割合を計算する"""
    return fc.count_colon(sentence) / fc.sentence_length(sentence)


def nonalphabetic_characters_frequency(sentence: list[str]) -> float:
    """文章内で出現する記号の割合を計算する"""
    return fc.count_nonalphabetic_characters(sentence) / fc.sentence_length(sentence)


def uncommon_word_frequency(sentence: list[str]) -> float:
    """ストップワードではない単語の割合を計算する"""
    return fc.count_uncommon_words(sentence) / fc.sentence_length(sentence)


def all_pos_frequency(sentence: list[str]) -> dict[str, int]:
    """文章中の全ての品詞の割合を計算する"""
    pos_list = nltk.pos_tag(sentence)
    fd = FreqDist(pos_list)

    total_tags = fd.N()
    return {tag[1]: count / total_tags for tag, count in fd.items()}
