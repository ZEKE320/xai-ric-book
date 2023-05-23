import nltk
import re
from nltk.corpus import stopwords
from nltk import FreqDist


class Counter:
    def sentence_length(self, sentence: list[str]) -> int:
        """文章中に出現する単語数を計算する"""
        return len(sentence)

    def count_word_types(self, sentence: list[str]) -> int:
        """文章中に出現する単語の種類数を計算する"""
        return len(set(sentence))

    def count_character(self, sentence: list[str], character: str) -> int:
        """文章内で出現する指定した文字の合計を計算する"""
        return sentence.count(character)

    def count_comma(self, sentence: list[str]) -> int:
        """文章内で出現するカンマの合計を計算する"""
        return self.count_character(sentence, ",")

    def count_period(self, sentence: list[str]) -> int:
        """文章内で出現するピリオドの合計を計算する"""
        return self.count_character(sentence, ".")

    def count_attention_mark(self, sentence: list[str]) -> int:
        """文章内で出現する感嘆符の合計を計算する"""
        return self.count_character(sentence, "!")

    def count_question_mark(self, sentence: list[str]) -> int:
        """文章内で出現する疑問符の合計を計算する"""
        return self.count_character(sentence, "?")

    def count_double_quotation(self, sentence: list[str]) -> int:
        """文章内で出現する二重引用符の合計を計算する"""
        return self.count_character(sentence, '"')

    def count_single_quotation(self, sentence: list[str]) -> int:
        """文章内で出現する一重引用符の合計を計算する"""
        return self.count_character(sentence, "'")

    def count_semicolon(self, sentence: list[str]) -> int:
        """文章内で出現するセミコロンの合計を計算する"""
        return self.count_character(sentence, ";")

    def count_colon(self, sentence: list[str]) -> int:
        """文章内で出現するコロンの合計を計算する"""
        return self.count_character(sentence, ":")

    def count_nonalphabetic_characters(self, sentence: list[str]) -> int:
        """文章内で出現する記号の合計を計算する"""
        pattern = r"[^a-zA-Z\s]"
        matches = re.findall(pattern, sentence)
        return len(matches)

    def count_uncommon_words(self, sentence: list[str]) -> int:
        """ストップワードではない単語の数を計算する"""
        stop_words = set(stopwords.words("english"))
        return len([word for word in sentence if word not in stop_words])


class FrequencyCalculator:
    ctr = Counter()

    def word_variation(self, sentence: list[str]) -> float:
        """文章中の単語の豊富さを計算する: (単語の種類の数)/(文章中の単語数)"""
        return self.ctr.count_word_types(sentence) / self.ctr.sentence_length(sentence)

    def average_word_length(self, sentence: list[str]) -> float:
        """文章中の単語の平均文字数を計算する"""
        return sum([len(word) for word in sentence]) / self.ctr.sentence_length(sentence)

    def comma_frequency(self, sentence: list[str]) -> float:
        """文章内で出現するカンマの割合を計算する"""
        return self.ctr.count_comma(sentence) / self.ctr.sentence_length(sentence)

    def period_frequency(self, sentence: list[str]) -> float:
        """文章内で出現するピリオドの割合を計算する"""
        return self.ctr.count_period(sentence) / self.ctr.sentence_length(sentence)

    def attention_mark_frequency(self, sentence: list[str]) -> float:
        """文章内で出現する感嘆符の割合を計算する"""
        return self.ctr.count_attention_mark(sentence) / self.ctr.sentence_length(sentence)

    def question_mark_frequency(self, sentence: list[str]) -> float:
        """文章内で出現する疑問符の割合を計算する"""
        return self.ctr.count_question_mark(sentence) / self.ctr.sentence_length(sentence)

    def double_quotation_frequency(self, sentence: list[str]) -> float:
        """文章内で出現する二重引用符の割合を計算する"""
        return self.ctr.count_double_quotation(sentence) / self.ctr.sentence_length(sentence)

    def single_quotation_frequency(self, sentence: list[str]) -> float:
        """文章内で出現する一重引用符の割合を計算する"""
        return self.ctr.count_single_quotation(sentence) / self.ctr.sentence_length(sentence)

    def semicolon_frequency(self, sentence: list[str]) -> float:
        """文章内で出現するセミコロンの割合を計算する"""
        return self.ctr.count_semicolon(sentence) / self.ctr.sentence_length(sentence)

    def colon_frequency(self, sentence: list[str]) -> float:
        """文章内で出現するコロンの割合を計算する"""
        return self.ctr.count_colon(sentence) / self.ctr.sentence_length(sentence)

    def nonalphabetic_characters_frequency(self, sentence: list[str]) -> float:
        """文章内で出現する記号の割合を計算する"""
        return self.ctr.count_nonalphabetic_characters(sentence) / self.ctr.sentence_length(
            sentence
        )

    def uncommon_word_frequency(self, sentence: list[str]) -> float:
        """ストップワードではない単語の割合を計算する"""
        return self.ctr.count_uncommon_words(sentence) / self.ctr.sentence_length(sentence)

    def all_pos_frequency(self, sentence: list[str]) -> dict[str, int]:
        """文章中の全ての品詞の割合を計算する"""
        pos_list = nltk.pos_tag(sentence)
        fd = FreqDist(pos_list)

        total_tags = fd.N()
        return {tag[1]: count / total_tags for tag, count in fd.items()}


class DatasetGenerator:
    ctr = Counter()
    fc = FrequencyCalculator()

    def __init__(self, tags: list[str] = None):
        self.columns = [
            "word variation",
            "uncommon word frequency",
            "sentence length",
            "average word length",
        ]
        self.columns.extend(tags)

    def generate_dataset_sent(
        self, sentence: list[str], tags: list[str], correctness: bool
    ) -> tuple[list[float], bool]:
        """文章のリストから特徴量のリストを生成する"""
        freq_dict = self.fc.all_pos_frequency(sentence)
        return (
            [
                self.fc.word_variation(sentence),
                self.fc.uncommon_word_frequency(sentence),
                self.ctr.sentence_length(sentence),
                self.fc.average_word_length(sentence),
            ]
            + [freq_dict.get(tag, 0) for tag in tags],
            correctness,
        )

    def generate_dataset_para(
        self, paragraph: list[list[str]], tags: list[str], correctness: bool
    ) -> tuple[list[float], bool]:
        """段落のリストから特徴量のリストを生成する"""
        sentence = [word for sentence in paragraph for word in sentence]
        return self.generate_dataset_sent(sentence, tags, correctness)


def para2sent(para: list[list[str]]) -> list[str]:
    """段落のリストを文章のリストに変換する"""
    return [word for sent in para for word in sent]
