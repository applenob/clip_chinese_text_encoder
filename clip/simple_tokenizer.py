import collections
from typing import Text, List


class SimpleCharTokenizer():
    def __init__(self, zh_vocab_file, unk: int = 0):
        self.zh_vocab_file = zh_vocab_file
        self._mapping = {}
        self._mapping
        self._freq = collections.defaultdict(int)
        self.unk = unk
        self.load_vocab(zh_vocab_file)
        self.encoder = self._mapping

    def init_map(self) -> None:
        self._mapping = {'<unk>': 0}

    def add(self, word: Text) -> None:
        if word not in self._mapping:
            self._mapping[word] = len(self._mapping)
        self._freq[word] += 1

    def trim(self, min_frequency: int) -> None:
        self._freq = sorted(self._freq.items(), key=lambda x: x[1], reverse=True)
        self.init_map()
        idx = len(self._mapping)

        for word, count in self._freq:
            if count < min_frequency:
                break
            if word in self._mapping:
                continue
            self._mapping[word] = idx
            idx += 1
        self._freq = dict(self._freq[:idx - 1])

    def fit(self, corpus: List[List[Text]], min_frequency: int = 5) -> None:
        for sen in corpus:
            for word in sen:
                self.add(word)
        self.trim(min_frequency)
        self._mapping["<|startoftext|>"] = max(self._mapping.values()) + 1
        self._mapping["<|endoftext|>"] = max(self._mapping.values()) + 1

    def save_vocab(self, vocab_file: Text) -> None:
        print(f"Saving vocab to {vocab_file}")
        with open(vocab_file, "w") as f:
            for word, idx in sorted(self._mapping.items(), key=lambda x: x[1]):
                f.write(f"{word}\t{idx}\n")

    def load_vocab(self, vocab_file: Text) -> None:
        print(f"Loading vocab from {vocab_file}")
        self.init_map()
        with open(vocab_file, "r") as f:
            for line in f:
                sp = line.strip().split('\t')
                if len(sp) < 2:
                    continue
                word, idx = sp[0], int(sp[1])
                self._mapping[word] = idx

    def encode(self, sentence):
        return self.tokenize_one_sentence(sentence)

    def tokenize_sentences(self, sentences: List[List[Text]]) -> List[List[int]]:
        res = []
        for sen in sentences:
            res.append([self._mapping.get(word, self.unk) for word in sen])
        return res

    def tokenize_one_sentence(self, sentence: List[Text]) -> List[int]:
        return [self._mapping.get(word, self.unk) for word in sentence]
