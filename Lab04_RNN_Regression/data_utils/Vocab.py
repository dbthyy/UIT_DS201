import os, torch, json, re, string
from typing import List

class Vocab:
    def __init__(self, data_dir: str, src_language: str, tgt_language: str) -> None:
        self._init_special_tokens()
        self.src_language = src_language
        self.tgt_language = tgt_language
        self.make_vocab(data_dir)

    def _init_special_tokens(self):
        self.pad_token = "<pad>"
        self.bos_token = "<bos>"
        self.eos_token = "<eos>"
        self.unk_token = "<unk>"

        self.specials = [self.pad_token, self.bos_token, self.eos_token, self.unk_token]

        self.pad_idx = 0
        self.bos_idx = 1
        self.eos_idx = 2
        self.unk_idx = 3
    
    def make_vocab(self, data_dir: str):
        src_words = set()
        tgt_words = set()

        for file in os.listdir(data_dir):
            path = os.path.join(data_dir, file)
            data = json.load(open(path, encoding="utf-8"))

            for item in data:
                src_tokens = self.preprocess_sentence(item[self.src_language])
                tgt_tokens = self.preprocess_sentence(item[self.tgt_language])
                src_words.update(src_tokens)
                tgt_words.update(tgt_tokens)

        src_i2s = self.specials + list(src_words)
        self.src_s2i = {tok: i for i, tok in enumerate(src_i2s)}
        self.src_i2s = {i: tok for tok, i in self.src_s2i.items()}

        tgt_i2s = self.specials + list(tgt_words)
        self.tgt_s2i = {tok: i for i, tok in enumerate(tgt_i2s)}
        self.tgt_i2s = {i: tok for tok, i in self.tgt_s2i.items()}

    @property
    def total_src_tokens(self) -> int:
        return len(self.src_i2s)

    @property
    def total_tgt_tokens(self) -> int:
        return len(self.tgt_i2s)

    def preprocess_sentence(self, sentence: str) -> List[str]:
        sentence = sentence.lower()

        punctuation = string.punctuation
        sentence = sentence.replace("\ufeff", "")
        sentence = re.sub(f"[{re.escape(punctuation)}]", " ", sentence)
        sentence = re.sub(r"[–—…“”•·]", " ", sentence)
        sentence = re.sub(r"\s+", " ", sentence)
        return sentence.strip().split()

    def encode_sentence(self, sentence: str, language: str) -> torch.Tensor:
        s2i = self.src_s2i if language == self.src_language else self.tgt_s2i
        tokens = self.preprocess_sentence(sentence)
        vec = [s2i[token] if token in s2i else self.unk_idx for token in tokens]
        vec = [self.bos_idx] + vec + [self.eos_idx]
        vec = torch.Tensor(vec).long()
        return vec

    def decode_sentence(self, tensor: torch.Tensor, language: str) -> str:
        i2s = self.src_i2s if language == self.src_language else self.tgt_i2s
        ids = tensor.tolist()
        
        words = []
        for idx in ids:
            if idx in {self.pad_idx, self.bos_idx, self.eos_idx}:
                continue
            words.append(i2s.get(idx, self.unk_token))
        return " ".join(words)