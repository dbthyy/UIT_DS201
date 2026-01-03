import re
import json
from typing import List
from pyvi import ViTokenizer

class ViOCDVocab:
    def __init__(self, data_path: str):
        words = set()
        labels = set()

        self.pad = "<pad>"
        self.unk = "<unk>"

        with open(data_path, encoding="utf-8") as f:
            raw = json.load(f)

        items = raw if isinstance(raw, list) else raw.values()

        for item in items:
            sent = self._preprocess(item["review"])
            for w in sent.split():
                words.add(w)
            labels.add(str(item["domain"]))

        # word vocab
        self.w2id = {self.pad: 0, self.unk: 1}
        for w in sorted(words):
            self.w2id[w] = len(self.w2id)
        self.id2w = {i: w for w, i in self.w2id.items()}

        # label vocab
        self.label2idx = {l: i for i, l in enumerate(sorted(labels))}
        self.idx2label = {i: l for l, i in self.label2idx.items()}

        self.pad_idx = self.w2id[self.pad]
        self.unk_idx = self.w2id[self.unk]

    def _preprocess(self, text: str) -> str:
        if not isinstance(text, str):
            return ""
        text = text.lower()
        text = re.sub(r"\s+", " ", text).strip()
        return ViTokenizer.tokenize(text)

    def encode_sentence(self, sentence: str) -> List[int]:
        tokens = self._preprocess(sentence).split()
        return [self.w2id.get(t, self.unk_idx) for t in tokens]