import os

class PhoNERVocab:
    def __init__(self, base: str, folder_path: str ="datasets/PhoNER_COVID19"):
        words = set()
        labels = set()

        self.pad = "<pad>"
        self.unk = "<unk>"

        for split in ["train", "dev"]:
            path = os.path.join(folder_path, base, f"{split}_{base}.conll")
            with open(path, encoding="utf-8") as file:
                for line in file:
                    line = line.strip()
                    if not line:
                        continue
                    token, tag = line.split()
                    words.add(token)
                    labels.add(tag)

        # word2id
        self.w2id = {self.pad: 0, self.unk: 1}
        for w in sorted(words):
            self.w2id[w] = len(self.w2id)

        # id2word
        self.id2w = {i: w for w, i in self.w2id.items()}

        # label2id
        labels = sorted(labels)
        assert "O" in labels

        self.label2idx = {"O": 0}
        for label in labels:
            if label != "O":
                self.label2idx[label] = len(self.label2idx)

        self.idx2label = {v: k for k, v in self.label2idx.items()}
