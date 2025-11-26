import os, torch
from torch.utils.data import Dataset
import torch.nn.functional as F

class PhoNERVocab:
    def __init__(self, base: str, folder_path: str ="datasets/PhoNER_COVID19"):
        words = set()
        labels = set()

        self.pad = "<pad>"
        self.unk = "<unk>"

        for split in ["train", "dev"]:
            path = os.path.join(folder_path, base, f"{split}_{base}.conll")
            print(path)
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
        self.l2id = {label: idx for idx, label in enumerate(sorted(labels))}
        self.id2l = {idx: label for label, idx in self.l2id.items()}

class PhoNERDataset(Dataset):
    def __init__(self, split_name: str, base: str, vocab: PhoNERVocab, folder_path: str = "datasets/PhoNER_COVID19"):
        self.vocab = vocab
        self.sentences = []
        self.labels = []

        path = os.path.join(folder_path, base, f"{split_name}_{base}.conll")

        sentence, tags = [], []

        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()

                if line == "":
                    if sentence:
                        self.sentences.append(sentence)
                        self.labels.append(tags)
                        sentence, tags = [], []
                    continue

                token, tag = line.split()
                sentence.append(token)
                tags.append(tag)

        if sentence:
            self.sentences.append(sentence)
            self.labels.append(tags)

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        tokens = self.sentences[idx]
        tags = self.labels[idx]

        ids = [self.vocab.w2id.get(tok, self.vocab.w2id[self.vocab.unk]) for tok in tokens]
        tag_ids = [self.vocab.l2id[tag] for tag in tags]

        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "label_ids": torch.tensor(tag_ids, dtype=torch.long)
        }

def collate_fn(batch):
    input_ids = [item["input_ids"] for item in batch]
    label_ids = [item["label_ids"] for item in batch]

    max_len = max(x.size(0) for x in input_ids)

    padded_inputs = []
    padded_labels = []

    for input_id, label in zip(input_ids, label_ids):
        pad_len = max_len - input_id.size(0)

        padded_inputs.append(F.pad(input_id, (0, pad_len), value=0))
        padded_labels.append(F.pad(label, (0, pad_len), value=0))

    return {
        "input_ids": torch.stack(padded_inputs),
        "label_ids": torch.stack(padded_labels)
    }