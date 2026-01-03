import os, torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from vocabs.PhoNER import PhoNERVocab

class PhoNERDataset(Dataset):
    def __init__(self, path: str, vocab: PhoNERVocab):
        self.vocab = vocab
        self.sentences = []
        self.labels = []

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
        tag_ids = [self.vocab.label2idx[tag] for tag in tags]

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
    attention_masks = []

    for input_id, label in zip(input_ids, label_ids):
        pad_len = max_len - input_id.size(0)

        padded_inputs.append(
            F.pad(input_id, (0, pad_len), value=0)
        )

        padded_labels.append(
            F.pad(label, (0, pad_len), value=-100)
        )

        attention_masks.append(
            torch.cat([
                torch.ones(input_id.size(0)),
                torch.zeros(pad_len)
            ])
        )

    return {
        "input_ids": torch.stack(padded_inputs),
        "labels": torch.stack(padded_labels),
        "attention_mask": torch.stack(attention_masks)
    }