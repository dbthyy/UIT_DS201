import json
import torch
from torch.utils.data import Dataset
from vocabs.ViOCD import ViOCDVocab

class ViOCDDataset(Dataset):
    def __init__(self, data_path: str, vocab = ViOCDVocab):
        self.vocab = vocab
        self.sentences = []
        self.labels = []

        with open(data_path, encoding="utf-8") as f:
            raw = json.load(f)

        items = raw if isinstance(raw, list) else raw.values()

        for item in items:
            self.sentences.append(vocab.encode_sentence(item["review"]))
            self.labels.append(vocab.label2idx[str(item["domain"])])

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(self.sentences[idx], dtype=torch.long),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long)
        }

def collate_fn(batch, pad_idx=0):
    input_ids = [b["input_ids"] for b in batch]
    labels = torch.stack([b["labels"] for b in batch])

    max_len = max(x.size(0) for x in input_ids)

    padded_inputs = torch.full(
        (len(input_ids), max_len),
        pad_idx,
        dtype=torch.long
    )

    attention_mask = torch.zeros(
        (len(input_ids), max_len),
        dtype=torch.long
    )

    for i, seq in enumerate(input_ids):
        padded_inputs[i, : seq.size(0)] = seq
        attention_mask[i, : seq.size(0)] = 1

    return {
        "input_ids": padded_inputs,
        "attention_mask": attention_mask,
        "labels": labels
    }