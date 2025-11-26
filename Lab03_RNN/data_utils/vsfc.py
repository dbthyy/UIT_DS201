import torch, json, os, string
from torch.utils.data import Dataset
import torch.nn.functional as F

class VSFCVocab:
    def __init__(self, folder_path: str = "datasets/UIT-VSFC"):
        all_words = set()
        labels = set()
        
        # Create a dict with train-dev file
        for file_name in ["UIT-VSFC-train.json", "UIT-VSFC-dev.json"]:
            data = json.load(open(os.path.join(folder_path, file_name), "r", encoding="utf-8-sig"))
            for item in data:
                sentence = self.sentence_preprocessing(item["sentence"])
                all_words.update(sentence.split())
                labels.add(item["topic"])

        # word <-> id
        self.pad = "<p>"
        self.w2id = {word: idx for idx, word in enumerate(all_words, start=1)}
        self.w2id[self.pad] = 0
        self.id2w = {idx: word for word, idx in self.w2id.items()}

        # label <-> id
        self.l2id = {label: idx for idx, label in enumerate(labels)}
        self.i2ld = {idx: label for label, idx in self.l2id.items()}

    def n_labels(self):
        return len(self.l2id)

    def sentence_preprocessing(self, sentence: str) -> str:
        sentence = sentence.lower()
        translator = str.maketrans("", "", string.punctuation) #delete punctuation
        return sentence.translate(translator)

    def sentence_encoding(self, sentence: str) -> torch.Tensor:
        cleaned_sentence = self.sentence_preprocessing(sentence)
        words = cleaned_sentence.split()

        ids = []
        for word in words:
            if word in self.w2id:
                ids.append(self.w2id[word])
            else:
                ids.append(0) #missing value - 0
                
        return torch.tensor(ids, dtype=torch.long)
    
    def label_encoding(self, label: str) -> int:
        if label not in self.l2id:
            raise ValueError(f"Label {label} not in Vocab labels")
        return self.l2id[label]

class VSFCDataset(Dataset):
    def __init__(self, split_name: str, vocab: VSFCVocab):
        super().__init__()
        path = f"datasets/UIT-VSFC/UIT-VSFC-{split_name}.json"
        self._data = json.load(open(path, "r", encoding="utf-8-sig"))
        self.vocab = vocab

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        item = self._data[idx]
        input_ids = self.vocab.sentence_encoding(item["sentence"])
        label = self.vocab.label_encoding(item["topic"])
        return {
            "input_ids": input_ids,
            "label": torch.tensor(label),
        }

def collate_fn(items):
    input_ids = [item["input_ids"] for item in items]
    label_ids = [item["label"] for item in items]

    max_len = max(input_id.size(0) for input_id in input_ids)
    padded = [
        F.pad(input_id, (0, max_len - input_id.size(0)), value=0).unsqueeze(0) for input_id in input_ids
    ]

    return {
        "input_ids": torch.cat(padded, dim=0),
        "label": torch.stack(label_ids)
    }