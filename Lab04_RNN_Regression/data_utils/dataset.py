import json
from data_utils.Vocab import Vocab
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

class PhoMTDataset(Dataset):
    def __init__(self, path: str, vocab: Vocab, src_language: str = "english", tgt_language: str = "vietnamese"):
        super().__init__()
        self.vocab = vocab
        self.src_language = src_language
        self.tgt_language = tgt_language
        with open(path, encoding="utf-8") as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        item = self.data[index]

        src_sent = item[self.src_language]
        tgt_sent = item[self.tgt_language]

        encoded_src = self.vocab.encode_sentence(src_sent, self.src_language)
        encoded_tgt = self.vocab.encode_sentence(tgt_sent, self.tgt_language)
        
        return {
            self.src_language: encoded_src,
            self.tgt_language: encoded_tgt
        }
    
def collate_fn(items, pad_idx, src_language="english", tgt_language="vietnamese"):
    src_sents = [item[src_language] for item in items]
    tgt_sents = [item[tgt_language] for item in items]
    
    src_padded = pad_sequence(src_sents, batch_first=True, padding_value=pad_idx)
    tgt_padded = pad_sequence(tgt_sents, batch_first=True, padding_value=pad_idx)

    return {
        src_language: src_padded,
        tgt_language: tgt_padded
    }