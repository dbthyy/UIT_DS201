import torch
import torch.nn as nn
from models.transformer import *

class TransformerModel(nn.Module):
    def __init__(self, d_model: int, head: int, n_layers: int, d_ff: int, dropout: float, vocab):
        super().__init__()

        self.vocab = vocab
        self.embedding = nn.Embedding(len(vocab.w2id), d_model)
        self.PE = PositionnalEncoding(d_model, dropout)
        self.encoder = TransformerEncoder(d_model, head, n_layers, d_ff, dropout)
        
        self.ln_head = nn.Linear(d_model, len(vocab.label2idx))
        self.dropout = nn.Dropout(dropout)
        
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor):
        attention_mask = generate_padding_mask(
          input_ids, 
          self.vocab.pad_idx
        )

        x = self.embedding(input_ids)
        x = self.PE(x)
        x = self.encoder(x, attention_mask)

        mask = (input_ids != self.vocab.pad_idx).unsqueeze(-1)
        x = (x * mask).sum(1) / mask.sum(1)

        logits = self.dropout(self.ln_head(x))

        return logits, self.loss_fn(logits, labels)