import torch
import torch.nn as nn
from models.transformer import *

class TransformerModel(nn.Module):
    def __init__(self, d_model: int, head: int, n_layers: int, d_ff: int, dropout: float, vocab):
        super().__init__()

        self.vocab = vocab
        self.embedding = nn.Embedding(
            num_embeddings=len(vocab.w2id),
            embedding_dim=d_model,
            padding_idx=vocab.w2id[vocab.pad]
        )
        self.PE = PositionnalEncoding(d_model, dropout)
        self.encoder = TransformerEncoder(
            d_model, head, n_layers, d_ff, dropout
        )

        self.classifier = nn.Linear(d_model, len(vocab.label2idx))

        self.dropout = nn.Dropout(dropout)

        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor):
        attention_mask = generate_padding_mask(
            input_ids,
            self.vocab.w2id[self.vocab.pad]
        )

        x = self.embedding(input_ids)        
        x = self.PE(x)
        x = self.encoder(x, attention_mask)  

        logits = self.classifier(self.dropout(x)) 

        loss = self.loss_fn(
            logits.view(-1, logits.size(-1)),  
            labels.view(-1)                    
        )
        
        return logits, loss