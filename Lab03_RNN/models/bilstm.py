import torch
import torch.nn as nn

class BiLSTMEncoder(nn.Module):
    def __init__(self,
                vocab_size: int,
                hidden_size: int,
                n_layers: int,
                n_labels: int,
                padding_idx: int = 0):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=hidden_size,
            padding_idx=padding_idx
        )

        self.bilstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=True
        ) #output dim = hidden_size * 2

        self.classifier = nn.Linear(
            in_features=hidden_size * 2,
            out_features=n_labels
        )

    def forward(self, input_ids: torch.Tensor):
        embedded = self.embedding(input_ids)  # (B, T, H)
        lstm_out, _ = self.bilstm(embedded)   # (B, T, H*2)
        logits = self.classifier(lstm_out)    # (B, T, n_labels)
        return logits