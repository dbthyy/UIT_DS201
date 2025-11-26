import torch
import torch.nn as nn

class LstmModule(nn.Module):
    def __init__(
            self,
            vocab_size: int,
            hidden_size: int,
            n_layer: int,
            n_labels: int,
            padding_idx: int = 0
            ):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=hidden_size,
            padding_idx=padding_idx
        )

        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            batch_first=True,
            num_layers=n_layer
        ) #return output, (hn, cn)

        self.classifier = nn.Linear(
            in_features=hidden_size,
            out_features=n_labels
        )

    def forward(self, inputs: torch.Tensor):
        embedded_feats = self.embedding(inputs) # (batch, seq_len, hidden_size)
        output, (hn, cn) = self.lstm(embedded_feats) 
        last_hidden = output[:, -1, :]
        logits = self.classifier(last_hidden)
        return logits