import torch
import torch.nn as nn
from data_utils.Vocab import Vocab

class BahdanauAttention(nn.Module):
    def __init__(self, enc_dim, dec_dim):
        super().__init__()
        self.W_h = nn.Linear(enc_dim, dec_dim, bias=False)
        self.W_s = nn.Linear(dec_dim, dec_dim, bias=False)
        self.v = nn.Linear(dec_dim, 1, bias=False)

    def forward(self, decoder_hidden, encoder_outputs, mask=None):
        dec = self.W_s(decoder_hidden).unsqueeze(1)
        enc = self.W_h(encoder_outputs)

        energy = self.v(torch.tanh(enc + dec)).squeeze(-1)
        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e9)

        attn_weights = torch.softmax(energy, dim=1)
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)

        return context.squeeze(1), attn_weights

class Seq2SeqLSTM(nn.Module):
    def __init__(
        self,
        d_model: int = 256,
        num_layers: int = 3,
        dropout: float = 0.1,
        vocab: Vocab = None
    ):
        super().__init__()

        self.vocab = vocab
        self.d_model = d_model

        self.src_embedding = nn.Embedding(
            num_embeddings=vocab.total_src_tokens,
            embedding_dim=d_model,
            padding_idx=vocab.pad_idx
        )

        self.tgt_embedding = nn.Embedding(
            num_embeddings=vocab.total_tgt_tokens,
            embedding_dim=2*d_model,
            padding_idx=vocab.pad_idx
        )

        self.encoder = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )

        self.decoder = nn.LSTM(
            input_size=4*d_model,  
            hidden_size=2*d_model,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=False
        )

        self.attention = BahdanauAttention(
            enc_dim=2*d_model,
            dec_dim=2*d_model
        )

        self.output_head = nn.Linear(
            in_features=4*d_model, 
            out_features=vocab.total_tgt_tokens
        )

        self.loss_fn = nn.CrossEntropyLoss(ignore_index=vocab.pad_idx)

    def forward_step(self, input_ids, hidden_states, cell_mem, encoder_outputs, src_mask=None):
        # Embed
        emb = self.tgt_embedding(input_ids)
        # Decoder input
        dec_input = torch.cat([emb, encoder_outputs[:, :1] * 0], dim=-1)
        # RNN step
        out, (hidden_states, cell_mem) = self.decoder(dec_input, (hidden_states, cell_mem))
        # Attention
        context, _ = self.attention(hidden_states[-1], encoder_outputs, src_mask)
        # Output projection
        combined = torch.cat([out.squeeze(1), context], dim=-1)
        logits = self.output_head(combined)

        return hidden_states, cell_mem, logits

    def forward(self, x, y):
        enc_emb = self.src_embedding(x)
        encoder_outputs, (h_n, c_n) = self.encoder(enc_emb)

        bs = x.size(0)
        num_layers = h_n.size(0) // 2
        h_n = h_n.view(num_layers, 2, bs, self.d_model)
        c_n = c_n.view(num_layers, 2, bs, self.d_model)
        h = torch.cat([h_n[:, 0], h_n[:, 1]], dim=-1)
        c = torch.cat([c_n[:, 0], c_n[:, 1]], dim=-1)

        decoder_input = y[:, :-1]

        outputs = []
        for t in range(decoder_input.size(1)):
            h, c, _, logits = self.forward_step(decoder_input[:, t].unsqueeze(1), h, c, encoder_outputs)
            outputs.append(logits.unsqueeze(1))

        return torch.cat(outputs, dim=1)

    def predict(self, x, max_len=100):
        enc_emb = self.src_embedding(x)
        encoder_outputs, (h_n, c_n) = self.encoder(enc_emb)

        bs = x.size(0)
        num_layers = h_n.size(0) // 2
        h_n = h_n.view(num_layers, 2, bs, self.d_model)
        c_n = c_n.view(num_layers, 2, bs, self.d_model)
        h = torch.cat([h_n[:, 0], h_n[:, 1]], dim=-1)
        c = torch.cat([c_n[:, 0], c_n[:, 1]], dim=-1)

        y_i = torch.full(
            size=(bs, 1), 
            fill_value=self.vocab.bos_idx,
            device=x.device, 
            dtype=torch.long
        )

        outputs = []
        for _ in range(max_len):
            h, c, logits = self.forward_step(y_i, h, c, encoder_outputs)
            y_i = logits.argmax(dim=-1, keepdim=True)
            outputs.append(y_i)

            if (y_i == self.vocab.eos_idx).all():
                break

        return torch.cat(outputs, dim=1)