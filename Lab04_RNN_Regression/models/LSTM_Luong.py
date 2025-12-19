import torch
import torch.nn as nn
from data_utils.Vocab import Vocab

class LuongAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.W_a = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, decoder_out, encoder_outputs, mask=None):
        enc_proj = self.W_a(encoder_outputs)
        scores = torch.bmm(decoder_out, enc_proj.transpose(1, 2)).squeeze(1)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn_weights = torch.softmax(scores, dim=-1)
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs).squeeze(1)
        return context, attn_weights
    
class Seq2SeqLSTM(nn.Module):
    def __init__(
        self,
        d_model=256,
        num_layers=3,
        dropout=0.1,
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

        self.attention = LuongAttention(hidden_dim=2*d_model)

        self.concat_fc = nn.Linear(4*d_model, 2*d_model)
        
        self.output_head = nn.Linear(
            in_features=2*d_model,
            out_features=vocab.total_tgt_tokens
        )

        self.loss_fn = nn.CrossEntropyLoss(ignore_index=vocab.pad_idx)

    def forward_step(self, input_ids, hidden_states, cell_mem, prev_context, encoder_outputs):
        # Embed
        emb = self.tgt_embedding(input_ids)
        # Decoder input
        dec_input = torch.cat([emb, prev_context.unsqueeze(1)], dim=-1)
        # RNN step
        out, (hidden_states, cell_mem) = self.decoder(dec_input, (hidden_states, cell_mem))
        # Attention
        context, _ = self.attention(out, encoder_outputs)
        # Output projection
        combined = torch.cat([out.squeeze(1), context], dim=-1)
        dec_tilde = torch.tanh(self.concat_fc(combined))
        logits = self.output_head(dec_tilde)

        return hidden_states, cell_mem, context, logits

    def forward(self, x, y):
        enc_emb = self.src_embedding(x)
        encoder_outputs, (h_n, c_n) = self.encoder(enc_emb)

        bs = x.size(0)
        h_n = h_n.view(-1, 2, bs, self.d_model)
        c_n = c_n.view(-1, 2, bs, self.d_model)
        h = torch.cat([h_n[:, 0], h_n[:, 1]], dim=-1)
        c = torch.cat([c_n[:, 0], c_n[:, 1]], dim=-1)

        prev_context = torch.zeros(bs, self.hidden_dim, device=x.device)
        
        outputs = []
        for t in range(y.size(1) - 1):
            h, c, _, logits = self.forward_step(y[:, t].unsqueeze(1), h, c, prev_context, encoder_outputs)
            outputs.append(logits.unsqueeze(1))

        return torch.cat(outputs, dim=1)

    def predict(self, x, max_len=100):
        enc_emb = self.src_embedding(x)
        encoder_outputs, (h_n, c_n) = self.encoder(enc_emb)
        
        bs = x.size(0)
        h_n = h_n.view(-1, 2, bs, self.d_model)
        c_n = c_n.view(-1, 2, bs, self.d_model)
        h = torch.cat([h_n[:, 0], h_n[:, 1]], dim=-1)
        c = torch.cat([c_n[:, 0], c_n[:, 1]], dim=-1)
        
        y_i = torch.full(
            size=(bs, 1), 
            fill_value=self.vocab.bos_idx,
            device=x.device, 
            dtype=torch.long
        )

        prev_context = torch.zeros(bs, self.hidden_dim, device=x.device)

        outputs = []
        for _ in range(max_len):
            h, c, prev_context, logits = self.forward_step(y_i, h, c, prev_context, encoder_outputs)
            y_i = logits.argmax(dim=-1, keepdim=True)
            outputs.append(y_i)

            if (y_i == self.vocab.eos_idx).all():
                break

        return torch.cat(outputs, dim=1)