import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class ScaledDotProductAttention(nn.Module):
    def __init__(self, head: int, d_model: int, dropout: float):
        super(ScaledDotProductAttention, self).__init__()

        self.d_model = d_model
        self.head = head
        self.d_q = d_model // head
        self.d_kv = d_model // head

        self.fc_q = nn.Linear(d_model, head * self.d_q)
        self.fc_k = nn.Linear(d_model, head * self.d_kv)
        self.fc_v = nn.Linear(d_model, head * self.d_kv)
        self.linear = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(dropout)

    def forward(self, queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor, attention_mask):
        b_s, nq = queries.shape[:2]
        nk = keys.shape[1]

        q = self.fc_q(queries).view(b_s, nq, self.head, self.d_q).permute(0, 2, 1, 3)  
        k = self.fc_k(keys).view(b_s, nk, self.head, self.d_kv).permute(0, 2, 3, 1)  
        v = self.fc_v(values).view(b_s, nk, self.head, self.d_kv).permute(0, 2, 1, 3) 

        att = torch.matmul(q, k) / math.sqrt(self.d_kv)
        
        if attention_mask is not None:
            att = att.masked_fill(attention_mask.bool(), float("-inf"))

        att = torch.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        out = torch.matmul(att, v).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.d_model)

        return self.linear(out)
    
class PositionnalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout, max_len=512):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) 
        self.register_buffer('pe', pe)

    def forward(self, x):
        pe = self.pe[:, :x.size(1)]
        pe = pe.expand(x.size(0), -1, -1)
        x = x + pe
        return self.dropout(x)

class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super(FeedForward, self).__init__()
        self.Linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.Linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor):
        return self.Linear2(self.dropout(F.relu(self.Linear1(x))))
    
class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model: int, head: int, d_ff: int, dropout: float):
        super().__init__()
        self.self_attn = ScaledDotProductAttention(head, d_model, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        
        self.ffw = FeedForward(d_model, d_ff, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, src: torch.Tensor, attention_mask: torch.Tensor):
        x = src
        attn_out = self.self_attn(x, x, x, attention_mask)
        x = x + self.dropout1(attn_out)
        x = self.norm1(x)

        ff_out = self.ffw(x)
        x = x + self.dropout2(ff_out)
        x = self.norm2(x)

        return x
        
class TransformerEncoder(nn.Module):
    def __init__(self, d_model: int, head: int, n_layers: int, d_ff: int, dropout: float):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, head, d_ff, dropout)
            for _ in range(n_layers)
        ])

    def forward(self, inputs: torch.Tensor, attention_mask: torch.Tensor):
        outputs = inputs
        for layer in self.layers:
            outputs = layer(outputs, attention_mask)
        return outputs

def generate_padding_mask(_seq: torch.Tensor, pad_value: int = 0) -> torch.Tensor:
    mask = (_seq == pad_value)
    return mask.unsqueeze(1).unsqueeze(1)