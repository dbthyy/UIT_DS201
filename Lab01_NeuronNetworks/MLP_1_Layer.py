import torch
from torch import nn
from torch.nn import functional as F

class MLP1Layer(torch.nn.Module):
    def __init__(self, input_size, output_size, dropout_rate):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        x = x.reshape(batch_size, -1)
        
        # Single layer - ReLU
        x = self.linear(x)
        x = self.dropout(x)
        x = F.log_softmax(x, dim=-1)
        return x