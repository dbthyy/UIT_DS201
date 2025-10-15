import torch
from torch import nn
from torch.nn import functional as F

class MLP3Layer(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size, dropout_rate):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size1)
        self.linear2 = nn.Linear(hidden_size1, hidden_size2)
        self.linear3 = nn.Linear(hidden_size2, output_size)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        x = x.reshape(batch_size, -1)

        # Two hidden layers - ReLU
        x = self.linear1(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.linear2(x)
        x = F.relu(x)
        x = self.dropout(x)

        # Output layer - Softmax
        x = self.linear3(x)
        x = F.log_softmax(x, dim=-1)
        return x