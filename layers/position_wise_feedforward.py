from torch import nn
from torch import Tensor


class PositionWiseFeedForward(nn.Module):
    def __init__(self, embedding_dimension: int, hidden_dimension: int, dropout_probability: float = 0.2):
        super().__init__()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_probability)
        self.hidden1 = nn.Linear(embedding_dimension, hidden_dimension)
        self.hidden2 = nn.Linear(hidden_dimension, embedding_dimension)

    def forward(self, x: Tensor):
        x = self.hidden1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.hidden2(x)
        return x
