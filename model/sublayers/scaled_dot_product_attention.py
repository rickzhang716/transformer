import torch
import typing
from typing import Optional
from torch import nn
from torch import Tensor
import math


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, query: Tensor, key: Tensor, value: Tensor, mask: Optional[Tensor] = None):
        # input: Tensor[batch_size, head_number, length, key_dimension]
        # output: Tensor[batch_size, head_number, length, value_dimension]

        batch_size, head_number, length, key_dimension = key.size()
        # formula to calculate attention score
        attention_score: Tensor = (query @ (key.transpose(-2, -1))) / math.sqrt(key_dimension)

        # handle mask
        if mask is not None:
            attention_score = attention_score.masked_fill(mask == 0, -1e9)

        # get weighted sum attention vectors, weighted with respect to attention scores
        # softmax to normalize, sum = 1
        attention_probability = self.softmax(attention_score)
        output = attention_probability @ value
        return output
