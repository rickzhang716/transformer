import torch
from torch import nn
import math
from torch import Tensor
import typing
from typing import Optional
from scaled_dot_product_attention import ScaledDotProductAttention


class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dimension: int, key_dimension: int, value_dimension: int, num_heads: int):
        super().__init__()

        assert (embedding_dimension % num_heads == 0,
                "model dimension must be divisible by number of heads.")
        self.model_dimension = embedding_dimension
        self.num_heads = num_heads
        self.key_dimension = key_dimension
        self.value_dimension = value_dimension

        self.scaled_dot_product_attention = ScaledDotProductAttention()

        self.W_q = nn.Linear(embedding_dimension, num_heads *
                             self.key_dimension)  # query weight matrix
        self.W_k = nn.Linear(embedding_dimension, num_heads *
                             self.key_dimension)  # keys weight matrix
        self.W_v = nn.Linear(embedding_dimension, num_heads *
                             self.value_dimension)  # values weight matrix
        self.W_o = nn.Linear(num_heads * self.value_dimension,
                             embedding_dimension)  # output weight matrix

    def split_heads(self, tensor: Tensor):
        # reshape input to have num_heads for multi-headed attention
        # use a single tensor with another dimension (representing the head number)
        # instead of making {num_heads} number of Modules, similar to grouped convolutions.

        # input: Tensor[batch_size, length, dimension]
        # output: Tensor[batch_size, head_number, length, tensor_dimension]
        #   note that tensor_dimension can be key_dimension or value_dimension,
        #   depending on which matrix we are splitting.

        batch_size, length, dimension = tensor.size()
        return tensor.view(batch_size, length, self.num_heads, self.key_dimension).transpose(1, 2)

    def combine_heads(self, tensor: Tensor):
        # combine heads back into original shape
        # input: Tensor[batch_size, head_number, length, tensor_dimension]
        #   we only ever combine attention matrices back together,
        #   thus it will always be tensor_dimension = value_dimension

        # output: Tensor[batch_size, length, dimension]

        batch_size, num_heads, length, tensor_dimension = tensor.size()
        return tensor.transpose(1, 2).contiguous().view(batch_size, length,
                                                        num_heads * tensor_dimension)

    def forward(self, query: Tensor, key: Tensor, value: Tensor, mask: Optional[int] = None):
        # multiply by weights
        query = self.W_q(query)
        key = self.W_k(key)
        value = self.W_v(value)

        # split heads
        query = self.split_heads(query)
        key = self.split_heads(key)
        value = self.split_heads(value)

        # get attention matrix
        attention_matrix = self.scaled_dot_product_attention(
            query, key, value, mask)

        # multiply by output weights to get attention vector
        output = self.W_o(self.combine_heads(attention_matrix))
        return output
