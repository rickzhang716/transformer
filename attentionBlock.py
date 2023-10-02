import torch
import torch.nn as nn
import math


class MultiHeadAttention(nn.Module):
    def __init__(self, model_dimension, num_heads):
        super().__init__()
        assert (model_dimension % num_heads == 0, "model dimension must be divisible by number of heads.")
        self.model_dimension = model_dimension
        self.num_heads = num_heads
        self.key_dimension = model_dimension // num_heads

        self.W_q = nn.Linear(model_dimension, model_dimension)  # query weight matrix
        self.W_k = nn.Linear(model_dimension, model_dimension)  # keys weight matrix
        self.W_v = nn.Linear(model_dimension, model_dimension)  # values weight matrix
        self.W_o = nn.Linear(model_dimension, model_dimension)  # output weight matrix

    def scaled_dot_product_attention(self, query, key, value, mask=None):
        # formula to calculate attention score
        attention_score = torch.matmul(query, key.transpose(2, 3) / math.sqrt(self.key_dimension))

        # handle mask
        if mask is not None:
            attention_score = attention_score.masked_fill(mask == 0, -1e9)

        # get weighted sum attention vectors, weighted with respect to attention scores
        # softmax to normalize, sum = 1
        attention_probability = torch.softmax(attention_score, dim=-1)
        output = torch.matmul(attention_probability, value)
        return output

    def split_heads(self, x):
        # reshape input to have num_heads for multi-headed attention
        batch_size, sequence_length, model_dimension = x.size()
        return x.view(batch_size, sequence_length, self.num_heads, self.key_dimension).transpose(1, 2)

    def combine_heads(self, x):
        # combine heads back into original shape
        batch_size, _, sequence_length = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, sequence_length, self.model_dimension)

    def forward(self, query, key, value, mask=None):
        # multiply by weights
        query = self.W_q(query)
        key = self.W_k(key)
        value = self.W_v(value)

        query = self.split_heads(query)
        key = self.split_heads(key)
        value = self.split_heads(value)

        # get attention matrix
        attention_matrix = self.scaled_dot_product_attention(query, key, value, mask)

        # multiply by output weights to get attention vector
        output = self.W_o(self.combine_heads(attention_matrix))
        return output
