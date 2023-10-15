import torch
from torch import nn
from torch import Tensor


class PositionalEncoding(nn.Module):
    def __init__(self, embedding_dimension: int, max_length: int):
        '''
        :param embedding_dimension: dimension of our word embeddings
        :param max_length: maximum length of a sequence of words
        '''
        super().__init__()
        self.embedding_dimension = embedding_dimension
        self.max_length = max_length
        self.positional_encoding = torch.zeros(max_length, embedding_dimension)
        position = torch.arange(0, max_length)
        _2i = torch.arange(0, embedding_dimension, step=2)
        self.positional_encoding[:, 0::2] = torch.sin(position / (10000 ** (_2i / embedding_dimension)))
        self.positional_encoding[:, 1::2] = torch.cos(position / (10000 ** (_2i / embedding_dimension)))

    def forward(self, tensor: Tensor):
        '''

        :param tensor: Tensor[batch_size,length,tensor_dimension]
        :return: tensor of positional embeddings Tensor[length,tensor_dimension]
        '''
        batch_size, length, tensor_dimension = tensor.size()
        return self.positional_encoding[0:length, :]
