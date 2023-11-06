from torch import nn
from torch import Tensor
import math


class TokenEmbedding(nn.Module):
    '''
    token embedding: convert word to its embedding vector
    '''

    def __init__(self, vocab_size: int, embedding_dimension: int):
        '''

        :param vocab_size: number of unique words in our dictionary
        :param embedding_dimension: length of embedding vector
        '''
        super().__init__()
        self.embedding_dimension = embedding_dimension
        self.embed = nn.Embedding(vocab_size, embedding_dimension)

    def forward(self, x: Tensor):
        return self.embed(x) * math.sqrt(self.embedding_dimension)
