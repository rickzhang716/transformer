from positional_encoding import PositionalEncoding
from token_embedding import TokenEmbedding
from torch import nn
from torch import Tensor


class TransformerEmbedding(nn.Module):
    def __init__(self, embedding_dimension: int, max_length: int, dropout_probability: int, vocab_size: int):
        '''

        :param embedding_dimension: length of embedding vectors
        :param max_length: maximum length of a sequence of words
        :param dropout_probability: probability of dropping out an element in a tensor
        :param vocab_size: number of unique words in our vocabulary
        '''
        super().__init__()
        self.token_embedding = TokenEmbedding(vocab_size, embedding_dimension)
        self.positional_encoding = PositionalEncoding(embedding_dimension, max_length)
        self.dropout = nn.Dropout(p=dropout_probability)

    def forward(self, x: Tensor):
        token_embedding = self.token_embedding(x)
        position_encoding = self.positional_encoding(x)
        y = self.dropout(token_embedding + position_encoding)
        return y


