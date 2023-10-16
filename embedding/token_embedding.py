from torch import nn


class TokenEmbedding(nn.Embedding):
    '''
    token embedding: convert word to its embedding vector

    '''

    def __init__(self, vocab_size: int, embedding_dimension: int):
        '''

        :param vocab_size: number of unique words in our dictionary
        :param embedding_dimension: length of embedding vector
        '''
        super().__init__(self, vocab_size, embedding_dimension, padding_idx=1)
