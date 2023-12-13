import torch
import copy
from torch import nn
from torch import Tensor
from torch.nn.functional import log_softmax
from .encoder import Encoder
from .decoder import Decoder

c = copy.deepcopy


class TransformerFinalLayer(nn.Module):
    def __init__(self, embedding_dimension: int, decoder_vocab_size: int):
        super().__init__()
        self.linear = nn.Linear(embedding_dimension, decoder_vocab_size)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, src: Tensor):
        return log_softmax(self.linear(src), dim=-1)


class Transformer(nn.Module):
    def __init__(self,
                 encoder_vocab_size: int,
                 decoder_vocab_size: int,

                 embedding_dimension: int = 512,
                 key_dimension: int = 64,
                 value_dimension: int = 64,
                 num_layers: int = 6,
                 num_heads: int = 8,

                 encoder_max_length: int = 256,
                 decoder_max_length: int = 256,
                 hidden_dimension: int = 2048,

                 epsilon: float = 1e-5,
                 dropout_probability: float = 0.1
                 ):
        '''
        :param num_layers: The number of encoder/decoder layers.
        :param embedding_dimension: Size of the model. How long is an embedding for a word?
        :param key_dimension: key vector size.
        :param value_dimension: value vector size.
        :param num_heads: number of attention heads in the multi-headed attention layer.
        :param encoder_max_length: maximum length that an input sentence can be.
        :param encoder_max_length: maximum length that an output sentence can be.
        :param encoder_vocab_size: size of the vocabulary for encoders.
        :param decoder_vocab_size: size of the vocabulary for decoders.
        :param hidden_dimension: size of hidden layer in the feedforward layers.
        :param epsilon: a constant used to avoid division by near zero. See https://www.pinecone.io/learn/batch-layer-normalization/
            default epsilon chosen from https://pytorch.org/docs/stable/generated/torch.ao.nn.quantized.LayerNorm.html#layernorm
        :param dropout_probability: dropout probability. See https://towardsdatascience.com/dropout-in-neural-networks-47a162d621d9
        '''
        super().__init__()
        self.embedding_dimension = embedding_dimension
        self.encoder = Encoder(num_layers, encoder_vocab_size, embedding_dimension, key_dimension, value_dimension,
                               encoder_max_length, num_heads, hidden_dimension, epsilon, dropout_probability)

        self.decoder = Decoder(num_layers, decoder_vocab_size, embedding_dimension, key_dimension, value_dimension,
                               decoder_max_length, num_heads, hidden_dimension, epsilon, dropout_probability)
        self.final_layer = TransformerFinalLayer(embedding_dimension, decoder_vocab_size)

    def forward(self, src: Tensor, tgt: Tensor, src_mask: Tensor, tgt_mask: Tensor):
        output = self.encode_decode(src, tgt, src_mask, tgt_mask)
        output = self.final_layer(output)
        return output

    def encode_decode(self, src: Tensor, tgt: Tensor, src_mask: Tensor, tgt_mask: Tensor):
        encoder_output = self.encoder(src, src_mask)
        output = self.decoder(tgt, encoder_output, tgt_mask, src_mask)
        return output
