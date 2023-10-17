import torch
from torch import nn
from torch import Tensor
from model.encoder import Encoder
from model.decoder import Decoder
from typing import Optional


# TODO: figure out masking
class Transformer(nn.Module):
    def __init__(self,
                 num_layers: int,
                 embedding_dimension: int,
                 key_dimension: int,
                 value_dimension: int,
                 num_heads: int,

                 encoder_max_length: int,
                 decoder_max_length: int,
                 encoder_vocab_size: int,
                 decoder_vocab_size: int,

                 hidden_dimension: int,
                 epsilon: float = 1e-5,
                 dropout_probability: float = 0.1,
                 input_mask: Optional[Tensor] = None):
        '''
        :param num_layers: The number of encoder/decoder layers.
        :param embedding_dimension: Size of the model. How long is an embedding for a word?
        :param key_dimension: key vector size
        :param value_dimension: value vector size
        :param num_heads: number of attention heads in the multi-headed attention layer
        :param encoder_max_length: maximum length that an input sentence can be
        :param encoder_max_length: maximum length that an output sentence can be
        :param encoder_vocab_size: size of the vocabulary for encoders
        :param decoder_vocab_size: size of the vocabulary for decoders
        :param hidden_dimension: size of hidden layer in the feedforward layers
        :param epsilon: a constant used to avoid division by near zero. See                  # default epsilon chosen from https://pytorch.org/docs/stable/generated/torch.ao.nn.quantized.LayerNorm.html#layernorm
        :param dropout_probability: dropout probability. See https://towardsdatascience.com/dropout-in-neural-networks-47a162d621d9
        '''
        super().__init__()
        self.encoder = Encoder(num_layers, encoder_vocab_size, embedding_dimension, key_dimension, value_dimension,
                               encoder_max_length, num_heads, hidden_dimension, epsilon, dropout_probability)
        self.decoder = Decoder(num_layers, decoder_vocab_size, embedding_dimension, key_dimension, value_dimension,
                               decoder_max_length, num_heads, hidden_dimension, epsilon, dropout_probability)
        self.input_mask = input_mask

    def forward(self, x: Tensor, y: Tensor):
        encoder_output = self.encoder(x, self.input_mask)
        output = self.decoder(x, encoder_output, self.input_mask)
        return output
