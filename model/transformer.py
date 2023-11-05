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

                 decoder_mask_index: int,
                 epsilon: float = 1e-5,
                 dropout_probability: float = 0.1,
                 encoder_input_mask: Optional[Tensor] = None,
                 encoder_decoder_mask: Optional[Tensor] = None):
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
        :param decoder_mask_index: index to be used to pad decoder input mask.
        :param epsilon: a constant used to avoid division by near zero. See https://www.pinecone.io/learn/batch-layer-normalization/
            default epsilon chosen from https://pytorch.org/docs/stable/generated/torch.ao.nn.quantized.LayerNorm.html#layernorm
        :param dropout_probability: dropout probability. See https://towardsdatascience.com/dropout-in-neural-networks-47a162d621d9
        :param encoder_input_mask: mask for encoder self-attention.
        :param encoder_decoder_mask: mask for encoder-decoder attention.
        '''
        super().__init__()
        self.encoder = Encoder(num_layers, encoder_vocab_size, embedding_dimension, key_dimension, value_dimension,
                               encoder_max_length, num_heads, hidden_dimension, epsilon, dropout_probability)
        self.decoder = Decoder(num_layers, decoder_vocab_size, embedding_dimension, key_dimension, value_dimension,
                               decoder_max_length, num_heads, hidden_dimension, epsilon, dropout_probability)
        self.decoder_input_mask_pad_idx = decoder_mask_index
        self.encoder_input_mask = encoder_input_mask
        self.encoder_decoder_mask = encoder_decoder_mask
        self.final_linear = nn.Linear(embedding_dimension, decoder_vocab_size)
        self.final_softmax = nn.Softmax(dim=2)

    def forward(self, x: Tensor, y: Tensor):
        encoder_output = self.encoder(x, self.input_mask)
        output = self.decoder(y, encoder_output, self.decoder_input_mask, self.encoder_decoder_mask)
        output = self.final_linear(output)
        output = self.final_softmax(output)
        return output

    def decoder_input_mask(self, y: Tensor):
        '''

        :param y: decoder input Tensor[batch_size, seq_length, embedding_dimension]
        :return: decoder_input_mask
        '''
        pad_mask = (y != self.decoder_input_mask_pad_idx).unsqueeze(1).unsqueeze(3)
        seq_length = y.shape[1]
        sub_mask = torch.tril(torch.ones(seq_length, seq_length))
        mask = pad_mask & sub_mask
        return mask
