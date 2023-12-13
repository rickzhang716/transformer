from torch import nn
from torch import Tensor
from model.sublayers.layer_normalization import LayerNormalization
from model.sublayers.multi_attention import MultiHeadAttention
from model.sublayers.position_wise_feedforward import PositionWiseFeedForward
from typing import Optional


class DecoderLayer(nn.Module):
    def __init__(self,
                 embedding_dimension: int,
                 key_dimension: int,
                 value_dimension: int,
                 num_heads: int,
                 hidden_dimension: int,
                 # default epsilon chosen from https://pytorch.org/docs/stable/generated/torch.ao.nn.quantized.LayerNorm.html#layernorm
                 epsilon: float = 1e-5,
                 dropout_probability: float = 0.1):
        super().__init__()
        self.self_attention = MultiHeadAttention(embedding_dimension, key_dimension, value_dimension, num_heads)
        self.dropout = nn.Dropout(dropout_probability)
        self.layer_norm = LayerNormalization(embedding_dimension, epsilon)

        self.encoder_decoder_attention = MultiHeadAttention(embedding_dimension, key_dimension, value_dimension,
                                                            num_heads)
        self.dropout2 = nn.Dropout(dropout_probability)
        self.layer_norm2 = LayerNormalization(embedding_dimension, epsilon)

        self.feedforward = PositionWiseFeedForward(embedding_dimension, hidden_dimension, dropout_probability)
        self.dropout3 = nn.Dropout(dropout_probability)
        self.layer_norm3 = LayerNormalization(embedding_dimension, epsilon)

    def forward(self, tgt: Tensor, encoder_output: Tensor, input_mask: Tensor,
                encoder_decoder_mask: Optional[Tensor] = None):
        '''
        :param tgt: Tensor[batch_size, head_number, length, tensor_dimension]
        :param encoder_output: Tensor[batch_size, head_number, length, tensor_dimension]
        :param encoder_decoder_mask: mask for encoder_output
        :param input_mask: mask for x
        :return:
        '''

        # self_attention, add + norm
        sublayer_output = self.dropout(self.self_attention(query=tgt,
                                                           key=tgt,
                                                           value=tgt,
                                                           mask=input_mask
                                                           ))
        y = self.layer_norm(tgt + sublayer_output)

        # encoder-decoder-attention, add + norm
        sublayer_output = self.dropout2(self.encoder_decoder_attention(query=y,
                                                                       key=encoder_output,
                                                                       value=encoder_output,
                                                                       mask=encoder_decoder_mask
                                                                       ))
        y2 = self.layer_norm(y + sublayer_output)

        # feedforward, add + norm
        sublayer_output = self.dropout3(self.feedforward(y2))
        output = self.layer_norm(y2 + sublayer_output)
        return output
