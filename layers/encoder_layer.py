from torch import nn
from torch import Tensor
from sublayers.layer_normalization import LayerNormalization
from sublayers.multi_attention import MultiHeadAttention
from sublayers.position_wise_feedforward import PositionWiseFeedForward


class EncoderLayer(nn.Module):
    def __init__(self,
                 embedding_dimension: int,
                 key_dimension: int,
                 value_dimension: int,
                 num_heads: int,
                 hidden_dimension: int,
                 epsilon: float = 1e-5,
                 # default epsilon chosen from https://pytorch.org/docs/stable/generated/torch.ao.nn.quantized.LayerNorm.html#layernorm
                 dropout_probability: float = 0.1):
        self.self_attention = MultiHeadAttention(embedding_dimension, key_dimension, value_dimension, num_heads)
        self.dropout = nn.Dropout(dropout_probability)

        self.layer_norm = LayerNormalization(embedding_dimension, epsilon)

        self.feedforward = PositionWiseFeedForward(embedding_dimension, hidden_dimension, dropout_probability)
        self.dropout2 = nn.Dropout(dropout_probability)
        self.layer_norm2 = LayerNormalization(embedding_dimension, epsilon)

    def forward(self, x: Tensor, input_mask: Tensor):
        '''
        :param x: Tensor[batch_size, head_number, length, tensor_dimension]
        :param input_mask: mask for x
        :return:
        '''
        # multi-attention, add + norm
        sublayer_output = self.dropout(self.self_attention(query=x, key=x, value=x, mask=input_mask))
        y = self.layer_norm(x + sublayer_output)

        # feedforward, add + norm
        sublayer_output = self.dropout2(self.feedforward(y))
        output = self.layer_norm2(y + sublayer_output)
        return output
