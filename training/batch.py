import torch
from torch import Tensor


class Batch:
    """A Batch of data for training"""
    def __init__(self, src: Tensor, tgt: Tensor, padding_index):
        self.src = src
        self.src_mask = (src != padding_index).unsqueeze(-2)
        if tgt is not None:
            self.tgt: Tensor = tgt[:, :-1]
            self.tgt_y: Tensor = tgt[:, 1:]
            self.tgt_mask: Tensor = self.make_decoder_input_mask(self.tgt, padding_index)
            self.num_tokens: int = (self.tgt_y != padding_index).data.sum()

    @staticmethod
    def make_decoder_input_mask(tgt: Tensor, padding_index: int):
        """
        :param tgt: Tensor[batch_size, seq_length]
        :param padding_index: value that we assign to padding.

        Create a mask to hide padding and future words.
        """
        tgt_mask = (tgt != padding_index).unsqueeze(-2)

        sequence_length: int = tgt.size(-1)
        no_peek_mask = torch.tril(torch.ones(sequence_length, sequence_length)).type_as(tgt_mask.data)
        mask = tgt_mask & no_peek_mask
        return mask

