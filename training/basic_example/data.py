
import torch
from training.batch import Batch


def data_gen(vocab_size: int, batch_size, num_batches):
    #  Generate random data for a src-tgt copy task.
    for i in range(num_batches):
        data = torch.randint(1, vocab_size, size=(batch_size, 10))
        data[:, 0] = 1
        src = data.requires_grad_(False).clone().detach()
        tgt = data.requires_grad_(False).clone().detach()
        yield Batch(src, tgt, 0)
