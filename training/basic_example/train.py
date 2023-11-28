import torch
from torch import nn
from torch.optim.lr_scheduler import LambdaLR
from training.util.utils import LabelSmoothing, rate, SimpleLossCompute, greedy_decode
from training.train import run_epoch
from .data import data_gen
from .dummy import DummyOptimizer, DummyScheduler
from model.transformer import Transformer


def example_simple_model():
    vocab_size = 11
    criterion = LabelSmoothing(size=vocab_size, padding_idx=0, smoothing=0.0)
    model = Transformer(vocab_size, vocab_size, num_layers=2)
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.5, betas=(0.9, 0.98), eps=1e-9
    )
    print(model.embedding_dimension)
    lr_scheduler = LambdaLR(
        optimizer=optimizer,
        lr_lambda=lambda step: rate(
            step, model_size=model.embedding_dimension, factor=1.0, warmup=400
        ),
    )
    batch_size = 80
    for epoch in range(20):
        model.train()
        run_epoch(
            data_gen(vocab_size, batch_size, 20),
            model,
            SimpleLossCompute(model.final_layer, criterion),
            optimizer,
            lr_scheduler,
            mode="train",
        )
        model.eval()
        run_epoch(
            data_gen(vocab_size, batch_size, 5),
            model,
            SimpleLossCompute(model.final_layer, criterion),
            DummyOptimizer(),
            DummyScheduler(),
            mode="eval",
        )[0]

    model.eval()
    src = torch.LongTensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]])
    max_len = src.shape[1]
    src_mask = torch.ones(1, 1, max_len)
    guess = greedy_decode(model, src, src_mask, max_len=max_len, start_symbol=0)
    print(guess.dtype)
    print(guess)



