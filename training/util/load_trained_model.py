import torch
from model.transformer import Transformer


def load_trained_model(config, vocab_src, vocab_tgt):
    model = Transformer(len(vocab_src), len(vocab_tgt), num_layers=6)
    model.load_state_dict(torch.load(f'{config["file_name"]}final.pt'))
    return model

