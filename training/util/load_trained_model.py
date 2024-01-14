import torch
from model.transformer import make_transformer_model
from training.sample import make_sample_model


def load_model(model_name, vocab_src, vocab_tgt):
    if "sample" in model_name:
        model = make_sample_model(len(vocab_src), len(vocab_tgt), num_layers=6)
    else:
        model = make_transformer_model(len(vocab_src), len(vocab_tgt), num_layers=6)
    model.load_state_dict(torch.load(f'{model_name}final.pt'))
    return model

