import torch
from model.transformer import Transformer
from training.utils import no_peek_mask


def inference_test():
    test_model = Transformer(11, 11, num_layers=2)
    test_model.eval()
    src = torch.LongTensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    src_mask = torch.ones(1, 1, 10)

    memory = test_model.encoder(src, src_mask)
    ys = torch.zeros(1, 1).type_as(src)

    for i in range(9):
        out = test_model.decoder(
            ys, memory,  no_peek_mask(ys.size(1)).type_as(src.data), src_mask
        )
        prob = test_model.final_layer(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data[0]
        ys = torch.cat(
            [ys, torch.empty(1, 1).type_as(src.data).fill_(next_word)], dim=1
        )

    print("Example Untrained Model Prediction - Transformer:", ys)

