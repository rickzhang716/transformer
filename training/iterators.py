import torch
from torch.nn.functional import pad

def collate_fn(
        batch,
        src_pipeline,
        tgt_pipeline,
        src_vocab,
        tgt_vocab,
        device,
        max_padding=128,  # this is max length of sentence
        pad_id=2
):
    sentence_begin_id = torch.Tensor([0], device=device).int()
    sentence_end_id = torch.Tensor([1], device=device).int()
    src_list, tgt_list = [], []
    for src_tgt_pair in batch:
        src = src_tgt_pair['de']
        tgt = src_tgt_pair['en']
        processed_src = torch.cat(
            [
                sentence_begin_id,
                torch.tensor(
                    src_vocab(src_pipeline(src)),
                    dtype=torch.int64,
                    device=device
                ),
                sentence_end_id,
            ],
            0,
        )

        processed_tgt = torch.cat(
            [
                sentence_begin_id,
                torch.tensor(
                    tgt_vocab(tgt_pipeline(tgt)),
                    dtype=torch.int64,
                    device=device
                ),
                sentence_end_id
            ],
            0,
        )
        src_list.append(
            pad(
                processed_src,
                # if max_padding - len is negative, then we will end up overwriting values.
                (0, max_padding - len(processed_src)),
                value=pad_id
            )
        )
        tgt_list.append(
            pad(
                processed_tgt,
                (0, max_padding - len(processed_tgt)),
                value=pad_id
            )
        )
    src = torch.stack(src_list).int()
    tgt = torch.stack(tgt_list).int()
    return src, tgt
