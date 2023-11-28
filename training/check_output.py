import torch
from .batch import Batch
from training.util.tokenizer import build_vocabulary, load_tokenizers
from training.util.utils import greedy_decode
from .data_loader import create_dataloaders
from torchtext.data.metrics import bleu_score
from training.util.load_trained_model import load_trained_model
from .config import config


def run_example(num_examples: int = 5) -> None:
    spacy_german, spacy_english = load_tokenizers()
    vocab_src, vocab_tgt = build_vocabulary(spacy_german, spacy_english)
    print("preprocessing data...")

    _, _, testing_dataloader = create_dataloaders(
        torch.device("cpu"),
        vocab_src,
        vocab_tgt,
        spacy_german,
        spacy_english,
        batch_size=2,
        distributed=False
    )

    model = load_trained_model(
        config,
        vocab_src,
        vocab_tgt
    )
    get_bleu(
        testing_dataloader,
        model,
        vocab_src,
        vocab_tgt,
        num_examples,
    )


def get_bleu(
    dataloader,
    model,
    vocab_src,
    vocab_tgt,
    num_examples=5,
    pad_idx=2,
    eos_string="</s>",
):
    results = []
    for idx in range(num_examples):
        print("\nExample %d ========\n" % idx)
        b = next(iter(dataloader))
        batch = Batch(b[0], b[1], pad_idx)
        model.eval()
        outputs = []

        src_tokens = []
        for src_sentence in batch.src:
            sentence_tokens = [
                vocab_src.get_itos()[x] for x in src_sentence if x != pad_idx
            ]
            src_tokens.append(sentence_tokens)

        tgt_tokens = []
        for tgt_sentence in batch.tgt:
            sentence_tokens = [
                vocab_tgt.get_itos()[x] for x in tgt_sentence if x != pad_idx
            ]
            tgt_tokens.append([sentence_tokens])

        for sentence in src_tokens:
            print(
                "Source Text (Input)        : "
                + " ".join(sentence).replace("\n", "")
            )
        for sentence in tgt_tokens:
            print(
                "Target Text (Ground Truth) : "
                + " ".join(sentence[0]).replace("\n", "")
            )
        with torch.no_grad():
            output = model(batch.src, batch.tgt, batch.src_mask, batch.tgt_mask)
            print(output)
            for sentence in output:
                predicted_sentence = sentence.max(dim=1)[1]
                sentence_txt = [vocab_tgt.get_itos()[x] for x in predicted_sentence if x != pad_idx]
                sentence_end = sentence_txt.index(eos_string)
                sentence_trimmed = sentence_txt[1:sentence_end]
                outputs.append(sentence_trimmed)
                results.append(" ".join(sentence_trimmed))

        # print(outputs)

        # model_txt = (
        #     " ".join(
        #         [vocab_tgt.get_itos()[x] for x in model_out if x != pad_idx]
        #     ).split(eos_string, 1)[0]
        #     + eos_string
        # )
        # output_tokens = model_txt.split(" ")
        # print("Model Output               : " + model_txt.replace("\n", ""))
        print(tgt_tokens)
        print(outputs)
        print(results)
        print("BLEU score:", bleu_score(outputs, tgt_tokens))

    return results

