import torch
from .batch import Batch
from training.util.tokenizer import build_vocabulary, load_tokenizers
from training.util.utils import greedy_decode
from .data_loader import create_dataloaders
from torchtext.data.metrics import bleu_score
from training.util.load_trained_model import load_model
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
        batch_size=1,
        distributed=False
    )

    model = load_model(
        config["file_name"],
        vocab_src,
        vocab_tgt
    )
    results, bleu_scores = get_bleu(
        testing_dataloader,
        model,
        vocab_src,
        vocab_tgt,
        num_examples,
    )

    print(f"total bleu score for model {config['file_name']}: {sum(bleu_scores)/len(bleu_scores)}")
    print(results)

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
    bleu_scores = []
    for idx in range(num_examples):
            print("\nExample %d ========\n" % idx)
            b = next(iter(dataloader))
            batch = Batch(b[0], b[1], pad_idx)
            model.eval()
            outputs = []


            src_sentence = [
                    vocab_src.get_itos()[x] for x in batch.src[0] if x != pad_idx
                ]
            src_trimmed = src_sentence[1:-1]
            tgt_sentence = [
                vocab_tgt.get_itos()[x] for x in batch.tgt[0] if x != pad_idx
            ]
            tgt_trimmed = tgt_sentence[1:-1]

            print(
                "Source Text (Input)        : "
                + " ".join(src_trimmed).replace("\n", "")
            )

            print(
                "Target Text (Ground Truth) : "
                + " ".join(tgt_trimmed).replace("\n", "")
            )
            with torch.no_grad():
                output = greedy_decode(model, batch.src, batch.src_mask, 72, 0)[0]
                # output = model(batch.src, batch.tgt, batch.src_mask, batch.tgt_mask)
                # print(output)
                sentence_txt = [vocab_tgt.get_itos()[x] for x in output if x != pad_idx]
                try:

                    sentence_end = sentence_txt.index(eos_string)
                    sentence_trimmed = sentence_txt[1:sentence_end]
                    # print(sentence_trimmed)
                    outputs.append(sentence_trimmed)
                    results.append(" ".join(sentence_trimmed))
                except ValueError:
                    print(sentence_txt)
                    continue
            # print(outputs)

            # model_txt = (
            #     " ".join(
            #         [vocab_tgt.get_itos()[x] for x in model_out if x != pad_idx]
            #     ).split(eos_string, 1)[0]
            #     + eos_string
            # )
            # output_tokens = model_txt.split(" ")
            # print("Model Output               : " + model_txt.replace("\n", ""))
            print(tgt_trimmed)
            print(outputs)
            # print(results)
            bleu = bleu_score(outputs, [[tgt_trimmed]])
            print("BLEU score:", bleu)
            bleu_scores.append(bleu)


    return results, bleu_scores

