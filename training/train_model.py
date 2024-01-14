import torch
import torch.multiprocessing as mp
import os
from .basic_example.dummy import DummyOptimizer, DummyScheduler
from .batch import Batch
from model.transformer import make_transformer_model
from .data_loader import create_dataloaders
from .train import run_epoch, TrainingState
from training.util.utils import LabelSmoothing, rate, SimpleLossCompute
from training.util.tokenizer import build_vocabulary, load_tokenizers
from torch.optim.lr_scheduler import LambdaLR
from .worker import train_worker
from training.sample import make_sample_model


class InvalidModelNameException(Exception):
    def __init__(self,message="model name already taken!!"):
        super().__init__(message)


def train_distributed_model(
    vocab_src,
    vocab_tgt,
    spacy_de,
    spacy_en,
    config
):
    ngpus = torch.cuda.device_count()
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12356"
    print(f"Number of GPUs detected: {ngpus}")
    print("Spawning training processes ...")
    mp.spawn(
        train_worker,
        nprocs=ngpus,
        args=(ngpus, vocab_src, vocab_tgt, spacy_de, spacy_en, config, True),
    )


def train_model(config):
    if os.path.isfile(f"{config['file_name']}final.pt"):
        raise InvalidModelNameException()
    if config["distributed"]:
        spacy_german, spacy_english = load_tokenizers()
        vocab_src, vocab_tgt = build_vocabulary(spacy_german, spacy_english)
        train_distributed_model(
            vocab_src, vocab_tgt, spacy_german, spacy_english, config
        )
    else:
        print("training non distributed model...")
        train_single_model("cpu", config)


def train_single_model(gpu, config):
    spacy_german, spacy_english = load_tokenizers()
    vocab_src, vocab_tgt = build_vocabulary(spacy_german, spacy_english)
    pad_idx = vocab_tgt["<blank>"]
    model_dimension = config["model_dimension"]
    distributed = config["distributed"]
    print("src_vocab_size:", len(vocab_src))
    model = make_transformer_model(
        len(vocab_src),
        len(vocab_tgt),
    )
    model.name = config["file_name"]

    criterion = LabelSmoothing(
        size=len(vocab_tgt), padding_idx=pad_idx, smoothing=0.1
    )

    training_dataloader, validation_dataloader, _ = create_dataloaders(
        gpu,
        vocab_src,
        vocab_tgt,
        spacy_german,
        spacy_english,
        batch_size=config["batch_size"],
        max_padding=config["max_padding"],
        distributed=distributed
    )

    optimizer = torch.optim.Adam(
        model.parameters(), lr=config["base_learning_rate"], betas=(0.9, 0.98), eps=1e-9
    )
    lr_scheduler = LambdaLR(
        optimizer=optimizer,
        lr_lambda=lambda step: rate(
            step, model_dimension, factor=1, warmup=config["warmup"]
        ),
    )

    train_state = TrainingState()
    for epoch in range(config["num_epochs"]):
        if distributed:
            training_dataloader.sampler.set_epoch(epoch)
            validation_dataloader.sampler.set_epoch(epoch)

        model.train()
        print(f"[GPU {gpu}] Epoch {epoch} Training ====", flush=True)

        _, train_state = run_epoch(
            (Batch(b[0], b[1], pad_idx) for b in training_dataloader),
            model,
            SimpleLossCompute(model.final_layer, criterion),
            optimizer,
            lr_scheduler,
            mode="train+log",
            iterations_to_accumulate=config["iterations_to_accumulate"],
            training_state=train_state
        )

        file_path = "%s%.2d.pt" % (config["file_name"], epoch)
        torch.save(model.state_dict(), file_path)
        torch.cuda.empty_cache()

        print(f"[GPU {gpu}] Epoch {epoch} Validation ====", flush=True)
        model.eval()
        sloss = run_epoch(
            (Batch(b[0], b[1], pad_idx) for b in validation_dataloader),
            model,
            SimpleLossCompute(model.final_layer, criterion),
            DummyOptimizer(),
            DummyScheduler(),
            mode="eval",
        )
        print(sloss)
        torch.cuda.empty_cache()

    file_path = "%sfinal.pt" % config["file_name"]
    torch.save(model.state_dict(), file_path)




