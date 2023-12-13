import GPUtil
import torch
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from .basic_example.dummy import DummyScheduler, DummyOptimizer
from training.util.utils import LabelSmoothing, rate, SimpleLossCompute
from .data_loader import create_dataloaders
from .batch import Batch
from .train import run_epoch, TrainingState
from torch.optim.lr_scheduler import LambdaLR

from model.transformer import Transformer


def train_worker(
    number_gpus_per_node,
    vocab_src,
    vocab_tgt,
    spacy_german,
    spacy_english,
    config,
    gpu="cpu",
    distributed=False
):
    print(f'training worker...')
    pad_idx = vocab_tgt["<blank>"]
    model_dimension = config["model_dimension"]
    model = Transformer(
        len(vocab_src),
        len(vocab_tgt),
        num_layers=6
    )

    module = model
    if distributed:
        dist.init_process_group(
            "nccl", init_method="env://", rank=gpu, world_size=number_gpus_per_node
        )
        model = DDP(model, device_ids=[gpu])
        module = model.module
        is_main_process = (gpu == 0)

    criterion = LabelSmoothing(
        size=len(vocab_tgt), padding_idx=pad_idx, smoothing=0.1
    )
    criterion.cuda(gpu)

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
        model.parameters(), lr=config["base_learning_rate"], betas=(0.9,0.98), eps=1e-9
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
        print(f"[GPU{gpu}] Epoch {epoch} Training ====", flush=True)

        _, train_state = run_epoch(
            (Batch(b[0],b[1],pad_idx) for b in training_dataloader),
            model,
            SimpleLossCompute(module.final_layer, criterion),
            optimizer,
            lr_scheduler,
            mode="train+log",
            iterations_to_accumulate=config["iterations_to_accumulate"],
            training_state=train_state
        )

        GPUtil.showUtilization()
        if is_main_process:
            file_path = "%s%.2d.pt" % (config["file_prefix"], epoch)
            torch.save(module.state_dict(), file_path)
        torch.cuda.empty_cache()

        print(f"[GPU{gpu}] Epoch {epoch} Validation ====", flush=True)
        model.eval()
        sloss = run_epoch(
            (Batch(b[0], b[1], pad_idx) for b in validation_dataloader),
            model,
            SimpleLossCompute(module.generator, criterion),
            DummyOptimizer(),
            DummyScheduler(),
            mode="eval",
        )
        print(sloss)
        torch.cuda.empty_cache()

    if is_main_process:
        file_path = "%sfinal.pt" % config["file_prefix"]
        torch.save(module.state_dict(), file_path)