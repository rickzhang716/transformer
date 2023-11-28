from training.util.tokenizer import tokenize
from .iterators import collate_fn as collate
from training.util.utils import dataset_to_torch_dataset
from torch.utils.data import DataLoader
from datasets import load_dataset


def create_dataloaders(
    device,
    vocab_src,
    vocab_tgt,
    spacy_german,
    spacy_english,
    batch_size=12000,
    max_padding=128,
    distributed=False
):
    def tokenize_german(text):
        return tokenize(text, spacy_german)

    def tokenize_english(text):
        return tokenize(text, spacy_english)

    def collate_fn(batch):
        return collate(
            batch,
            tokenize_german,
            tokenize_english,
            vocab_src,
            vocab_tgt,
            device,
            max_padding=max_padding,
            pad_id=vocab_src.get_stoi()["<blank>"]
        )

    dataset = load_dataset("bentrevett/multi30k")

    training_iterable = dataset['train']
    validation_iterable = dataset['validation']
    testing_iterable = dataset['test']

    # training_iterable_map = to_map_style_dataset(training_iterable)
    # training_sampler = DistributedSampler(training_iterable_map) if distributed else None
    #
    # validation_iterable_map = to_map_style_dataset(validation_iterable)
    # validation_sampler = DistributedSampler(validation_iterable_map) if distributed else None
    training_dataloader = DataLoader(
        training_iterable,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )

    validation_dataloader = DataLoader(
        dataset_to_torch_dataset(validation_iterable),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )

    testing_dataloader = DataLoader(
        dataset_to_torch_dataset(testing_iterable),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )

    return training_dataloader, validation_dataloader, testing_dataloader
