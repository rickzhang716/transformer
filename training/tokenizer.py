import spacy
import os

import torchtext.datasets as datasets
from torchtext.vocab import build_vocab_from_iterator
from copy import deepcopy as c


# spacy tokenizer model. Includes:
# tok2vec - token to vectors
# tagger - assign tags to words, like verb, noun, adverb, etc.
# morphologizer - predict morphological features and POS tags **only for german pipeline**
# dependency parser (parser) - assign sections based on grammatical dependencies.
#   ie. "I prefer the morning flight through Denver." -> "Denver" depends on "flight"
# lemmatizer - converts words into their base form: mice -> mouse, ran -> run
# sentence recognizer (senter) - identifying and segmenting sentences
# attribute ruler - tokens based on patterns of words
# named entity recognition (ner) - labelling proper nouns


def load_german_tokenizer():
    try:
        spacy_german = spacy.load("de_core_news_sm")
    except IOError:
        os.system("python -m spacy download de_core_news_sm")
        spacy_german = spacy.load("de_core_news_sm")
    return spacy_german


def load_english_tokenizer():
    try:
        spacy_english = spacy.load("en_core_web_sm")
    except IOError:
        os.system("python -m spacy download de_core_news_sm")
        spacy_english = spacy.load("en_core_web_sm")

    return spacy_english


def tokenize(text, tokenizer):
    return [token.text for token in tokenizer.tokenizer(text)]


def yield_tokens(data_iterable, tokenizer, index):
    for from_to_tuple in data_iterable:
        yield tokenizer(from_to_tuple[index])


def build_vocabulary(tokenizer_german,tokenizer_english):
    def tokenize_german(text):
        return tokenize(text, tokenizer_german)

    def tokenize_english(text):
        return tokenize(text, tokenizer_english)

    training_set, validation_set, testing_set = datasets.Multi30k(language_pair=("de","en"))
    vocab_src = build_vocab_from_iterator(
        yield_tokens(c(training_set + validation_set + testing_set), tokenize_german, index=0),
        min_freq=2,
        specials=["<s>", "</s>", "<blank>", "<unk>"]  # sentence start, end, blank, unknown
    )

    vocab_tgt = build_vocab_from_iterator(
        yield_tokens(c(training_set + validation_set + testing_set), tokenize_english, index=0),
        min_freq=2,
        specials=["<s>", "</s>", "<blank>", "<unk>"]  # sentence start, end, blank, unknown
    )

    vocab_src.set_default_index(vocab_src["<unk>"])
    vocab_tgt.set_default_index(vocab_src["<unk>"])


    return vocab_src, vocab_tgt


def load_tokenizers():
    spacy_german = load_german_tokenizer()
    spacy_english = load_english_tokenizer()
    return spacy_german, spacy_english
