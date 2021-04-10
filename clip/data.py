from random import choice
import os
from tqdm import tqdm
from PIL import Image
from pathlib import Path
from typing import Union, List
import pandas as pd
import numpy as np
import torch
import random
import h5py
from torchvision import transforms as T
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
from torchvision.utils import make_grid, save_image
from torchvision.transforms import Compose, Resize, CenterCrop, \
    ToTensor, Normalize, RandomCrop

from clip.simple_tokenizer import SimpleCharTokenizer


clip_dir = os.path.dirname(os.path.realpath(__file__))
model_dir = os.path.join(clip_dir, '..', 'model')


def my_transform(n_px):
    return Compose([
        Resize(n_px, interpolation=Image.BICUBIC),
        CenterCrop(n_px),
        lambda image: image.convert("RGB"),
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])


def training_transform(n_px):
    return Compose([
        ToTensor(),
        RandomCrop(n_px),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])


def infer_transform(n_px):
    return Compose([
        ToTensor(),
        CenterCrop(n_px),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

def tokenize(tokenizer, texts: Union[str, List[str]], context_length: int = 77) -> torch.LongTensor:
    """
    Returns the tokenized representation of given input string(s)

    Parameters
    ----------
    texts : Union[str, List[str]]
        An input string or a list of input strings to tokenize

    context_length : int
        The context length to use; all CLIP models use 77 as the context length

    Returns
    -------
    A two-dimensional tensor containing the resulting tokens, shape = [number of input strings, context_length]
    """
    if isinstance(texts, str):
        texts = [texts]

    sot_token = tokenizer.encoder["<|startoftext|>"]
    eot_token = tokenizer.encoder["<|endoftext|>"]
    all_tokens = [[sot_token] + tokenizer.encode(text) + [eot_token] for text in texts]

    return tokens_to_tensor(all_tokens, context_length)


def tokens_to_tensor(all_tokens, context_length):
    result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)

    for i, tokens in enumerate(all_tokens):
        if len(tokens) > context_length:
            tokens = tokens[:context_length]
            # raise RuntimeError(f"Input {texts[i]} is too long for context length {context_length}")
        result[i, :len(tokens)] = torch.tensor(tokens)
    return result

