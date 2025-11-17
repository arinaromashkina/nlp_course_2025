import random

from typing import Dict, List

import torch

from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer


class TextDataset(Dataset):
    def __init__(
        self,
        texts: List[str],
        tokenizer,
        max_length: int = 512,
        stride: int = 256,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.stride = stride
        self.examples = []

        for text in texts:
            tokenized = tokenizer(
                text,
                truncation=False,
                return_attention_mask=False,
            )

            input_ids = tokenized["input_ids"]

            for i in range(0, len(input_ids), stride):
                chunk = input_ids[i : i + max_length]
                if len(chunk) >= 32:
                    self.examples.append(chunk)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return torch.tensor(self.examples[idx], dtype=torch.long)


def collate_fn(batch: List[torch.Tensor], pad_token_id: int = 0) -> Dict[str, torch.Tensor]:
    max_len = max(len(x) for x in batch)

    input_ids = []
    attention_mask = []

    for x in batch:
        padding_length = max_len - len(x)
        padded = torch.cat([x, torch.full((padding_length,), pad_token_id, dtype=torch.long)])
        mask = torch.cat(
            [torch.ones(len(x), dtype=torch.long), torch.zeros(padding_length, dtype=torch.long)]
        )

        input_ids.append(padded)
        attention_mask.append(mask)

    input_ids = torch.stack(input_ids)
    attention_mask = torch.stack(attention_mask)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": input_ids.clone(),
    }


def prepare_wikitext_data(
    tokenizer_name: str = "EleutherAI/pythia-1.4b",
    max_length: int = 512,
    dataset_name: str = "wikitext",
    dataset_config: str = "wikitext-2-raw-v1",
) -> tuple:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset(dataset_name, dataset_config)

    train_texts = [text for text in dataset["train"]["text"] if text.strip()]
    val_texts = [text for text in dataset["validation"]["text"] if text.strip()]

    train_dataset = TextDataset(train_texts, tokenizer, max_length=max_length)
    val_dataset = TextDataset(val_texts, tokenizer, max_length=max_length)

    return train_dataset, val_dataset, tokenizer


def prepare_custom_data(
    texts: List[str],
    tokenizer_name: str = "EleutherAI/pythia-1.4b",
    max_length: int = 512,
    train_split: float = 0.9,
) -> tuple:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    random.shuffle(texts)
    split_idx = int(len(texts) * train_split)
    train_texts = texts[:split_idx]
    val_texts = texts[split_idx:]

    train_dataset = TextDataset(train_texts, tokenizer, max_length=max_length)
    val_dataset = TextDataset(val_texts, tokenizer, max_length=max_length)

    return train_dataset, val_dataset, tokenizer


def create_dataloaders(
    train_dataset: Dataset,
    val_dataset: Dataset,
    batch_size: int = 8,
    num_workers: int = 4,
    pad_token_id: int = 0,
) -> tuple:
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=lambda x: collate_fn(x, pad_token_id),
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=lambda x: collate_fn(x, pad_token_id),
        pin_memory=True,
    )

    return train_loader, val_loader
