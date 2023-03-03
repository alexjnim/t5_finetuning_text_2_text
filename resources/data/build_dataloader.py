import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from transformers import T5Tokenizer


class _dataset(Dataset):
    def __init__(
        self,
        tokenizer: T5Tokenizer,
        max_token_len: int,
        original_text: pd.Series,
        target_text: pd.Series,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_token_len
        self.original_text = original_text.tolist()
        self.target_text = target_text.tolist()

    def __len__(self) -> int:
        return len(self.original_text)

    def __getitem__(self, index: int) -> dict:
        original_text = str(self.original_text[index])

        encoded_ot_dict = self.tokenizer.batch_encode_plus(
            [original_text],
            max_length=self.max_length,
            add_special_tokens=True,
            truncation=True,
            pad_to_max_length=True,
            padding="max_length",
            return_tensors="pt",
        )

        original_input_ids = encoded_ot_dict["input_ids"].squeeze()
        original_attention_mask = encoded_ot_dict["attention_mask"].squeeze()

        target_text = str(self.target_text[index])
        target_text = " ".join(target_text.split())
        encoded_tt_dict = self.tokenizer.batch_encode_plus(
            [target_text],
            max_length=self.max_length,
            add_special_tokens=True,
            truncation=True,
            pad_to_max_length=True,
            padding="max_length",
            return_tensors="pt",
        )

        target_input_ids = encoded_tt_dict["input_ids"].squeeze()
        target_attention_mask = encoded_tt_dict["attention_mask"].squeeze()

        return {
            "original_text": original_text,
            "original_input_ids": original_input_ids.to(dtype=torch.long),
            "original_attention_mask": original_attention_mask.to(dtype=torch.long),
            "target_text": target_text,
            "target_input_ids": target_input_ids.to(dtype=torch.long),
            "target_attention_mask": target_attention_mask.to(dtype=torch.long),
        }


def build_dataloader(
    args,
    tokenizer: T5Tokenizer,
    max_token_len: int,
    original_text: pd.Series,
    target_text: pd.Series,
    trainset: bool = False,
) -> DataLoader:
    dataset = _dataset(tokenizer, max_token_len, original_text, target_text)

    if trainset:
        sampler = RandomSampler(original_text)
    else:
        sampler = SequentialSampler(original_text)

    return DataLoader(dataset, batch_size=args.batch_size, sampler=sampler)
