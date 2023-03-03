import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, SequentialSampler
from transformers import T5Tokenizer


class _dataset(Dataset):
    def __init__(
        self, tokenizer: T5Tokenizer, max_token_len: int, test_data: pd.DataFrame
    ):
        self.tokenizer = tokenizer
        self.max_token_len = max_token_len
        self.test_data = test_data

    def __len__(self) -> int:
        return len(self.test_data)

    def __getitem__(self, index: int) -> dict:
        original_text = str(self.test_data.iloc[index]["text"])

        encoded_ot_dict = self.tokenizer.batch_encode_plus(
            [original_text],
            max_length=self.max_token_len,
            add_special_tokens=True,
            truncation=True,
            pad_to_max_length=True,
            padding="max_length",
            return_tensors="pt",
        )

        original_input_ids = encoded_ot_dict["input_ids"].squeeze()
        original_attention_mask = encoded_ot_dict["attention_mask"].squeeze()

        target_text = self.test_data.iloc[index]['summary']
        return {
            "original_text": original_text,
            "original_input_ids": original_input_ids.to(dtype=torch.long),
            "original_attention_mask": original_attention_mask.to(dtype=torch.long),
            "target_text": target_text,
        }


def build_test_dataloader(
    args,
    tokenizer: T5Tokenizer,
    max_token_len: int,
    test_data: pd.DataFrame,
) -> DataLoader:
    dataset = _dataset(tokenizer, max_token_len, test_data)

    original_text = test_data["text"]
    sampler = SequentialSampler(original_text)

    return DataLoader(dataset, batch_size=args.batch_size, sampler=sampler)
