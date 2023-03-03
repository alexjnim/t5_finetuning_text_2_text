import argparse

import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

from resources.training.test import test_model
from resources.data.build_test_dataloader import build_test_dataloader
from resources.data.load_raw_data import load_test_data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_saved_model(args) -> None:
    model = T5ForConditionalGeneration.from_pretrained(args.save_model_path)
    model.to(device)

    tokenizer = T5Tokenizer.from_pretrained(args.save_model_path)

    test_data = load_test_data()

    test_dataloader = build_test_dataloader(
        args=args,
        tokenizer=tokenizer,
        max_length=args.max_token_len,
        test_data=test_data,
    )

    test_model(
        args=args, model=model, tokenizer=tokenizer, test_dataloader=test_dataloader
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--save_model_path",
        "-smp",
        default="./model/30_epochs_13_02_2023_t5-small_model",
        action="store",
        help="name of huggingface bert model",
    )

    parser.add_argument(
        "--max_token_len",
        "-mtl",
        default=50,
        action="store",
        help="max token length of input text",
    )

    parser.add_argument(
        "--batch_size",
        "-tbs",
        default=10,
        action="store",
        help="batch size of training",
    )

    args = parser.parse_args()

    test_saved_model(
        args=args,
    )
