import argparse

from sklearn.model_selection import train_test_split
from torch import cuda
from transformers import T5Tokenizer, T5ForConditionalGeneration

from resources.data.load_raw_data import (
    load_training_data,
    load_validation_data,
    process_data,
)
from resources.data.build_dataloader import build_dataloader
from resources.training.train import train_model

device = "cuda" if cuda.is_available() else "cpu"


def train_t5(args) -> None:
    process_data()
    training_data = load_training_data()
    validation_data = load_validation_data()

    model = T5ForConditionalGeneration.from_pretrained(args.model_name)
    model = model.to(device)
    tokenizer = T5Tokenizer.from_pretrained(args.model_name)

    train_dataloader = build_dataloader(
        args=args,
        tokenizer=tokenizer,
        max_token_len=args.max_token_len,
        original_text=training_data["text"],
        target_text=training_data["summary"],
        trainset=True,
    )

    val_dataloader = build_dataloader(
        args=args,
        tokenizer=tokenizer,
        max_token_len=args.max_token_len,
        original_text=validation_data["text"],
        target_text=validation_data["summary"],
        trainset=False,
    )

    train_model(
        args=args,
        model=model,
        tokenizer=tokenizer,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_name",
        "-m",
        default="t5-small",
        action="store",
        help="name of huggingface bert model",
    )

    parser.add_argument(
        "--max_token_len",
        "-mtl",
        default=500,
        action="store",
        help="max token length of input text",
    )

    parser.add_argument(
        "--batch_size",
        "-tbs",
        default=128,
        action="store",
        help="batch size of training",
    )

    parser.add_argument(
        "--learning_rate",
        "-lr",
        default=1e-4,
        action="store",
        help="learning rate during training",
    )

    parser.add_argument(
        "--epochs",
        "-e",
        default=30,
        action="store",
        help="number of epochs to run training",
    )

    parser.add_argument(
        "--patience",
        "-p",
        default=3,
        action="store",
        help="number of epochs to improve before early stopping",
    )

    parser.add_argument(
        "--savedir",
        "-sd",
        default="./model/",
        action="store",
        help="path to save model",
    )

    args = parser.parse_args()

    train_t5(
        args=args,
    )
