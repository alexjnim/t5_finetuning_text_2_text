import argparse

import torch
from resources.inference.model import T5SUM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(args) -> None:
    model = T5SUM(
        save_model_path=args.save_model_path, max_token_len=args.max_token_len
    )

    args.text = input("Enter your chosen text to summarise: ")
    summary = model.generate_summary(args.text)

    print(f"Text input:\t{args.text}")
    print(f"Summary output:\t{summary}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--save_model_path",
        "-smp",
        default="./model/30_epochs_03_03_2023_t5-small_model",
        action="store",
        help="name of huggingface bert model",
    )

    parser.add_argument(
        "--max_token_len",
        "-mtl",
        default=100,
        action="store",
        help="max token length of input text",
    )

    args = parser.parse_args()

    main(
        args=args,
    )
