import argparse

import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

from resources.inference.model import T5SUM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(args) -> None:
    model = T5SUM(
        save_model_path=args.save_model_path, max_token_len=args.max_token_len
    )

    args.grapheme = input("Enter your grapheme to process: ")
    phoneme = model.generate_phonemes(args.grapheme)

    print(f"Grapheme input:\t{args.grapheme}")
    print(f"Phoneme output:\t{phoneme}")


def generate_phonemes(
    args, model: T5ForConditionalGeneration, tokenizer: T5Tokenizer
) -> str:
    model.to(device)
    task_prefix = "Summarise: "
    model_input = task_prefix + args.grapheme

    encoding = tokenizer(
        [model_input],
        padding="max_length",
        pad_to_max_length=True,
        add_special_tokens=True,
        max_length=args.max_token_len,
        truncation=True,
        return_tensors="pt",
    )

    input_ids = encoding["input_ids"]
    attention_mask = encoding["attention_mask"]

    generated_ids = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_length=args.max_token_len,
        num_beams=2,
        repetition_penalty=2.5,
        length_penalty=1.0,
        early_stopping=True,
    )
    prediction = tokenizer.decode(
        generated_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=True
    )
    return prediction


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

    args = parser.parse_args()

    main(
        args=args,
    )
