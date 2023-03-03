import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class T5SUM:
    def __init__(self, save_model_path: str, max_token_len: int) -> None:
        self.model = T5ForConditionalGeneration.from_pretrained(save_model_path)
        self.model.to(device)
        self.tokenizer = T5Tokenizer.from_pretrained(save_model_path)
        self.max_token_len = max_token_len

    def generate_phonemes(self, grapheme: str) -> str:
        task_prefix = "Summarise: "

        model_input = task_prefix + grapheme

        encoding = self.tokenizer(
            [model_input],
            padding="max_length",
            pad_to_max_length=True,
            add_special_tokens=True,
            max_length=self.max_token_len,
            truncation=True,
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"]
        attention_mask = encoding["attention_mask"]

        generated_ids = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=self.max_token_len,
            num_beams=2,
            repetition_penalty=2.5,
            length_penalty=1.0,
            early_stopping=True,
        )
        prediction = self.tokenizer.decode(
            generated_ids[0],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        return prediction
