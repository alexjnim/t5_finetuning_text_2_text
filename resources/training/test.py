from rouge_score import rouge_scorer
import torch
from torch.utils.data import DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration


def test_model(
    args,
    model: T5ForConditionalGeneration,
    tokenizer: T5Tokenizer,
    test_dataloader: DataLoader,
) -> None:
    model.eval()

    # Tracking variables
    all_rouge1_scores, all_rougeL_scores = [], []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for _, batch in enumerate(test_dataloader):
        b_input_ids = batch["original_input_ids"].to(device)
        b_attention_mask = batch["original_attention_mask"].to(device)

        # Telling the model not to compute or store gradients, saving memory and speeding up prediction
        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=b_input_ids,
                attention_mask=b_attention_mask,
                max_length=150,
                num_beams=2,
                repetition_penalty=2.5,
                length_penalty=1.0,
                early_stopping=True,
            )
        preds = [
            tokenizer.decode(
                g, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )
            for g in generated_ids
        ]

        test_targets = batch["target_text"]

        for pred, targets in zip(preds, test_targets):
            targets = targets.split("___")
            wer_scores = [wer(pred, t) for t in targets]
            cer_scores = [cer(pred, t) for t in targets]

            all_wer_scores.append(max(wer_scores))
            all_cer_scores.append(max(cer_scores))

    avg_wer = sum(all_wer_scores) / len(all_wer_scores)
    avg_cer = sum(all_cer_scores) / len(all_cer_scores)
    print(f"Average WER: {avg_wer}")
    print(f"Average CER: {avg_cer}")
