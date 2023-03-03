import time

from rouge_score import rouge_scorer
import torch
from torch.utils.data import DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration

from resources.utils.helper_functions import format_time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate(
    model: T5ForConditionalGeneration,
    tokenizer: T5Tokenizer,
    val_dataloader: DataLoader,
) -> tuple[float, float, float, float]:
    """After the completion of each training epoch, measure the model's performance
    on our validation set.
    """
    # Put the model into the evaluation mode. The dropout layers are disabled during
    # the test time.
    t0 = time.time()
    model.eval()

    # Tracking outputs
    predictions = []
    correct_answers = []
    val_loss_set = []
    rouge1_score_set = []
    rougeL_score_set = []

    # scorer
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    
    # For each batch in our validation set...
    for batch in val_dataloader:
        b_input_ids = batch["original_input_ids"].to(device, dtype=torch.long)
        b_attention_mask = batch["original_attention_mask"].to(device, dtype=torch.long)

        y = batch["target_input_ids"].to(device, dtype=torch.long)
        y_ids = y[:, :-1].contiguous()
        lm_labels = y[:, 1:].clone().detach()
        lm_labels[y[:, 1:] == tokenizer.pad_token_id] = -100

        # get loss
        outputs = model(
            input_ids=b_input_ids,
            attention_mask=b_attention_mask,
            decoder_input_ids=y_ids,
            labels=lm_labels,
        )
        loss = outputs[0]
        val_loss_set.append(loss.item())

        # Get ROUGE scores
        # Telling the model not to compute or store gradients, saving memory and speeding up prediction
        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=b_input_ids,
                attention_mask=b_attention_mask,
                max_length=50,
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

        target_input_ids = batch["target_input_ids"].to(device, dtype=torch.long)
        targets = [
            tokenizer.decode(
                t, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )
            for t in target_input_ids
        ]
        predictions.extend(preds)
        correct_answers.extend(targets)
    # Compute the average accuracy and loss over the validation set.
    validation_time = format_time(time.time() - t0)
    
    avg_val_loss = sum(val_loss_set) / len(val_loss_set)
    
    for ans, pred in zip(correct_answers, predictions):
        scores = scorer.score(ans,pred)
        rouge1 = scores['rouge1']
        rougeL = scores['rougeL']
        rouge1_score_set.append(rouge1.fmeasure)
        rougeL_score_set.append(rougeL.fmeasure)
        
    avg_rouge1 = sum(rouge1_score_set)/len(rouge1_score_set)
    avg_rougeL = sum(rougeL_score_set)/len(rougeL_score_set)
    return avg_val_loss, avg_rouge1, avg_rougeL, validation_time
