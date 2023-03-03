from datetime import date
import json
import time

import torch
from torch.utils.data import DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration

from resources.training.evaluate import evaluate
from resources.utils.helper_functions import format_time
from resources.utils.pytorchtools import build_optimizer_scheduler, EarlyStopping

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_model(
    args,
    model: T5ForConditionalGeneration,
    tokenizer: T5Tokenizer,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
) -> None:
    """Train and validate the NER BERT model."""
    training_stats, train_loss_set = [], []
    total_loss, batch_loss, batch_counts = 0, 0, 0

    model.to(device)

    name = (
        args.savedir
        + str(args.epochs)
        + "_epochs_"
        + str(date.today().strftime("%d_%m_%Y"))
        # + str(args.model_name)
    )
    save_model_path = name + "_" + args.model_name.replace("/", "_") + "_model"
    save_results_path = name + "_results.json"

    early_stopping = EarlyStopping(
        patience=args.patience, verbose=True, path=save_model_path
    )
    optimizer, scheduler = build_optimizer_scheduler(
        model=model, epochs=args.epochs, train_dataloader=train_dataloader
    )

    print("Start training...\n")
    for epoch_i in range(args.epochs):
        # =======================================
        #               Training
        # =======================================
        print(
            f"{'Epoch':^7} | {'Batch':^7} | {'Train Loss':^12} | {'Val Loss':^10} | {'Val ROUGE1':^9} | {'Val ROUGEL':^9} | {'Elapsed':^9}"
        )
        print("-" * 80)
        t0 = time.time()
        t0_epoch, t0_batch = time.time(), time.time()

        model.train()

        for step, batch in enumerate(train_dataloader):
            batch_counts += 1
            b_input_ids = batch["original_input_ids"].to(device, dtype=torch.long)
            b_attention_mask = batch["original_attention_mask"].to(
                device, dtype=torch.long
            )

            y = batch["target_input_ids"].to(device, dtype=torch.long)
            y_ids = y[:, :-1].contiguous()
            lm_labels = y[:, 1:].clone().detach()
            lm_labels[y[:, 1:] == tokenizer.pad_token_id] = -100

            outputs = model(
                input_ids=b_input_ids,
                attention_mask=b_attention_mask,
                decoder_input_ids=y_ids,
                labels=lm_labels,
            )

            loss = outputs[0]
            batch_loss += loss.item()
            total_loss += loss.item()
            train_loss_set.append(loss.item())

            # backpropagation
            optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            # Print the loss values and time elapsed for every 20 batches
            if (step % 20 == 0) or (step == len(train_dataloader) - 1):
                time_elapsed = time.time() - t0_batch
                print(
                    f"{epoch_i + 1:^7} | {step:^7} | {batch_loss / batch_counts:^12.6f} | {'-':^10} | {'-':^9} | {'-':^9} | {time_elapsed:^9.2f}"
                )
                # Reset batch tracking variables
                batch_loss, batch_counts = 0, 0
                t0_batch = time.time()

        # Calculate the average loss over the entire training data
        avg_train_loss = total_loss / len(train_dataloader)
        training_time = format_time(time.time() - t0)

        print("-" * 80)

        # =======================================
        #               Evaluation
        # =======================================
        avg_val_loss, avg_val_accuracy, validation_time = None, None, None
        if val_dataloader:
            avg_val_loss, avg_rouge1, avg_rougeL, validation_time = evaluate(
                model, tokenizer, val_dataloader
            )
            time_elapsed = time.time() - t0_epoch
            print(
                f"{epoch_i + 1:^7} | {'-':^7} | {avg_train_loss:^12.6f} | {avg_val_loss:^10.6f} | {avg_rouge1:^9.2f} | {avg_rougeL:^9.2f} | {time_elapsed:^9.2f}"
            )
            print("-" * 80)

            early_stopping(avg_val_loss, model, tokenizer)
        else:
            early_stopping(avg_train_loss, model, tokenizer)

        if early_stopping.early_stop:
            print("Early stopping")
            break

        training_stats.append(
            {
                "epoch": epoch_i + 1,
                "Training Loss": avg_train_loss,
                "Valid. Loss": avg_val_loss,
                "Valid. Accur.": avg_val_accuracy,
                "Training Time": training_time,
                "Validation Time": validation_time,
            }
        )

        with open(save_results_path, "w") as f:
            json.dump(training_stats, f)

    print("\n")
    print("Training complete!")
    print("Time taken to complete training: {}".format(time.time() - t0))
