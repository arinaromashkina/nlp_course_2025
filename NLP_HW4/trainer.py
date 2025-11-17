import os

import numpy as np
import torch
import torch.nn as nn

from NLP_HW4.utils import (
    AverageMeter,
    MemoryTracker,
    log_to_comet,
    save_checkpoint,
)
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import get_linear_schedule_with_warmup


class LoRATrainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        device: str = "cuda",
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
        log_interval: int = 10,
        eval_interval: int = 100,
        save_dir: str = "./checkpoints",
        experiment=None,
        use_amp: bool = True,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.device = device
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.log_interval = log_interval
        self.eval_interval = eval_interval
        self.save_dir = save_dir
        self.experiment = experiment

        self.use_amp = use_amp and torch.cuda.is_available() and device == "cuda"
        if self.use_amp:
            self.scaler = GradScaler()
            print("Using AMP")
        else:
            self.scaler = None

        os.makedirs(save_dir, exist_ok=True)

        self.global_step = 0
        self.epoch = 0

        self.train_loss_meter = AverageMeter()
        self.val_loss_meter = AverageMeter()

    def train_epoch(self, epoch: int, scheduler=None):
        self.model.train()
        self.train_loss_meter.reset()

        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch}")

        for step, batch in enumerate(progress_bar):
            batch = {k: v.to(self.device) for k, v in batch.items()}

            if self.use_amp:
                with autocast():
                    outputs = self.model(**batch)
                    loss = outputs.loss / self.gradient_accumulation_steps
            else:
                outputs = self.model(**batch)
                loss = outputs.loss / self.gradient_accumulation_steps

            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            if (step + 1) % self.gradient_accumulation_steps == 0:
                if self.use_amp:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.optimizer.step()

                if scheduler is not None:
                    scheduler.step()
                self.optimizer.zero_grad()

                self.global_step += 1

            self.train_loss_meter.update(loss.item() * self.gradient_accumulation_steps)

            if (step + 1) % self.log_interval == 0:
                lr = self.optimizer.param_groups[0]["lr"]
                progress_bar.set_postfix(
                    {
                        "loss": f"{self.train_loss_meter.avg:.4f}",
                        "lr": f"{lr:.2e}",
                    }
                )

                metrics = {
                    "train/loss": self.train_loss_meter.avg,
                    "train/learning_rate": lr,
                    "train/step": self.global_step,
                }

                if self.device == "cuda":
                    mem_stats = MemoryTracker.get_gpu_memory()
                    metrics["memory/allocated_mb"] = mem_stats["allocated"]
                    metrics["memory/reserved_mb"] = mem_stats["reserved"]

                log_to_comet(self.experiment, metrics, self.global_step)

            if (step + 1) % self.eval_interval == 0:
                val_loss = self.validate()
                self.model.train()

                print(f"\nStep {self.global_step} - Val Loss: {val_loss:.4f}\n")

        return self.train_loss_meter.avg

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        self.val_loss_meter.reset()

        for batch in tqdm(self.val_loader, desc="Validation", leave=False):
            batch = {k: v.to(self.device) for k, v in batch.items()}

            if self.use_amp:
                with autocast():
                    outputs = self.model(**batch)
                    loss = outputs.loss
            else:
                outputs = self.model(**batch)
                loss = outputs.loss

            self.val_loss_meter.update(loss.item())

        metrics = {
            "val/loss": self.val_loss_meter.avg,
            "val/perplexity": np.exp(self.val_loss_meter.avg),
        }
        log_to_comet(self.experiment, metrics, self.global_step)

        return self.val_loss_meter.avg

    def train(
        self,
        num_epochs: int,
        warmup_steps: int = 0,
        save_best: bool = True,
    ):
        total_steps = len(self.train_loader) * num_epochs // self.gradient_accumulation_steps
        scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        best_val_loss = float("inf")

        for epoch in range(num_epochs):
            self.epoch = epoch

            train_loss = self.train_epoch(epoch, scheduler)

            val_loss = self.validate()
            val_perplexity = np.exp(val_loss)

            print(
                f"\nEpoch {epoch} Summary: Train Loss: {train_loss:.4f} Val Loss: {val_loss:.4f} "
            )

            checkpoint_path = os.path.join(self.save_dir, f"checkpoint_epoch_{epoch}.pt")
            save_checkpoint(
                self.model,
                self.optimizer,
                epoch,
                val_loss,
                checkpoint_path,
                lora_only=True,
            )

            if save_best and val_loss < best_val_loss:
                best_val_loss = val_loss
                best_path = os.path.join(self.save_dir, "best_model.pt")
                save_checkpoint(
                    self.model,
                    self.optimizer,
                    epoch,
                    val_loss,
                    best_path,
                    lora_only=True,
                )
                print(f"saved new best model with val_loss: {val_loss:.4f}")
        return best_val_loss

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        tokenizer,
        max_length: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.95,
    ) -> str:
        self.model.eval()

        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.device)

        generated_ids = input_ids

        for _ in range(max_length):
            outputs = self.model(generated_ids)
            next_token_logits = outputs.logits[:, -1, :] / temperature

            if top_k > 0:
                indices_to_remove = (
                    next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                )
                next_token_logits[indices_to_remove] = float("-inf")

            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                next_token_logits[indices_to_remove] = float("-inf")

            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            generated_ids = torch.cat([generated_ids, next_token], dim=1)

            if next_token.item() == tokenizer.eos_token_id:
                break

        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        return generated_text
