import os

import numpy as np
import torch
import torch.nn as nn

from NLP_HW4.dpo import DPOTrainer as DPOLossTrainer
from NLP_HW4.dpo import print_dpo_stats
from NLP_HW4.utils import (
    AverageMeter,
    MemoryTracker,
    log_to_comet,
    save_checkpoint,
)
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import get_linear_schedule_with_warmup


class DPOFullTrainer:
    def __init__(
        self,
        model: nn.Module,
        ref_model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        device: str = "cuda",
        beta: float = 0.1,
        label_smoothing: float = 0.0,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
        log_interval: int = 10,
        eval_interval: int = 100,
        save_dir: str = "./dpo_checkpoints",
        experiment=None,
    ):
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.log_interval = log_interval
        self.eval_interval = eval_interval
        self.save_dir = save_dir
        self.experiment = experiment

        self.model = model
        self.ref_model = ref_model

        self.dpo_trainer = DPOLossTrainer(
            model=model,
            ref_model=ref_model,
            beta=beta,
            label_smoothing=label_smoothing,
            device=device,
        )

        os.makedirs(save_dir, exist_ok=True)

        self.global_step = 0
        self.epoch = 0

        self.train_loss_meter = AverageMeter()
        self.train_reward_margin_meter = AverageMeter()
        self.train_accuracy_meter = AverageMeter()


    def train_epoch(self, epoch: int, scheduler=None):
        self.model.train()
        self.train_loss_meter.reset()
        self.train_reward_margin_meter.reset()
        self.train_accuracy_meter.reset()

        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch}")

        for step, batch in enumerate(progress_bar):
            batch_device = {}
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch_device[k] = v.to(self.device)
                else:
                    batch_device[k] = v

            loss, metrics = self.dpo_trainer.compute_loss(batch_device)
            loss = loss / self.gradient_accumulation_steps

            loss.backward()

            if (step + 1) % self.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad], self.max_grad_norm
                )
                self.optimizer.step()

                if scheduler is not None:
                    scheduler.step()

                self.optimizer.zero_grad()
                self.global_step += 1

            self.train_loss_meter.update(loss.item() * self.gradient_accumulation_steps)
            self.train_reward_margin_meter.update(metrics["rewards/margins"])
            self.train_accuracy_meter.update(metrics["rewards/accuracy"])

            if (step + 1) % self.log_interval == 0:
                lr = self.optimizer.param_groups[0]["lr"]
                progress_bar.set_postfix(
                        {
                            "loss": f"{self.train_loss_meter.avg:.4f}",
                            "acc": f"{self.train_accuracy_meter.avg:.3f}",
                            "margin": f"{self.train_reward_margin_meter.avg:.3f}",
                            "lr": f"{lr:.2e}",
                        }
                )

                log_metrics = {
                        "train/loss": self.train_loss_meter.avg,
                        "train/reward_margin": self.train_reward_margin_meter.avg,
                        "train/accuracy": self.train_accuracy_meter.avg,
                        "train/learning_rate": lr,
                        "train/step": self.global_step,
                    }
                log_metrics.update({f"train/{k}": v for k, v in metrics.items()})

                if self.device == "cuda":
                    mem_stats = MemoryTracker.get_gpu_memory()
                    log_metrics["memory/allocated_mb"] = mem_stats["allocated"]

                log_to_comet(self.experiment, log_metrics, self.global_step)

            if (step + 1) % self.eval_interval == 0:
                val_metrics = self.validate()
                self.model.train()

                print(f"\nStep {self.global_step} - Val Metrics:")
                print_dpo_stats(val_metrics, prefix="  ")
                print()


        return self.train_loss_meter.avg

    @torch.no_grad()
    def validate(self):
        self.model.eval()

        all_metrics = {
            "loss": [],
            "rewards/chosen": [],
            "rewards/rejected": [],
            "rewards/margins": [],
            "rewards/accuracy": [],
        }

        for batch in tqdm(self.val_loader, desc="Validation", leave=False):
            batch_device = {}
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch_device[k] = v.to(self.device)
                else:
                    batch_device[k] = v

            loss, metrics = self.dpo_trainer.compute_loss(batch_device)

            all_metrics["loss"].append(loss.item())
            for key, value in metrics.items():
                if key in all_metrics:
                    all_metrics[key].append(value)

        avg_metrics = {k: np.mean(v) for k, v in all_metrics.items()}

        log_metrics = {f"val/{k}": v for k, v in avg_metrics.items()}
        log_to_comet(self.experiment, log_metrics, self.global_step)

        return avg_metrics

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

        best_reward_margin = -float("inf")


        for epoch in range(num_epochs):
            self.epoch = epoch

            train_loss = self.train_epoch(epoch, scheduler)

            val_metrics = self.validate()

            print(f"\nEpoch {epoch} Summary: Train Loss: {train_loss:.4f} Reward Margin: {self.train_reward_margin_meter.avg:.4f}")
            print(f"  Val Loss: {val_metrics['loss']:.4f} Accuracy: {val_metrics['rewards/accuracy']:.4f} Reward Margin: {val_metrics['rewards/margins']:.4f}")


            checkpoint_path = os.path.join(self.save_dir, f"checkpoint_epoch_{epoch}.pt")
            save_checkpoint(
                self.model,
                self.optimizer,
                epoch,
                val_metrics["loss"],
                checkpoint_path,
                lora_only=True,
            )

            if save_best and val_metrics["rewards/margins"] > best_reward_margin:
                best_reward_margin = val_metrics["rewards/margins"]
                best_path = os.path.join(self.save_dir, "best_model.pt")
                save_checkpoint(
                    self.model,
                    self.optimizer,
                    epoch,
                    val_metrics["loss"],
                    best_path,
                    lora_only=True,
                )

        return best_reward_margin
