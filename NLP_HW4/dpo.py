from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class DPOLoss(nn.Module):
    def __init__(
        self,
        beta: float = 0.1,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.beta = beta
        self.label_smoothing = label_smoothing

    def forward(
        self,
        policy_chosen_logps: torch.Tensor,
        policy_rejected_logps: torch.Tensor,
        reference_chosen_logps: torch.Tensor,
        reference_rejected_logps: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        
        ref_logratios = reference_chosen_logps - reference_rejected_logps

        policy_logratios = policy_chosen_logps - policy_rejected_logps

        logits = policy_logratios - ref_logratios

        if self.label_smoothing > 0:
            losses = (
                -F.logsigmoid(self.beta * logits) * (1 - self.label_smoothing)
                - F.logsigmoid(-self.beta * logits) * self.label_smoothing
            )
        else:
            losses = -F.logsigmoid(self.beta * logits)

        loss = losses.mean()

        with torch.no_grad():
            chosen_rewards = self.beta * (policy_chosen_logps - reference_chosen_logps)
            rejected_rewards = self.beta * (policy_rejected_logps - reference_rejected_logps)

            accuracy = (chosen_rewards > rejected_rewards).float().mean()

            metrics = {
                "rewards/chosen": chosen_rewards.mean().item(),
                "rewards/rejected": rejected_rewards.mean().item(),
                "rewards/margins": (chosen_rewards - rejected_rewards).mean().item(),
                "rewards/accuracy": accuracy.item(),
                "logps/chosen": policy_chosen_logps.mean().item(),
                "logps/rejected": policy_rejected_logps.mean().item(),
                "logits/chosen": policy_logratios.mean().item(),
            }

        return loss, metrics


def get_batch_logps(
    logits: torch.Tensor,
    labels: torch.Tensor,
    average_log_prob: bool = False,
) -> torch.Tensor:
    
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    vocab_size = shift_logits.size(-1)
    valid_labels_mask = (shift_labels >= 0) & (shift_labels < vocab_size)

    ignore_mask = shift_labels == -100

    safe_mask = valid_labels_mask | ignore_mask

    if not safe_mask.all():
        invalid_labels = shift_labels[~safe_mask]
        shift_labels = torch.where(safe_mask, shift_labels, torch.zeros_like(shift_labels))
        
    log_probs = F.log_softmax(shift_logits, dim=-1)
    gather_labels = shift_labels.clone()
    gather_labels[gather_labels == -100] = 0

    per_token_logps = torch.gather(log_probs, dim=2, index=gather_labels.unsqueeze(2)).squeeze(2)

    loss_mask = shift_labels != -100

    if average_log_prob:
        result = (per_token_logps * loss_mask).sum(-1) / (loss_mask.sum(-1) + 1e-10)
    else:
        result = (per_token_logps * loss_mask).sum(-1)

    return result


def concatenated_forward(
    model: nn.Module,
    batch: Dict[str, torch.Tensor],
    device: str = "cuda",
) -> Tuple[torch.Tensor, torch.Tensor]:
    batch_size = batch["input_ids"].shape[0] // 2

    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)

    vocab_size = model.config.vocab_size
    if (input_ids >= vocab_size).any():
        input_ids = torch.clamp(input_ids, 0, vocab_size - 1)

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
    )

    logits = outputs.logits
    all_logps = get_batch_logps(logits, labels, average_log_prob=False)

    chosen_logps = all_logps[:batch_size]
    rejected_logps = all_logps[batch_size:]

    return chosen_logps, rejected_logps


class DPOTrainer:
    def __init__(
        self,
        model: nn.Module,
        ref_model: nn.Module,
        beta: float = 0.1,
        label_smoothing: float = 0.0,
        device: str = "cuda",
    ):
        self.model = model
        self.ref_model = ref_model
        self.device = device

        for param in self.ref_model.parameters():
            param.requires_grad = False
        self.ref_model.eval()

        self.dpo_loss = DPOLoss(
            beta=beta,
            label_smoothing=label_smoothing,
        )

        self.vocab_size = model.config.vocab_size


    def compute_loss(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        self._validate_batch(batch)

        policy_chosen_logps, policy_rejected_logps = concatenated_forward(
            self.model, batch, self.device
        )

        with torch.no_grad():
            reference_chosen_logps, reference_rejected_logps = concatenated_forward(
                self.ref_model, batch, self.device
            )

        loss, metrics = self.dpo_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            reference_chosen_logps,
            reference_rejected_logps,
        )

        return loss, metrics

    def _validate_batch(self, batch: Dict[str, torch.Tensor]):
        input_ids = batch["input_ids"]
        labels = batch["labels"]

        if input_ids.max() >= self.vocab_size:
            max_id = input_ids.max().item()
            invalid_mask = input_ids >= self.vocab_size

        valid_labels = labels[labels != -100]
        if len(valid_labels) > 0 and valid_labels.max() >= self.vocab_size:
            max_label = valid_labels.max().item()
            invalid_mask = (labels != -100) & (labels >= self.vocab_size)


    def prediction_step(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, float]:
        with torch.no_grad():
            _, metrics = self.compute_loss(batch)
        return metrics


def print_dpo_stats(metrics: Dict[str, float], prefix: str = ""):
    print(f"{prefix}DPO Metrics:")
    print(f"rewards/reward: {metrics.get('rewards/chosen', 0):.4f}")
    print(f"rewards/reward: {metrics.get('rewards/rejected', 0):.4f}")
    print(f"rewards/margin: {metrics.get('rewards/margins', 0):.4f}")
    print(f"rewards/accuracy: {metrics.get('rewards/accuracy', 0):.4f}")
