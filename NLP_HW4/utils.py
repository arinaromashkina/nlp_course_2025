import os
import time

from typing import Any, Dict

import numpy as np
import psutil
import torch


class MemoryTracker:
    @staticmethod
    def get_gpu_memory() -> Dict[str, float]:
        if torch.cuda.is_available():
            return {
                "allocated": torch.cuda.memory_allocated() / 1024**2,
                "reserved": torch.cuda.memory_reserved() / 1024**2,
                "max_allocated": torch.cuda.max_memory_allocated() / 1024**2,
            }
        return {"allocated": 0, "reserved": 0, "max_allocated": 0}

    @staticmethod
    def get_cpu_memory() -> float:
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024**2

    @staticmethod
    def reset_peak_memory():
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()


class Timer:
    def __init__(self, name: str = "Operation", sync_cuda: bool = True):
        self.name = name
        self.sync_cuda = sync_cuda
        self.elapsed = 0

    def __enter__(self):
        if self.sync_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()
        self.start_time = time.time()
        return self

    def __exit__(self, *args):
        if self.sync_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()
        self.elapsed = time.time() - self.start_time
        print(f"{self.name}: {self.elapsed:.4f} seconds")


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def format_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        return f"{seconds / 60:.2f}m"
    else:
        return f"{seconds / 3600:.2f}h"


def set_seed(seed: int = 42):
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def print_model_stats(model: torch.nn.Module, model_name: str = "Model"):
    from NLP_HW4.lora import count_parameters

    stats = count_parameters(model)

    print(f"\n{'=' * 60}")
    print(f"{model_name}")
    print(f"total_params: {stats['total_params']:,}")
    print(f"trainable_params: {stats['trainable_params']:,}")
    print(f"lora_param: {stats['lora_params']:,}")
    print(f"trainable_percentage %: {stats['trainable_percentage']:.2f}%")
    print(f"{'=' * 60}\n")


def compare_models_memory(model_baseline, model_lora, batch_size=4, seq_len=128, device="cuda"):
    results = {}
    dummy_input = torch.randint(0, 1000, (batch_size, seq_len)).to(device)

    for name, model in [("Baseline", model_baseline), ("LoRA", model_lora)]:
        model = model.to(device)
        model.train()

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        with Timer(f"{name} Forward", sync_cuda=True) as timer_fwd:
            outputs = model(dummy_input, labels=dummy_input)
            loss = outputs.loss

        mem_after_forward = MemoryTracker.get_gpu_memory()

        with Timer(f"{name} Backward", sync_cuda=True) as timer_bwd:
            loss.backward()

        mem_after_backward = MemoryTracker.get_gpu_memory()

        results[name] = {
            "forward_time": timer_fwd.elapsed,
            "backward_time": timer_bwd.elapsed,
            "memory_after_forward": mem_after_forward["allocated"],
            "memory_after_backward": mem_after_backward["allocated"],
            "peak_memory": mem_after_backward["max_allocated"],
        }

        del outputs, loss
        model.zero_grad()
        model = model.cpu()
        torch.cuda.empty_cache()

    return results


def print_comparison_results(results: Dict[str, Any]):
    print(f"\n{'=' * 60}")
    baseline = results["Baseline"]
    lora = results["LoRA"]

    print("forward_time:")
    print(f"Baseline: {baseline['forward_time']:.4f}s")
    print(f"LoRA:     {lora['forward_time']:.4f}s")
    print(f"Speedup:  {baseline['forward_time'] / lora['forward_time']:.2f}x")

    print("backward_time:")
    print(f"Baseline: {baseline['backward_time']:.4f}s")
    print(f"LoRA:     {lora['backward_time']:.4f}s")
    print(f"Speedup:  {baseline['backward_time'] / lora['backward_time']:.2f}x")

    print("memory_usage_peak:")
    print(f"Baseline Peak: {baseline['peak_memory']:.2f} MB")
    print(f"LoRA Peak:     {lora['peak_memory']:.2f} MB")
    print(f"Savings:       {(1 - lora['peak_memory'] / baseline['peak_memory']) * 100:.2f}%")
    print(f"{'=' * 60}\n")


def log_to_comet(experiment, metrics: Dict[str, Any], step: int):
    if experiment is not None:
        experiment.log_metrics(metrics, step=step)


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    path: str,
    lora_only: bool = True,
):
    if lora_only:
        state_dict = {k: v for k, v in model.state_dict().items() if "lora_" in k}
    else:
        state_dict = model.state_dict()

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": state_dict,
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
    }
    torch.save(checkpoint, path)
    print(f"checkpoint saved to {path}")


def load_checkpoint(
    model: torch.nn.Module, optimizer: torch.optim.Optimizer, path: str, lora_only: bool = True
):
    checkpoint = torch.load(path)

    if lora_only:
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    else:
        model.load_state_dict(checkpoint["model_state_dict"])

    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return checkpoint["epoch"], checkpoint["loss"]
