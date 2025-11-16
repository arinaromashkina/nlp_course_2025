import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List
import math


class LoRALayer(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 4,
        alpha: float = 1.0,
        dropout: float = 0.0,
        merge_weights: bool = False,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.merge_weights = merge_weights
        self.merged = False

        self.weight = nn.Parameter(torch.zeros(out_features, in_features), requires_grad=False)
        self.bias = None
        
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        self.lora_dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = F.linear(x, self.weight, self.bias)
        
        if self.rank > 0 and not self.merged:
            # x @ A^T @ B^T
            lora_out = self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T
            result = result + lora_out * self.scaling
        
        return result
    
    def merge_weights(self):
        if self.rank > 0 and not self.merged:
            # W = W0 + (B @ A) * scaling
            self.weight.data += (self.lora_B @ self.lora_A) * self.scaling
            self.merged = True
    
    def unmerge_weights(self):
        if self.rank > 0 and self.merged:
            self.weight.data -= (self.lora_B @ self.lora_A) * self.scaling
            self.merged = False
    
    @classmethod
    def from_linear(cls, linear: nn.Linear, rank: int = 4, alpha: float = 1.0, 
                    dropout: float = 0.0) -> 'LoRALayer':
        lora_layer = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
        )
        lora_layer.weight.data = linear.weight.data.clone()
        if linear.bias is not None:
            lora_layer.bias = nn.Parameter(linear.bias.data.clone())
        return lora_layer


def mark_only_lora_as_trainable(model: nn.Module, bias: str = 'none') -> None:
    for name, param in model.named_parameters():
        if 'lora_' not in name:
            param.requires_grad = False
    
    if bias == 'none':
        return
    elif bias == 'all':
        for name, param in model.named_parameters():
            if 'bias' in name:
                param.requires_grad = True
    elif bias == 'lora_only':
        for name, param in model.named_parameters():
            if 'bias' in name and 'lora_' in name:
                param.requires_grad = True


def apply_lora_to_model(
    model: nn.Module,
    target_modules: List[str] = ["q_proj", "v_proj", "k_proj"],
    rank: int = 4,
    alpha: float = 1.0,
    dropout: float = 0.0,
) -> nn.Module:
    for name, module in model.named_modules():
        if any(target in name for target in target_modules):
            if isinstance(module, nn.Linear):
                parent_name = '.'.join(name.split('.')[:-1])
                child_name = name.split('.')[-1]
                
                if parent_name:
                    parent = model.get_submodule(parent_name)
                else:
                    parent = model
                lora_layer = LoRALayer.from_linear(
                    module, 
                    rank=rank, 
                    alpha=alpha, 
                    dropout=dropout
                )
                setattr(parent, child_name, lora_layer)

    mark_only_lora_as_trainable(model, bias='none')
    return model


def get_lora_parameters(model: nn.Module) -> List[nn.Parameter]:
    return [p for n, p in model.named_parameters() if 'lora_' in n and p.requires_grad]


def count_parameters(model: nn.Module) -> dict:
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    lora_params = sum(p.numel() for n, p in model.named_parameters() if 'lora_' in n)
    
    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'lora_params': lora_params,
        'trainable_percentage': 100 * trainable_params / total_params if total_params > 0 else 0,
    }


def merge_lora_weights(model: nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, LoRALayer):
            module.merge_weights()


def unmerge_lora_weights(model: nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, LoRALayer):
            module.unmerge_weights()


def save_lora_weights(model: nn.Module, path: str) -> None:
    lora_state_dict = {
        name: param for name, param in model.state_dict().items() 
        if 'lora_' in name
    }
    torch.save(lora_state_dict, path)


def load_lora_weights(model: nn.Module, path: str) -> None:
    lora_state_dict = torch.load(path)
    model.load_state_dict(lora_state_dict, strict=False)
