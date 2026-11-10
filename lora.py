"""Minimal LoRA for the DINOv3 ViT used by INSID3.

Hand-written instead of `peft` because DINOv3's qkv layer is `LinearKMaskedBias`
(an nn.Linear subclass with a masked key bias). Wrapping the module and calling
it unchanged keeps that behaviour, and merging only touches `.weight`, so the
merged state dict has exactly the original keys and loads with
`torch.hub.load(..., weights=<path>)` (strict=True).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float = 0.0):
        super().__init__()
        self.base = base
        self.rank = rank
        self.scaling = alpha / rank
        self.enabled = True
        self.lora_A = nn.Parameter(torch.empty(rank, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank))  # B = 0: starts as the identity edit
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    # Expose the attributes DINOv3 code reads off the linear (e.g. qkv.in_features).
    @property
    def in_features(self):
        return self.base.in_features

    @property
    def out_features(self):
        return self.base.out_features

    @property
    def weight(self):
        return self.base.weight

    @property
    def bias(self):
        return self.base.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        if not self.enabled:
            return out
        a = self.lora_A.to(x.dtype)
        b = self.lora_B.to(x.dtype)
        return out + (self.dropout(x) @ a.T @ b.T) * self.scaling

    @torch.no_grad()
    def merged_weight(self) -> torch.Tensor:
        delta = (self.lora_B.float() @ self.lora_A.float()) * self.scaling
        return (self.base.weight.float() + delta).to(self.base.weight.dtype)


def inject_lora(encoder: nn.Module, blocks: list[int], targets: list[str], rank: int, alpha: float,
                dropout: float = 0.0) -> list[str]:
    """Replace `blocks[i].<target>` linears with LoRALinear. Targets: qkv, proj, fc1, fc2."""
    paths = {"qkv": ("attn", "qkv"), "proj": ("attn", "proj"), "fc1": ("mlp", "fc1"), "fc2": ("mlp", "fc2")}
    wrapped = []
    for i in blocks:
        blk = encoder.blocks[i]
        for t in targets:
            parent_name, child_name = paths[t]
            parent = getattr(blk, parent_name)
            base = getattr(parent, child_name)
            if not isinstance(base, nn.Linear):
                raise TypeError(f"blocks.{i}.{parent_name}.{child_name} is {type(base)}, not nn.Linear")
            lora = LoRALinear(base, rank, alpha, dropout).to(base.weight.device)
            setattr(parent, child_name, lora)
            wrapped.append(f"blocks.{i}.{parent_name}.{child_name}")
    return wrapped


def lora_modules(encoder: nn.Module):
    return [m for m in encoder.modules() if isinstance(m, LoRALinear)]


def set_lora_enabled(encoder: nn.Module, enabled: bool) -> None:
    for m in lora_modules(encoder):
        m.enabled = enabled


def lora_state_dict(encoder: nn.Module) -> dict:
    return {k: v.detach().cpu() for k, v in encoder.state_dict().items() if "lora_" in k}


def merged_state_dict(encoder: nn.Module) -> dict:
    """Full encoder state dict with LoRA folded into the weights and original key names."""
    merged_weights = {}
    for name, m in encoder.named_modules():
        if isinstance(m, LoRALinear):
            merged_weights[f"{name}.weight"] = m.merged_weight().cpu()
    out = {}
    for k, v in encoder.state_dict().items():
        if "lora_" in k:
            continue
        k_orig = k.replace(".base.", ".")
        out[k_orig] = merged_weights.get(k_orig, v.detach().cpu()) if k_orig.endswith(".weight") else v.detach().cpu()
    return out
