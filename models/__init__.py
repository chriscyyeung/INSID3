"""Model construction utilities for INSID3."""

from __future__ import annotations

import torch

from models.insid3 import INSID3

_HUB_NAMES = {
    "small": "dinov3_vits16",
    "base": "dinov3_vitb16",
    "large": "dinov3_vitl16",
}

_WEIGHTS = {
    "small": "pretrain/dinov3_vits16_pretrain_lvd1689m-08c60483.pth",
    "base": "pretrain/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth",
    "large": "pretrain/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth",
}


def _build_encoder(model_size: str = "large", weights: str | None = None):
    if weights is None:
        return torch.hub.load(
            "facebookresearch/dinov3",
            _HUB_NAMES[model_size],
            weights=_WEIGHTS[model_size],
        )
    # Override checkpoint (e.g. a LoRA-merged backbone from train_lora_lumpnav.py), loaded strictly
    # from disk. Hub's weights= would copy it into ~/.cache/torch/hub/checkpoints and reuse it by
    # file name, so a retrained checkpoint with the same name would silently load stale weights.
    encoder = torch.hub.load("facebookresearch/dinov3", _HUB_NAMES[model_size], pretrained=False)
    encoder.load_state_dict(torch.load(weights, map_location="cpu"), strict=True)
    return encoder


def build_insid3(
    *,
    model_size: str = "large",
    image_size: int = 1024,
    svd_components: int = 500,
    tau: float = 0.6,
    merge_threshold: float = 0.2,
    mask_refiner: str = "bilinear",
    crf_size: int = 640,
    resize_to_orig_size: bool = True,
    device: str = "cuda",
    encoder_weights: str | None = None,
):
    encoder = _build_encoder(model_size, encoder_weights)
    model = INSID3(
        encoder=encoder,
        image_size=image_size,
        svd_components=svd_components,
        tau=tau,
        merge_threshold=merge_threshold,
        mask_refiner=mask_refiner,
        crf_size=crf_size,
        resize_to_orig_size=resize_to_orig_size,
        device=device,
    )
    for param in model.parameters():
        param.requires_grad = False
    return model


def build_insid3_from_args(args):
    return build_insid3(
        model_size=args.model_size,
        image_size=args.image_size,
        svd_components=int(args.svd_comps),
        tau=args.tau,
        merge_threshold=args.merge_thresh,
        mask_refiner='crf' if getattr(args, 'crf_mask_refinement', False) else 'bilinear',
        crf_size=args.crf_size,
        resize_to_orig_size=False,
        device=args.device,
        encoder_weights=getattr(args, 'encoder_weights', None),
    )
