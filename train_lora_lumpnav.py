"""Train LoRA adapters on INSID3's DINOv3 backbone for one LumpNav CV fold.

INSID3 itself is unchanged: training only changes the backbone weights it reads
features from. At the end the best adapter is merged into the weights and saved
as a plain DINOv3 state dict (original key names), which inference loads with
`inference_segmentation.py --encoder-weights <file>`.

Training is episodic, the way INSID3 is used. Each episode is a target frame
plus 1..max_shots non-empty reference frames from *other* training patients.
Losses, all on the last-layer patch features INSID3 reads:

  proto  INSID3's forward match: target patches should have cosine > 0 to the
         reference tumour prototype iff they are tumour (debiased space,
         sigmoid(sim / t), balanced BCE + soft Dice).
  nn     INSID3's backward match: each target patch's soft nearest neighbour
         among a reference's patches should land inside that reference's mask
         iff the target patch is tumour (debiased space, balanced BCE).
  gram   Gram anchoring: the patch-to-patch cosine matrix should stay close to
         the frozen DINOv3's, so INSID3's clustering (raw features, tau=0.6)
         and positional debiasing still behave.

Debiasing uses the positional basis INSID3 computes (SVD of the features of a
constant image), recomputed from the current weights every --basis-every steps.

CV hygiene: only the fold's training patients are used. --inner-val-patients of
them are held out (seeded) and used only to pick the checkpoint, by running the
real INSID3 pipeline on them. The fold's validation patients are never touched.

Example (from the INSID3 repo root):
    python train_lora_lumpnav.py --data-root $SLURM_TMPDIR/cv --fold 0 \
        --output-dir output/insid3_lora/fold_0
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
import time
from os.path import join

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.checkpoint import checkpoint
from torch.utils.data import DataLoader, Dataset

REPO_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_DIR)

from datasets.lumpnav import DatasetLumpNav  # noqa: E402
from lora import inject_lora, lora_modules, lora_state_dict, merged_state_dict, set_lora_enabled  # noqa: E402
from models.insid3 import INSID3  # noqa: E402
from utils.data import downsample_mask  # noqa: E402

HUB_NAMES = {"small": "dinov3_vits16", "base": "dinov3_vitb16", "large": "dinov3_vitl16"}
WEIGHTS = {
    "small": "pretrain/dinov3_vits16_pretrain_lvd1689m-08c60483.pth",
    "base": "pretrain/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth",
    "large": "pretrain/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth",
}
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def get_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # Data
    p.add_argument("--data-root", required=True, help="CV root containing fold_<k>/{train,val}/{images,masks}")
    p.add_argument("--fold", type=int, required=True)
    p.add_argument("--inner-val-patients", type=int, default=3,
                   help="Training patients held out for checkpoint selection")
    p.add_argument("--p-empty-target", type=float, default=0.1,
                   help="Probability that an episode's target is an empty-mask frame")
    p.add_argument("--max-shots", type=int, default=3, help="References per training episode: uniform in 1..max")
    p.add_argument("--no-augment", action="store_true")
    # Model / LoRA
    p.add_argument("--model-size", default="large", choices=list(HUB_NAMES))
    p.add_argument("--weights", default=None, help="Base DINOv3 checkpoint (default: INSID3's pretrain/ file)")
    p.add_argument("--image-size", type=int, default=1024, help="Training resolution (INSID3 infers at 1024)")
    p.add_argument("--lora-last-blocks", type=int, default=12, help="Add LoRA to the last N blocks")
    p.add_argument("--lora-targets", default="qkv,proj", help="Comma list from qkv,proj,fc1,fc2")
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--alpha", type=float, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.0)
    p.add_argument("--grad-ckpt", action="store_true", help="Gradient checkpointing per block (less memory)")
    # Loss
    p.add_argument("--w-proto", type=float, default=1.0)
    p.add_argument("--w-nn", type=float, default=1.0)
    p.add_argument("--w-gram", type=float, default=1.0)
    p.add_argument("--t-proto", type=float, default=0.1, help="Temperature for sigmoid(sim / t)")
    p.add_argument("--t-nn", type=float, default=0.05, help="Softmax temperature for soft nearest neighbour")
    p.add_argument("--gram-patches", type=int, default=1024, help="Patches sampled per image for the Gram loss")
    p.add_argument("--svd-comps", type=int, default=500, help="Positional subspace size (INSID3 default)")
    p.add_argument("--basis-every", type=int, default=50, help="Recompute the positional basis every N steps")
    # Optimisation
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--episodes-per-step", type=int, default=4, help="Gradient accumulation over episodes")
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--clip", type=float, default=1.0)
    # Inner validation (real INSID3 pipeline)
    p.add_argument("--eval-every", type=int, default=250)
    p.add_argument("--eval-targets", type=int, default=100, help="Non-empty inner-val targets per evaluation")
    p.add_argument("--eval-shots", type=int, nargs="+", default=[1, 5])
    p.add_argument("--eval-image-size", type=int, default=1024)
    # Runtime
    p.add_argument("--output-dir", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num-workers", type=int, default=6)
    p.add_argument("--device", default="cuda")
    p.add_argument("--log-every", type=int, default=20)
    return p.parse_args()


# ──────── Data ────────

def load_frame(img_path, msk_path, size, rng=None):
    """(3,S,S) normalised image and (S,S) {0,1} mask, with optional augmentation."""
    img = np.asarray(Image.open(img_path).convert("L"), dtype=np.float32) / 255.0
    msk = DatasetLumpNav.read_mask(msk_path).numpy().astype(np.float32)
    if rng is not None:
        if rng.random() < 0.5:
            img, msk = img[:, ::-1], msk[:, ::-1]
        img = np.clip(img * rng.uniform(0.85, 1.15) + rng.uniform(-0.05, 0.05), 0, 1) ** rng.uniform(0.8, 1.25)
    img_t = torch.from_numpy(np.ascontiguousarray(img))[None, None]
    msk_t = torch.from_numpy(np.ascontiguousarray(msk))[None, None]
    img_t = F.interpolate(img_t, size=(size, size), mode="bilinear", align_corners=False, antialias=True)[0]
    msk_t = F.interpolate(msk_t, size=(size, size), mode="nearest")[0, 0]
    img_t = (img_t.expand(3, -1, -1) - MEAN) / STD
    return img_t, msk_t


class EpisodeDataset(Dataset):
    """Each item is one random episode: references first, target last."""

    def __init__(self, frames, size, max_shots, p_empty, augment, seed, length=10**8):
        self.size, self.max_shots, self.p_empty, self.augment = size, max_shots, p_empty, augment
        self.seed, self.length = seed, length
        self.nonempty = [f for f in frames if f["nonempty"]]
        self.empty = [f for f in frames if not f["nonempty"]]
        self.by_patient = {}
        for f in self.nonempty:
            self.by_patient.setdefault(f["patient"], []).append(f)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        rng = np.random.default_rng([self.seed, idx])
        pool = self.empty if (self.empty and rng.random() < self.p_empty) else self.nonempty
        tgt = pool[rng.integers(len(pool))]
        k = int(rng.integers(1, self.max_shots + 1))
        others = [p for p in self.by_patient if p != tgt["patient"]]
        # References from distinct patients where possible (adjacent frames are near-duplicates)
        ref_patients = rng.choice(others, size=k, replace=len(others) < k)
        refs = [self.by_patient[p][rng.integers(len(self.by_patient[p]))] for p in ref_patients]
        aug = rng if self.augment else None
        items = [load_frame(f["img"], f["msk"], self.size, aug) for f in refs + [tgt]]
        return torch.stack([i for i, _ in items]), torch.stack([m for _, m in items])


def list_pairs(fold_dir, split):
    """(image, mask) path pairs, matched by frame name (same rules as DatasetLumpNav)."""
    import glob
    name = DatasetLumpNav.frame_name
    images = {name(p): p for p in glob.glob(join(fold_dir, split, "images", "*.png"))}
    masks = {name(p): p for p in glob.glob(join(fold_dir, split, "masks", "*.png"))}
    missing = sorted(set(masks) - set(images))
    if missing or not masks:
        raise RuntimeError(f"{fold_dir}/{split}: {len(masks)} masks, {len(missing)} without an image")
    return [(images[k], masks[k]) for k in sorted(masks)]


def list_frames(fold_dir, split):
    frames = []
    for img, msk in list_pairs(fold_dir, split):
        frames.append({"img": img, "msk": msk, "name": DatasetLumpNav.frame_name(msk),
                       "patient": DatasetLumpNav.patient_id(msk),
                       "nonempty": bool(DatasetLumpNav.read_mask(msk).any())})
    return frames


# ──────── Backbone ────────

def load_encoder(model_size, weights):
    """Build the DINOv3 ViT and load `weights` strictly from disk.

    Not via hub's `weights=`: that copies the file into ~/.cache/torch/hub/checkpoints and
    reuses it by file name, so a retrained checkpoint with the same name would load stale weights.
    """
    hub_local = join(torch.hub.get_dir(), "facebookresearch_dinov3_main")
    if os.path.isdir(hub_local):  # compute nodes have no internet
        enc = torch.hub.load(hub_local, HUB_NAMES[model_size], source="local", pretrained=False)
    else:
        enc = torch.hub.load("facebookresearch/dinov3", HUB_NAMES[model_size], pretrained=False)
    enc.load_state_dict(torch.load(weights, map_location="cpu"), strict=True)
    return enc


def encode(encoder, x, first_lora_block, grad_ckpt, with_teacher):
    """Last-layer normalised patch tokens (B, N, C) as INSID3 reads them, plus the frozen-model ones.

    Blocks before the first LoRA block are frozen and shared by student and teacher.
    """
    tokens, (H, W) = encoder.prepare_tokens_with_masks(x)
    rope = encoder.rope_embed(H=H, W=W) if encoder.rope_embed is not None else None
    n_skip = encoder.n_storage_tokens + 1
    blocks = encoder.blocks
    with torch.no_grad():
        for blk in blocks[:first_lora_block]:
            tokens = blk(tokens, rope)
    teacher = None
    if with_teacher:
        with torch.no_grad():
            set_lora_enabled(encoder, False)
            t = tokens
            for blk in blocks[first_lora_block:]:
                t = blk(t, rope)
            teacher = encoder.norm(t[:, n_skip:])
            set_lora_enabled(encoder, True)
    s = tokens
    for blk in blocks[first_lora_block:]:
        s = checkpoint(blk, s, rope, use_reentrant=False) if grad_ckpt else blk(s, rope)
    student = encoder.norm(s[:, n_skip:])
    return student, teacher


@torch.no_grad()
def positional_basis(encoder, size, n_comps, device):
    """Same estimate as INSID3._build_positional_basis, at the given resolution."""
    img = ((torch.zeros(3, size, size) - MEAN) / STD)[None].to(device)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=img.is_cuda):
        f = encoder.get_intermediate_layers(img, n=1, reshape=True)[0]
    f = F.normalize(f.float(), dim=1)
    E = f.flatten(2)[0]  # (C, h*w)
    E = E - E.mean(dim=1, keepdim=True)
    U, _, _ = torch.linalg.svd(E, full_matrices=False)
    return U[:, :n_comps].contiguous()


def debias(f, basis):
    """f: (..., C) unit vectors. Project out the positional subspace and renormalise."""
    return F.normalize(f - (f @ basis) @ basis.T, dim=-1)


# ──────── Loss ────────

def balanced_bce_logits(logits, labels):
    fg, bg = labels > 0.5, labels <= 0.5
    loss = logits.new_zeros(())
    terms = 0
    l = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    if fg.any():
        loss, terms = loss + l[fg].mean(), terms + 1
    if bg.any():
        loss, terms = loss + l[bg].mean(), terms + 1
    return loss / max(terms, 1)


def episode_loss(student, teacher, masks, basis, h, w, args):
    """student/teacher: (K+1, N, C) for K references then the target; masks: (K+1, S, S)."""
    K = student.shape[0] - 1
    f = F.normalize(student.float(), dim=-1)
    fd = debias(f, basis)
    patch = masks.shape[-1] // h
    soft = F.avg_pool2d(masks[:, None], patch)[:, 0].flatten(1)          # (K+1, N) tumour fraction
    ref_bin = torch.stack([downsample_mask(masks[s][None, None], h, w).flatten() for s in range(K)])  # INSID3 style

    tgt_soft = soft[K]
    tgt_bin = (tgt_soft > 0.5).float()
    tgt_nonempty = bool(masks[K].any())
    logs = {}

    # proto: forward matching against the shot-averaged prototype
    protos = [fd[s][ref_bin[s]].mean(0) for s in range(K)]
    proto = F.normalize(torch.stack(protos).mean(0), dim=0)
    sim = fd[K] @ proto
    logits = sim / args.t_proto
    l_proto = balanced_bce_logits(logits, tgt_soft)
    if tgt_nonempty:
        p = torch.sigmoid(logits)
        l_proto = l_proto + 1 - (2 * (p * tgt_soft).sum() + 1) / (p.sum() + tgt_soft.sum() + 1)
    logs["proto"] = l_proto.item()

    # nn: soft nearest neighbour of each target patch in each reference
    l_nn = 0.0
    for s in range(K):
        attn = torch.softmax((fd[K] @ fd[s].T) / args.t_nn, dim=-1)       # (N_t, N_r)
        p_fg = (attn @ ref_bin[s].float()).clamp(1e-6, 1 - 1e-6)
        l_nn = l_nn + balanced_bce_logits(torch.logit(p_fg), tgt_bin)
    l_nn = l_nn / K
    logs["nn"] = float(l_nn)

    # gram: keep the patch-similarity structure of the frozen model
    l_gram = torch.zeros((), device=f.device)
    if teacher is not None and args.w_gram > 0:
        ft = F.normalize(teacher.float(), dim=-1)
        n = f.shape[1]
        idx = torch.randperm(n, device=f.device)[: min(args.gram_patches, n)]
        gs = f[:, idx] @ f[:, idx].transpose(1, 2)
        gt = ft[:, idx] @ ft[:, idx].transpose(1, 2)
        l_gram = (gs - gt).pow(2).mean()
    logs["gram"] = l_gram.item()

    total = args.w_proto * l_proto + args.w_nn * l_nn + args.w_gram * l_gram
    logs["total"] = total.item()
    return total, logs


# ──────── Inner validation (real INSID3) ────────

def build_eval_episodes(eval_frames, ref_frames, n_targets, shots_list, seed):
    rng = np.random.default_rng(seed)
    targets = [f for f in eval_frames if f["nonempty"]]
    if len(targets) > n_targets:
        targets = [targets[i] for i in sorted(rng.choice(len(targets), n_targets, replace=False))]
    max_shots = max(shots_list)
    episodes = []
    for t in targets:
        refs = [ref_frames[i] for i in rng.choice(len(ref_frames), max_shots, replace=False)]
        episodes.append((t, refs))  # k-shot uses the first k references
    return episodes


@torch.no_grad()
def evaluate(encoder, episodes, shots_list, args):
    model = INSID3(encoder=encoder, image_size=args.eval_image_size, svd_components=args.svd_comps,
                   resize_to_orig_size=False, device=args.device)
    size = args.eval_image_size
    cache = {}

    def get(f):
        if f["name"] not in cache:
            img, msk = load_frame(f["img"], f["msk"], size)
            cache[f["name"]] = (img.to(args.device), msk.to(args.device) > 0.5)
        return cache[f["name"]]

    results = {}
    for k in shots_list:
        dices = []
        for tgt, refs in episodes:
            ref_imgs = torch.stack([get(r)[0] for r in refs[:k]])
            ref_msks = torch.stack([get(r)[1] for r in refs[:k]])
            tgt_img, tgt_msk = get(tgt)
            pred = model.predict_mask(ref_imgs, ref_msks, tgt_img[None])
            inter = (pred & tgt_msk).sum().item()
            denom = pred.sum().item() + tgt_msk.sum().item()
            dices.append(2 * inter / denom if denom else float("nan"))
        results[f"dice_{k}shot"] = float(np.nanmean(dices))
    results["dice_mean"] = float(np.mean([results[f"dice_{k}shot"] for k in shots_list]))
    del model
    return results


# ──────── Main ────────

def main():
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = args.device
    with open(join(args.output_dir, "config.json"), "w") as fp:
        json.dump(vars(args) | {"argv": sys.argv}, fp, indent=2)

    # Data: training patients of this fold only
    fold_dir = join(args.data_root, f"fold_{args.fold}")
    frames = list_frames(fold_dir, "train")
    val_patients = {DatasetLumpNav.patient_id(m) for _, m in list_pairs(fold_dir, "val")}
    patients = sorted({f["patient"] for f in frames})
    if set(patients) & val_patients:
        raise RuntimeError(f"Train/val patient overlap: {sorted(set(patients) & val_patients)}")
    rng = np.random.default_rng(args.seed + 1000 * args.fold)
    inner_val = sorted(rng.choice(patients, args.inner_val_patients, replace=False).tolist())
    fit_frames = [f for f in frames if f["patient"] not in inner_val]
    eval_frames = [f for f in frames if f["patient"] in inner_val]
    print(f"Fold {args.fold}: {len(patients)} train patients; inner-val {inner_val}; "
          f"fit {len(fit_frames)} frames ({sum(f['nonempty'] for f in fit_frames)} non-empty); "
          f"inner-val {len(eval_frames)} frames. Fold val patients (never used): {sorted(val_patients)}")
    with open(join(args.output_dir, "split.json"), "w") as fp:
        json.dump({"fit_patients": sorted(set(patients) - set(inner_val)), "inner_val_patients": inner_val,
                   "fold_val_patients_unused": sorted(val_patients)}, fp, indent=2)

    loader = DataLoader(
        EpisodeDataset(fit_frames, args.image_size, args.max_shots, args.p_empty_target,
                       not args.no_augment, args.seed + 1000 * args.fold),
        batch_size=None, shuffle=False, num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0, prefetch_factor=2 if args.num_workers > 0 else None)
    eval_refs = [f for f in fit_frames if f["nonempty"]]
    episodes = build_eval_episodes(eval_frames, eval_refs, args.eval_targets, args.eval_shots, args.seed)

    # Backbone + LoRA
    weights = args.weights or join(REPO_DIR, WEIGHTS[args.model_size])
    encoder = load_encoder(args.model_size, weights).to(device).eval()
    for prm in encoder.parameters():
        prm.requires_grad = False
    depth = len(encoder.blocks)
    first = depth - min(args.lora_last_blocks, depth)
    wrapped = inject_lora(encoder, list(range(first, depth)), args.lora_targets.split(","),
                          args.rank, args.alpha, args.lora_dropout)
    params = [p for m in lora_modules(encoder) for p in (m.lora_A, m.lora_B)]
    print(f"LoRA on {len(wrapped)} linears in blocks {first}-{depth - 1}: "
          f"{sum(p.numel() for p in params):,} trainable params")

    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / args.warmup) * 0.5 * (1 + math.cos(math.pi * min(s, args.steps) / args.steps)))

    log_path = join(args.output_dir, "eval_log.csv")
    eval_fields = ["step"] + [f"dice_{k}shot" for k in args.eval_shots] + ["dice_mean"]
    with open(log_path, "w", newline="") as fp:
        csv.writer(fp).writerow(eval_fields)

    def run_eval(step):
        t0 = time.time()
        res = evaluate(encoder, episodes, args.eval_shots, args)
        with open(log_path, "a", newline="") as fp:
            csv.writer(fp).writerow([step] + [f"{res[k]:.4f}" for k in eval_fields[1:]])
        print(f"[eval] step {step}: " + ", ".join(f"{k}={v:.4f}" for k, v in res.items())
              + f" ({time.time() - t0:.0f}s)", flush=True)
        return res["dice_mean"]

    best = run_eval(0)  # baseline: unadapted DINOv3 (LoRA B=0)
    best_step = 0
    torch.save(lora_state_dict(encoder), join(args.output_dir, "lora_best.pt"))

    basis = None
    data_iter = iter(loader)
    patch = encoder.patch_size
    h = w = args.image_size // patch
    t0 = time.time()
    running = {}
    for step in range(1, args.steps + 1):
        if basis is None or (step - 1) % args.basis_every == 0:
            basis = positional_basis(encoder, args.image_size, args.svd_comps, device)
        opt.zero_grad(set_to_none=True)
        for _ in range(args.episodes_per_step):
            imgs, masks = next(data_iter)
            imgs, masks = imgs.to(device, non_blocking=True), masks.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=imgs.is_cuda):
                student, teacher = encode(encoder, imgs, first, args.grad_ckpt, args.w_gram > 0)
            loss, logs = episode_loss(student, teacher, masks, basis, h, w, args)
            (loss / args.episodes_per_step).backward()
            for k, v in logs.items():
                running[k] = running.get(k, 0.0) + v / args.episodes_per_step
        gnorm = torch.nn.utils.clip_grad_norm_(params, args.clip)
        if not torch.isfinite(gnorm):
            raise RuntimeError(f"Non-finite gradient norm at step {step}")
        opt.step()
        sched.step()

        if step % args.log_every == 0:
            el = time.time() - t0
            mem = torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0
            print(f"step {step}/{args.steps} " + " ".join(f"{k}={v / args.log_every:.4f}" for k, v in running.items())
                  + f" gnorm={gnorm:.3f} lr={sched.get_last_lr()[0]:.2e} {el / step:.2f}s/step mem={mem:.1f}G",
                  flush=True)
            running = {}

        if step % args.eval_every == 0 or step == args.steps:
            torch.save(lora_state_dict(encoder), join(args.output_dir, "lora_last.pt"))
            score = run_eval(step)
            if score > best:
                best, best_step = score, step
                torch.save(lora_state_dict(encoder), join(args.output_dir, "lora_best.pt"))

    # Merge the best adapter and save a drop-in DINOv3 checkpoint
    encoder.load_state_dict(torch.load(join(args.output_dir, "lora_best.pt")), strict=False)
    # "-<8 chars>.pth" suffix: DINOv3 hub entries parse it as a hash if the file is ever passed as weights=
    out = join(os.path.abspath(args.output_dir),
               f"{HUB_NAMES[args.model_size]}_lumpnav_lora_fold{args.fold}-lora{args.fold:04d}.pth")
    torch.save(merged_state_dict(encoder), out)
    summary = {"best_step": best_step, "best_inner_val_dice_mean": best, "merged_weights": out}
    with open(join(args.output_dir, "summary.json"), "w") as fp:
        json.dump(summary, fp, indent=2)
    print(f"Best inner-val mean Dice {best:.4f} at step {best_step}. Merged weights: {out}")
    if best_step == 0:
        print("WARNING: no adapted checkpoint beat the unadapted backbone on inner-val; merged weights = original.")


if __name__ == "__main__":
    main()
