"""In-context segmentation inference script with INSID3."""

import argparse
import datetime
import os
import random
import sys
import time
from os.path import join

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from PIL import Image
from tqdm import tqdm

import opts
from datasets import build_dataset
from models import build_insid3_from_args
from utils.metrics import Evaluator, AverageMeter


def main(args: argparse.Namespace) -> float:
    print(args)

    # ──────── Reproducibility and logging setup ────────
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available() and torch.cuda.get_device_properties(0).major >= 8:
        # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    log_file = join(args.output_dir, 'log.txt')
    with open(log_file, 'w') as fp:
        fp.write(" ".join(sys.argv) + '\n')
        fp.write(str(vars(args)) + '\n\n')

    # ──────── Model setup ────────
    model = build_insid3_from_args(args)
    model.to(args.device)
    model.eval()

    print(f'Parameters: {sum(p.numel() for p in model.parameters()):,}')
    print('Start inference')

    start_time = time.time()
    miou = evaluate(args, model, log_file)
    print(f'Total inference time: {time.time() - start_time:.1f}s')
    return miou


def evaluate(args: argparse.Namespace, model: torch.nn.Module, log_file: str) -> float:
    # ──────── Dataset and loader setup ────────
    ds = build_dataset(args.dataset, args=args)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=args.num_workers,
                        collate_fn=lambda x: x[0])
    meter = AverageMeter(args.dataset, ds.class_ids, device=args.device)
    pred_dir = join(args.output_dir, 'pred_masks')
    if args.save_preds:
        os.makedirs(pred_dir, exist_ok=True)

    # ──────── Evaluation loop ────────
    pbar = tqdm(loader, ncols=80)
    for idx, batch in enumerate(pbar):

        ref_imgs = batch['ref_imgs']    # list of PIL Images
        ref_masks = batch['ref_masks']  # list of tensors
        tgt_img = batch['tgt_img']      # PIL Image
        tgt_mask = batch['tgt_mask']    # tensor

        # Set all references
        for i in range(len(ref_imgs)):
            model.set_reference(ref_imgs[i], ref_masks[i])
        # Set target
        model.set_target(tgt_img)
        # Segment
        pred_mask = model.segment()
        if args.save_preds:
            w, h = tgt_img.size   # original frame size
            pred_orig = F.interpolate(pred_mask.float()[None, None], size=(h, w), mode='nearest')[0, 0] > 0.5
            name = batch.get('tgt_name', f'{idx:06d}')
            Image.fromarray(pred_orig.cpu().numpy().astype(np.uint8) * 255).save(join(pred_dir, f'{name}.png'))

        tgt_mask = F.interpolate(
            tgt_mask.unsqueeze(0).unsqueeze(0).float(),
            size=pred_mask.shape, mode='nearest',
        ).squeeze(0).squeeze(0) > 0.5

        area_inter, area_union = Evaluator.classify_prediction(
            pred_mask, tgt_mask,
            tgt_ignore_idx=batch.get('tgt_ignore_idx'),
        )
        meter.update(area_inter, area_union, batch['class_id'].to(args.device))

        if (idx + 1) % 50 == 0:
            miou = meter.compute_iou()[0]
            pbar.set_description(f'mIoU: {miou:.1f}')

    # ──────── Final results ────────
    miou = meter.compute_iou()[0].item()
    out_str = f'mIoU = {miou:.1f}'
    print(out_str)
    with open(log_file, 'a') as fp:
        fp.write(out_str + '\n')
    return miou


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        'INSID3 inference on in-context segmentation',
        parents=[opts.get_args_parser()],
    )
    args = parser.parse_args()
    timestamp = datetime.datetime.now().strftime('%m%d_%H%M')
    args.output_dir = join(args.output_dir, f'{args.exp_name}_{timestamp}')
    os.makedirs(args.output_dir, exist_ok=True)
    main(args)
