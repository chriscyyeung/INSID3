"""LumpNav breast ultrasound in-context segmentation dataset (patient-wise CV folds).

Expects the 5-fold split produced by generate_5fold_cv_split.py:

    <data-root>/fold_<k>/train/images/LNXXX_frameNNNNN.png
    <data-root>/fold_<k>/train/masks/LNXXX_frameNNNNN.png
    <data-root>/fold_<k>/val/images/...
    <data-root>/fold_<k>/val/masks/...

Targets are every frame of the fold's validation patients (deterministic order).
References are sampled from the fold's training patients only, and only from
frames with a non-empty tumour mask, so no validation patient is ever a reference.
"""
from __future__ import annotations

import os
import glob
import re

from torch.utils.data import Dataset
import torch
import PIL.Image as Image
import numpy as np

_FRAME_RE = re.compile(r'^(LN\d+)_frame(\d+)(?:_0000)?$')


class DatasetLumpNav(Dataset):
    def __init__(self, datapath: str, fold: int, shot: int, seed: int = 0,
                 nonempty_targets_only: bool = False, max_targets: int = 0):
        self.benchmark = 'lumpnav'
        self.shot = shot
        self.seed = seed

        self.base_path = os.path.join(datapath, f'fold_{fold}')
        self.categories = ['tumor']
        self.class_ids = range(0, 1)

        trn_pairs = self.list_pairs('train')
        val_pairs = self.list_pairs('val')

        # References: non-empty training frames only (an empty reference mask has no prototype)
        self.ref_pool = [(img, msk) for img, msk in trn_pairs if self.read_mask(msk).any()]
        if len(self.ref_pool) < shot:
            raise RuntimeError(f'Only {len(self.ref_pool)} non-empty training frames in {self.base_path}')

        trn_patients = {self.patient_id(m) for _, m in trn_pairs}
        val_patients = {self.patient_id(m) for _, m in val_pairs}
        overlap = trn_patients & val_patients
        if overlap:
            raise RuntimeError(f'Patients in both train and val of fold {fold}: {sorted(overlap)}')

        if nonempty_targets_only:
            val_pairs = [(img, msk) for img, msk in val_pairs if self.read_mask(msk).any()]
        if max_targets > 0 and len(val_pairs) > max_targets:
            keep = np.random.default_rng(seed).choice(len(val_pairs), max_targets, replace=False)
            val_pairs = [val_pairs[i] for i in sorted(keep)]
        self.tgt_pairs = val_pairs

        print(f'LumpNav fold {fold}: {len(self.tgt_pairs)} targets from {len(val_patients)} val patients, '
              f'{len(self.ref_pool)} non-empty reference frames from {len(trn_patients)} train patients')

    def __len__(self) -> int:
        return len(self.tgt_pairs)

    def __getitem__(self, idx: int) -> dict:
        tgt_img_path, tgt_mask_path = self.tgt_pairs[idx]
        # Per-target RNG: the same references for a target on every run, independent of loader order
        rng = np.random.default_rng([self.seed, idx])
        ref_ids = rng.choice(len(self.ref_pool), self.shot, replace=False)
        refs = [self.ref_pool[i] for i in ref_ids]

        batch = {'tgt_img': Image.open(tgt_img_path).convert('RGB'),
                 'tgt_mask': self.read_mask(tgt_mask_path).float(),
                 'ref_imgs': [Image.open(img).convert('RGB') for img, _ in refs],
                 'ref_masks': [self.read_mask(msk) for _, msk in refs],
                 'class_id': torch.tensor(0),
                 'tgt_name': self.frame_name(tgt_mask_path),
                 'ref_names': [self.frame_name(msk) for _, msk in refs]}
        return batch

    def list_pairs(self, split: str) -> list[tuple[str, str]]:
        img_dir = os.path.join(self.base_path, split, 'images')
        msk_dir = os.path.join(self.base_path, split, 'masks')
        images = {self.frame_name(p): p for p in glob.glob(os.path.join(img_dir, '*.png'))}
        masks = {self.frame_name(p): p for p in glob.glob(os.path.join(msk_dir, '*.png'))}
        missing = sorted(set(masks) - set(images))
        if missing:
            raise RuntimeError(f'{len(missing)} masks in {msk_dir} have no image, e.g. {missing[:3]}')
        if not masks:
            raise RuntimeError(f'No masks found in {msk_dir}')
        return [(images[k], masks[k]) for k in sorted(masks)]

    @staticmethod
    def frame_name(path: str) -> str:
        # Strip the nnU-Net channel suffix so images and masks pair up either way
        name = os.path.splitext(os.path.basename(path))[0]
        return name[:-5] if name.endswith('_0000') else name

    @staticmethod
    def patient_id(path: str) -> str:
        m = _FRAME_RE.match(DatasetLumpNav.frame_name(path))
        if m is None:
            raise ValueError(f'Unexpected frame name: {path}')
        return m.group(1)

    @staticmethod
    def read_mask(img_name: str) -> torch.Tensor:
        # Any non-zero pixel is tumour: works for 0/255, {0,1} and mode "1" (boolean) PNGs
        mask = np.array(Image.open(img_name)) > 0
        if mask.ndim == 3:
            mask = mask.any(axis=-1)
        return torch.from_numpy(mask).to(torch.uint8)


def build(args) -> DatasetLumpNav:
    dataset = DatasetLumpNav(datapath=args.data_root, fold=args.fold, shot=args.shots, seed=args.seed,
                             nonempty_targets_only=getattr(args, 'nonempty_targets_only', False),
                             max_targets=getattr(args, 'max_targets', 0))
    return dataset
