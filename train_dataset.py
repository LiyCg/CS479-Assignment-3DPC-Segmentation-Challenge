import glob
import os

import numpy as np
import torch
from torch.utils.data import Dataset


class NubzukiTrainDataset(Dataset):
    """
    Training dataset for Nubzuki instance segmentation.

    Loads generated .npy files (scene + Nubzuki composites) and applies:
      - Point subsampling to fixed size
      - The same normalization as the evaluation dataset.py
      - Optional data augmentation (rotation, scaling, color jitter, flip)

    Returns:
      - features: [9, N] (xyz + rgb + normal)
      - instance_labels: [N], background=0, instances=1,2,...
      - sem_labels: [N], binary foreground(1) / background(0)
    """

    def __init__(
        self,
        data_dir: str,
        num_points: int = 80000,
        augment: bool = True,
        bg_color_jitter: bool = False,
        split: str = "train",
    ):
        self.data_dir = data_dir
        self.num_points = num_points
        self.augment = augment
        self.bg_color_jitter = bg_color_jitter

        if split in ("train", "val"):
            search_dir = os.path.join(data_dir, split)
        else:
            search_dir = data_dir

        self.files = sorted(glob.glob(os.path.join(search_dir, "**", "*.npy"), recursive=True))
        if len(self.files) == 0:
            raise ValueError(f"No .npy files found under: {search_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = np.load(self.files[idx], allow_pickle=True).item()

        xyz = np.asarray(data["xyz"], dtype=np.float32)
        rgb = np.asarray(data["rgb"], dtype=np.float32)
        if rgb.max() > 1.0:
            rgb = rgb / 255.0
        normal = np.asarray(data["normal"], dtype=np.float32)
        instance_labels = np.asarray(data["instance_labels"], dtype=np.int64)

        # Binary semantic label: 0=background, 1=foreground
        sem_labels = (instance_labels > 0).astype(np.int64)

        # --- Subsample to fixed number of points ---
        n = xyz.shape[0]
        if n >= self.num_points:
            indices = np.random.choice(n, self.num_points, replace=False)
        else:
            indices = np.random.choice(n, self.num_points, replace=True)

        xyz = xyz[indices]
        rgb = rgb[indices]
        normal = normal[indices]
        instance_labels = instance_labels[indices]
        sem_labels = sem_labels[indices]

        # --- Data augmentation (before normalization, on raw coords) ---
        if self.augment:
            xyz, rgb, normal = self._augment(xyz, rgb, normal, instance_labels)

        # --- Normalization (must match dataset.py exactly) ---
        centroid = np.mean(xyz, axis=0)
        xyz = xyz - centroid
        radius = np.max(np.sqrt(np.sum(xyz ** 2, axis=1)))
        if radius > 1e-8:
            xyz = xyz / radius

        normal_norm = np.linalg.norm(normal, axis=1, keepdims=True)
        normal = np.divide(normal, normal_norm, out=normal, where=normal_norm != 0)

        # --- Pack features [9, N] ---
        features = np.concatenate([xyz, rgb, normal], axis=1).T  # [9, N]

        return {
            "features": torch.tensor(features, dtype=torch.float32),
            "instance_labels": torch.tensor(instance_labels, dtype=torch.long),
            "sem_labels": torch.tensor(sem_labels, dtype=torch.long),
        }

    def _augment(self, xyz, rgb, normal, instance_labels):
        # 1) Random rotation around z-axis
        theta = np.random.uniform(0, 2 * np.pi)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        rot_z = np.array([
            [cos_t, -sin_t, 0],
            [sin_t,  cos_t, 0],
            [0,      0,     1],
        ], dtype=np.float32)
        xyz = xyz @ rot_z.T
        normal = normal @ rot_z.T

        # 2) Random flip along x or y axis
        if np.random.random() < 0.5:
            xyz[:, 0] = -xyz[:, 0]
            normal[:, 0] = -normal[:, 0]
        if np.random.random() < 0.5:
            xyz[:, 1] = -xyz[:, 1]
            normal[:, 1] = -normal[:, 1]

        # 3) Random isotropic scaling
        scale = np.random.uniform(0.8, 1.2)
        xyz = xyz * scale

        # 4) Independent color jitter per Nubzuki instance (matches test generation)
        # Background: optionally jitter (off by default, test data has no bg jitter)
        if self.bg_color_jitter:
            bg_mask = instance_labels == 0
            bg_jitter = np.random.uniform(0.8, 1.2, size=(1, 3)).astype(np.float32)
            rgb[bg_mask] = rgb[bg_mask] * bg_jitter

        # Each Nubzuki instance: independent jitter
        for inst_id in np.unique(instance_labels):
            if inst_id == 0:
                continue
            inst_mask = instance_labels == inst_id
            inst_jitter = np.random.uniform(0.5, 2.5, size=(1, 3)).astype(np.float32)
            rgb[inst_mask] = rgb[inst_mask] * inst_jitter

        rgb = np.clip(rgb, 0.0, 1.0)

        # 5) Random color drop (with small probability)
        if np.random.random() < 0.1:
            rgb = np.zeros_like(rgb)

        return xyz, rgb, normal
