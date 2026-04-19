import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import SegmentationModel, voxelize, VOXEL_SIZE
from losses import InstanceSegLoss
from train_dataset import NubzukiTrainDataset


def train_one_epoch(model, loader, loss_fn, optimizer, device, epoch):
    model.train()
    total_loss = 0.0
    loss_counts = {"semantic": 0.0, "embedding": 0.0, "pull": 0.0, "push": 0.0}
    num_batches = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch}")
    for batch in pbar:
        features = batch["features"].to(device)        # [B, 9, N]
        instance_labels = batch["instance_labels"]      # [B, N]

        B = features.shape[0]
        batch_sem_logits = []
        batch_embeddings = []
        batch_labels = []

        for b in range(B):
            feat = features[b]          # [9, N]
            xyz = feat[:3].T            # [N, 3]
            rgb = feat[3:6].T           # [N, 3]
            normal = feat[6:9].T        # [N, 3]
            point_feat = torch.cat([rgb, normal], dim=1)  # [N, 6]

            sparse_input, inverse_map = voxelize(
                xyz, point_feat, voxel_size=VOXEL_SIZE, device=device
            )

            sem_logits, embeddings = model(sparse_input)  # [M, 2], [M, E]

            # Map voxel labels from points: majority vote per voxel
            pt_labels = instance_labels[b].to(device)  # [N]
            num_voxels = sem_logits.shape[0]
            # Use the label of the first point mapped to each voxel
            perm = torch.arange(inverse_map.shape[0], device=device)
            first_idx = torch.zeros(num_voxels, device=device, dtype=torch.long)
            first_idx.scatter_(0, inverse_map, perm)
            voxel_labels = pt_labels[first_idx]  # [M]

            batch_sem_logits.append(sem_logits)
            batch_embeddings.append(embeddings)
            batch_labels.append(voxel_labels)

        # Concatenate all voxels in batch
        all_sem = torch.cat(batch_sem_logits, dim=0)
        all_embed = torch.cat(batch_embeddings, dim=0)
        all_labels = torch.cat(batch_labels, dim=0)

        loss, loss_dict = loss_fn(all_sem, all_embed, all_labels)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()

        total_loss += loss_dict["total"]
        for k in loss_counts:
            loss_counts[k] += loss_dict.get(k, 0.0)
        num_batches += 1

        pbar.set_postfix({
            "loss": f"{loss_dict['total']:.4f}",
            "sem": f"{loss_dict['semantic']:.4f}",
            "emb": f"{loss_dict['embedding']:.4f}",
        })

    avg = {k: v / max(num_batches, 1) for k, v in loss_counts.items()}
    avg["total"] = total_loss / max(num_batches, 1)
    return avg


@torch.no_grad()
def validate(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    loss_counts = {"semantic": 0.0, "embedding": 0.0}
    num_batches = 0

    # Also track semantic accuracy
    correct = 0
    total_pts = 0

    for batch in tqdm(loader, desc="Validation"):
        features = batch["features"].to(device)
        instance_labels = batch["instance_labels"]
        B = features.shape[0]

        batch_sem_logits = []
        batch_embeddings = []
        batch_labels = []

        for b in range(B):
            feat = features[b]
            xyz = feat[:3].T
            rgb = feat[3:6].T
            normal = feat[6:9].T
            point_feat = torch.cat([rgb, normal], dim=1)

            sparse_input, inverse_map = voxelize(
                xyz, point_feat, voxel_size=VOXEL_SIZE, device=device
            )

            sem_logits, embeddings = model(sparse_input)

            pt_labels = instance_labels[b].to(device)
            num_voxels = sem_logits.shape[0]
            perm = torch.arange(inverse_map.shape[0], device=device)
            first_idx = torch.zeros(num_voxels, device=device, dtype=torch.long)
            first_idx.scatter_(0, inverse_map, perm)
            voxel_labels = pt_labels[first_idx]

            batch_sem_logits.append(sem_logits)
            batch_embeddings.append(embeddings)
            batch_labels.append(voxel_labels)

            # Semantic accuracy (per voxel)
            sem_pred = sem_logits.argmax(dim=1)
            sem_gt = (voxel_labels > 0).long()
            correct += (sem_pred == sem_gt).sum().item()
            total_pts += sem_gt.shape[0]

        all_sem = torch.cat(batch_sem_logits, dim=0)
        all_embed = torch.cat(batch_embeddings, dim=0)
        all_labels = torch.cat(batch_labels, dim=0)

        loss, loss_dict = loss_fn(all_sem, all_embed, all_labels)

        total_loss += loss_dict["total"]
        for k in loss_counts:
            loss_counts[k] += loss_dict.get(k, 0.0)
        num_batches += 1

    avg = {k: v / max(num_batches, 1) for k, v in loss_counts.items()}
    avg["total"] = total_loss / max(num_batches, 1)
    avg["sem_acc"] = correct / max(total_pts, 1)
    return avg


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Dataset
    train_ds = NubzukiTrainDataset(
        data_dir=args.data_dir,
        num_points=args.num_points,
        augment=True,
        bg_color_jitter=args.bg_color_jitter,
        split="train",
    )
    val_ds = NubzukiTrainDataset(
        data_dir=args.data_dir,
        num_points=args.num_points,
        augment=False,
        split="val",
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    print(f"Train: {len(train_ds)} samples, Val: {len(val_ds)} samples")

    # Model
    model = SegmentationModel(
        in_channels=6,
        base_channels=args.base_channels,
        embed_dim=args.embed_dim,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    # Loss & optimizer
    loss_fn = InstanceSegLoss(sem_weight=args.sem_weight, embed_weight=args.embed_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # Training loop
    os.makedirs(args.save_dir, exist_ok=True)
    best_val_loss = float("inf")
    train_history = []
    val_history = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_avg = train_one_epoch(model, train_loader, loss_fn, optimizer, device, epoch)
        val_avg = validate(model, val_loader, loss_fn, device)

        scheduler.step()
        lr = scheduler.get_last_lr()[0]
        elapsed = time.time() - t0

        train_history.append(train_avg)
        val_history.append(val_avg)

        print(
            f"[Epoch {epoch}/{args.epochs}] "
            f"train_loss={train_avg['total']:.4f} "
            f"val_loss={val_avg['total']:.4f} "
            f"sem_acc={val_avg['sem_acc']:.4f} "
            f"lr={lr:.6f} "
            f"time={elapsed:.1f}s"
        )

        # Save checkpoint
        ckpt = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_avg["total"],
            "val_loss": val_avg["total"],
        }

        torch.save(ckpt, os.path.join(args.save_dir, "last.pth"))

        if val_avg["total"] < best_val_loss:
            best_val_loss = val_avg["total"]
            torch.save(ckpt, os.path.join(args.save_dir, "best.pth"))
            print(f"  -> New best model saved (val_loss={best_val_loss:.4f})")

    # Save training history
    np.savez(
        os.path.join(args.save_dir, "history.npz"),
        train=[h["total"] for h in train_history],
        val=[h["total"] for h in val_history],
        sem_acc=[h["sem_acc"] for h in val_history],
    )
    print(f"\nTraining complete. Best val_loss: {best_val_loss:.4f}")
    print(f"Checkpoints saved to: {args.save_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Data
    parser.add_argument("--data-dir", type=str, default="data/generated_datas/generated_datas")
    parser.add_argument("--num-points", type=int, default=80000)
    parser.add_argument("--bg-color-jitter", action="store_true", default=False)

    # Model
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--embed-dim", type=int, default=16)

    # Training
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--sem-weight", type=float, default=1.0)
    parser.add_argument("--embed-weight", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=2)

    # Save
    parser.add_argument("--save-dir", type=str, default="checkpoints")

    main(parser.parse_args())
