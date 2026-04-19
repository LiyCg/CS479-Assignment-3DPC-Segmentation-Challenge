import torch
import torch.nn as nn


class DiscriminativeLoss(nn.Module):
    """
    Discriminative loss for instance embedding (de Brabandere et al., 2017).

    Three terms:
      - L_pull: pull embeddings toward their instance mean
      - L_push: push different instance means apart
      - L_reg:  regularize embedding magnitudes

    Args:
        delta_pull: margin for pull loss (embeddings within this distance are free)
        delta_push: margin for push loss (means beyond this distance are free)
        alpha, beta, gamma: weights for pull, push, reg
    """

    def __init__(self, delta_pull=0.5, delta_push=1.5, alpha=1.0, beta=1.0, gamma=0.001):
        super().__init__()
        self.delta_pull = delta_pull
        self.delta_push = delta_push
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def forward(self, embeddings, instance_labels):
        """
        Args:
            embeddings: [M, E] per-voxel embeddings
            instance_labels: [M] instance IDs (0=background, 1,2,...=instances)

        Returns:
            loss: scalar
            loss_dict: dict with individual loss components
        """
        # Only compute on foreground points
        fg_mask = instance_labels > 0
        if fg_mask.sum() == 0:
            zero = torch.tensor(0.0, device=embeddings.device, requires_grad=True)
            return zero, {"pull": 0.0, "push": 0.0, "reg": 0.0}

        fg_embed = embeddings[fg_mask]          # [F, E]
        fg_labels = instance_labels[fg_mask]    # [F]

        unique_ids = torch.unique(fg_labels)
        C = len(unique_ids)  # number of instances

        if C == 0:
            zero = torch.tensor(0.0, device=embeddings.device, requires_grad=True)
            return zero, {"pull": 0.0, "push": 0.0, "reg": 0.0}

        # Compute instance means
        means = []
        for inst_id in unique_ids:
            mask = fg_labels == inst_id
            means.append(fg_embed[mask].mean(dim=0))
        means = torch.stack(means)  # [C, E]

        # --- L_pull: pull embeddings to their instance mean ---
        pull_losses = []
        for i, inst_id in enumerate(unique_ids):
            mask = fg_labels == inst_id
            inst_embed = fg_embed[mask]  # [Ni, E]
            dist = torch.norm(inst_embed - means[i].unsqueeze(0), dim=1)  # [Ni]
            pull = torch.clamp(dist - self.delta_pull, min=0.0) ** 2
            pull_losses.append(pull.mean())
        pull_loss = torch.stack(pull_losses).mean()

        # --- L_push: push different instance means apart ---
        if C > 1:
            push_losses = []
            for i in range(C):
                for j in range(i + 1, C):
                    dist = torch.norm(means[i] - means[j])
                    push = torch.clamp(2 * self.delta_push - dist, min=0.0) ** 2
                    push_losses.append(push)
            push_loss = torch.stack(push_losses).mean()
        else:
            push_loss = torch.zeros(1, device=embeddings.device, requires_grad=True).squeeze()

        # --- L_reg: regularize mean magnitudes ---
        reg_loss = torch.mean(torch.norm(means, dim=1))

        total = self.alpha * pull_loss + self.beta * push_loss + self.gamma * reg_loss

        loss_dict = {
            "pull": pull_loss.item(),
            "push": push_loss.item(),
            "reg": reg_loss.item(),
        }
        return total, loss_dict


class InstanceSegLoss(nn.Module):
    """
    Combined loss for instance segmentation:
      - Semantic loss: CrossEntropy for fg/bg classification
      - Embedding loss: Discriminative loss for instance clustering

    Args:
        sem_weight: weight for semantic loss
        embed_weight: weight for embedding loss
    """

    def __init__(self, sem_weight=1.0, embed_weight=1.0):
        super().__init__()
        self.sem_weight = sem_weight
        self.embed_weight = embed_weight
        self.sem_loss_fn = nn.CrossEntropyLoss()
        self.embed_loss_fn = DiscriminativeLoss()

    def forward(self, sem_logits, embeddings, instance_labels):
        """
        Args:
            sem_logits: [M, 2] semantic logits (fg/bg)
            embeddings: [M, E] instance embeddings
            instance_labels: [M] ground truth (0=bg, 1,2,...=instances)

        Returns:
            total_loss: scalar
            loss_dict: dict with breakdown
        """
        # Semantic: binary fg/bg
        sem_targets = (instance_labels > 0).long()
        sem_loss = self.sem_loss_fn(sem_logits, sem_targets)

        # Embedding: discriminative
        embed_loss, embed_dict = self.embed_loss_fn(embeddings, instance_labels)

        total = self.sem_weight * sem_loss + self.embed_weight * embed_loss

        loss_dict = {
            "total": total.item(),
            "semantic": sem_loss.item(),
            "embedding": embed_loss.item(),
            "pull": embed_dict["pull"],
            "push": embed_dict["push"],
            "reg": embed_dict["reg"],
        }

        return total, loss_dict
