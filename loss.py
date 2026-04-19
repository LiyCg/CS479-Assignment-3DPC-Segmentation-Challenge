import torch
import torch.nn as nn


def semantic_loss(pred, gt):
    """
        pred["sem_labels"]:
            logits of shape [B, N] or [B, 1, N]

        gt["sem_labels"]:
            binary labels of shape [B, N], values in {0, 1}
    """
    pred_sem = pred["sem_labels"].float()
    gt_sem = gt["sem_labels"].float()

    # [B, 1, N] -> [B, N]
    if pred_sem.dim() == 3 and pred_sem.size(1) == 1:
        pred_sem = pred_sem.squeeze(1)

    if pred_sem.shape != gt_sem.shape:
        raise ValueError(f"Shape mismatch in semantic_loss: pred {pred_sem.shape}, gt {gt_sem.shape}")

    criterion = nn.BCEWithLogitsLoss(reduction="mean")
    loss = criterion(pred_sem, gt_sem)

    return loss


_loss_matching_ = {
    "semantic": semantic_loss,
}

def get_loss_function(ltype):
    if ltype not in _loss_matching_:
        raise ValueError(f"Unknown loss type: {ltype}")
    return _loss_matching_[ltype]


def compute_loss(loss_cfg, pred, gt):
    losses = {}

    device = next(iter(pred.values())).device
    loss_sum = torch.tensor(0.0, device=device)

    for key, weight in loss_cfg.items():
        if weight <= 1e-8:
            continue

        ftn = get_loss_function(key)

        try:
            loss = ftn(pred, gt)
        except Exception as e:
            print(f"{key} error occured: {e}")
            raise

        if torch.isnan(loss).any():
            raise ValueError(f"{key} NaN occured")

        weighted_loss = loss * weight
        losses[key] = weighted_loss
        loss_sum = loss_sum + weighted_loss

    losses["total"] = loss_sum
    return losses