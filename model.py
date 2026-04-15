import torch
import torch.nn as nn
from scipy.spatial import cKDTree


class DummyModel(nn.Module):
    """
    Instance segmentation baseline model.
    """
    
    # ! [CHANGE HERE]
    def __init__(self, cfg):
        super().__init__()
        
        self.in_channels = cfg.input_channel
        self.conv = torch.nn.Conv1d(self.in_channels, 1, 1)

    # ! [CHANGE HERE]    
    def forward(self, input):
        return self.conv(input)


def initialize_model(
    ckpt_path: str,
    device: torch.device,
    in_channels: int = 9,
    num_classes: int = 2,
) -> nn.Module:
    model = DummyModel.to(device)

    checkpoint = torch.load(ckpt_path, map_location=device)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        checkpoint = checkpoint["model_state_dict"]

    if isinstance(checkpoint, dict) and any(k.startswith("module.") for k in checkpoint.keys()):
        checkpoint = {k.replace("module.", "", 1): v for k, v in checkpoint.items()}

    model.load_state_dict(checkpoint, strict=False)
    model.eval()
    return model


def run_inference(
    model: nn.Module,
    features: torch.Tensor,
    **kwargs
) -> torch.Tensor:
    """
    Returns per-point instance labels [B, N], background=0, instances start at 1.
    """
    

    return torch.zeros_like(features[:, 0, :])
