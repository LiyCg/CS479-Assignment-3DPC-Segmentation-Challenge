import torch
import torch.nn as nn
import numpy as np
import spconv.pytorch as spconv
from sklearn.cluster import MeanShift


# ============================================================
# Sparse UNet Backbone (spconv-based)
# ============================================================

class ResBlock(nn.Module):
    """Sparse residual block: SubMConv3d -> BN -> ReLU -> SubMConv3d -> BN + skip."""

    def __init__(self, channels):
        super().__init__()
        self.net = spconv.SparseSequential(
            spconv.SubMConv3d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
            spconv.SubMConv3d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm1d(channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.net(x)
        out = out.replace_feature(self.relu(out.features + x.features))
        return out


class SparseUNet(nn.Module):
    """
    Sparse 3D UNet encoder-decoder with skip connections.

    Encoder: progressively downsample with SparseConv3d (stride=2)
    Decoder: progressively upsample with SparseInverseConv3d

    Channel progression: in_channels -> 32 -> 64 -> 128 -> 256 (bottleneck) -> decode back
    """

    def __init__(self, in_channels=6, base_channels=32):
        super().__init__()
        ch = base_channels  # 32

        # --- Encoder ---
        self.input_conv = spconv.SparseSequential(
            spconv.SubMConv3d(in_channels, ch, 3, padding=1, bias=False),
            nn.BatchNorm1d(ch),
            nn.ReLU(inplace=True),
        )
        self.enc1 = ResBlock(ch)  # 32

        self.down1 = spconv.SparseSequential(
            spconv.SparseConv3d(ch, ch * 2, 2, stride=2, bias=False, indice_key="down1"),
            nn.BatchNorm1d(ch * 2),
            nn.ReLU(inplace=True),
        )
        self.enc2 = ResBlock(ch * 2)  # 64

        self.down2 = spconv.SparseSequential(
            spconv.SparseConv3d(ch * 2, ch * 4, 2, stride=2, bias=False, indice_key="down2"),
            nn.BatchNorm1d(ch * 4),
            nn.ReLU(inplace=True),
        )
        self.enc3 = ResBlock(ch * 4)  # 128

        self.down3 = spconv.SparseSequential(
            spconv.SparseConv3d(ch * 4, ch * 8, 2, stride=2, bias=False, indice_key="down3"),
            nn.BatchNorm1d(ch * 8),
            nn.ReLU(inplace=True),
        )
        self.bottleneck = ResBlock(ch * 8)  # 256

        # --- Decoder ---
        self.up3 = spconv.SparseSequential(
            spconv.SparseInverseConv3d(ch * 8, ch * 4, 2, bias=False, indice_key="down3"),
            nn.BatchNorm1d(ch * 4),
            nn.ReLU(inplace=True),
        )
        self.skip_proj3 = nn.Sequential(nn.Linear(ch * 8, ch * 4), nn.ReLU(inplace=True))
        self.dec3 = ResBlock(ch * 4)

        self.up2 = spconv.SparseSequential(
            spconv.SparseInverseConv3d(ch * 4, ch * 2, 2, bias=False, indice_key="down2"),
            nn.BatchNorm1d(ch * 2),
            nn.ReLU(inplace=True),
        )
        self.skip_proj2 = nn.Sequential(nn.Linear(ch * 4, ch * 2), nn.ReLU(inplace=True))
        self.dec2 = ResBlock(ch * 2)

        self.up1 = spconv.SparseSequential(
            spconv.SparseInverseConv3d(ch * 2, ch, 2, bias=False, indice_key="down1"),
            nn.BatchNorm1d(ch),
            nn.ReLU(inplace=True),
        )
        self.skip_proj1 = nn.Sequential(nn.Linear(ch * 2, ch), nn.ReLU(inplace=True))
        self.dec1 = ResBlock(ch)

        self.out_channels = ch  # 32

    def forward(self, x):
        # Encoder
        x0 = self.input_conv(x)
        x1 = self.enc1(x0)

        x2 = self.down1(x1)
        x2 = self.enc2(x2)

        x3 = self.down2(x2)
        x3 = self.enc3(x3)

        x4 = self.down3(x3)
        x4 = self.bottleneck(x4)

        # Decoder with skip connections
        d3 = self.up3(x4)
        d3 = d3.replace_feature(
            self.skip_proj3(torch.cat([d3.features, x3.features], dim=1))
        )
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = d2.replace_feature(
            self.skip_proj2(torch.cat([d2.features, x2.features], dim=1))
        )
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = d1.replace_feature(
            self.skip_proj1(torch.cat([d1.features, x1.features], dim=1))
        )
        d1 = self.dec1(d1)

        return d1


# ============================================================
# Segmentation Model (backbone + heads)
# ============================================================

class SegmentationModel(nn.Module):
    """
    Instance segmentation model:
      - Sparse UNet backbone -> per-voxel features
      - Semantic head: binary foreground/background
      - Embedding head: discriminative embedding for instance clustering
    """

    def __init__(self, in_channels=6, base_channels=32, embed_dim=16):
        super().__init__()
        self.backbone = SparseUNet(in_channels=in_channels, base_channels=base_channels)

        feat_ch = self.backbone.out_channels

        self.semantic_head = nn.Sequential(
            nn.Linear(feat_ch, feat_ch),
            nn.ReLU(inplace=True),
            nn.Linear(feat_ch, 2),
        )

        self.embed_head = nn.Sequential(
            nn.Linear(feat_ch, feat_ch),
            nn.ReLU(inplace=True),
            nn.Linear(feat_ch, embed_dim),
        )

        self.embed_dim = embed_dim

    def forward(self, sparse_input):
        out = self.backbone(sparse_input)
        feats = out.features

        sem_logits = self.semantic_head(feats)
        embeddings = self.embed_head(feats)

        return sem_logits, embeddings


# ============================================================
# Voxelization utilities
# ============================================================

def voxelize(xyz, features, voxel_size=0.02, device='cuda'):
    """
    Convert a single point cloud to spconv.SparseConvTensor.

    Args:
        xyz: [N, 3] point coordinates
        features: [N, C] point features (rgb + normal)
        voxel_size: voxel grid resolution
        device: target device

    Returns:
        sparse_tensor: spconv.SparseConvTensor
        inverse_map: [N] maps each point to its voxel index
    """
    coords = torch.floor(xyz / voxel_size).int()
    coords_min = coords.min(dim=0).values
    coords = coords - coords_min

    spatial_shape = (coords.max(dim=0).values + 1).tolist()

    coord_hash = coords[:, 0].long() * (spatial_shape[1] * spatial_shape[2]) + \
                 coords[:, 1].long() * spatial_shape[2] + coords[:, 2].long()

    _, inverse, counts = torch.unique(coord_hash, return_inverse=True, return_counts=True)
    num_voxels = counts.shape[0]

    # Scatter-mean features into voxels
    voxel_features = torch.zeros(num_voxels, features.shape[1], device=device)
    voxel_features.scatter_add_(0, inverse.unsqueeze(1).expand(-1, features.shape[1]), features)
    voxel_features = voxel_features / counts.unsqueeze(1).float()

    # Voxel coords: use first occurrence
    perm = torch.arange(inverse.shape[0], device=device)
    first_idx = torch.zeros(num_voxels, device=device, dtype=torch.long)
    first_idx.scatter_(0, inverse, perm)
    voxel_coords = coords[first_idx]

    batch_idx = torch.zeros(num_voxels, 1, device=device, dtype=torch.int32)
    voxel_coords_batch = torch.cat([batch_idx, voxel_coords], dim=1)

    sparse_tensor = spconv.SparseConvTensor(
        features=voxel_features,
        indices=voxel_coords_batch,
        spatial_shape=spatial_shape,
        batch_size=1,
    )

    return sparse_tensor, inverse


# ============================================================
# Clustering for inference
# ============================================================

def cluster_embeddings(embeddings, bandwidth=0.5):
    """
    Cluster foreground point embeddings into instance IDs using MeanShift.

    Args:
        embeddings: [M, E] numpy array
    Returns:
        instance_ids: [M] numpy array, starting from 1
    """
    if len(embeddings) == 0:
        return np.array([], dtype=np.int64)

    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True, n_jobs=-1)
    ms.fit(embeddings)
    labels = ms.labels_ + 1  # 0-based -> 1-based
    return labels.astype(np.int64)


# ============================================================
# Required interface: initialize_model / run_inference
# ============================================================

VOXEL_SIZE = 0.02
EMBED_DIM = 16
BASE_CHANNELS = 32
FG_THRESHOLD = 0.5
CLUSTER_BANDWIDTH = 0.5


def initialize_model(
    ckpt_path: str,
    device: torch.device,
    in_channels: int = 9,
    num_classes: int = 2,
) -> nn.Module:
    # xyz is used for voxel coordinates, rgb+normal (6ch) as input features
    model = SegmentationModel(
        in_channels=6,
        base_channels=BASE_CHANNELS,
        embed_dim=EMBED_DIM,
    ).to(device)

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
    **kwargs,
) -> torch.Tensor:
    """
    Args:
        model: SegmentationModel
        features: [B, 9, N] (xyz, rgb, normal)

    Returns:
        instance_labels: [B, N] — 0=background, 1,2,...=instances
    """
    device = features.device
    B, _, N = features.shape
    result = torch.zeros(B, N, dtype=torch.long, device=device)

    for b in range(B):
        feat = features[b]      # [9, N]
        xyz = feat[:3].T        # [N, 3]
        rgb = feat[3:6].T       # [N, 3]
        normal = feat[6:9].T    # [N, 3]

        point_features = torch.cat([rgb, normal], dim=1)  # [N, 6]

        # Voxelize
        sparse_input, inverse_map = voxelize(
            xyz, point_features, voxel_size=VOXEL_SIZE, device=device
        )

        # Forward
        sem_logits, embeddings = model(sparse_input)

        # Map voxel predictions back to points
        point_sem = sem_logits[inverse_map]
        point_embed = embeddings[inverse_map]

        # Foreground mask
        fg_prob = torch.softmax(point_sem, dim=1)[:, 1]
        fg_mask = fg_prob > FG_THRESHOLD

        # Cluster foreground embeddings
        labels = torch.zeros(N, dtype=torch.long, device=device)
        if fg_mask.sum() > 0:
            fg_embeddings = point_embed[fg_mask].detach().cpu().numpy()
            instance_ids = cluster_embeddings(fg_embeddings, bandwidth=CLUSTER_BANDWIDTH)
            labels[fg_mask] = torch.from_numpy(instance_ids).to(device)

        result[b] = labels

    return result
