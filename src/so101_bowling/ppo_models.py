"""Dual-camera Asymmetric PPO networks and standalone deployable actor."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.distributions import Normal

ACTOR_OBS_KEYS: tuple[str, ...] = (
    "wrist_rgb",
    "tripod_rgb",
    "camera_meta",
    "proprio",
)
PRIVILEGED_STATE_DIM = 46
PROPRIO_DIM = 12
CAMERA_META_DIM = 4
AUX_TARGET_DIM = 6
AUX_MASK_DIM = 3
ACTION_DIM = 4


def _ortho_init(layer: nn.Linear, std: float = math.sqrt(2.0)) -> nn.Linear:
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, 0.0)
    return layer


def rgb_stack_to_chw(x: Tensor) -> Tensor:
    """Convert (B, k, H, W, 3) uint8/float tensor into (B, 3*k, H, W) float32 in [0, 1]."""
    b, k, h, w, c = x.shape
    chw = x.permute(0, 1, 4, 2, 3).reshape(b, k * c, h, w)
    if not chw.is_floating_point():
        return chw.to(dtype=torch.float32) * (1.0 / 255.0)
    return chw.to(dtype=torch.float32)


def apply_view_shift(x: Tensor, shifts: Tensor | None) -> Tensor:
    """Apply per-sample integer (dy, dx) edge-padded shift identically across channels."""
    if shifts is None:
        return x
    shifts = shifts.to(device=x.device, dtype=torch.long)
    if not torch.any(shifts):
        return x
    b, _, h, w = x.shape
    dy = shifts[:, 0].view(b, 1, 1)
    dx = shifts[:, 1].view(b, 1, 1)
    ys = (torch.arange(h, device=x.device).view(1, h, 1) - dy).clamp(0, h - 1)
    xs = (torch.arange(w, device=x.device).view(1, 1, w) - dx).clamp(0, w - 1)
    b_idx = torch.arange(b, device=x.device).view(b, 1, 1)
    return x.permute(0, 2, 3, 1)[b_idx, ys, xs].permute(0, 3, 1, 2).contiguous()


class SpatialCameraEncoder(nn.Module):
    """Lightweight strided ConvNet preserving 2D spatial layout without BatchNorm."""

    def __init__(
        self,
        frame_stack: int = 2,
        out_dim: int = 128,
        encoder_type: str = "spatial_grid",
        img_hw: tuple[int, int] = (64, 112),
    ) -> None:
        super().__init__()
        if encoder_type not in ("spatial_grid", "spatial_softmax"):
            raise ValueError(f"Unsupported encoder_type: {encoder_type}")
        self.frame_stack = frame_stack
        self.out_dim = out_dim
        self.encoder_type = encoder_type

        in_ch = 3 * frame_stack
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 32, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
            nn.Conv2d(32, 48, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 48),
            nn.SiLU(),
            nn.Conv2d(48, 64, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
        )
        feat_h = (img_hw[0] + 15) // 16
        feat_w = (img_hw[1] + 15) // 16

        if encoder_type == "spatial_grid":
            self.grid_proj = nn.Conv2d(64, 8, kernel_size=1)
            head_in = 8 * feat_h * feat_w
        else:
            self.grid_proj = None
            pos_y = torch.linspace(-1.0, 1.0, feat_h).view(1, 1, feat_h, 1)
            pos_x = torch.linspace(-1.0, 1.0, feat_w).view(1, 1, 1, feat_w)
            self.register_buffer("pos_y", pos_y.expand(1, 1, feat_h, feat_w).clone())
            self.register_buffer("pos_x", pos_x.expand(1, 1, feat_h, feat_w).clone())
            head_in = 64 * 2

        self.head = nn.Sequential(
            _ortho_init(nn.Linear(head_in, out_dim)),
            nn.LayerNorm(out_dim),
            nn.SiLU(),
        )

    def forward(self, x_chw: Tensor) -> Tensor:
        feat = self.conv(x_chw)
        if self.encoder_type == "spatial_grid":
            assert self.grid_proj is not None
            spatial = self.grid_proj(feat).flatten(start_dim=1)
        else:
            b, c, h, w = feat.shape
            weights = F.softmax(feat.view(b, c, h * w), dim=-1).view(b, c, h, w)
            exp_x = (weights * self.pos_x).sum(dim=(-2, -1))
            exp_y = (weights * self.pos_y).sum(dim=(-2, -1))
            spatial = torch.cat((exp_x, exp_y), dim=-1)
        return self.head(spatial)


class DualCameraActor(nn.Module):
    """Standalone deployable actor reading ONLY dual cameras, meta, and proprio."""

    def __init__(
        self,
        frame_stack: int = 2,
        action_dim: int = ACTION_DIM,
        encoder_type: str = "spatial_grid",
        img_hw: tuple[int, int] = (64, 112),
    ) -> None:
        super().__init__()
        self.frame_stack = frame_stack
        self.action_dim = action_dim
        self.img_hw = img_hw
        self.encoder_type = encoder_type

        self.wrist_encoder = SpatialCameraEncoder(
            frame_stack=frame_stack, out_dim=128, encoder_type=encoder_type, img_hw=img_hw
        )
        self.tripod_encoder = SpatialCameraEncoder(
            frame_stack=frame_stack, out_dim=128, encoder_type=encoder_type, img_hw=img_hw
        )
        self.proprio_encoder = nn.Sequential(
            _ortho_init(nn.Linear(PROPRIO_DIM + CAMERA_META_DIM, 64)),
            nn.LayerNorm(64),
            nn.SiLU(),
        )
        self.trunk = nn.Sequential(
            _ortho_init(nn.Linear(128 + 128 + 64, 256)),
            nn.LayerNorm(256),
            nn.SiLU(),
            _ortho_init(nn.Linear(256, 256)),
            nn.LayerNorm(256),
            nn.SiLU(),
        )
        self.mean_head = _ortho_init(nn.Linear(256, action_dim), std=0.01)
        self.log_std = nn.Parameter(torch.full((action_dim,), -0.5))

    def encode_visual(
        self,
        wrist_rgb: Tensor,
        tripod_rgb: Tensor,
        wrist_shifts: Tensor | None = None,
        tripod_shifts: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        w_chw = apply_view_shift(rgb_stack_to_chw(wrist_rgb), wrist_shifts)
        t_chw = apply_view_shift(rgb_stack_to_chw(tripod_rgb), tripod_shifts)
        return self.wrist_encoder(w_chw), self.tripod_encoder(t_chw)

    def forward_actor(
        self,
        obs: Mapping[str, Tensor],
        wrist_shifts: Tensor | None = None,
        tripod_shifts: Tensor | None = None,
    ) -> tuple[Normal, Tensor]:
        # Explicitly slice ONLY deployable actor keys (never read privileged_* keys).
        wrist_rgb = obs["wrist_rgb"]
        tripod_rgb = obs["tripod_rgb"]
        proprio = obs["proprio"].to(dtype=torch.float32)
        camera_meta = obs["camera_meta"].to(dtype=torch.float32)

        z_wrist, z_tripod = self.encode_visual(
            wrist_rgb, tripod_rgb, wrist_shifts=wrist_shifts, tripod_shifts=tripod_shifts
        )
        z_vis = torch.cat((z_wrist, z_tripod), dim=-1)
        z_prop = self.proprio_encoder(torch.cat((proprio, camera_meta), dim=-1))
        h = self.trunk(torch.cat((z_vis, z_prop), dim=-1))
        mean = self.mean_head(h)
        std = torch.exp(self.log_std.clamp(-5.0, 2.0)).expand_as(mean)
        return Normal(mean, std), z_vis

    @torch.no_grad()
    def act(
        self,
        obs: Mapping[str, np.ndarray | Tensor],
        deterministic: bool = False,
    ) -> Tensor:
        device = self.log_std.device
        prop_raw = obs["proprio"]
        unbatched = (
            prop_raw.ndim == 1
            if isinstance(prop_raw, (np.ndarray, Tensor))
            else np.asarray(prop_raw).ndim == 1
        )
        actor_obs: dict[str, Tensor] = {}
        for key in ACTOR_OBS_KEYS:
            val = obs[key]
            t = val if isinstance(val, Tensor) else torch.as_tensor(val)
            if unbatched:
                t = t.unsqueeze(0)
            actor_obs[key] = t.to(device=device)

        dist, _ = self.forward_actor(actor_obs)
        raw = dist.mean if deterministic else dist.sample()
        action = torch.clamp(raw, -1.0, 1.0)
        return action.squeeze(0) if unbatched else action

    def export_payload(self) -> dict[str, Any]:
        return {
            "state_dict": {k: v.detach().cpu() for k, v in self.state_dict().items()},
            "config": {
                "frame_stack": self.frame_stack,
                "action_dim": self.action_dim,
                "encoder_type": self.encoder_type,
                "img_hw": list(self.img_hw),
            },
        }

    def save_actor(self, path: str | Path) -> Path:
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.export_payload(), out_path)
        return out_path


def load_actor(
    path: str | Path, map_location: str | torch.device = "cpu"
) -> DualCameraActor:
    """Load a standalone exported DualCameraActor checkpoint without critic weights."""
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    cfg = dict(payload["config"])
    height, width = cfg.get("img_hw", (64, 112))
    cfg["img_hw"] = (height, width)
    actor = DualCameraActor(**cfg)
    actor.load_state_dict(payload["state_dict"], strict=True)
    actor.to(map_location)
    actor.eval()
    return actor


class AsymmetricPPOAgent(nn.Module):
    """Training wrapper pairing deployable DualCameraActor with privileged Critic & aux head."""

    def __init__(
        self,
        frame_stack: int = 2,
        action_dim: int = ACTION_DIM,
        encoder_type: str = "spatial_grid",
        img_hw: tuple[int, int] = (64, 112),
    ) -> None:
        super().__init__()
        self.actor = DualCameraActor(
            frame_stack=frame_stack,
            action_dim=action_dim,
            encoder_type=encoder_type,
            img_hw=img_hw,
        )
        critic_in = PRIVILEGED_STATE_DIM + PROPRIO_DIM + CAMERA_META_DIM
        self.critic = nn.Sequential(
            _ortho_init(nn.Linear(critic_in, 256)),
            nn.LayerNorm(256),
            nn.SiLU(),
            _ortho_init(nn.Linear(256, 256)),
            nn.LayerNorm(256),
            nn.SiLU(),
            _ortho_init(nn.Linear(256, 1), std=1.0),
        )
        self.aux_head = nn.Sequential(
            _ortho_init(nn.Linear(256, 64)),
            nn.SiLU(),
            _ortho_init(nn.Linear(64, AUX_TARGET_DIM), std=0.01),
        )

    def get_value(self, obs: Mapping[str, Tensor]) -> Tensor:
        x = torch.cat(
            (
                obs["privileged_state"].to(dtype=torch.float32),
                obs["proprio"].to(dtype=torch.float32),
                obs["camera_meta"].to(dtype=torch.float32),
            ),
            dim=-1,
        )
        return self.critic(x).squeeze(-1)

    def get_action_and_value(
        self,
        obs: Mapping[str, Tensor],
        wrist_shifts: Tensor | None = None,
        tripod_shifts: Tensor | None = None,
        action: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        dist, z_vis = self.actor.forward_actor(
            obs, wrist_shifts=wrist_shifts, tripod_shifts=tripod_shifts
        )
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        value = self.get_value(obs)
        aux_pred = self.aux_head(z_vis)
        return action, log_prob, entropy, value, aux_pred


def masked_aux_loss(aux_pred: Tensor, aux_targets: Tensor, aux_mask: Tensor) -> Tensor:
    """Compute visibility-masked SmoothL1 loss on (ball_xy, rack_xy, rel_xy)."""
    weights = aux_mask.to(dtype=aux_pred.dtype).repeat_interleave(2, dim=-1)
    per_elem = F.smooth_l1_loss(aux_pred, aux_targets, reduction="none") * weights
    return per_elem.sum() / weights.sum().clamp(min=1.0)
