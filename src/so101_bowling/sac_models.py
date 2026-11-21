"""Dual-camera Asymmetric SAC networks: privileged twin-Q critic and training wrapper."""

from __future__ import annotations

import copy
import math
from typing import Mapping

import torch
from torch import Tensor, nn

from so101_bowling.ppo_models import (
    ACTION_DIM,
    AUX_TARGET_DIM,
    CAMERA_META_DIM,
    PRIVILEGED_STATE_DIM,
    PROPRIO_DIM,
    DualCameraActor,
    _ortho_init,
)


def _build_q_mlp(input_dim: int) -> nn.Sequential:
    return nn.Sequential(
        _ortho_init(nn.Linear(input_dim, 256)),
        nn.LayerNorm(256),
        nn.SiLU(),
        _ortho_init(nn.Linear(256, 256)),
        nn.LayerNorm(256),
        nn.SiLU(),
        _ortho_init(nn.Linear(256, 1), std=1.0),
    )


class PrivilegedTwinQ(nn.Module):
    """Twin Q-value critic consuming privileged state, proprio, camera meta, and action."""

    def __init__(self, action_dim: int = ACTION_DIM) -> None:
        super().__init__()
        self.input_dim = PRIVILEGED_STATE_DIM + PROPRIO_DIM + CAMERA_META_DIM + action_dim
        self.q1 = _build_q_mlp(self.input_dim)
        self.q2 = _build_q_mlp(self.input_dim)

    def forward(self, obs: Mapping[str, Tensor], action: Tensor) -> tuple[Tensor, Tensor]:
        x = torch.cat(
            (
                obs["privileged_state"].float(),
                obs["proprio"].float(),
                obs["camera_meta"].float(),
                action.float(),
            ),
            dim=-1,
        )
        return self.q1(x).squeeze(-1), self.q2(x).squeeze(-1)


class AsymmetricSACAgent(nn.Module):
    """Asymmetric SAC agent pairing squashed DualCameraActor with PrivilegedTwinQ."""

    def __init__(
        self,
        frame_stack: int = 2,
        action_dim: int = ACTION_DIM,
        encoder_type: str = "spatial_grid",
        img_hw: tuple[int, int] = (64, 112),
        init_alpha: float = 0.2,
        target_entropy: float | None = None,
    ) -> None:
        super().__init__()
        if not math.isfinite(init_alpha) or init_alpha <= 0.0:
            raise ValueError(f"init_alpha must be positive and finite, got {init_alpha}")

        self.actor = DualCameraActor(
            frame_stack=frame_stack,
            action_dim=action_dim,
            encoder_type=encoder_type,
            img_hw=img_hw,
            squashed=True,
        )
        self.critic = PrivilegedTwinQ(action_dim=action_dim)
        self.critic_target = copy.deepcopy(self.critic)
        for param in self.critic_target.parameters():
            param.requires_grad_(False)

        self.aux_head = nn.Sequential(
            _ortho_init(nn.Linear(256, 64)),
            nn.SiLU(),
            _ortho_init(nn.Linear(64, AUX_TARGET_DIM), std=0.01),
        )
        self.log_alpha = nn.Parameter(
            torch.tensor(math.log(float(init_alpha)), dtype=torch.float32)
        )
        self.target_entropy = (
            -float(action_dim) if target_entropy is None else float(target_entropy)
        )

    @property
    def alpha(self) -> Tensor:
        """Clamped entropy temperature derived from log_alpha."""
        return self.log_alpha.clamp(-10.0, 2.0).exp()

    @torch.no_grad()
    def soft_update_target(self, tau: float) -> None:
        """Polyak-average online critic parameters into frozen target critic."""
        for p_targ, p_src in zip(
            self.critic_target.parameters(), self.critic.parameters(), strict=True
        ):
            p_targ.lerp_(p_src, tau)
