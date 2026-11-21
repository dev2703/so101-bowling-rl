"""Public API for the SO-101 bowling simulator."""

from so101_bowling.config import load_calibration, load_config
from so101_bowling.env import SO101BowlingEnv
from so101_bowling.scorer import DeliveryPhase, DeliveryScorer, EpisodeOutcome, ScorerConfig

__all__ = [
    "load_config",
    "load_calibration",
    "SO101BowlingEnv",
    "DeliveryScorer",
    "ScorerConfig",
    "DeliveryPhase",
    "EpisodeOutcome",
    "PPOConfig",
    "load_ppo_config",
    "DualCameraActor",
    "AsymmetricPPOAgent",
    "load_actor",
    "SACConfig",
    "load_sac_config",
    "AsymmetricSACAgent",
    "PrivilegedTwinQ",
    "RawFrameReplayBuffer",
]


def __getattr__(name: str):
    if name in ("PPOConfig", "load_ppo_config"):
        from so101_bowling import ppo

        return getattr(ppo, name)
    if name in ("AsymmetricPPOAgent", "DualCameraActor", "load_actor"):
        from so101_bowling import ppo_models

        return getattr(ppo_models, name)
    if name in ("SACConfig", "load_sac_config"):
        from so101_bowling import sac

        return getattr(sac, name)
    if name in ("AsymmetricSACAgent", "PrivilegedTwinQ"):
        from so101_bowling import sac_models

        return getattr(sac_models, name)
    if name == "RawFrameReplayBuffer":
        from so101_bowling import replay_buffer

        return getattr(replay_buffer, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
