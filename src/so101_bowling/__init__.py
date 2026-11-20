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
]


def __getattr__(name: str):
    if name in ("PPOConfig", "load_ppo_config"):
        from so101_bowling import ppo

        return getattr(ppo, name)
    if name in ("AsymmetricPPOAgent", "DualCameraActor", "load_actor"):
        from so101_bowling import ppo_models

        return getattr(ppo_models, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
