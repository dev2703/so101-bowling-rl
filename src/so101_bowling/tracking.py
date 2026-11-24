"""Optional Weights & Biases (W&B) experiment tracking wrapper."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

_VALID_MODES = ("online", "offline", "disabled")


@dataclass(frozen=True)
class WandbConfig:
    """Configuration for optional Weights & Biases experiment tracking."""

    project: str | None = None
    entity: str | None = None
    run_name: str | None = None
    group: str | None = None
    tags: tuple[str, ...] = ()
    mode: str = "online"

    def __post_init__(self) -> None:
        if self.mode not in _VALID_MODES:
            raise ValueError(f"wandb mode must be one of {_VALID_MODES}, got {self.mode!r}")

    @property
    def enabled(self) -> bool:
        return bool(self.project) and self.mode != "disabled"


def parse_wandb_tags(raw: str | None) -> tuple[str, ...]:
    """Parse a comma-separated tag string into a deduplicated tuple."""
    if not raw:
        return ()
    seen: list[str] = []
    for part in raw.split(","):
        tag = part.strip()
        if tag and tag not in seen:
            seen.append(tag)
    return tuple(seen)


class WandbTracker:
    """Lifecycle-safe W&B logger that becomes a clean no-op when disabled."""

    def __init__(
        self,
        cfg: WandbConfig | None,
        *,
        run_dir: Path,
        algo: str,
        train_config: Mapping[str, Any],
        wandb_module: Any | None = None,
    ) -> None:
        self.cfg = cfg
        self._wandb: Any | None = None
        self._run: Any | None = None
        self._finished = False
        self.run_url: str | None = None

        if cfg is None or not cfg.enabled:
            return

        if wandb_module is not None:
            wb = wandb_module
        else:
            try:
                import wandb as wb  # type: ignore[import-not-found]
            except ImportError as exc:
                raise ImportError(
                    "wandb is required when --wandb-project is set; "
                    "install with `uv sync --extra tracking`"
                ) from exc

        tags = list(cfg.tags) if cfg.tags else [algo, "so101-bowling"]
        init_kwargs: dict[str, Any] = {
            "project": cfg.project,
            "entity": cfg.entity,
            "name": cfg.run_name,
            "group": cfg.group,
            "tags": tags,
            "mode": cfg.mode,
            "dir": str(run_dir),
            "config": dict(train_config),
            "reinit": True,
        }
        self._wandb = wb
        self._run = wb.init(**init_kwargs)
        url = getattr(self._run, "url", None)
        self.run_url = str(url) if url else None

    @property
    def active(self) -> bool:
        return self._run is not None and not self._finished

    def log_metrics(
        self,
        prefix: str,
        metrics: Mapping[str, Any],
        *,
        step: int,
        exclude_keys: tuple[str, ...] = ("global_step", "update"),
    ) -> None:
        if not self.active:
            return
        payload: dict[str, float] = {}
        for key, val in metrics.items():
            if key in exclude_keys:
                continue
            if isinstance(val, (int, float)) and not isinstance(val, bool):
                fval = float(val)
                if math.isfinite(fval):
                    payload[f"{prefix}/{key}"] = fval
        if payload:
            self._run.log(payload, step=int(step))

    def log_video(
        self,
        key: str,
        video_path: str | Path | None,
        *,
        step: int,
        fps: int = 20,
    ) -> None:
        if not self.active or video_path is None or self._wandb is None:
            return
        vp = Path(video_path)
        if not vp.is_file():
            return
        try:
            video_cls = getattr(self._wandb, "Video", None)
            if video_cls is None:
                return
            self._run.log({key: video_cls(str(vp), fps=int(fps), format="mp4")}, step=int(step))
        except Exception as exc:
            print(f"[wandb] warning: failed to log video {vp}: {exc}", flush=True)

    def update_summary(self, summary: Mapping[str, Any]) -> None:
        if not self.active:
            return
        run_summary = getattr(self._run, "summary", None)
        if run_summary is None:
            return
        for scalar_key in ("global_step", "total_timesteps", "nan_inf_detected", "eval_aux_loss"):
            if scalar_key in summary and summary[scalar_key] is not None:
                run_summary[scalar_key] = summary[scalar_key]
        for nested_key in ("best_eval", "final_eval", "final_losses"):
            nested = summary.get(nested_key)
            if isinstance(nested, Mapping):
                for k, v in nested.items():
                    if isinstance(v, (int, float, bool)) and (
                        isinstance(v, bool) or math.isfinite(float(v))
                    ):
                        run_summary[f"{nested_key}/{k}"] = v

    def finish(self, exit_code: int = 0) -> None:
        if self._run is None or self._finished:
            return
        self._finished = True
        try:
            self._run.finish(exit_code=int(exit_code))
        except Exception as exc:
            print(f"[wandb] warning: finish failed: {exc}", flush=True)
