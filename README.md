# so101-bowling-rl

Sim-to-real dual-camera reinforcement learning (Asymmetric PPO & Asymmetric SAC) for tabletop 3-pin push-bowling on the SO-101 robotic arm.

## Overview

- **Simulator (`M1`)**: MuJoCo 3-pin tabletop bowling environment (`SO101BowlingEnv`) with vendored SO-101 STL/MJCF assets, procedural hollow bowling pin meshes, `condim=6` rolling/sliding contact physics, domain randomization, a 4-phase delivery scorer (`DeliveryScorer`), and a 4-DOF damped-least-squares IK scripted baseline (`ScriptedPushController`).
- **Asymmetric PPO (`M2`)**: Dual-camera (`wrist_cam` + `iphone_continuity_cam`, `64×112` frame-stacked RGB) deployable actor (`DualCameraActor`) trained with a privileged critic, visibility-masked auxiliary head, DrQ view shifts, and spawn-based vectorized environments (`SubprocVecBowlingEnv`).
- **Real-Transfer Safety Bridge (`M3`)**: Thread-backed timestamped OpenCV cameras, full collision-volume MuJoCo FK safety guard with 16-corner servo-lag verification (`SafetyGuard`), LeRobot `SO101Follower` bridge (`SO101BowlingHardwareBridge`), passive camera gate, and trial harness (`scripts/run_real_transfer.py`).
- **Asymmetric SAC (`M4`)**: Off-policy dual-camera SAC (`AsymmetricSACAgent`, `PrivilegedTwinQ`) with a generation-checked deduplicated raw-frame replay buffer (`RawFrameReplayBuffer`) and paired-seed baseline comparison (`scripts/compare_baselines.py`).

## Setup

```bash
uv sync --extra dev
# Optional: for live SO-101 hardware
uv sync --extra dev --extra hardware
# Optional: for Hugging Face Hub uploads/downloads
uv sync --extra dev --extra hub
# Optional: for Weights & Biases experiment tracking
uv sync --extra dev --extra tracking
```

## Testing

```bash
uv run pytest -m "not hardware"
```

On headless Linux (CI / Colab), set `MUJOCO_GL=egl` (GPU) or `MUJOCO_GL=osmesa` (CPU).

## Training & Evaluation

```bash
# Train Asymmetric PPO
uv run python scripts/train_ppo.py --config configs/ppo.yaml --run-dir runs/ppo

# Train Asymmetric SAC
uv run python scripts/train_sac.py --config configs/sac.yaml --out-dir runs/sac

# Evaluate a standalone exported actor (PPO or SAC)
uv run python scripts/eval_ppo.py --checkpoint runs/sac/actor_best.pt --episodes 20 --ablate-cameras

# Paired-seed comparison: scripted 4-DOF vs PPO vs SAC
uv run python scripts/compare_baselines.py \
  --episodes 20 \
  --ppo-checkpoint runs/ppo/best_actor.pt \
  --sac-checkpoint runs/sac/actor_best.pt
```

## Real-Transfer Harness

```bash
# 1. Passive camera gate (no motor commands)
uv run python scripts/run_real_transfer.py passive-gate --out-dir runs/gate --samples 40

# 2. Dry-run trials (safe, no motor commands)
uv run python scripts/run_real_transfer.py run-trials \
  --checkpoint runs/sac/actor_best.pt --policy-name sac --trials 5 --out-dir runs/dry

# 3. Live trials (requires calibrated SO-101 JSON)
uv run python scripts/run_real_transfer.py run-trials \
  --checkpoint runs/sac/actor_best.pt --policy-name sac --trials 10 \
  --calibration path/to/follower.json --live --out-dir runs/live
```

## Datasets & Training

```bash
# Collect dual-camera scripted 4-DOF demo episodes
uv run python scripts/collect_dataset.py --out-dir data/demos --episodes 100 --dr --only-legal

# Warm-start SAC from demos and resume from checkpoint if interrupted
uv run python scripts/train_sac.py --out-dir runs/sac --dr --demo-dataset data/demos
uv run python scripts/train_sac.py --out-dir runs/sac --dr --resume runs/sac/checkpoint_latest.pt

# End-to-end headless GPU training on Colab Pro or modal script (MUJOCO_GL=egl + auto HF Hub & W&B)
uv run python scripts/colab_train.py \
  --algo sac --total-timesteps 100000 --collect-demos 100 --dr \
  --wandb-project so101-bowling-rl \
  --hf-dataset-repo <user>/so101-bowling-demos \
  --hf-model-repo <user>/so101-bowling-sac

# Manually push model or dataset to Hugging Face Hub
uv run python scripts/push_to_hub.py model --run-dir runs/sac --repo-id <user>/so101-bowling-sac
uv run python scripts/push_to_hub.py dataset --dataset-dir data/demos --repo-id <user>/so101-bowling-demos
```
