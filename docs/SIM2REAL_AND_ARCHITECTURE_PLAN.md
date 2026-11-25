# Sim-to-Real Transfer & RL Architecture Plan (`so101-bowling-rl`)

This roadmap synthesizes the repository's environment/physics, RL pipeline, and sim-to-real hardware audits into concrete engineering milestones. It separates what is addressed by **PRs 1–3** from the remaining simulation, reward, architecture, and physical hardware calibration work required for reliable zero-shot sim-to-real bowling on the SO-101 arm.

---

## 1. Bottleneck & Bug Summary

### Addressed in PRs 1–3

| Area | Issues Fixed |
| :--- | :--- |
| **PR 1 (`fix/env-perf-and-physics`)** | Fixed 6D vs 3D pin velocity coasting bug (`env.py:93`) that caused `unresolved_motion=True` and 1,250-substep stalls; vectorized `_snapshot()` contact scan and cached `MjData` in IK; fixed action-lag double-clipping (`env.py:210-217`) and `ScriptedPushController` lag compensation; exposed `w_push_phi` in `RewardConfig` / `workspace.yaml`, aligned `make_vec_env` `dr_enabled` default, deduplicated look-at math, and removed 3 unused STLs. |
| **PR 2 (`fix/rl-perf-dagger-and-hygiene`)** | Eliminated per-minibatch `.item()` CUDA stream sync storms (`ppo.py`, `sac.py`, `teacher.py`); cached `RawFrameReplayBuffer._valid_mask()` with a dirty flag; switched DAgger `beta` mixing to per-episode selection, fixed squashed MSE saturation, deduplicated DAgger frame storage, and added `Teacher` critic warmup after BC; tuned `configs/ppo*.yaml` (`minibatch_size`, `num_epochs`, `lr`) and removed rollout-time double DrQ shifting; cleaned CLI/Hub/doc drift. |
| **PR 3 (`fix/sim2real-safety-and-roadmap`)** | Commanded an explicit safe hold (`q_active` if `check_pose` is safe, else E-stop) instead of dropping `send_action()` on lag-corner guard rejection (`hardware.py`); cleared `_last_sent_active_rad` cleanly on `reset_episode()` and added `HELD_JOINTS` (`wrist_roll`, `gripper`) hold targets to live commands; tolerated repeated frames within `max_camera_age_s` by emitting `fresh=0.0` in `camera_meta` instead of false-tripping E-stop; verified/logged negotiated `OpenCVTimestampedCamera` resolution/FPS; aligned `HardwareConfig.tripod_cam_index` default (`2`), added `--wrist-index`/`--tripod-index` to `run-trials`, and saved high-res raw frames when available in `real_eval.py` for human review. |

### Remaining Open Bottlenecks & Gaps

1. **RGB Rendering Throughput (`~40–75 env-steps/s`):** Even with Python substep overhead removed, rendering two `64×112` cameras twice per step against `165,857` vertices / `332,713` faces (`so101_new_calib_camera.xml`) with shadows (`assets.py:431`) consumes ~90% of wall-clock step time on CPU/CGL, and `SubprocVecBowlingEnv` lockstep synchronization stalls workers on long coast episodes. Next step: replace the 15 raw STL collision geoms with primitive capsules/boxes/convex hulls, add decimated visual LOD meshes for training renders, and move off-policy/visual training to headless Linux EGL or MJX/Madrona batch rendering.
2. **Physical Sim-to-Real Dynamics & Perception Gaps:**
   - `so101_new_calib_camera.xml` uses `kp=998.22, kv=2.731, forcerange=±3.35` (torque-saturating at `0.19°` error) and defines `<default class="backlash">` without attaching it to any joint.
   - `connect_robot()` does not verify or enforce STS3215 `P_Coefficient` (`16` vs `32`), `D_Coefficient`, `Maximum_Acceleration`, or `Torque_Limit`.
   - Sim lacks joint zero-offset DR (`±2–5°`), 12-bit encoder quantization/noise, variable-`dt` velocity estimation noise, `pin_desk_friction` / `pusher_ball` friction / restitution / table-tilt DR, motion blur, lens distortion, and `z_floor_margin_m=0.015` parity.
   - `run-trials` lacks an automated/guided `move_to_home()` + inter-trial ball/pin reset handshake, and `run_trial_episode` always runs `max_steps=80` instead of holding after ball release.

---

## 2. Reward Engineering & Scorer Improvements

Follow-up `scorer.py` / `env.py` PRs should fix four reward-design flaws identified in the environment audit:

1. **Close the Wobble Full-Credit Loophole (`scorer.py:173-175, 307-309`):**
   - *Current behavior:* Any impacted pin tilting `>= 12°` for `>= 20 ms` receives full `r_pin_credit = 5.0` and counts toward `r_strike_bonus = 5.0` (`20.0` total), even if it rocks back upright.
   - *Fix:* Split terminal credit into `full_knock_pins + off_table_pins` (`1.0 * r_pin_credit`) vs `wobble_only_pins` (`0.25 * r_pin_credit` or `0.0`), and require `full_knock_pins + off_table_pins == 3` to award `r_strike_bonus`. This aligns sim scoring with physical `TrialRecord` grading (`legal_strike` requires 3 downed pins).
2. **Reject `RELEASED` + `delivery_timeout` Short Rolls (`scorer.py:263-267, 288`):**
   - *Current behavior:* A slow push (`< 0.25 m/s`) that separates and stops on the mousepad short of `release_line_x = 0.44` times out at `max_steps = 80` with `legal_delivery = True` and `penalty_reward = 0.0`.
   - *Fix:* Treat `DeliveryPhase.RELEASED` + `delivery_timeout` (ball never crossed `release_line_x`) as an invalid/incomplete delivery (`legal_delivery = False`, applying `r_illegal_penalty` or scaled stall penalty).
3. **Break the "Freeze-at-Home" Local Optimum (`config.py:156-165`, `scorer.py:122-141`):**
   - *Current behavior:* At home pose, `phi_0 = 0.140` and `phi_terminal = 0.0`; doing nothing yields `~0.0` return, while exploratory swings frequently hit `-2.0` (`r_illegal_penalty`) vs `<= 1.2` shaping for a legal miss.
   - *Fix:* Combine (a) BC/scripted warmup or Teacher-guided exploration before pure RL, (b) a small legal-launch bonus when `ball_vel_x >= min_launch_speed_mps` and `ball_x > release_line_x` (so legal misses strictly dominate idling), and (c) graded pin-proximity / kinetic-energy shaping on `RELEASED` only if `legal_delivery=True` and gated to prevent push-scamming.
4. **Enforce PBRS `gamma` & `z_floor` Parity (`env.py:104-105, 133`, `hardware.py:480-513`):**
   - Ensure `ScorerConfig.gamma` is always synchronized with the RL algorithm's `gamma` (`make_vec_env(..., gamma=algo_cfg.gamma)`).
   - Mirror `HardwareConfig.z_floor_margin_m = 0.015` in `env.py` / `scorer.py` (either as an invalidation or soft penalty below `0.015 m`) so policies never rely on grazing trajectories that `SafetyGuard` rejects on hardware.

---

## 3. Policy Architecture & Training Methods

### Recommended Path: Privileged Teacher -> History-Conditioned / Residual Student

1. **Asymmetric Actor-Critic + Privileged Physics/Lag Context (Immediate):**
   - Append the sampled DR parameters (`ball_mass`, `pin_masses`, `pad_roll_friction`, `desk_roll_friction`, `slide_friction`, `actuator_kp_scale`, `joint_damping_scale`, `action_lag`, `camera_delay`) to `privileged_state` (`env.py:374`). Currently `PrivilegedTeacherAgent` (`teacher.py:47`) is a memoryless MLP that cannot observe dynamics parameters, forcing conservative average policies under DR.
2. **RMA-Style Proprioceptive History Encoder (`Kumar et al., RSS 2021`):**
   - Replace the Markovian `frame_stack=2` `proprio` input with a lightweight 1D-CNN or GRU adaptation module consuming the last `H = 8–16` ticks (`0.4–0.8 s`) of `(q_active, dq_active, prev_applied_delta)` (plus optionally `wrist_rgb`/`tripod_rgb` embeddings). Online history lets the deployable student infer ball mass, surface rolling resistance, and STS3215 servo lag without privileged inputs.
3. **Residual RL over `ScriptedPushController` / Action Chunking (`TossingBot` / `ResiP` / `ACT`):**
   - Pure 20 Hz closed-loop joint-delta RL is vulnerable to servo lag, encoder jitter, and narrow contact timing.
   - *Recommended medium-term architecture:* Use `ScriptedPushController` (`kinematics.py:93`) or an **ACT** (`Zhao et al., RSS 2023`) chunked base trajectory (`a_base(t)` or parameterized `PushParams`: `aim_yaw_offset_rad`, `push_speed_mps`, `follow_through_m`, `push_height_z`) and train a bounded **residual policy** (`a = clip(a_base + alpha * Delta_a, -1, 1)`, `alpha ~ 0.25–0.35`) via PPO/SAC to react visually to ball spawn shift, pin rack offset, and contact slip.
4. **Algorithm Trade-offs (`PPO` vs `SAC` vs `FastTD3` vs `TD-MPC2`):**
   - **Teacher (state-only, `~1,500+ env-steps/s`):** **PPO** (`teacher.py`) is simple and reliable once `privileged_state` includes DR context; **FastTD3** (`Seo et al., 2025` — large-batch off-policy TD3 with distributional critic) is an attractive upgrade if sample efficiency or continuous fine-tuning becomes a bottleneck.
   - **Visual Student:** **Teacher -> Online DAgger** (`dagger.py`) remains the fastest path while `render_rgb=True` is capped at `40–75 steps/s`, especially once action-chunking or residual parameterization smooths the target trajectories. **SAC** (`sac.py`) with BC/CQL demo regularization is preferred for pure off-policy visual fine-tuning or real-world residual adaptation; **TD-MPC2** (`Hansen et al., ICLR 2024`) is strong for latent planning over contact dynamics but adds inference latency and hyperparameter complexity for a 20 Hz dual-RGB loop.

---

## 4. Hyperparameter Recommendations

| Algorithm | Key Parameters (Current vs Recommended) | Rationale |
| :--- | :--- | :--- |
| **Teacher PPO** (`configs/teacher_dr.yaml`) | Keep `lr=3e-4`, `num_envs=8..16`, `rollout_steps=128..256`, `minibatch_size=256`, `num_epochs=4..6`, `gamma=0.99`, `gae_lambda=0.95`, `ent_coef=0.005`; add **critic warmup** (`5–10` value-only epochs after `bc_warmup_episodes`) and anneal/initialize `log_std=-1.2` after BC. | Prevents post-BC value-error spike (`V≈0` vs `~20` return) and excessive initial exploration noise (`std≈0.61`) from destroying the BC policy. |
| **DAgger Student** (`configs/dagger_dr.yaml`) | Per-episode `beta` schedule (`1.0 -> 0.0` over `12–16` rounds); `lr=3e-4` (cosine decay to `5e-5`); `batch_size=128`; `epochs_per_round=6..8`; `drq_pad_px=4`; use **unsquashed bounded MSE or Beta/tanh-NLL** (or clip teacher targets to `±0.98` when `squashed=True`); enable `RawFrameReplayBuffer`-style deduplication. | Eliminates mid-episode teacher/student zig-zagging and avoids `tanh` gradient saturation when teacher commands `±1.0`. |
| **Visual PPO** (`configs/ppo*.yaml`) | `lr: 1e-4` (from `3e-5`), `num_epochs: 4` (from `10`), `minibatch_size: 64` (from `16`), `rollout_steps: 128`, `num_envs: 8` (headless EGL), `ent_coef: 0.005`, `aux_coef: 0.5`, **disable rollout-time `drq_pad_px`** (apply DrQ only in gradient updates or rely on `env.py` `max_shift_px`). | Fixes noisy minibatch `approx_kl` early stopping, reduces gradient steps per rollout from 320 to 32, and keeps rollout actions consistent with `evaluate_actor`. |
| **Visual SAC** (`configs/sac*.yaml`) | `lr: 3e-4`, `batch_size: 128..256`, `learning_starts: 2000` (and enforce replay refill guard on `resume_from`), `utd_ratio: 1`, `gamma: 0.99`, `tau: 0.005`, **state-dependent `log_std` head**, and **BC/behavior regularization** during `demo_pretrain_steps`. | Prevents OOD Q-overestimation on narrow demos and allows tight action variance during contact/release vs broader approach. |

---

## 5. Sim-to-Real Bridge Protocol for SO-101

### 5.1 STS3215 Servo System Identification & Motor Parity
1. **Standardize Register Configuration on Connect (`hardware.py`):** Explicitly read and verify (or set) STS3215 `P_Coefficient` (LeRobot default `16`, factory `32`), `I_Coefficient` (`0`), `D_Coefficient` (`0`/`32`), `Maximum_Acceleration`, and `Torque_Limit` during `connect_robot()`, and record supply voltage (`7.4V` vs `12V`).
2. **Trajectory-Based SysID (`SAGE` / `GapONet` approach):** Collect 20 Hz step and multi-frequency chirp trajectories (`q_cmd`, `q_meas`, timestamps) on the real follower arm (no ball) across the bowling workspace. Fit MuJoCo `<position>` `kp`, `kv`, `armature`, `damping`, `frictionloss`, backlash (`<default class="backlash">`, `±0.5°`), and loop transport delay (`1–3` ticks) to minimize `(q_sim - q_real)` RMS error—replacing the current `kp=998.22` bang-bang placeholder in `assets/so101/so101_new_calib_camera.xml:28`.
3. **Wider Actuator & Sensor DR (`randomization.py`, `env.py`):**
   - Joint zero-offset DR: `±0.04 rad` (`±2.3°`) per active joint applied consistently to kinematics/observations (matching manual `lerobot-calibrate` midpoint error).
   - 12-bit encoder quantization (`2π / 4096 ≈ 0.001534 rad`) + `±1` tick noise + `dt` jitter (`±3 ms`) on `CausalJointVelocityEstimator` (or switch both sim and real to a causal 2-tap/EMA velocity filter).
   - Add DR over `pin_desk_friction`, `ball-pusher` friction, `solref` contact restitution, and subtle table tilt (`±0.5°`).

### 5.2 Camera Intrinsics, Extrinsics & Visual Parity
1. **Calibrate Real UVC Cameras:** Run a ChArUco / checkerboard calibration on `wrist_cam` and `tripod_cam` at `1280×720` to measure true vertical FOV (`fovy`) and radial/tangential distortion; update `configs/workspace.yaml:61-66` (`wrist_pos`, `wrist_xyaxes`, `wrist_fovy`, `tripod_pos`, `tripod_target`, `tripod_fovy`) and optionally undistort live frames before `preprocess_camera_frame`.
2. **Lock Exposure & White Balance:** Disable auto-exposure and auto-white-balance on both UVC webcams during trials so frame intervals do not drop below `20 Hz` under indoor lighting and ball/pin hues stay stable.
3. **Visual DR Additions:** Add directional motion blur along fast wrist motion and mild background/table texture randomization beyond `studio_sky`.

### 5.3 Hardware Deployment Checklist (`scripts/run_real_transfer.py`)
1. **Preflight Calibration & Bus Check:** Run `lerobot-calibrate` if joints were reassembled; verify `configs/so101_calibration.yaml` / live calibration JSON matches motor EEPROM (`connect_robot()` enforces `use_degrees=True`, `max_relative_target=None`, `is_calibrated=True`, and exact field parity).
2. **Passive Camera Gate:** Run `run_real_transfer.py passive-gate` (`--samples 60 --checkpoint <actor.pt>`) to verify negotiated `1280×720@25fps`, inter-camera skew `< 50 ms`, `stale_rate` near `0`, and `actor_inference` p95 latency well under `20 ms`. Inspect `crops_side_by_side.png` against MuJoCo `tripod_cam` / `wrist_cam` renders for viewpoint alignment.
3. **Dry-Run Verification:** Run `run_real_transfer.py run-trials --trials 3 --checkpoint <actor.pt> --policy-name <name> --out-dir <dir>` (without `--live`) while manually verifying `StepTrace` `guard_reason`, `camera_meta`, and high-res `.mp4` output.
4. **Live Evaluation Protocol:**
   - Home the arm cleanly (`wrist_roll` & `gripper` at `home_qpos`) and place the ball (`[0.28, 0.0]`) and 3 pins (`[0.62, 0.0]`, `[0.695, ±0.044]`) using table alignment marks.
   - Start with `--trials 1 --live --labeler operator_cli` with a clear workspace and physical power cutoff within reach; inspect `trial_001_trace.jsonl` and `trial_001.mp4` before scaling to `--trials 10–20` interleaved A/B trials (`run_real_trials`).
   - Confirm `torque_disable_unconfirmed_trial_ids == []` on any E-stop and verify no persistent `z_floor_violation` or `foul_line_violation` guard triggers.
