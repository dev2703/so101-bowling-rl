"""Measure dual-camera rollout and learner throughput on Apple silicon."""

import argparse
import multiprocessing as mp
import os
import platform
import subprocess
from contextlib import ExitStack
from time import perf_counter

os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import torch
from torch import nn

from so101_bowling import SO101BowlingEnv


def make_env():
    return SO101BowlingEnv(render_rgb=True, frame_stack=2, dr_enabled=True)


class Rollout:
    def __init__(self, env, seed):
        self.env = env
        self.seed = seed
        self.action = np.zeros(4, dtype=np.float32)

    def reset(self):
        obs, _ = self.env.reset(seed=self.seed)
        self.seed = None
        return obs

    def step(self):
        obs, reward, terminated, truncated, info = self.env.step(self.action)
        done = terminated or truncated
        reset_obs = self.reset() if done else None
        return obs, reward, done, info, reset_obs


def worker(connection, seed):
    torch.set_num_threads(1)
    with connection, make_env() as env:
        rollout = Rollout(env, seed)
        connection.send(None)
        try:
            while True:
                command = connection.recv()
                if command == "close":
                    return
                connection.send(rollout.reset() if command == "reset" else rollout.step())
        except EOFError:
            return


class DualCNN(nn.Module):
    def __init__(self):
        super().__init__()

        def encoder():
            return nn.Sequential(
                nn.Conv2d(6, 16, 5, stride=2, padding=2),
                nn.ReLU(),
                nn.Conv2d(16, 32, 3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 32, 3, stride=2, padding=1),
                nn.ReLU(),
                nn.AvgPool2d((4, 7)),
                nn.Flatten(),
            )

        self.wrist = encoder()
        self.tripod = encoder()
        self.head = nn.Sequential(nn.Linear(2 * 32 * 2 * 2 + 16, 128), nn.ReLU(), nn.Linear(128, 4))

    def forward(self, wrist, tripod, state):
        return self.head(torch.cat((self.wrist(wrist), self.tripod(tripod), state), dim=-1))


def train_step(model, optimizer, observations, rewards, device):
    def images(key):
        frames = np.stack([obs[key] for obs in observations])
        batch, stack, height, width, channels = frames.shape
        frames = frames.transpose(0, 1, 4, 2, 3).reshape(batch, stack * channels, height, width)
        return torch.from_numpy(np.ascontiguousarray(frames)).to(device, dtype=torch.float32) / 255

    state = np.stack([np.concatenate((obs["proprio"], obs["camera_meta"])) for obs in observations])
    target = torch.tensor(rewards, dtype=torch.float32, device=device)
    optimizer.zero_grad(set_to_none=True)
    prediction = model(images("wrist_rgb"), images("tripod_rgb"), torch.from_numpy(state).to(device))
    nn.functional.mse_loss(prediction.mean(dim=-1), target).backward()
    optimizer.step()


def benchmark(steps, workers, device):
    torch.manual_seed(1)
    model = DualCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)
    with ExitStack() as stack:
        connections = []
        processes = []
        if workers == 1:
            rollout = Rollout(stack.enter_context(make_env()), seed=100)
        else:
            context = mp.get_context("spawn")
            for index in range(workers):
                parent, child = context.Pipe()
                process = context.Process(target=worker, args=(child, 100 + index))
                process.start()
                child.close()
                connections.append(stack.enter_context(parent))
                processes.append(process)

        def exchange(command):
            for connection in connections:
                connection.send(command)
            return [connection.recv() for connection in connections]

        try:
            for connection in connections:
                connection.recv()
            started = perf_counter()
            if connections:
                exchange("reset")
            else:
                rollout.reset()
            resets, coasts = workers, 0
            for _ in range(steps):
                transitions = exchange("step") if connections else [rollout.step()]
                # Optimize on the transition's next observation, including terminal frames.
                observations, rewards, dones, infos, _ = zip(*transitions)
                resets += sum(dones)
                coasts += sum(info.get("end_reason") == "delivery_timeout" for info in infos)
                train_step(model, optimizer, observations, rewards, device)
            if device == "mps":
                torch.mps.synchronize()
            elapsed = perf_counter() - started
        finally:
            for connection in connections:
                try:
                    connection.send("close")
                except (BrokenPipeError, EOFError, OSError):
                    pass
            for process in processes:
                process.join(timeout=5)
                if process.is_alive():
                    process.terminate()
                    process.join()
    return elapsed, resets, coasts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=2000, help="Timed transitions per environment")
    parser.add_argument("--workers", type=int, nargs="+", choices=(1, 2, 4), default=[1, 2, 4])
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")
    torch.set_num_threads(1)
    hardware = platform.machine()
    if platform.system() == "Darwin":
        hardware = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
    devices = ["cpu", "mps"] if torch.backends.mps.is_available() else ["cpu"]
    print(f"{hardware} | two 64×112 RGB cameras | stack=2 | DR on | {args.steps} steps per env", flush=True)
    if "mps" not in devices:
        print("MPS unavailable on this host.", flush=True)
    for device in devices:
        for workers in args.workers:
            elapsed, resets, coasts = benchmark(args.steps, workers, device)
            transitions = args.steps * workers
            rate = transitions / elapsed
            print(
                f"{device.upper():3} | {workers} env | {transitions:6} transitions"
                f" | {elapsed:.3f} s | {rate:.2f} transitions/s"
                f" | {resets} resets | {coasts} timeout coasts"
                f" | 500k {500_000 / rate / 3600:.3f} h | 1M {1_000_000 / rate / 3600:.3f} h",
                flush=True,
            )


if __name__ == "__main__":
    main()
