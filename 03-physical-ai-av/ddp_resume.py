"""
Step 2: checkpoint and resume for DDP on 2 CPU processes (macOS).

Run:     physical-ai-av/bin/python ddp_resume.py
Kill:    kill -9 on the process (or Ctrl+C). If one rank dies, the others wait
         in a collective, so kill all of them: pkill -9 -f ddp_resume.py
Resume:  run the same command again. It loads checkpoints/ckpt.pt.
Fresh:   delete checkpoints/ and logs/ first.
"""
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn, optim
from torch.nn.parallel import DistributedDataParallel as DDP

ROOT = Path(__file__).parent
CKPT_PATH = ROOT / "checkpoints" / "ckpt.pt"
LOG_PATH = ROOT / "logs" / "loss_resume.log"

WORLD_SIZE = 2
TOTAL_STEPS = 200
SAVE_EVERY = 10
LR = 0.001
SEED = 0
STEP_DELAY = 0.05  # seconds per step, so there is time to kill the run


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12356"
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net1 = nn.Linear(10, 10)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(10, 5)

    def forward(self, x):
        return self.net2(self.relu(self.net1(x)))


def batch_for(step, rank):
    # The same (step, rank) always gives the same batch, before and after a restart.
    # That is what makes the loss curve comparable across the kill.
    g = torch.Generator().manual_seed(SEED + 1000 * step + rank)
    x = torch.randn(32, 10, generator=g)
    labels = torch.randn(32, 5, generator=g)
    return x, labels


def save_checkpoint(ddp_model, optimizer, step, rank, world_size):
    # Every rank contributes its own RNG state. All ranks must call this.
    rng_states = [None] * world_size
    dist.all_gather_object(rng_states, torch.get_rng_state())
    dist.barrier()
    if rank == 0:
        CKPT_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = CKPT_PATH.with_suffix(".tmp")
        torch.save(
            {
                "model": ddp_model.module.state_dict(),
                "optimizer": optimizer.state_dict(),
                "rng": rng_states,
                "step": step,
            },
            tmp,
        )
        os.replace(tmp, CKPT_PATH)  # atomic: a kill mid-save never leaves a half-written file
    dist.barrier()


def load_checkpoint(model, optimizer, rank):
    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    torch.set_rng_state(ckpt["rng"][rank])
    return ckpt["step"]


def truncate_log(step):
    # Lines after the checkpoint belong to steps that will be run again.
    # Drop them so the log has one line per step.
    if not LOG_PATH.exists():
        return
    kept = [line for line in LOG_PATH.read_text().splitlines() if int(line.split()[0]) <= step]
    LOG_PATH.write_text("".join(line + "\n" for line in kept))


def train(rank, world_size):
    setup(rank, world_size)

    try:
        torch.manual_seed(SEED)
        model = ToyModel()
        optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9)

        step = 0
        if CKPT_PATH.exists():
            step = load_checkpoint(model, optimizer, rank)
            if rank == 0:
                truncate_log(step)
            print(f"rank={rank} resumed from step {step}", flush=True)

        ddp_model = DDP(model)
        loss_fn = nn.MSELoss()

        if rank == 0:
            LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

        while step < TOTAL_STEPS:
            x, labels = batch_for(step, rank)

            optimizer.zero_grad()
            loss = loss_fn(ddp_model(x), labels)
            loss.backward()
            optimizer.step()
            step += 1

            avg_loss = loss.detach().clone()
            dist.all_reduce(avg_loss)
            avg_loss /= world_size
            if rank == 0:
                with open(LOG_PATH, "a") as f:
                    f.write(f"{step} {avg_loss.item():.6f}\n")

            if step % SAVE_EVERY == 0:
                save_checkpoint(ddp_model, optimizer, step, rank, world_size)

            time.sleep(STEP_DELAY)
    finally:
        cleanup()


if __name__ == "__main__":
    # NOTE: Spawned children re-import your script, so everything that launches
    # processes must sit under if __name__ == "__main__":.
    mp.spawn(train, args=(WORLD_SIZE,), nprocs=WORLD_SIZE, join=True, start_method="spawn")
