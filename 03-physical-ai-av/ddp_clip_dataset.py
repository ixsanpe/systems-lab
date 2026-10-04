"""
Toy DDP training on 2 CPU processes (macOS), with checkpoint/resume (Step 2)
and clip-level sharded data (Step 3).
"""
import json
import os
import random
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn, optim
from torch.nn.parallel import DistributedDataParallel as DDP

ROOT = Path(__file__).parent
SAMPLES_JSON = ROOT / "data" / "samples.json"
CKPT_PATH = ROOT / "checkpoints" / "ckpt.pt"
LOG_PATH = ROOT / "logs" / "loss.log"

WORLD_SIZE = 2
BATCH_PER_RANK = 4
TOTAL_STEPS = 60
SAVE_EVERY = 10
LR = 0.01
SEED = 0
DEVICE = torch.device("cpu")


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


class ClipDataset:
    """One item per clip: 7 camera features and a target.

    Features are file sizes in MB (log-scaled), so no video is decoded.
    The target is a placeholder: the mean of the features.
    """

    def __init__(self):
        samples = json.loads(SAMPLES_JSON.read_text())["samples"]
        # Step 1: group the entries by uuid
        clips = {}
        # clips = {
        #     "8d669f31-90e7-4828-a99b-264bf6dedd14": {
        #         "camera_cross_left_120_fov": Path(".../data/8d669f31-...camera_cross_left_120fov.mp4"),
        #         "camera_front_wide_120_fov": Path(".../data/8d669f31-...camera_front_wide_120fov.mp4"),
        #         ...  # 7 cameras per clip
        #     },
        #     ...  # 100 clips
        # }
        for s in samples:
            clips.setdefault(s["uuid"], {})[s["group"]["name"]] = ROOT / "data" / s["filepath"]
        # Step 2: Sort camera names so that the features order is the same for every camera
        self.cameras = sorted({s["group"]["name"] for s in samples})
        self.uuids = sorted(clips)
        self.features = torch.zeros(len(self.uuids), len(self.cameras))
        # Step 3: Feature is the file size, no video is decoded.
        for i, uuid in enumerate(self.uuids):
            for j, cam in enumerate(self.cameras):
                size_mb = os.path.getsize(clips[uuid][cam]) / 1e6
                self.features[i, j] = torch.log1p(torch.tensor(size_mb))

    def __len__(self):
        return len(self.uuids)

    def batch(self, idx):
        x = self.features[idx]
        y = x.mean(dim=1, keepdim=True)
        return x, y


class ToyModel(nn.Module):
    def __init__(self, in_features=7):
        super().__init__()
        self.net1 = nn.Linear(in_features, 16)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(16, 1)

    def forward(self, x):
        return self.net2(self.relu(self.net1(x)))


def shard_for(epoch, rank, world_size, num_clips):
    """Clip indices for this rank in this epoch.

    Every rank shuffles with the same seed (SEED + epoch), so the ranks agree
    on the order without communicating. Each rank then takes every
    world_size-th clip, giving disjoint shards with the same length.

    Example with 8 clips and 2 ranks (after the shuffle):

        pos:   0   1   2   3   4   5   6   7     position in shuffled order
        clip:  5   2   7   0   3   6   1   4     clip index at that position (shuffle)
        rank:  0   1   0   1   0   1   0   1     order[rank::2] takes these

        rank 0 -> [5, 7, 3, 1]
        rank 1 -> [2, 0, 6, 4]

    Each epoch the shuffle changes, so the same clip can land on a different
    rank next epoch. Clips never overlap within an epoch.
    """
    order = list(range(num_clips))
    random.Random(SEED + epoch).shuffle(order)
    per_rank = num_clips // world_size
    return order[rank::world_size][:per_rank]


def save_checkpoint(ddp_model, optimizer, step, rank, world_size):
    # Every rank contributes its own RNG state. The collective must run on all ranks.
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


def train(rank, world_size):
    setup(rank, world_size)

    try:
        dataset = ClipDataset()
        num_clips = len(dataset)
        steps_per_epoch = (num_clips // world_size) // BATCH_PER_RANK

        torch.manual_seed(SEED)
        model = ToyModel(in_features=len(dataset.cameras)).to(DEVICE)
        optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9)

        step = 0
        if CKPT_PATH.exists():
            ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
            model.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            # Each rank's generator is checkpointed
            # Does not affect the clip order, because that's the Python generator
            torch.set_rng_state(ckpt["rng"][rank])
            step = ckpt["step"]
            print(f"rank={rank} resumed from step {step}", flush=True)

        ddp_model = DDP(model)
        loss_fn = nn.MSELoss()

        if rank == 0:
            LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

        current_epoch = None
        while step < TOTAL_STEPS:
            # Global step -> (epoch, position within the epoch). Every rank computes
            # the same position, so their batches line up.
            epoch, k = divmod(step, steps_per_epoch)
            if epoch != current_epoch:
                # The shard only changes between epochs, so compute it once per epoch.
                shard = shard_for(epoch, rank, world_size, num_clips)
                current_epoch = epoch
            idx = shard[k * BATCH_PER_RANK:(k + 1) * BATCH_PER_RANK]
            x, y = dataset.batch(idx)

            optimizer.zero_grad()
            loss = loss_fn(ddp_model(x.to(DEVICE)), y.to(DEVICE))
            loss.backward()
            optimizer.step()
            step += 1

            avg_loss = loss.detach().clone() # NOTE: important to acumulate properly
            dist.all_reduce(avg_loss)
            avg_loss /= world_size
            if rank == 0:
                with open(LOG_PATH, "a") as f:
                    f.write(f"{step} {avg_loss.item():.6f}\n")

            if step % SAVE_EVERY == 0:
                checksum = sum(p.detach().sum().item() for p in model.parameters())
                print(f"rank={rank} step={step} checksum={checksum:.6f}", flush=True)
                save_checkpoint(ddp_model, optimizer, step, rank, world_size)
    finally:
        cleanup()


if __name__ == "__main__":
    # NOTE: Spawned children re-import your script, so everything that launches
    # processes must sit under if __name__ == "__main__":.
    mp.spawn(train, args=(WORLD_SIZE,), nprocs=WORLD_SIZE, join=True, start_method="spawn")
