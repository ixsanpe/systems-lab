
"""
Implement toy example of 2 multiple processes that run in CPU
on MacOS.
"""
import os
import tempfile

import torch
from torch import nn, optim
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.multiprocessing as mp

def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    # We want to be able to train our model on an accelerator such as CUDA, MPS,
    # MTIA, or XPU.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backend = dist.get_default_backend_for_device(device)
    dist.init_process_group(backend, rank=rank, world_size=world_size)


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


def train(rank, world_size):
    print(f"Running basic DDP example on rank {rank}.")
    setup(rank, world_size)

    try:
        model = ToyModel()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        # ddp_model = DDP(model, device_ids=[rank]) #FIXME: this is for GPU
        ddp_model = DDP(model)

        # This file is accessed across nodes
        CHECKPOINT_PATH = tempfile.gettempdir() + "/model.checkpoint"
        if rank == 0:
            # All processes should see same parameters as they all start from same
            # random parameters and gradients are synchronized in backward passes.
            # Therefore, saving it in one process is sufficient.
            print("Saving to storage")
            torch.save(ddp_model.state_dict(), CHECKPOINT_PATH)

        # Use a barrier() to make sure that process 1 loads the model after process
        # 0 finishes saving it.
        dist.barrier()
        # NOTE: We want to be able to train our model on an `accelerator <https://pytorch.org/docs/stable/torch.html#accelerators>`__
        # such as CUDA, MPS, MTIA, or XPU.
        #acc = torch.accelerator.current_accelerator()
        # configure map_location properly
        # map_location = {f'{acc}:0': f'{acc}:{rank}'}
        ddp_model.load_state_dict(
            torch.load(CHECKPOINT_PATH, map_location='cpu', weights_only=True))

        loss_fn = nn.MSELoss()
        optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

        optimizer.zero_grad()
        x = torch.randn(32, 10, device=device)
        outputs = ddp_model(x)
        checksum = sum(out.detach().sum().item() for out in outputs)
        print(f"rank={rank} outputs_checksum={checksum:.6f}", flush=True)
        print(f"rank={rank}, output_shape={tuple(outputs.shape)}")

        labels = torch.randn(32,5).to(device) #FIXME: this is for GPU, MPS: .to(rank)
        print(f"Make sure all tensors are in same device: {x.device}, {outputs.device}, {labels.device}")
        loss_fn(outputs, labels).backward()
        optimizer.step()
        # Make sure the optimizer worked for both processes:
        checksum = sum(p.detach().sum().item() for p in model.parameters())
        print(f"rank={rank} after optim step checksum={checksum:.6f}", flush=True)

    finally:
        cleanup()


# torchrun: a launcher, not part of the training logic.
#
# What it does
#   - Starts N processes (one per rank) for you.
#   - Sets the env vars that setup() reads by hand today: RANK, LOCAL_RANK,
#     WORLD_SIZE, MASTER_ADDR, MASTER_PORT.
#   - Watches the workers. With --max-restarts, if one dies it restarts the group.
#   - With --nnodes and a rendezvous, the same command works across machines.
#
# Where it fits here
#   - It replaces the mp.spawn call below, and the hard-coded MASTER_ADDR/PORT in setup().
#   - train(), the checkpoint logic and the checksums stay the same.
#   - Command (from the project folder, using the venv):
#       physical-ai-av/bin/torchrun --nproc_per_node=2 ddp.py
#
# Connection to the kill test
#   - A killed rank leaves the others stuck in a collective. We cleaned up by hand.
#   - With --max-restarts, torchrun does that cleanup and restarts the group.
#     The checkpoint resume then continues from the last saved step.
#

if __name__ == "__main__":
    # NOTE: Spawned children re-import your script, so everything that launches
    # processes must sit under if __name__ == "__main__":.
    world_size = 2
    mp.spawn(train, args=(world_size,), nprocs=world_size, join=True, start_method="spawn")
