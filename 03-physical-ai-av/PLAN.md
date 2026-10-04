For learning purposes, these are the steps prepared to build distributed training fundamentals in PyTorch.
Claude is used as a mentor.

Goal is to understand what DDP actually does under the hood (not just call it), then make a training loop that survives a crash — the two things "robust abstractions for distributed training and checkpointing" actually means day to day.

### Step 1: DDP from first principles
- Multi-process training loop using `torch.distributed` directly: manually init the process group (gloo backend is fine on CPU/single machine), spawn N processes with `torch.multiprocessing.spawn`.
- Wrap a tiny model in `DDP` and train it on a toy dataset (synthetic regression or MNIST-sized).
- Prove to yourself DDP is doing an all-reduce: before wrapping in DDP, manually diverge gradients across ranks (different data per rank, no sync) and show the model weights drift apart across processes. Then wrap in DDP and show they stay identical after each step.

### Step 2: Checkpoint / resume under failure
- Save model state, optimizer state, and RNG state every N steps.
- Kill a rank mid-training (literally `kill -9` or raise an exception) and resume from the last checkpoint.
- Verify the loss curve is continuous across the kill/resume boundary (log loss per step to a file, plot before vs after).

### Step 3: Distributed data loading with resumable shards
- Data source: [`dgural/PhysicalAI-Autonomous-Vehicles-Sample`](https://huggingface.co/datasets/dgural/PhysicalAI-Autonomous-Vehicles-Sample) — a small (~700 row) sample of NVIDIA's real multi-sensor AV dataset (7 cameras + lidar + radar per clip, chunked as per-sensor parquet/mp4 with UUIDs cross-linking sensors). Structurally identical to the full gated [`nvidia/PhysicalAI-Autonomous-Vehicles`](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles) (133TB, same chunk format), so what you build here generalizes.
- Build a sampler that shards by chunk across ranks (don't just use `DistributedSampler` — implement the indexing logic yourself first, then compare against it).
- To keep the focus on sharding mechanics rather than video I/O: start from the per-sensor parquet metadata (timestamps, UUIDs) only, and treat the mp4/sensor payload as a lazy lookup by UUID — decode frames only for a handful of samples, don't make decoding part of the sharding logic.
- Handle the hard part: resuming mid-epoch without re-shuffling already-seen data or desyncing ranks, and without a rank ending up with a different mix of sensors/modalities than another. This is the part that's genuinely annoying in production and a common source of silent bugs.

### Step 4: Stretch — FSDP comparison
- Take the same toy model and wrap it in FSDP instead of DDP.
- Compare peak memory usage between DDP and FSDP (log via `torch.cuda.max_memory_allocated` if on GPU, or just parameter memory accounting on CPU).
- Don't go deep on wrapping policies / mixed precision here — the goal is just to see *why* FSDP exists (parameter sharding vs. full replication), not master it.

### Future steps
#### Observability
Add structured metrics logging (loss, throughput, grad norm) in a format Prometheus/Grafana could scrape — doesn't need real Prometheus running, just the right shape of output.

#### Ray
Once this project's training loop + checkpointing is solid, orchestrate it with Ray Train instead of hand-rolled `torch.multiprocessing.spawn`. This is meant to come *after*, so Ray is wrapping something you already understand rather than hiding it.

#### FSDP depth
Wrapping policies, mixed precision, activation checkpointing — real depth here is its own project, not a weekend stretch goal.
