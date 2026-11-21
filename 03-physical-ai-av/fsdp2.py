# https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html
# ZeRO (Zero Redundancy Optimizer) in small words
#
# DDP: every rank keeps a full copy of everything.
#
# ZeRO-1: shard the optimizer state
#   - Optimizer state = extra numbers the optimizer keeps (e.g. SGD momentum, Adam's two moments).
#   - Each rank keeps only its slice of that state.
#   - Weights and gradients are still full copies.
#   - PyTorch: torch.distributed.optim.ZeroRedundancyOptimizer
#
# ZeRO-2: shard the gradients too
#   - Each rank keeps only the gradients for its own slice.
#   - Weights are still full copies.
#   - FSDP name: SHARD_GRAD_OP
#
# ZeRO-3: shard the weights too
#   - Each rank keeps only its slice of the weights.
#   - Full layers are gathered just before they run, then freed.
#   - FSDP name: FULL_SHARD. FSDP2 (fully_shard) works this way.
#
# Rule of thumb
#   - More stages = less memory per rank = more communication.
#   - All three stages are still data parallel: every rank runs the full model on its own data.
#
# Reading
#   - https://pytorch.org/docs/stable/distributed.optim.html
#   - https://pytorch.org/docs/stable/distributed.fsdp.fully_shard.html
