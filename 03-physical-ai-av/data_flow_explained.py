 """
How data moves through training in this project, for two options:

  A. Camera only   -> what "ddp copy.py" does today
  B. Everything    -> camera + lidar + radar (design only, not implemented)

This file explains; it does not train. Write the code yourself.
Line references point at "ddp copy.py" unless stated otherwise.

Read it top to bottom once, then check each claim against the code.
"""


# =============================================================================
# 0. The shape of one training step (same for both options)
# =============================================================================
#
#   samples.json  ->  one feature row per clip  ->  shard per rank  ->
#   mini-batch per rank  ->  forward  ->  loss  ->  backward (DDP all-reduces
#   gradients)  ->  optimizer.step  ->  every 10 steps: checkpoint
#
# Every rank runs the same loop. Each rank sees a different set of clips.
# DDP makes sure the weights stay identical across ranks after each step.
#
# Numbers in the current code (ddp copy.py, lines 21-24):
#   WORLD_SIZE      = 2     two processes, CPU, gloo backend
#   BATCH_PER_RANK  = 4     clips per rank per step
#   TOTAL_STEPS     = 60
#   SAVE_EVERY      = 10


# =============================================================================
# 1. OPTION A: camera only (what the code does now)
# =============================================================================

# ---- 1a. Discover clips (ClipDataset.__init__, starts line 40) --------------
#
# samples.json has 700 entries: 100 clips x 7 cameras, one entry per file.
# Each entry has:
#   uuid          the clip id, shared by all 7 cameras of that clip
#   group.name    which camera, e.g. "camera_front_wide_120_fov"
#   filepath      "data/<uuid>.<camera>.mp4"
#
# Step 1: group the entries by uuid
#     clips[uuid][camera_name] = path to that camera's mp4
#     -> 100 clips, each with 7 paths
#
# Step 2: the column order is the sorted list of camera names. The same
#         order is used for every clip, so column j always means the same
#         camera.
#
# Step 3: the clip order is the sorted list of uuids. Row i is clip i in
#         every rank, every epoch.

# ---- 1b. Build the feature table (ClipDataset.__init__, end of it) ----------
#
# For every (clip, camera) pair:
#     size_bytes = os.path.getsize(mp4_path)     # file size on disk
#     size_mb    = size_bytes / 1e6
#     feature    = log1p(size_mb)
#
# Result: a tensor of shape (100, 7), kept in RAM for the whole run.
#
# Important: no video is decoded. The "feature" is only the file size. That
# is a placeholder that lets the pipeline run. It is not a perception
# feature. A real model would decode frames or use per-frame embeddings.

# ---- 1c. The target (ClipDataset.batch, line 64) ----------------------------
#
# y = mean of the 7 features of that clip (shape (batch, 1)).
# This is also a placeholder. The task is "predict the average camera
# size", which lets us check that the loop learns something.

# ---- 1d. The model (ToyModel, line 70) --------------------------------------
#
# input  (batch, 7)
#   -> Linear 7 -> 16
#   -> ReLU
#   -> Linear 16 -> 1
# output (batch, 1)

# ---- 1e. Shard the clips across ranks (shard_for, line 81) ------------------
#
# Each epoch:
#   1. Every rank shuffles the list 0..99 with the SAME seed (SEED + epoch).
#      Same seed -> same permutation on both ranks, with no communication.
#   2. Rank r takes order[r::2], i.e. every 2nd clip starting at r.
#   3. Each rank keeps only the first 50 (= 100 // 2) so both ranks have the
#      same length.
#
# Result: rank 0 and rank 1 get disjoint halves of the clips. The split
# changes every epoch because the seed changes.

# ---- 1f. Global step -> (epoch, batch position) (train loop, line 145) -----
#
# steps_per_epoch = (100 // 2) // 4 = 12
#
#   epoch, k = divmod(step, steps_per_epoch)
#
# step 0..11  -> epoch 0, k = 0..11
# step 12..23 -> epoch 1, k = 0..11
# ...
#
# Each rank takes shard[k*4 : (k+1)*4] as its batch. 12 steps x 4 clips =
# 48 of the 50 clips in the shard. The last 2 clips are skipped this epoch.
# They are different each epoch because the shuffle changes.

# ---- 1g. One step (train loop, after line 145) ------------------------------
#
# 1. x, y = dataset.batch(idx)          x: (4, 7) camera features
#                                       y: (4, 1) target
# 2. optimizer.zero_grad()
# 3. out  = ddp_model(x)                forward, local to this rank
# 4. loss = MSE(out, y)                 local loss over 4 clips
# 5. loss.backward()                    DDP averages gradients across ranks
#                                       here (all-reduce), so both ranks
#                                       now hold the same gradient
# 6. optimizer.step()                   both ranks apply the same update,
#                                       so weights stay identical
#
# Effective batch per step = 2 ranks x 4 clips = 8 clips.

# ---- 1h. Logging (train loop, after the step) -------------------------------
#
# Each rank's loss is all-reduced and divided by world_size, giving the
# average loss over the 8 clips. Rank 0 writes "step avg_loss" to
# logs/loss.log.

# ---- 1i. Checkpoint and resume (save_checkpoint, line 94) -------------------
#
# Every SAVE_EVERY = 10 steps:
#   - every rank contributes its RNG state (all_gather_object)
#   - rank 0 saves model, optimizer, RNG list, and step to ckpt.pt
#     (written to a .tmp file and then renamed, so a kill mid-write is safe)
#
# On restart: load the model and optimizer, set this rank's RNG state, and
# continue from the saved step. divmod(step, 12) gives the same epoch and
# position, so the same clips are used next. No clip is seen twice in one
# epoch and none is skipped.


# =============================================================================
# 2. OPTION B: everything (camera + lidar + radar), design only
# =============================================================================
#
# Status: NOT implemented. The local sample has cameras only (see the
# previous answer). This option needs the full dataset or another sample
# that includes lidar/radar. Everything below is what would need to change.

# ---- 2a. Discovery: join on uuid across sensor folders ----------------------
#
# The full NVIDIA layout has:
#   camera/<name>/...zip          one folder per camera
#   lidar/lidar_top_360fov/...zip one lidar
#   radar/<name>/...zip           one folder per radar (several)
#   metadata/sensor_presence.parquet   which sensors each clip HAS
#
# A clip does not necessarily have every sensor. The presence table is the
# source of truth. Do not assume that every uuid has all 7 cameras, 1 lidar,
# and N radars.
#
# Open question for you: where do the lidar and radar files come from? The
# sample on disk does not have them. Options: download chunks from the gated
# full dataset (lidar zips are about 20 GB each), or find a sample that
# includes them.

# ---- 2b. Feature per modality ------------------------------------------------
#
# Each modality needs its own cheap feature, computed without decoding:
#
#   camera  (per camera)  file size in MB, log1p  (same as option A)
#   lidar   (1 per clip)  per-clip parquet holds 200 spins. Feature idea:
#                         mean point count per spin, or the file size of the
#                         clip's parquet. Reduce to a fixed size.
#   radar   (per radar)   detection count per clip from its parquet, or file
#                         size. One value per radar sensor.
#
# Key rule: every clip must end up with a FIXED-LENGTH feature vector, even
# though lidar has many spins and radars vary by clip. Reduce variable data
# to one number (or a fixed number) per sensor before batching.

# ---- 2c. Missing sensors ----------------------------------------------------
#
# If a clip lacks a sensor, its feature slot has no value. Two common
# choices:
#   - fill with 0 and add a 0/1 "present" flag per sensor (mask)
#   - train only on clips that have every sensor (smaller dataset)
#
# This is the "sensor mix" problem from PLAN.md Step 3. If ranks get
# different mixes, one rank's batch looks different from the other's, and
# the gradients are no longer comparable. Check the per-rank counts of each
# sensor type, not just the number of clips.

# ---- 2d. Feature table shape ------------------------------------------------
#
# Option A:  (100 clips, 7 cameras)                  -> 7 input features
# Option B:  (N clips, 7 cameras + 1 lidar + R radar + R present-flags)
#            e.g. with R = 4 radars and masks: 7 + 1 + 4 + 4 = 16 features.
#
# ToyModel(in_features=...) must change to match. In ddp copy.py this is
# already a parameter (line 71), so only the caller changes.

# ---- 2e. Sharding and steps -------------------------------------------------
#
# Still shard by clip uuid (shard_for, unchanged). Each rank gets whole
# clips, so every clip brings all its sensors with it.
#
# steps_per_epoch depends on how many clips survive the filter in 2c. If
# you drop clips that lack a sensor, N changes and steps_per_epoch changes
# with it. Recompute it from the filtered count, not from the 100 in the
# sample.

# ---- 2f. What does not change -----------------------------------------------
#
#   - DDP setup, all-reduce of gradients, same-seed shuffle
#   - divmod(step, steps_per_epoch) for position
#   - checkpoint format and resume logic (apart from in_features)
#
# What does change: ClipDataset (the feature builder), in_features, the
# steps_per_epoch count, and how missing sensors are handled.


# =============================================================================
# 3. Side by side
# =============================================================================
#
#                       A: camera only            B: everything
# ---------------------------------------------------------------------------
# clips                 100 (sample)              depends on filter (2c)
# features per clip     7                         7 + lidar + radars + masks
# feature source        mp4 file size             mp4, parquet, zip sizes/counts
# missing sensors       none (all 7 present)      must handle (fill + mask)
# batch shape           (4, 7)                    (4, F), F set by 2d
# sharding              by uuid, same seed        by uuid, same seed
# model input           7                         F
# status                running                   design only
