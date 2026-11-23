For learning purposes, these are the steps to get hands-on with world models using my own robot vacuum (Ecovacs DEEBOT T90 Omni).
Claude is used as a mentor.

Goal: understand what a world model actually is — a model that predicts the *next observation given the current observation and an action* — by collecting real data from the vacuum, training a small action-conditioned model on it, and using it to plan. Not to train a giant video generator.

Reality check that shapes the plan: a weekend of vacuum driving is a few hours of low-res video. That is far too little to train a world model from pixels from scratch, but plenty to train a **small dynamics predictor on top of a frozen pretrained visual encoder** (the DINO-WM / V-JEPA 2-AC recipe).

### Step 0: Data pipeline from the robot (runs on my machine, not here)
- Video: get the front camera live view via the Home Assistant integration (`imcdo17-cyber/home-assistant-ecovacs-live`), or lift its streaming code into a standalone Python script. Expect a `deebot-client` version pin to fix the `DeviceVerificationRequiredError` import error.
- Actions: test whether manual move commands (forward / left / right / stop) work on the T90 via `deebot-client` or `ecovacs-deebot.js`.
  - If yes: script random-ish drives and log every command with a timestamp. Best case — actions are exact.
  - If no: let it clean normally, log its map position + heading over time, and derive actions as pose deltas between frames.
- Fallback for video: phone strapped on top.
- Measure latency between "command sent" and "frame shows motion" early — it decides whether actions need a time shift.
- [ ] PENDING — relocalization: after the robot is carried by hand, `getPos` reports `invalid: 1` and the T90 refuses `setRelocationState` (20003 "rcp not support"); returning to the dock fixed it. Run `probe.py --listen` (and the camera) while the robot relocalizes (return-to-dock from the app, or the app's relocate if it has one) to capture which commands/events happen, so `record.py` can trigger relocalization itself instead of needing the dock.

### Step 1: Episode format + dataset
- Resample everything to a fixed rate (start with 2–4 Hz; the vacuum is slow) and save episodes as `{frames[T,H,W,3], actions[T,A], pose[T,3], t[T]}`.
- Action space: discrete `{fwd, left, right, stop}` if commanded, otherwise continuous `(Δx, Δy, Δθ)` in the robot frame.
- Write a viewer that plays an episode with the action/pose overlaid. If the overlay looks wrong, everything downstream is wrong.
- Train/val split **by episode**, not by frame (adjacent frames leak).
- Privacy: recordings stay in `data/` (gitignored). Blur people/screens/documents before sharing any clip.
- [ ] PENDING — recording in a smaller space: in tight rooms the random policy pushes against furniture in 24–33% of forward steps (vs ~10% in open space). Make it escape faster (detect being blocked after 2 steps, back up 1–2 steps, turn ~75–150°, shorter forward runs), then record more 2-min driven episodes there.

### Step 2: Baselines before anything fancy
- Copy-last-frame baseline: predict `o_{t+1} = o_t`. Every model must beat this, and on a slow robot it is surprisingly strong.
- Inverse-dynamics sanity check: train a tiny classifier `(o_t, o_{t+1}) -> a_t`. If it can't recover the action from consecutive frames, the actions are mislabelled or desynced — fix Step 0/1 first.
- Done (`wm/data.py`, `wm/copy_last.py`, `wm/inverse_dynamics.py`). Results on the time-based test split:
  - Copy-last PSNR at 0.5 / 1 / 2 / 4 s: clean 15.6 / 14.1 / 13.0 / 12.6 dB, drive 16.3 / 14.9 / 13.8 / 13.1 dB (forward ~19 dB, turns ~13 dB at 0.5 s).
  - Inverse dynamics (3 seeds): clean 74.7% vs 53.6% always-majority, drive 68.2% vs 47.7%. Clean is clearly best at offset 0.
  - Video shift sweep on driven episodes (0.0–0.7 s): 0.42 s and 0.55 s tie (68–70%), smaller or larger shifts are worse → keep 0.42 s, which matches the measured latency. Driven test set is only 111 steps, so differences under ~5 points are noise.

### Step 3: Latent world model on a frozen encoder (the core)
- Encode frames with a frozen pretrained encoder (DINOv2 patch features; optionally compare V-JEPA 2). Cache embeddings to disk once.
- Train a small transformer predictor `f(z_{t-k..t}, a_t) -> z_{t+1}` in latent space (teacher forcing, L2/L1 loss on features).
- Evaluate:
  - 1-step and multi-step open-loop rollout error vs. the copy-last baseline, per horizon.
  - Action sensitivity: same start, different actions -> do predicted latents actually diverge in the right direction? (A model that ignores actions can still get low loss.)
  - Nearest-neighbour decode: for a predicted latent, show the closest real frame in the dataset — cheap way to "see" predictions without a decoder.
- First version done (`wm/encode.py`, `wm/predictor.py`): DINOv2 ViT-S/14 patches cached (2.6 ms/frame), 2x2-pooled to a 5x9 grid, 3-step context, 3.4 M-param transformer, epoch picked on a validation slice.
  - Test error vs copy-last at 0.5 / 1 / 2 / 4 s: clean 0.80 / 0.76 / 0.78 / 0.88, drive 0.86 / 0.88 / 0.86 / 0.89. Turns improve most (~0.72), forward/stop least (~0.9).
  - Action sensitivity: turnLeft vs turnRight predictions now differ, but stop vs forward barely (clean pick-the-true-action 47%, stop recall 18%).
- Second version (default now): backward dropped from the actions the planner may pick (~1% of data; the first model matched 39% of real stops to it) + an action loss (weight 0.5): for half of each batch, predict under all 4 actions and cross-entropy on which prediction is closest to reality.
  - Picks the true action: clean 79% (was 47%), drive 82% (was 58%) — above the Step 2 pixel classifier (75% / 68%). Stop is still weak (28%; inferred clean stops are noisy).
  - Costs some prediction error, still better than copy-last everywhere: clean 0.86 / 0.84 / 0.86 / 0.91, drive 0.85 / 0.87 / 0.91 / 0.90. Weight 2 was too strong (worse than copy-last beyond 1 s).
  - Figure: `uv run python -m wm.plot_results` -> `figures/step3_results.png`.
- [ ] PENDING — improve the world model before Step 4 (planning needs both good predictions and actions that lead to different futures):
  - Stop is recognised only 28% of the time (half of real stops look like forward). Check the inferred clean "stop" labels in the viewer (many are probably slow turns or tiny moves between pose updates), tighten `label_from_motion` or drop doubtful stops, and add stops to driven episodes.
  - Win back the prediction accuracy the action loss cost (0.76–0.80 → 0.84–0.86 vs copy-last at 0.5–2 s): validation error was still falling at epoch 25, so train longer (40–60 epochs); try weights between 0.2 and 0.5.
  - Try the full 10x18 grid (`--pool 1`, ~7x slower): forward over 0.5 s is a small change that 2x2 pooling may blur.
  - More driven data (see the Step 1 small-space item): the driven test set is only 109 steps, so driven numbers swing by ±5 points.

### Step 4: Plan with it
- Goal-reaching in latent space: given current frame and a goal frame (e.g. "facing the couch"), use CEM / random shooting over action sequences, roll out with the world model, pick the sequence minimizing distance to the goal embedding.
- First evaluate offline (does the planned first action match what the robot actually did in the logged episode?), then — if manual control works — run it closed-loop on the real vacuum at low speed.

### Future steps
#### Pixel decoder
Train a small decoder from latents back to frames to visualize dreamed rollouts. Nice to look at, not needed for planning.

#### Learn the encoder too
Compare frozen DINOv2 vs. a VAE + RSSM trained end to end (World Models / DreamerV3 style) on the same data — shows why pretrained features matter in the low-data regime.

#### Big pretrained video world models
Try zero-shot or fine-tuned action-conditioned prediction with a large model (e.g. NVIDIA Cosmos, V-JEPA 2-AC) on my episodes. Needs real GPU time; only worth it once Steps 1–3 give a trustworthy eval.

#### Tie back to project 03
Episodes are chunked video + actions, same shape of problem as the AV data — the resumable sharded loader from 03 can feed training here.
