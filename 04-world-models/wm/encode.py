"""Step 3a: encode every episode frame once with a frozen DINOv2 and cache it.

    uv run python -m wm.encode                # all episodes, skips ones already done
    uv run python -m wm.encode --force        # re-encode everything

Encoder: DINOv2 ViT-S/14 (facebookresearch/dinov2 via torch.hub, ~85 MB, downloaded
once to ~/.cache/torch/hub). It is never trained here: the world model learns to
predict in its feature space, which is the DINO-WM recipe.

Input: episode.npz frames (144x256 uint8) -> resized to 140x252 (multiples of the
14 px patch size) -> ImageNet normalisation. Output per episode, in
data/features/dinov2_vits14/<episode>/:

    patches.npy  [T, 180, 384] float16   patch tokens on a 10 x 18 grid (row-major)
    cls.npy      [T, 384]      float16   global token (for quick checks / retrieval)

Both are the encoder's final layer-normed outputs. Frame i of the episode is row i,
so the actions / keep mask from wm.data line up with them directly.

At the end it prints, per episode, how similar consecutive frames are in feature
space (mean cosine of patch tokens between step i and i+1) per action: stop should
be close to 1, forward lower, turns lowest. Same ordering as the pixel copy-last
baseline = the features see motion.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch
from torch.nn import functional as F

from wm.data import ACTIONS, EPISODES, ROOT, load_episodes

MODEL = "dinov2_vits14"
H, W = 140, 252  # 10 x 18 patches of 14 px
GRID = (H // 14, W // 14)
OUT = ROOT / "data" / "features" / MODEL
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_encoder(dev: torch.device) -> torch.nn.Module:
    model = torch.hub.load("facebookresearch/dinov2", MODEL, verbose=False)
    return model.eval().to(dev)


@torch.no_grad()
def encode(model: torch.nn.Module, frames: np.ndarray, dev: torch.device,
           batch: int = 64) -> tuple[np.ndarray, np.ndarray]:
    patches, cls = [], []
    mean, std = MEAN.to(dev), STD.to(dev)
    for s in range(0, len(frames), batch):
        x = torch.from_numpy(frames[s : s + batch]).to(dev).permute(0, 3, 1, 2).float() / 255
        x = F.interpolate(x, size=(H, W), mode="bilinear", antialias=True, align_corners=False)
        out = model.forward_features((x - mean) / std)
        patches.append(out["x_norm_patchtokens"].half().cpu())
        cls.append(out["x_norm_clstoken"].half().cpu())
    return torch.cat(patches).numpy(), torch.cat(cls).numpy()


def similarity_report(name: str, patches: np.ndarray, actions: np.ndarray, keep: np.ndarray) -> None:
    p = torch.from_numpy(patches).float()
    cos = F.cosine_similarity(p[:-1], p[1:], dim=-1).mean(-1).numpy()  # [T-1]
    ok = keep[:-1] & keep[1:]
    parts = []
    for a, act in enumerate(ACTIONS):
        m = ok & (actions[:-1] == a)
        if m.sum() >= 5:
            parts.append(f"{act} {cos[m].mean():.3f}")
    print(f"    {name}: consecutive-frame cosine  " + ", ".join(parts))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--force", action="store_true", help="re-encode cached episodes")
    args = parser.parse_args()
    dev = device()
    episodes = load_episodes(EPISODES)
    todo = [e for e in episodes if args.force or not (OUT / e.name / "patches.npy").exists()]
    print(f"{len(episodes)} episodes, {len(todo)} to encode with {MODEL} on {dev} "
          f"({H}x{W} -> {GRID[0]}x{GRID[1]} patches)")
    if todo:
        model = load_encoder(dev)
        for e in todo:
            t0 = time.time()
            patches, cls = encode(model, e.frames, dev)
            d = OUT / e.name
            d.mkdir(parents=True, exist_ok=True)
            np.save(d / "patches.npy", patches)
            np.save(d / "cls.npy", cls)
            dt = time.time() - t0
            print(f"  {e.name}: {len(e)} frames in {dt:.1f} s ({1000 * dt / len(e):.1f} ms/frame), "
                  f"{patches.nbytes / 1e6:.0f} MB")
    print("sanity check (higher = less change between steps):")
    for e in episodes:
        patches = np.load(OUT / e.name / "patches.npy", mmap_mode="r")
        assert len(patches) == len(e), f"{e.name}: {len(patches)} features vs {len(e)} steps"
        similarity_report(e.name, np.array(patches), e.actions, e.keep)


if __name__ == "__main__":
    main()
