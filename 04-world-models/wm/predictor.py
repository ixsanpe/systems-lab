"""Step 3b: the world model — predict the next DINOv2 features from the last few + actions.

    uv run python -m wm.encode            # once: cache features (Step 3a)
    uv run python -m wm.predictor         # train, then evaluate on the test split
    uv run python -m wm.predictor --eval  # evaluate the saved model only

Model f(z_{i-2..i}, a_{i-2..i}) -> z_{i+1}, where z is a frame's grid of DINOv2
patch features: the cached 10 x 18 grid averaged over 2 x 2 blocks to 5 x 9
(45 tokens x 384; --pool 1 keeps all 180, ~7x slower). It is a small transformer
over all 3 x 45 tokens. Each token gets: its features projected to 256 dims + an embedding
of the action taken at that step + a learned "which patch" and "which step"
embedding. Attention is causal over steps (a token can see its own step and the
earlier ones), so in one pass the output at step t predicts z_{t+1} for all three
steps (teacher forcing). The output is a *change* added to z_t, with the last
layer initialised to zero: an untrained model is exactly copy-last, and training
only has to learn what moves.

Evaluation on the test split (same time-based split as Step 2), per mode:
- open-loop rollouts at 1, 2, 4, 8 steps (0.5-4 s): the model is fed its own
  predictions and the true actions; error = MSE of the features vs the real frame,
  next to the copy-last error (z_{i+k} = z_i). ratio < 1 = better than copy-last.
- per action at 1 step: where it helps (turns should improve most).
- action sensitivity: for each test step, predict z_{i+1} under every candidate
  action (stop, forward, turnLeft, turnRight; backward is ~1% of the data, too rare
  to learn or to plan with) and pick the one closest to the real z_{i+1}. If the model ignored actions this is a
  coin flip; if it uses them it recovers the action (compare with Step 2's
  inverse-dynamics classifier: ~75% clean, ~68% drive). Planning (Step 4) relies on
  exactly this: different actions must lead to different, correct futures.

Action loss (--action-loss, weight lambda): plain next-step MSE lets the model get
most of the way by continuing the motion it sees in the history, and largely ignore
the action (first version: stop vs forward barely told apart). So for half of each
batch the last step is also predicted under all 4 candidate actions, the distances
to the real next features become scores (-MSE / tau) and a cross-entropy term asks
the true action's prediction to be the closest. It trains exactly what the planner
will use. The epoch is still picked on the validation 1-step error.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from wm.data import ACTIONS, ROOT, Episode, load_episodes, split
from wm.encode import GRID
from wm.encode import MODEL as ENCODER

FEATURES = ROOT / "data" / "features" / ENCODER
SAVE = ROOT / "data" / "models" / "predictor.pt"
CONTEXT = 3  # steps of history (1.5 s)
HORIZONS = (1, 2, 4, 8)
CANDIDATES = ["stop", "forward", "turnLeft", "turnRight"]  # actions the planner may choose
POOL = 2  # average pool x pool patches: 10 x 18 -> 5 x 9 grid, ~7x faster to train


def pool_patches(z: np.ndarray, pool: int) -> torch.Tensor:
    """[T, 180, D] cached patches -> [T, (10/pool)*(18/pool), D] float16."""
    t = torch.from_numpy(np.array(z)).float()
    if pool > 1:
        t = t.reshape(len(t), *GRID, -1).permute(0, 3, 1, 2)
        t = F.avg_pool2d(t, pool).permute(0, 2, 3, 1).reshape(len(t), -1, t.shape[1])
    return t.half()


def device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Latents:
    """A split's cached features and actions as flat tensors (one row per step)."""

    def __init__(self, episodes: list[Episode], dev: torch.device, pool: int = POOL) -> None:
        cache: dict[str, np.ndarray] = {}
        z, a, keep, piece, modes = [], [], [], [], []
        for j, e in enumerate(episodes):
            if e.name not in cache:
                cache[e.name] = np.load(FEATURES / e.name / "patches.npy", mmap_mode="r")
            z.append(pool_patches(cache[e.name][e.start : e.start + len(e)], pool))
            a.append(torch.from_numpy(e.actions))
            keep.append(torch.from_numpy(e.keep))
            piece.append(torch.full((len(e),), j))
            modes += [e.mode] * len(e)
        self.z = torch.cat(z).to(dev)  # [N, P, 384] float16 (P = 45 with pool 2)
        self.a = torch.cat(a).to(dev)  # [N] index into ACTIONS
        self.keep, self.piece = torch.cat(keep), torch.cat(piece)
        self.modes = np.array(modes)

    def starts(self, before: int, after: int) -> torch.Tensor:
        """Rows g where rows g-before .. g+after are all usable and in one piece."""
        n = len(self.keep)
        g = torch.arange(before, max(n - after, before))
        ok = torch.ones(len(g), dtype=torch.bool)
        for d in range(-before, after + 1):
            ok &= self.keep[g + d] & (self.piece[g + d] == self.piece[g])
        return g[ok]


class Predictor(nn.Module):
    def __init__(self, dim: int = 384, d: int = 256, layers: int = 4, heads: int = 8,
                 n_patches: int = 180, context: int = CONTEXT) -> None:
        super().__init__()
        self.inp = nn.Linear(dim, d)
        self.act = nn.Embedding(len(ACTIONS), d)
        self.pos = nn.Parameter(torch.randn(1, 1, n_patches, d) * 0.02)
        self.time = nn.Parameter(torch.randn(1, context, 1, d) * 0.02)
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.1, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.blocks = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        self.out = nn.Linear(d, dim)
        nn.init.zeros_(self.out.weight)  # start as copy-last
        nn.init.zeros_(self.out.bias)
        step = torch.arange(context).repeat_interleave(n_patches)
        self.register_buffer("mask", step[None, :] > step[:, None])  # True = may not attend

    def forward(self, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """z [B, T, P, dim] float, a [B, T] -> predicted z_{t+1} for each t, [B, T, P, dim]."""
        b, t, p, _ = z.shape
        h = self.inp(z) + self.act(a)[:, :, None] + self.pos + self.time[:, :t]
        h = self.blocks(h.reshape(b, t * p, -1), mask=self.mask[: t * p, : t * p])
        return z + self.out(self.norm(h)).reshape(b, t, p, -1)


def windows(data: Latents, g: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Features z_{g-2..g+1} [B, 4, P, D] and actions a_{g-2..g} [B, 3]."""
    idx = (g[:, None] + torch.arange(-CONTEXT + 1, 2)).to(data.z.device)
    return data.z[idx].float(), data.a[idx[:, :CONTEXT]]


@torch.no_grad()
def rollout(model: Predictor, data: Latents, g: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    """From the real history up to row g, apply actions[:, h] for h = 0..H-1.

    Returns predicted z_{g+1..g+H} [B, H, P, D]; predictions are fed back in.
    """
    idx = (g[:, None] + torch.arange(-CONTEXT + 1, 1)).to(data.z.device)
    zs = list(data.z[idx].float().unbind(1))
    acts = list(data.a[idx[:, :-1]].unbind(1)) + [actions[:, 0]]
    preds = []
    for h in range(actions.shape[1]):
        nxt = model(torch.stack(zs[-CONTEXT:], 1), torch.stack(acts[-CONTEXT:], 1))[:, -1]
        preds.append(nxt)
        zs.append(nxt)
        if h + 1 < actions.shape[1]:
            acts.append(actions[:, h + 1])
    return torch.stack(preds, 1)


def action_scores(model: Predictor, z: torch.Tensor, a: torch.Tensor, cand: torch.Tensor,
                  tau: float) -> torch.Tensor:
    """Score each candidate as the last action: -MSE(prediction, real z) / tau, [B, C].

    z [B, CONTEXT+1, P, D] (history + real next), a [B, CONTEXT] (last one is replaced).
    """
    b, c = len(a), len(cand)
    aa = a.repeat_interleave(c, 0).clone()
    aa[:, -1] = cand.repeat(b)
    pred = model(z[:, :-1].repeat_interleave(c, 0), aa)[:, -1].reshape(b, c, *z.shape[2:])
    return -((pred - z[:, -1, None]) ** 2).mean(dim=(-1, -2)) / tau


def mse(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return ((x - y) ** 2).mean(dim=(-1, -2))  # per sample


def chunks(g: torch.Tensor, size: int = 256):
    for s in range(0, len(g), size):
        yield g[s : s + size]


@torch.no_grad()
def evaluate(model: Predictor, data: Latents) -> dict:
    """Print the test report; also return it as plain numbers (used by wm.plot_results)."""
    model.eval()
    dev = data.z.device
    res: dict = {"rollout": {}, "per_action": {}, "sensitivity": {}}
    print("\nopen-loop rollouts (feature MSE, lower is better; ratio = model / copy-last)")
    for mode in sorted(set(data.modes)):
        print(f"  [{mode}]  {'steps':>5} {'sec':>4} {'model':>7} {'copy':>7} {'ratio':>6} {'n':>5}")
        for k in HORIZONS:
            g = data.starts(CONTEXT - 1, k)
            g = g[torch.from_numpy(data.modes[g.numpy()] == mode)]
            if not len(g):
                continue
            m_err, c_err = [], []
            for gc in chunks(g):
                acts = data.a[(gc[:, None] + torch.arange(k)).to(dev)]
                pred = rollout(model, data, gc, acts)[:, -1]
                target = data.z[(gc + k).to(dev)].float()
                m_err.append(mse(pred, target).cpu())
                c_err.append(mse(data.z[gc.to(dev)].float(), target).cpu())
            m, c = torch.cat(m_err).mean().item(), torch.cat(c_err).mean().item()
            res["rollout"].setdefault(mode, {})[k] = {"model": m, "copy": c, "n": len(g)}
            print(f"  {'':8} {k:5d} {k / 2:4.1f} {m:7.4f} {c:7.4f} {m / c:6.2f} {len(g):5d}")

    print("\nper action, 1 step (ratio = model / copy-last)")
    g = data.starts(CONTEXT - 1, 1)
    errs, copies, acts = [], [], []
    for gc in chunks(g):
        a = data.a[gc.to(dev)]
        pred = rollout(model, data, gc, a[:, None])[:, 0]
        target = data.z[(gc + 1).to(dev)].float()
        errs.append(mse(pred, target).cpu())
        copies.append(mse(data.z[gc.to(dev)].float(), target).cpu())
        acts.append(a.cpu())
    err, copy, act = torch.cat(errs), torch.cat(copies), torch.cat(acts)
    modes = data.modes[g.numpy()]
    for mode in sorted(set(modes)):
        mm = torch.from_numpy(modes == mode)
        parts = []
        for name in CANDIDATES:
            sel = mm & (act == ACTIONS.index(name))
            if sel.sum() >= 5:
                ratio = (err[sel].mean() / copy[sel].mean()).item()
                res["per_action"].setdefault(mode, {})[name] = {"ratio": ratio, "n": int(sel.sum())}
                parts.append(f"{name} {ratio:.2f} (n={int(sel.sum())})")
        print(f"  [{mode}] " + ", ".join(parts))

    print("\naction sensitivity: which action's prediction is closest to what really happened?")
    cand = torch.tensor([ACTIONS.index(c) for c in CANDIDATES], device=dev)
    picks, spread = [], []
    for gc in chunks(g, 128):
        target = data.z[(gc + 1).to(dev)].float()
        preds = torch.stack([rollout(model, data, gc, c.expand(len(gc))[:, None])[:, 0] for c in cand], 1)
        d = ((preds - target[:, None]) ** 2).mean(dim=(-1, -2))  # [B, C]
        picks.append(cand[d.argmin(1)].cpu())
        left, right = CANDIDATES.index("turnLeft"), CANDIDATES.index("turnRight")
        spread.append(mse(preds[:, left], preds[:, right]).cpu())
    pick = torch.cat(picks)
    gap = torch.cat(spread)
    in_cand = torch.isin(act, cand.cpu())  # steps whose true action can be picked
    for mode in sorted(set(modes)):
        mm = torch.from_numpy(modes == mode) & in_cand
        y, p = act[mm], pick[mm]
        acc = (p == y).float().mean().item()
        majority = torch.bincount(y).max().item() / len(y)
        recall = []
        for name in CANDIDATES:
            sel = y == ACTIONS.index(name)
            if sel.sum() >= 5:
                recall.append(f"{name} {(p[sel] == y[sel]).float().mean():.0%}")
        print(f"  [{mode}] picked the true action {acc:.1%} (always-majority {majority:.1%}), n={len(y)}")
        print("    recall: " + ", ".join(recall))
        print(f"    turnLeft vs turnRight predictions differ by MSE {gap[mm].mean():.4f} "
              f"(copy-last error at 1 step: {copy[mm].mean():.4f})")
        conf = np.zeros((len(CANDIDATES), len(CANDIDATES)), int)  # rows = true, cols = picked
        for i, ti in enumerate(CANDIDATES):
            for j, pj in enumerate(CANDIDATES):
                conf[i, j] = int(((y == ACTIONS.index(ti)) & (p == ACTIONS.index(pj))).sum())
        res["sensitivity"][mode] = {"accuracy": acc, "majority": majority, "confusion": conf.tolist()}
    return res


def train(epochs: int, seed: int, pool: int, action_loss: float, tau: float) -> None:
    torch.manual_seed(seed)
    dev = device()
    train_eps, test_eps = split(load_episodes())
    # Validation = the last 10% (in time) of the training part, to pick the epoch.
    # The test split is only looked at once, at the end.
    train_eps, val_eps = split(train_eps, test_fraction=0.1)
    tr, va, te = (Latents(e, dev, pool) for e in (train_eps, val_eps, test_eps))
    g_train, g_val = tr.starts(CONTEXT - 1, 1), va.starts(CONTEXT - 1, 1)
    model = Predictor(n_patches=tr.z.shape[1]).to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"train {len(g_train)} windows, val {len(g_val)}, {tr.z.shape[1]} tokens/step, "
          f"{n_params / 1e6:.1f} M params, {dev}, action loss {action_loss} (tau {tau})")
    cand = torch.tensor([ACTIONS.index(c) for c in CANDIDATES], device=dev)
    to_cand = torch.full((len(ACTIONS),), -1, device=dev)
    to_cand[cand] = torch.arange(len(cand), device=dev)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.05)
    batch = 64
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 6e-4, total_steps=epochs * (len(g_train) // batch + 1))
    t0 = time.time()
    best, best_state, best_ep = float("inf"), None, 0
    for ep in range(epochs):
        model.train()
        perm = g_train[torch.randperm(len(g_train))]
        total = 0.0
        for gc in chunks(perm, batch):
            z, a = windows(tr, gc)
            loss = F.mse_loss(model(z[:, :-1], a), z[:, 1:])
            if action_loss > 0:
                sub = torch.nonzero(to_cand[a[:, -1]] >= 0).squeeze(1)[: batch // 2]
                scores = action_scores(model, z[sub], a[sub], cand, tau)
                loss = loss + action_loss * F.cross_entropy(scores, to_cand[a[sub, -1]])
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            total += loss.item() * len(gc)
        model.eval()
        with torch.no_grad():
            errs, copies, hits = [], [], []
            for gc in chunks(g_val):
                z, a = windows(va, gc)
                errs.append(mse(model(z[:, :-1], a)[:, -1], z[:, -1]))
                copies.append(mse(z[:, -2], z[:, -1]))
                ok = to_cand[a[:, -1]] >= 0
                picked = action_scores(model, z[ok], a[ok], cand, tau).argmax(1)
                hits.append((picked == to_cand[a[ok, -1]]).float())
            ratio = (torch.cat(errs).mean() / torch.cat(copies).mean()).item()
            val_pick = torch.cat(hits).mean().item()
        if ratio < best:
            best, best_ep = ratio, ep + 1
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if ep == 0 or (ep + 1) % 5 == 0 or ep == epochs - 1:
            print(f"  epoch {ep + 1:3d}  train loss {total / len(g_train):.4f}  "
                  f"val 1-step ratio vs copy-last {ratio:.3f}, picks true action {val_pick:.1%}  "
                  f"({time.time() - t0:.0f} s)")
    print(f"best epoch {best_ep} (val ratio {best:.3f}); using it for the test")
    model.load_state_dict(best_state)
    SAVE.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state": model.state_dict(), "pool": pool}, SAVE)
    print(f"saved {SAVE}")
    evaluate(model, te)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pool", type=int, default=POOL, help="1 = full 10x18 grid (slow)")
    parser.add_argument("--action-loss", type=float, default=0.5,
                        help="weight of the action loss (0 = plain MSE; 2 hurt prediction badly)")
    parser.add_argument("--tau", type=float, default=0.05, help="action-loss temperature")
    parser.add_argument("--eval", action="store_true", help="only evaluate the saved model")
    args = parser.parse_args()
    if args.eval:
        dev = device()
        saved = torch.load(SAVE, map_location=dev)
        te = Latents(split(load_episodes())[1], dev, saved["pool"])
        model = Predictor(n_patches=te.z.shape[1]).to(dev)
        model.load_state_dict(saved["state"])
        evaluate(model, te)
    else:
        train(args.epochs, args.seed, args.pool, args.action_loss, args.tau)


if __name__ == "__main__":
    main()
