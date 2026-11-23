"""Step 2b: can a tiny CNN tell the action from two consecutive frames?

    uv run python -m wm.inverse_dynamics                 # train + test, offset 0
    uv run python -m wm.inverse_dynamics --offsets -1 0 1  # alignment check

Model: f(o_i, o_{i+1}) -> a_i, a small CNN on 72x128 frames (the two frames plus
their difference, 9 channels). Classes: stop, forward, turnLeft, turnRight,
backward ("unknown" steps are never used).

Why: if the actions, the 0.42 s video shift or the frame alignment were wrong,
consecutive frames wouldn't carry the action and accuracy would sit near the
"always predict the most common class" line. High accuracy = Step 0/1 are sound.

--offsets trains one model per offset d, pairing a_i with (o_{i+d}, o_{i+1+d}).
If the shift is right, d = 0 should win clearly; if d = +1 wins, frames lag the
actions by about a step (raise --shift in record.py), if d = -1 wins they lead.

Training details: Adam, cross-entropy with class weights (backward is rare),
left-right mirror augmentation that swaps turnLeft <-> turnRight (a mirrored
left turn looks exactly like a right turn), mild brightness jitter. Driven
episodes have exact labels, cleaning ones inferred labels; both are trained on
together and reported separately on the test split.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from wm.data import ACTIONS, ROOT, load_episodes, split, transitions

CLASSES = ["stop", "forward", "turnLeft", "turnRight", "backward"]
MIRROR = [0, 1, 3, 2, 4]  # class index after a left-right flip
H, W = 72, 128


def device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Pairs:
    """All usable (o_a, o_b, label, mode) pairs of a split, frames kept as uint8."""

    def __init__(self, episodes, offset: int) -> None:
        frames, a_idx, b_idx, labels, modes = [], [], [], [], []
        base = 0
        for e, idx in transitions(episodes, offset):
            small = F.interpolate(  # 144x256 -> 72x128 once, stored as uint8
                torch.from_numpy(e.frames).permute(0, 3, 1, 2).float(), size=(H, W),
                mode="area").round().to(torch.uint8)
            frames.append(small)
            names = [ACTIONS[a] for a in e.actions[idx]]
            labels += [CLASSES.index(n) for n in names]
            a_idx.append(base + idx + offset)
            b_idx.append(base + idx + 1 + offset)
            modes += [e.mode] * len(idx)
            base += len(e)
        self.frames = torch.cat(frames)
        self.a = torch.from_numpy(np.concatenate(a_idx))
        self.b = torch.from_numpy(np.concatenate(b_idx))
        self.y = torch.tensor(labels)
        self.modes = np.array(modes)

    def __len__(self) -> int:
        return len(self.y)

    def batch(self, idx: torch.Tensor, dev: torch.device, augment: bool):
        oa = self.frames[self.a[idx]].to(dev).float() / 255
        ob = self.frames[self.b[idx]].to(dev).float() / 255
        y = self.y[idx].to(dev)
        if augment:
            flip = torch.rand(len(idx), device=dev) < 0.5
            oa = torch.where(flip[:, None, None, None], oa.flip(-1), oa)
            ob = torch.where(flip[:, None, None, None], ob.flip(-1), ob)
            y = torch.where(flip, torch.tensor(MIRROR, device=dev)[y], y)
            gain = 1 + 0.2 * (torch.rand(len(idx), 1, 1, 1, device=dev) - 0.5)
            oa, ob = (oa * gain).clamp(0, 1), (ob * gain).clamp(0, 1)
        return torch.cat([oa, ob, ob - oa], dim=1), y


class Net(nn.Module):
    def __init__(self, n_classes: int = len(CLASSES)) -> None:
        super().__init__()

        def block(cin: int, cout: int, k: int) -> nn.Sequential:
            return nn.Sequential(nn.Conv2d(cin, cout, k, stride=2, padding=k // 2),
                                 nn.BatchNorm2d(cout), nn.ReLU())

        self.features = nn.Sequential(block(9, 32, 5), block(32, 64, 3), block(64, 128, 3),
                                      block(128, 128, 3))  # 72x128 -> 5x8
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(0.3), nn.Linear(128 * 5 * 8, n_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


def evaluate(model: Net, data: Pairs, dev: torch.device) -> np.ndarray:
    model.eval()
    preds = []
    with torch.no_grad():
        for s in range(0, len(data), 256):
            x, _ = data.batch(torch.arange(s, min(s + 256, len(data))), dev, augment=False)
            preds.append(model(x).argmax(1).cpu())
    return torch.cat(preds).numpy()


def train(offset: int, epochs: int, seed: int) -> dict[str, float]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = device()
    train_eps, test_eps = split(load_episodes())
    tr, te = Pairs(train_eps, offset), Pairs(test_eps, offset)
    counts = torch.bincount(tr.y, minlength=len(CLASSES)).float()
    weight = (counts.sum() / counts.clamp(min=1)).sqrt()  # sqrt-inverse frequency
    model = Net().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=epochs * (len(tr) // 64 + 1))
    print(f"\noffset {offset:+d}: train {len(tr)} pairs, test {len(te)} pairs, device {dev}")
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(tr))
        total = 0.0
        for s in range(0, len(tr), 64):
            x, y = tr.batch(perm[s : s + 64], dev, augment=True)
            loss = F.cross_entropy(model(x), y, weight=weight.to(dev))
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            total += loss.item() * len(y)
        if ep in (0, epochs - 1) or (ep + 1) % 5 == 0:
            acc = (evaluate(model, te, dev) == te.y.numpy()).mean()
            print(f"  epoch {ep + 1:2d}  loss {total / len(tr):.3f}  test acc {acc:.1%}  "
                  f"({time.time() - t0:.0f} s)")

    pred, y = evaluate(model, te, dev), te.y.numpy()
    out = {}
    for mode in sorted(set(te.modes)):
        m = te.modes == mode
        acc = (pred[m] == y[m]).mean()
        majority = np.bincount(y[m]).max() / m.sum()
        present = [c for c in range(len(CLASSES)) if (y[m] == c).any()]
        recall = {CLASSES[c]: (pred[m][y[m] == c] == c).mean() for c in present}
        balanced = np.mean(list(recall.values()))
        out[mode] = float(acc)
        print(f"  [{mode}] accuracy {acc:.1%} (always-majority {majority:.1%}), "
              f"balanced {balanced:.1%}, n={m.sum()}")
        print("    recall: " + ", ".join(f"{k} {v:.0%}" for k, v in recall.items()))
        conf = np.zeros((len(CLASSES), len(CLASSES)), int)
        np.add.at(conf, (y[m], pred[m]), 1)
        print("    confusion (rows = true, cols = predicted " + " ".join(c[:5] for c in CLASSES) + ")")
        for c in present:
            print(f"      {CLASSES[c]:<10}" + "".join(f"{v:6d}" for v in conf[c]))
    if offset == 0:
        path = ROOT / "data" / "models" / "inverse_dynamics.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), path)
        print(f"  saved {path}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--offsets", type=int, nargs="+", default=[0])
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    results = {d: train(d, args.epochs, args.seed) for d in args.offsets}
    if len(results) > 1:
        print("\nalignment check (test accuracy per offset):")
        for d, r in results.items():
            print(f"  offset {d:+d}: " + ", ".join(f"{m} {a:.1%}" for m, a in r.items()))


if __name__ == "__main__":
    main()
