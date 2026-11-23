"""Step 4a: plan with the world model, and test the planner offline on recorded episodes.

    uv run python -m wm.plan                  # offline test on the test split
    uv run python -m wm.plan --horizons 2 4   # which plan lengths to test (steps of 0.5 s)

Planner (used offline here and live by robot/plan_live.py): given the last
CONTEXT steps (features + actions) and a goal frame's features, imagine every
action sequence of length H over {stop, forward, turnLeft, turnRight} with the
world model (4^H sequences: 16 for 1 s, 256 for 2 s, all in one batch), score each
by how close the imagined final features are to the goal (MSE), and return the
best sequence. Live, only its first action is executed and the plan is redone
every step (model-predictive control). With 4 actions exhaustive search is cheap,
so no CEM is needed yet.

Offline test, on the test split: take a real moment i and use the real frame k
steps later as the goal. The robot really got there with actions a_i..a_{i+k-1},
so a good planner should
  - rank that real sequence near the top of all 4^k (rank percentile; random = 50%),
  - pick the same first action (random = 25%, always-majority shown too),
  - when the robot really turned >= 30 degrees on the way, turn the same way
    (net turnLeft - turnRight steps of the plan vs the pose's total heading change).
"""

from __future__ import annotations

import argparse
import itertools
import json

import numpy as np
import torch

from wm.data import ACTIONS, ROOT, load_episodes, split
from wm.predictor import CANDIDATES, CONTEXT, SAVE, Latents, Predictor, device

RESULTS = ROOT / "data" / "results" / "step4_offline.json"
TURN_DEG = 30.0


def load_model(dev: torch.device) -> tuple[Predictor, int]:
    saved = torch.load(SAVE, map_location=dev)
    model = Predictor(n_patches=saved["state"]["pos"].shape[2]).to(dev).eval()
    model.load_state_dict(saved["state"])
    return model, saved["pool"]


class Planner:
    """Exhaustive search over all action sequences of a fixed length."""

    def __init__(self, model: Predictor, horizon: int, dev: torch.device) -> None:
        self.model, self.horizon = model, horizon
        n = len(CANDIDATES)
        self.seqs = torch.tensor(list(itertools.product(range(n), repeat=horizon)), device=dev)
        self.cand = torch.tensor([ACTIONS.index(c) for c in CANDIDATES], device=dev)
        self.place = n ** torch.arange(horizon - 1, -1, -1, device=dev)  # seq -> row index

    def index(self, seq: torch.Tensor) -> int:
        """Row of a sequence given as CANDIDATES indices."""
        return int((seq * self.place).sum())

    @torch.no_grad()
    def imagine(self, z_ctx: torch.Tensor, a_ctx: torch.Tensor,
                prefix: tuple[int, ...] = ()) -> torch.Tensor:
        """Imagined final features of every sequence [4^H, P, D] (see costs)."""
        n = len(self.seqs)
        acts = self.cand[self.seqs]
        if prefix:
            acts = torch.cat([torch.tensor(prefix, device=acts.device).expand(n, len(prefix)), acts], 1)
        zs = [z.expand(n, *z.shape) for z in z_ctx.float()]
        past = [a.expand(n) for a in a_ctx]
        for h in range(acts.shape[1]):
            past.append(acts[:, h])
            zs.append(self.model(torch.stack(zs[-CONTEXT:], 1), torch.stack(past[-CONTEXT:], 1))[:, -1])
        return zs[-1]

    @torch.no_grad()
    def costs(self, z_ctx: torch.Tensor, a_ctx: torch.Tensor, goal: torch.Tensor,
              prefix: tuple[int, ...] = ()) -> torch.Tensor:
        """Cost of every sequence [4^H]: MSE of the imagined final features vs the goal.

        z_ctx [CONTEXT, P, D] real history, a_ctx [CONTEXT-1] the actions taken at its
        first CONTEXT-1 steps (ACTIONS indices), goal [P, D]. prefix: actions (ACTIONS
        indices) already committed after the last real frame; they are imagined first,
        then each candidate sequence (live, the newest frame is ~0.5 s old, so the
        action being executed right now is the prefix).
        """
        n = len(self.seqs)
        acts = self.cand[self.seqs]  # [N, H] ACTIONS indices
        if prefix:
            pre = torch.tensor(prefix, device=acts.device).expand(n, len(prefix))
            acts = torch.cat([pre, acts], 1)
        zs = [z.expand(n, *z.shape) for z in z_ctx.float()]
        past = [a.expand(n) for a in a_ctx]  # actions aligned with zs[:-1]
        for h in range(acts.shape[1]):
            past.append(acts[:, h])
            zs.append(self.model(torch.stack(zs[-CONTEXT:], 1), torch.stack(past[-CONTEXT:], 1))[:, -1])
        return ((zs[-1] - goal.float()) ** 2).mean(dim=(-1, -2))

    def plan(self, z_ctx: torch.Tensor, a_ctx: torch.Tensor, goal: torch.Tensor,
             prefix: tuple[int, ...] = ()) -> tuple[list[str], torch.Tensor]:
        """Best sequence (action names, prefix excluded) and the costs of all sequences."""
        c = self.costs(z_ctx, a_ctx, goal, prefix)
        best = self.seqs[int(c.argmin())]
        return [CANDIDATES[int(i)] for i in best], c


def offline(horizons: tuple[int, ...], stride: int) -> dict:
    dev = device()
    model, pool = load_model(dev)
    te = Latents(split(load_episodes())[1], dev, pool)
    to_cand = torch.full((len(ACTIONS),), -1, dtype=torch.long)
    for j, c in enumerate(CANDIDATES):
        to_cand[ACTIONS.index(c)] = j
    a_cpu = te.a.cpu()
    left, right = CANDIDATES.index("turnLeft"), CANDIDATES.index("turnRight")
    results: dict = {}
    for k in horizons:
        planner = Planner(model, k, dev)
        g_all = te.starts(CONTEXT - 1, k)
        steps = g_all[:, None] + torch.arange(k)
        g_all = g_all[(to_cand[a_cpu[steps]] >= 0).all(1)]  # real sequence must be plannable
        print(f"\nhorizon {k} steps ({k / 2:g} s): {len(planner.seqs)} sequences per plan")
        for mode in sorted(set(te.modes)):
            g = g_all[torch.from_numpy(te.modes[g_all.numpy()] == mode)][::stride]
            pct, first_ok, first_real, turn_ok, turn_n = [], [], [], 0, 0
            for gi in g.tolist():
                real = to_cand[a_cpu[gi : gi + k]].to(dev)
                c = planner.costs(te.z[gi - CONTEXT + 1 : gi + 1], te.a[gi - CONTEXT + 1 : gi],
                                  te.z[gi + k])
                best = planner.seqs[int(c.argmin())]
                pct.append(float((c < c[planner.index(real)]).float().mean()))
                first_ok.append(int(best[0] == real[0]))
                first_real.append(int(real[0]))
                dtheta = float(te.motion[gi : gi + k, 2].nansum())
                if abs(dtheta) >= TURN_DEG:
                    net = int((best == left).sum() - (best == right).sum())
                    turn_n += 1
                    turn_ok += int(np.sign(net) == np.sign(dtheta))
            if not g.numel():
                continue
            majority = np.bincount(first_real).max() / len(first_real)
            row = {
                "n": len(g),
                "rank_percentile": float(np.median(pct)),
                "real_in_top10": float(np.mean(np.array(pct) < 0.10)),
                "first_action": float(np.mean(first_ok)),
                "first_majority": float(majority),
                "turn_direction": turn_ok / turn_n if turn_n else None,
                "turn_n": turn_n,
            }
            results.setdefault(str(k), {})[mode] = row
            turn = f"{row['turn_direction']:.0%} of {turn_n}" if turn_n else "n/a"
            print(f"  [{mode}] n={row['n']}: real sequence ranked at median {row['rank_percentile']:.0%} "
                  f"(random 50%), in top 10% {row['real_in_top10']:.0%}; first action "
                  f"{row['first_action']:.0%} (random 25%, majority {majority:.0%}); "
                  f"turned the right way {turn}")
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps(results, indent=1))
    print(f"\nwrote {RESULTS}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--horizons", type=int, nargs="+", default=[2, 4])
    parser.add_argument("--stride", type=int, default=2, help="use every n-th start (speed)")
    args = parser.parse_args()
    offline(tuple(args.horizons), args.stride)


if __name__ == "__main__":
    main()
