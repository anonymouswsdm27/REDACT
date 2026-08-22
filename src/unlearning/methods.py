"""Practical unlearning baselines for MF (§11), evaluated against the retrain floor.

Wired here:
- `gradient_ascent_unlearn` — generic FU baseline (NegGrad): ascend the BPR loss on the forgotten
  (A,X), nudging the user embedding AND the item's *shared* embedding so X ranks below negatives.
  Touches shared params (unlike naive-delete), so it can approach the floor — at a utility cost.
- `finetune_unlearn` — approximate unlearning: warm-start from the trained model and continue BPR
  on D\\{(A,X)} for a few epochs. Cheaper than a cold retrain; a standard "fine-tune" baseline.

`retrain` (the oracle/floor) lives in models.mf.train_mf; `naive local-delete` is
models.mf.refit_user. FRU (log-rollback) and FedShare (snapshot) are most faithful with a real
FedAvg loop and are sequenced for P1.1/P1.4 (see docs/EXPERIMENT_PLAN.md).
"""
from __future__ import annotations

import numpy as np
import torch

from ..models.mf import MFModel, refit_user, train_mf


def gradient_ascent_unlearn(base: MFModel, user: int, item: int, n_items: int, *,
                            steps: int = 10, lr: float = 0.02, n_neg: int = 16,
                            seed: int = 0) -> MFModel:
    """NegGrad: push score(user, item) below sampled negatives by ascending the BPR loss,
    updating the user embedding and the item's shared (q, b). Returns a new MFModel."""
    g = np.random.default_rng(seed)
    p = base.p.copy(); q = base.q.copy(); b = base.b.copy()
    pu = torch.tensor(p[user], requires_grad=True)
    qx = torch.tensor(q[item], requires_grad=True)
    bx = torch.tensor(float(b[item]), requires_grad=True)
    opt = torch.optim.Adam([pu, qx, bx], lr=lr)
    qf = torch.from_numpy(q); bf = torch.from_numpy(b)
    for _ in range(steps):
        neg = torch.from_numpy(g.integers(0, n_items, size=n_neg))
        s_x = (pu * qx).sum() + bx
        s_neg = (qf[neg] * pu).sum(1) + bf[neg]
        # MINIMISE logsigmoid(s_x - s_neg) == gradient ASCENT on the original BPR objective.
        loss = torch.nn.functional.logsigmoid(s_x - s_neg).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    p[user] = pu.detach().numpy()
    q[item] = qx.detach().numpy()
    b[item] = float(bx.detach())
    return MFModel(p=p, q=q, b=b, dim=base.dim)


def finetune_unlearn(base: MFModel, train_minus: np.ndarray, n_users: int, n_items: int, *,
                     dim: int, epochs: int = 5, seed: int = 0) -> MFModel:
    """Continue BPR training warm-started from `base` on D\\{(A,X)} for a few epochs."""
    return train_mf(train_minus, n_users, n_items, dim=dim, epochs=epochs, seed=seed, init=base)


def _contrib_rollback(base: MFModel, user: int, item: int, hist_minus: np.ndarray,
                      n_items: int, *, dim: int, seed: int) -> MFModel:
    """Subtract user `user`'s logged contribution to the shared (q_item, b_item) and re-fit the
    user embedding on its remaining history. Requires the federated contribution logs in
    base.meta (src/federated/simulator.py). Returns a corrected MFModel."""
    cq = (base.meta or {}).get("contribs_q", {}).get(user, {})
    cb = (base.meta or {}).get("contribs_b", {}).get(user, {})
    q, b = base.q.copy(), base.b.copy()
    if item in cq:
        q[item] = q[item] - cq[item]
        b[item] = b[item] - float(cb.get(item, 0.0))
    p = base.p.copy()
    p[user] = refit_user(q, b, hist_minus, n_items, dim=dim, seed=seed)
    return MFModel(p=p, q=q, b=b, dim=base.dim)


def fru_unlearn(base: MFModel, user: int, item: int, hist_minus: np.ndarray, n_items: int, *,
                dim: int, seed: int = 0) -> MFModel:
    """FRU (federated unlearning, §3/§11): roll a user's logged contribution back out of the
    **global** shared model. Faithful given per-round contribution logs. Reaches the retrain
    floor — but the rollback changes q_item for *every* user (collateral; measured in the sweep)."""
    return _contrib_rollback(base, user, item, hist_minus, n_items, dim=dim, seed=seed)


def ours_unlearn(base: MFModel, user: int, item: int, hist_minus: np.ndarray, n_items: int, *,
                 dim: int, seed: int = 0) -> MFModel:
    """OURS (P3 — on-device residual self-suppression): the *same* contribution-rollback as FRU,
    but applied only to the user's **private** inference copy, never to the global model. Protects
    the **on-device / colluding-client** threat (a probe of A's own device model), reaching ~the
    retrain floor at **zero collateral** to other users (global q_item untouched).

    Threat-model note (§15.2): ours does NOT reduce what an honest-but-curious **server** infers from
    the *global* q_item — that residue is the fundamental limit and is only removable by touching
    OTHER users' data (see `neighbor_delete_unlearn`). Empirically ours can land *slightly below* the
    floor because the static contribution subtraction removes a touch more of A's own signal than the
    re-optimized retrain does; it still cannot remove the cross-user collaborative signal (§2)."""
    return _contrib_rollback(base, user, item, hist_minus, n_items, dim=dim, seed=seed)


def ours_suppress(base: MFModel, user: int, item: int, hist_minus: np.ndarray, n_items: int, *,
                  dim: int, alpha: float, seed: int = 0) -> MFModel:
    """OURS with a tunable suppression strength `alpha` (the privacy knob for the frontier, §6-P3),
    applied to the user's PRIVATE inference copy only (always zero collateral):
      alpha=0  -> naive (X's shared embedding intact),
      alpha=1  -> full contribution rollback == `ours_unlearn` (reaches the retrain floor),
      alpha>1  -> over-suppress along the user's own contribution direction (pushes inferability
                  toward/below the floor, but at the user's OWN held-out utility cost — it distorts
                  the user's private view of X's neighborhood).
    Sweeping alpha traces ours' privacy-vs-(own)-utility curve; it can reach the floor but the
    sub-floor region costs the requesting user's utility (never other users' — that needs §B2)."""
    cq = (base.meta or {}).get("contribs_q", {}).get(user, {})
    cb = (base.meta or {}).get("contribs_b", {}).get(user, {})
    q, b = base.q.copy(), base.b.copy()
    if item in cq:
        q[item] = q[item] - alpha * cq[item]
        b[item] = b[item] - alpha * float(cb.get(item, 0.0))
    p = base.p.copy()
    p[user] = refit_user(q, b, hist_minus, n_items, dim=dim, seed=seed)
    return MFModel(p=p, q=q, b=b, dim=base.dim)


def neighbor_delete_unlearn(base: MFModel, user: int, item: int, hist_minus: np.ndarray,
                            n_items: int, *, other_users: list[int], frac: float, dim: int,
                            seed: int = 0) -> tuple[MFModel, int]:
    """SRU-style collaborative deletion (§3): to push the requesting user's inferability BELOW the
    retrain floor you must delete *other* users' interactions with X too. Rolls back A's logged
    contribution AND a fraction `frac` of the OTHER X-consumers' contributions out of the GLOBAL
    shared q_X. This DOES reach sub-floor privacy — but it degrades those other users' utility for X
    (measured in the sweep). It is the baseline that traces the fundamental frontier below the floor;
    our whole point is that it is only reachable by harming others. Returns (model, #others deleted)."""
    cq = (base.meta or {}).get("contribs_q", {})
    cb = (base.meta or {}).get("contribs_b", {})
    q, b = base.q.copy(), base.b.copy()
    if item in cq.get(user, {}):                              # remove A's own contribution first
        q[item] = q[item] - cq[user][item]
        b[item] = b[item] - float(cb.get(user, {}).get(item, 0.0))
    others = [o for o in other_users if o != user and item in cq.get(o, {})]
    rng = np.random.default_rng(seed)
    k = int(round(frac * len(others)))
    for o in rng.permutation(others)[:k] if others else []:
        q[item] = q[item] - cq[o][item]
        b[item] = b[item] - float(cb.get(o, {}).get(item, 0.0))
    p = base.p.copy()
    p[user] = refit_user(q, b, hist_minus, n_items, dim=dim, seed=seed)
    return MFModel(p=p, q=q, b=b, dim=base.dim), k


def fedshare_unlearn(base: MFModel, user: int, item: int, hist_minus: np.ndarray, n_items: int, *,
                     dim: int, seed: int = 0) -> MFModel:
    """FedShare (arXiv 2603.11610, the CLOSEST NEIGHBOR — §3/§17): federated learn-unlearn that
    removes the representation *induced by the requesting user's own (unshared) data* via embedding
    snapshots. Faithfully, that is the difference between the current global q_X and a snapshot with
    A's contribution removed — i.e. exactly A's logged contribution to q_X. So structurally FedShare
    lands at the SAME floor residue as FRU/ours (it removes A's influence, not the collaborative
    signal). We implement it as the snapshot-difference rollback of A's contribution applied to the
    GLOBAL model (like FRU, hence non-zero collateral), so the probe can test its 'removed the
    influence' claim against our floor — the paper's foil. Reaches the floor, cannot beat it."""
    return _contrib_rollback(base, user, item, hist_minus, n_items, dim=dim, seed=seed)
