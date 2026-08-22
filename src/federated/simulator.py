"""FedAvg training of a GMF recommender — the GMF instantiation of FedNCF (§8/§11).

**Vectorized** (no per-client Python loop): a round trains every client's local update at once via
scatter/gather, so it is 10-50x faster than looping over clients — the load-bearing speedup for
full-scale federated sweeps on a cluster (the old loop was ~16 min/retrain on full ML-1M).

Semantics (unchanged): one client = one user. Item embeddings `q` and biases `b` are the SHARED
federated object — each round every client does local BPR on its interactions from the round-start
global model, and the server **FedAvg-averages** the per-item deltas over the clients that touched
each item (the `/counts` averaging IS the cross-user dilution, §2). The user embedding `p_u` is
PRIVATE. We log each user's contribution to the shared params (its averaged-in delta per item,
accumulated over rounds) so FRU can roll it back globally and *ours* locally (see src/unlearning/).

Vectorization detail: within a round we keep a *per-interaction* local copy of each touched item
embedding (so clients don't see each other's in-round updates — the FedAvg property) and update the
private `p_u` by scatter-reducing gradients over each user's interactions. Deterministic given
`seed`. Returns an `MFModel` (drop-in for the probe/metrics) with logs under `.meta`.
"""
from __future__ import annotations

import numpy as np
import torch

from ..models.mf import MFModel


def train_fedgmf(train: np.ndarray, n_users: int, n_items: int, *, dim: int = 32,
                 rounds: int = 15, local_epochs: int = 3, lr: float = 0.1, reg: float = 1e-5,
                 n_neg: int = 1, clients_per_round: int | None = None, seed: int = 0,
                 clip: float = 10.0, dp_sigma: float = 0.0, participation: float = 1.0,
                 log_contribs: bool = True) -> MFModel:
    """FedAvg-trained GMF, vectorized. `clients_per_round` is accepted for API compatibility;
    full participation (all clients each round) is used by default, standard for this simulator.

    `dp_sigma` > 0 adds Gaussian noise (std = dp_sigma·lr per round) to each item's AVERAGED global
    update — the DP-FedAvg / secure-aggregation-noise ablation (§6-P5, F4). This perturbs the
    aggregate but does NOT remove any single user's *contribution* to it, so the cross-user
    collaborative residue is expected to persist (the point of the ablation).

    `participation` < 1.0 selects a random fraction of clients (users) EACH round — the realistic
    non-IID / partial-participation FL ablation (§6-P5, F1). Only the selected users' interactions
    contribute to that round's aggregation and only their private p_u updates; over many rounds all
    users still contribute (a user participates in ~`participation` of rounds). Set to 1.0 (default)
    reproduces full participation *bit-for-bit* (no extra RNG is drawn)."""
    g = np.random.default_rng(seed)
    torch.manual_seed(seed)
    q = torch.from_numpy((g.standard_normal((n_items, dim)) * 0.01).astype(np.float32))
    b = torch.zeros(n_items)
    p = torch.from_numpy((g.standard_normal((n_users, dim)) * 0.01).astype(np.float32))
    u = torch.from_numpy(train[:, 0].astype(np.int64))
    it = torch.from_numpy(train[:, 1].astype(np.int64))
    n = len(train)
    ones = torch.ones(n)
    cq_acc = torch.zeros(n, dim) if log_contribs else None      # per-interaction contribution to q[i]
    cb_acc = torch.zeros(n) if log_contribs else None

    def cliprows(x):   # max-norm clip: scale down only rows whose L2 norm exceeds `clip`
        if not clip:
            return x
        return x * (clip / x.norm(dim=1, keepdim=True).clamp(min=1e-12)).clamp(max=1.0)

    for _ in range(rounds):
        j = torch.from_numpy(g.integers(0, n_items, size=n))    # one negative per interaction/round
        # partial participation: sample the round's clients; w=1 if an interaction's user is in.
        # (guarded: full participation draws NO extra RNG and uses `ones`, so it is bit-identical.)
        if participation < 1.0:
            part_users = torch.from_numpy(g.random(n_users) < participation)
            w = part_users[u].float()                           # (n,) per-interaction participation
        else:
            part_users = None
            w = ones
        p_loc = p.clone()                                       # private local user embeddings
        qlp, blp = q[it].clone(), b[it].clone()                 # per-interaction local POSITIVE copies
        qln, bln = q[j].clone(), b[j].clone()                   # per-interaction local NEGATIVE copies
        for _e in range(local_epochs):
            pu = p_loc[u]                                       # (n, dim) gather
            gc = torch.sigmoid((pu * qln).sum(1) + bln - (pu * qlp).sum(1) - blp).unsqueeze(1)
            if participation < 1.0:
                gc = gc * w.unsqueeze(1)                        # non-participating clients: 0 gradient
            gp = torch.zeros_like(p_loc)
            gp.index_add_(0, u, gc * (qlp - qln))               # scatter grads to each user's p
            p_loc = p_loc + lr * (gp - reg * p_loc)
            qlp = qlp + lr * (gc * pu - reg * qlp)              # push positives up
            qln = qln - lr * (gc * pu)                          # push negatives down
            blp = (blp + lr * gc.squeeze(1)).clamp(-clip, clip) if clip else blp + lr * gc.squeeze(1)
            bln = (bln - lr * gc.squeeze(1)).clamp(-clip, clip) if clip else bln - lr * gc.squeeze(1)
            p_loc, qlp, qln = cliprows(p_loc), cliprows(qlp), cliprows(qln)   # prevent divergence

        # FedAvg: average per-item deltas over the (participating) interactions that touched each item.
        cnt = torch.zeros(n_items)
        cnt.index_add_(0, it, w); cnt.index_add_(0, j, w)
        dqp, dbp = qlp - q[it], blp - b[it]                     # local deltas (vs round-start global)
        dq = torch.zeros_like(q); db = torch.zeros_like(b)
        if participation < 1.0:
            wq = w.unsqueeze(1)
            dq.index_add_(0, it, wq * dqp);  dq.index_add_(0, j, wq * (qln - q[j]))
            db.index_add_(0, it, w * dbp);   db.index_add_(0, j, w * (bln - b[j]))
        else:
            dq.index_add_(0, it, dqp);  dq.index_add_(0, j, qln - q[j])
            db.index_add_(0, it, dbp);  db.index_add_(0, j, bln - b[j])
        if log_contribs:                                        # user's SHARE of the applied update
            safe = cnt[it].clamp(min=1e-12) if participation < 1.0 else cnt[it]
            cq_acc += (w.unsqueeze(1) * dqp if participation < 1.0 else dqp) / safe.unsqueeze(1)
            cb_acc += (w * dbp if participation < 1.0 else dbp) / safe
        m = cnt > 0
        upd_q = dq[m] / cnt[m].unsqueeze(1)
        upd_b = db[m] / cnt[m]
        if dp_sigma:                                            # DP-FedAvg noise on the aggregate (F4)
            gen = torch.Generator().manual_seed(int(g.integers(0, 2**31)))
            upd_q = upd_q + dp_sigma * lr * torch.randn(upd_q.shape, generator=gen)
            upd_b = upd_b + dp_sigma * lr * torch.randn(upd_b.shape, generator=gen)
        q[m] += upd_q
        b[m] += upd_b
        if clip:
            q.copy_(cliprows(q)); b.clamp_(-clip, clip)         # bound the shared model each round
        if part_users is not None:                              # only participating users' p_u update
            p_next = p.clone(); p_next[part_users] = p_loc[part_users]; p = p_next
        else:
            p = p_loc                                          # private (already clipped above)

    meta = None
    if log_contribs:
        cq: dict[int, dict[int, np.ndarray]] = {}
        cb: dict[int, dict[int, float]] = {}
        u_np, i_np = train[:, 0], train[:, 1]
        cq_np, cb_np = cq_acc.numpy(), cb_acc.numpy()
        for k in range(n):                                     # one-off: (u,i) is unique per interaction
            uu, ii = int(u_np[k]), int(i_np[k])
            cq.setdefault(uu, {})[ii] = cq_np[k]
            cb.setdefault(uu, {})[ii] = float(cb_np[k])
        meta = {"contribs_q": cq, "contribs_b": cb, "backbone": "fedgmf"}
    return MFModel(p=p.numpy().copy(), q=q.numpy().copy(), b=b.numpy().copy(), dim=dim, meta=meta)
