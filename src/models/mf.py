"""Matrix-factorization collaborative backbone for the Phase 0 spike (§11).

score(u, i) = <p_u, q_i> + b_i, trained with BPR (implicit pairwise ranking).
- q_i, b_i are the *shared* item parameters (the federated object of interest, §10): every
  user who interacts with item i reinforces q_i. This is where the collaborative residue lives.
- p_u is the *private* per-user embedding (stays on the client in the FL setting).

This is the GMF half of NCF and the cleanest testbed for the collaborative-residue signal
(§11: "begin with the simplest that exhibits the effect"). Training is fully seeded so a
from-scratch retrain excluding an interaction is a deterministic, provably-correct unlearning
oracle (§4.1). CPU-only, vectorized for speed (§14.2).
"""
from __future__ import annotations

import dataclasses

import numpy as np
import torch


@dataclasses.dataclass
class MFModel:
    p: np.ndarray   # (n_users, dim)
    q: np.ndarray   # (n_items, dim)
    b: np.ndarray   # (n_items,)
    dim: int
    meta: dict | None = None   # federated extras (e.g. per-user contribution logs for FRU/ours)

    def scores(self, user: int, items: np.ndarray) -> np.ndarray:
        """Score one user against an array of item ids."""
        return self.q[items] @ self.p[user] + self.b[items]


def train_mf(
    train: np.ndarray,
    n_users: int,
    n_items: int,
    *,
    dim: int = 32,
    epochs: int = 40,
    batch: int = 8192,
    lr: float = 0.01,
    reg: float = 1e-5,
    seed: int = 0,
    init: "MFModel | None" = None,
    verbose: bool = False,
) -> MFModel:
    """Train MF with BPR. Deterministic given `seed` (so retrain == correct unlearning, §4.1).

    `init` warm-starts P/Q/B from an existing model — used by fine-tune unlearning (continue
    training on D\\{(A,X)} for a few epochs) so it is cheaper than a cold retrain.
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    dev = torch.device("cpu")
    torch.set_num_threads(max(1, torch.get_num_threads()))

    # sparse embeddings + SparseAdam: only the items/users touched in a batch are updated, so cost
    # is O(batch), independent of the item-catalog size -> feasible for large catalogs (Gowalla,
    # Yelp) as well as dense ones. L2 is applied to the looked-up rows (below) to keep grads sparse.
    P = torch.nn.Embedding(n_users, dim, sparse=True).to(dev)
    Q = torch.nn.Embedding(n_items, dim, sparse=True).to(dev)
    B = torch.nn.Embedding(n_items, 1, sparse=True).to(dev)
    if init is not None:
        with torch.no_grad():
            P.weight.copy_(torch.from_numpy(init.p))
            Q.weight.copy_(torch.from_numpy(init.q))
            B.weight.copy_(torch.from_numpy(init.b).unsqueeze(1))
    else:
        torch.nn.init.normal_(P.weight, std=0.01)
        torch.nn.init.normal_(Q.weight, std=0.01)
        torch.nn.init.zeros_(B.weight)
    opt = torch.optim.SparseAdam([P.weight, Q.weight, B.weight], lr=lr)

    u_all = torch.from_numpy(train[:, 0].astype(np.int64))
    i_all = torch.from_numpy(train[:, 1].astype(np.int64))
    n = train.shape[0]

    for ep in range(epochs):
        perm = torch.from_numpy(rng.permutation(n))
        tot = 0.0
        for s in range(0, n, batch):
            idx = perm[s:s + batch]
            u = u_all[idx]
            ipos = i_all[idx]
            ineg = torch.from_numpy(rng.integers(0, n_items, size=len(idx)))
            pu, qp, qn = P(u), Q(ipos), Q(ineg)
            spos = (pu * qp).sum(1) + B(ipos).squeeze(1)
            sneg = (pu * qn).sum(1) + B(ineg).squeeze(1)
            loss = (-torch.nn.functional.logsigmoid(spos - sneg).mean()
                    + reg * (pu.pow(2).sum(1) + qp.pow(2).sum(1) + qn.pow(2).sum(1)).mean())
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
        if verbose and (ep % 10 == 0 or ep == epochs - 1):
            print(f"  ep{ep:02d} bpr={tot / n:.4f}")

    return MFModel(
        p=P.weight.detach().numpy().copy(),
        q=Q.weight.detach().numpy().copy(),
        b=B.weight.detach().numpy().squeeze(1).copy(),
        dim=dim,
    )


def refit_user(q: np.ndarray, b: np.ndarray, pos_items: np.ndarray, n_items: int,
               *, dim: int, steps: int = 200, lr: float = 0.05, n_neg: int = 8,
               seed: int = 0) -> np.ndarray:
    """Re-fit a single user embedding p_u against FROZEN shared item params (q, b).

    This is the on-device action of **naive local-delete** (§11): the client drops an
    interaction and retrains *its own* embedding, but the shared item embeddings q (which still
    carry the user's historical contribution) are untouched. Cold-started and fit to convergence
    -> the most *generous* local re-fit (best case for the strawman). Returns p_u."""
    if len(pos_items) == 0:
        return np.zeros(dim, dtype=np.float32)
    g = np.random.default_rng(seed)
    qt = torch.from_numpy(q); bt = torch.from_numpy(b)
    pos = torch.from_numpy(pos_items.astype(np.int64))
    # init from this function's own seeded RNG (NOT the global torch RNG) so the result is
    # deterministic regardless of call order / global state (tests/test_sweep.py).
    pu = torch.tensor((g.standard_normal(dim) * 0.01).astype(np.float32), requires_grad=True)
    opt = torch.optim.Adam([pu], lr=lr)
    P = len(pos_items)
    for _ in range(steps):
        ineg = torch.from_numpy(g.integers(0, n_items, size=(P, n_neg)))
        spos = (qt[pos] * pu).sum(1) + bt[pos]                      # (P,)
        sneg = (qt[ineg] * pu).sum(2) + bt[ineg]                    # (P, n_neg)
        loss = -torch.nn.functional.logsigmoid(spos[:, None] - sneg).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return pu.detach().numpy().copy()


def evaluate_ranking(model: MFModel, eval_pairs: np.ndarray, user_items: list[np.ndarray],
                     n_items: int, k: int = 10, n_neg: int = 99, seed: int = 123) -> dict:
    """Leave-one-out HR@k / NDCG@k against `n_neg` sampled negatives (sanity that the model
    actually learned to rank, §12). Skips users with no held-out item."""
    rng = np.random.default_rng(seed)
    hr, ndcg, m = 0.0, 0.0, 0
    for u, pos in eval_pairs:
        if pos < 0:
            continue
        seen = set(user_items[u].tolist())
        negs = []
        while len(negs) < n_neg:
            c = int(rng.integers(0, n_items))
            if c != pos and c not in seen:
                negs.append(c)
        cand = np.array([pos] + negs)
        sc = model.scores(int(u), cand)
        rank = int((sc > sc[0]).sum())  # rank of the true item (0 = top)
        if rank < k:
            hr += 1.0
            ndcg += 1.0 / np.log2(rank + 2)
        m += 1
    return {"HR@%d" % k: hr / m, "NDCG@%d" % k: ndcg / m, "n_eval": m}
