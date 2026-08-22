"""Sequential recommender (GRU4Rec-style) for the SEQUENTIAL-residue claim (§2, §6-P1,
§11). Predicts the next item from a user's history sequence; item embeddings are the SHARED
federated object (as in MF), and the GRU carries the *within-user* autoregressive dependence — the
two channels the sequential residue lives in.

We keep the model tiny and fully seeded so a from-scratch retrain excluding an interaction is a
deterministic unlearning oracle (§4.1). `predict_vec(seq)` returns the next-item prediction vector
h so that score(seq, i) = <h, item_emb[i]> — mirroring MFModel.scores, so the same probe/controls
machinery applies. Federated extension (one client = one user's sequence, FedAvg over item+GRU
params) is a drop-in; the centralized version is the cleanest demonstration of the effect first.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import torch
from torch import nn


@dataclasses.dataclass
class SeqModel:
    item_emb: np.ndarray            # (n_items, dim) shared item embeddings (the residue substrate)
    _net: "GRU4Rec"
    n_items: int
    dim: int

    def predict_vec(self, seq: list[int] | np.ndarray) -> np.ndarray:
        """Next-item prediction vector h for a history `seq` (list of item ids, chronological)."""
        return self._net.predict_vec(seq, self.n_items)

    def score(self, seq: list[int] | np.ndarray, items: np.ndarray) -> np.ndarray:
        h = self.predict_vec(seq)
        return self.item_emb[items] @ h


class GRU4Rec(nn.Module):
    def __init__(self, n_items: int, dim: int = 64, hidden: int = 64):
        super().__init__()
        self.emb = nn.Embedding(n_items + 1, dim, padding_idx=n_items)   # id n_items = PAD
        self.gru = nn.GRU(dim, hidden, batch_first=True)
        self.out = nn.Linear(hidden, dim)
        self.n_items = n_items

    def forward(self, seqs: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """seqs (B,L) padded with n_items; returns per-position prediction vectors (B,L,dim)."""
        e = self.emb(seqs)
        h, _ = self.gru(e)
        return self.out(h)

    @torch.no_grad()
    def predict_vec(self, seq: list[int] | np.ndarray, n_items: int) -> np.ndarray:
        if len(seq) == 0:
            return np.zeros(self.out.out_features, dtype=np.float32)
        s = torch.tensor(np.asarray(seq, dtype=np.int64)[None, :])
        e = self.emb(s)
        h, _ = self.gru(e)
        return self.out(h[0, -1]).numpy().astype(np.float32)


def train_gru4rec(sequences: list[np.ndarray], n_items: int, *, dim: int = 64, hidden: int = 64,
                  epochs: int = 15, lr: float = 5e-3, n_neg: int = 8, max_len: int = 50,
                  reg: float = 1e-6, seed: int = 0) -> SeqModel:
    """Train GRU4Rec with sampled BPR over next-item transitions. `sequences[u]` = user u's
    chronological train item ids. Deterministic given `seed`."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net = GRU4Rec(n_items, dim, hidden)
    nn.init.normal_(net.emb.weight, std=0.01)
    with torch.no_grad():
        net.emb.weight[n_items].zero_()
    opt = torch.optim.Adam(net.parameters(), lr=lr)

    # build (prefix -> next) training pairs as padded sequences, truncated to max_len.
    seqs = [s[-(max_len + 1):] for s in sequences if len(s) >= 2]
    for _ep in range(epochs):
        order = rng.permutation(len(seqs))
        for bstart in range(0, len(order), 128):
            idx = order[bstart:bstart + 128]
            batch = [seqs[i] for i in idx]
            L = max(len(s) - 1 for s in batch)
            inp = np.full((len(batch), L), n_items, dtype=np.int64)
            tgt = np.full((len(batch), L), -1, dtype=np.int64)
            for r, s in enumerate(batch):
                inp[r, :len(s) - 1] = s[:-1]
                tgt[r, :len(s) - 1] = s[1:]
            inp_t = torch.from_numpy(inp)
            pred = net(inp_t, None)                                   # (B,L,dim)
            mask = torch.from_numpy(tgt >= 0)
            tgt_c = np.clip(tgt, 0, n_items - 1)
            pos_e = net.emb(torch.from_numpy(tgt_c))                  # (B,L,dim)
            neg_ids = torch.from_numpy(rng.integers(0, n_items, size=(len(batch), L, n_neg)))
            neg_e = net.emb(neg_ids)                                  # (B,L,n_neg,dim)
            s_pos = (pred * pos_e).sum(-1)                            # (B,L)
            s_neg = (pred.unsqueeze(2) * neg_e).sum(-1)               # (B,L,n_neg)
            loss_pos = -torch.nn.functional.logsigmoid(s_pos.unsqueeze(-1) - s_neg).mean(-1)
            loss = (loss_pos * mask).sum() / mask.sum().clamp(min=1)
            loss = loss + reg * (pred.pow(2).sum(-1) * mask).sum() / mask.sum().clamp(min=1)
            opt.zero_grad(); loss.backward(); opt.step()

    item_emb = net.emb.weight.detach().numpy()[:n_items].copy()
    return SeqModel(item_emb=item_emb, _net=net, n_items=n_items, dim=dim)
