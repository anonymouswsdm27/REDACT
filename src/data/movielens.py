"""ML-1M preprocessing for the Phase 0 collaborative-residue kill gate (§5).

Implicit-feedback leave-one-out split (standard recsys eval, §10). For each item we
precompute the cross-user *redundancy* = number of distinct users with that item in
their TRAIN history; this is the central stratifier for the residue-vs-redundancy
signature (§2). A user may be subsampled to keep from-scratch retrains cheap on CPU
(§14.2) while preserving a natural long-tail redundancy distribution that includes the
redundancy-0 stratum (the mandatory negative control, §4.3).
"""
from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]  # repo root (src/data/movielens.py -> ../../..)
PROC = _ROOT / "data" / "processed"

# dataset registry: name -> loader spec. `csv` = delimited (MovieLens); `jsonl` = one JSON per
# line (Amazon Reviews'23). IDs may be strings; preprocess factorizes them. The rest of the
# pipeline is dataset-agnostic (§10) — add a new domain here + a config, nothing else changes.
_RAW = _ROOT / "data" / "raw"
# `cols` = 0-based column indices (rating omitted => implicit feedback = 1). ISO/epoch timestamps
# both sort correctly. `fields` = JSON keys for jsonl. Dense domains only (the residue-vs-redundancy
# analysis needs items with many co-users, §10) — MovieLens/Steam/Gowalla/Yelp/LastFM.
SOURCES = {
    "ml1m":   {"kind": "csv", "path": _RAW / "ml-1m" / "ratings.dat", "sep": "::",
               "cols": {"user": 0, "item": 1, "rating": 2, "ts": 3}},
    "ml100k": {"kind": "csv", "path": _RAW / "ml-100k" / "u.data", "sep": "\t",
               "cols": {"user": 0, "item": 1, "rating": 2, "ts": 3}},
    "gowalla": {"kind": "csv", "path": _RAW / "gowalla" / "loc-gowalla_totalCheckins.txt",
                "sep": "\t", "cols": {"user": 0, "item": 4, "ts": 1}},              # POI check-ins
    "lastfm": {"kind": "csv", "path": _RAW / "lastfm" / "user_artists.dat", "sep": "\t",
               "skiprows": 1, "cols": {"user": 0, "item": 1, "rating": 2}},         # music (no ts)
    # Steam (games): McAuley review dump — one PYTHON-DICT (not JSON) per line, gzip-read directly.
    "steam": {"kind": "pydict", "path": _RAW / "steam" / "steam_reviews.json.gz",
              "fields": ("username", "product_id", None, "date")},                  # rating omitted
    # Yelp (business reviews): GATED download (accept terms at yelp.com/dataset), standard JSONL.
    "yelp": {"kind": "jsonl", "path": _RAW / "yelp" / "yelp_academic_dataset_review.json",
             "fields": ("user_id", "business_id", "stars", "date")},
    "amazon-beauty": {"kind": "jsonl", "path": _RAW / "amazon-beauty" / "All_Beauty.jsonl",
                      "fields": ("user_id", "asin", "rating", "timestamp")},         # sparse (contrast only)
}
RAW = SOURCES["ml1m"]["path"]  # back-compat default


def _read_raw(dataset: str) -> pd.DataFrame:
    """Load a dataset's raw interactions as a DataFrame[user, item, rating, ts] (IDs may be str)."""
    src = SOURCES[dataset]
    if src["kind"] == "csv":
        c = src["cols"]
        engine = "c" if len(src["sep"]) == 1 else "python"   # C engine (fast) for 1-char seps
        raw = pd.read_csv(src["path"], sep=src["sep"], engine=engine, header=None,
                          skiprows=src.get("skiprows", 0), usecols=sorted(set(c.values())))
        df = pd.DataFrame({"user": raw[c["user"]], "item": raw[c["item"]],
                           "rating": raw[c["rating"]] if "rating" in c else 1})
        df["ts"] = raw[c["ts"]] if "ts" in c else np.arange(len(df))   # no timestamp -> file order
        return df
    if src["kind"] in ("jsonl", "pydict"):
        # jsonl = one JSON per line (Amazon/Yelp); pydict = one PYTHON dict repr per line (Steam,
        # ast.literal_eval). Both are read directly from .gz if the path ends in .gz (avoids
        # decompressing multi-GB dumps). `fields`=(user,item,rating|None,ts|None); None rating -> 1,
        # None ts -> file order.
        import ast
        import gzip
        import json
        parse = json.loads if src["kind"] == "jsonl" else ast.literal_eval
        fu, fi, fr, ft = src["fields"]
        opener = gzip.open if str(src["path"]).endswith(".gz") else open
        u, it, r, t = [], [], [], []
        with opener(src["path"], "rt") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                d = parse(line)
                u.append(d[fu]); it.append(d[fi])
                r.append(d[fr] if fr else 1)
                t.append(d[ft] if ft else len(t))
        return pd.DataFrame({"user": u, "item": it, "rating": r, "ts": t})
    raise ValueError(f"unknown source kind {src['kind']}")


@dataclasses.dataclass
class Dataset:
    n_users: int
    n_items: int
    train: np.ndarray          # (N, 2) int32 [user, item]
    val: np.ndarray            # (n_users, 2) int32 [user, item]  (held-out, -1 item if none)
    test: np.ndarray           # (n_users, 2) int32
    item_pop: np.ndarray       # (n_items,) int32  distinct TRAIN users per item (redundancy)
    user_items: list[np.ndarray]  # per-user sorted train item ids
    seed: int

    @property
    def n_train(self) -> int:
        return int(self.train.shape[0])


def preprocess(n_users: int | None, seed: int = 0, min_inter: int = 5,
               dataset: str = "ml1m") -> Dataset:
    """Load a MovieLens variant, optionally subsample users, reindex, leave-one-out split.

    Args:
        n_users: subsample this many users (None = all). Densest are kept implicitly via
            the min_inter filter; selection among eligible users is random under `seed`.
        seed: RNG seed for user subsampling (§4.6).
        min_inter: drop users with fewer than this many interactions (need >=3 for LOO).
        dataset: key into SOURCES (e.g. "ml1m", "ml100k").
    """
    df = _read_raw(dataset)
    # implicit feedback: every rating is a positive interaction.
    cnt = df.groupby("user").size()
    eligible = cnt[cnt >= min_inter].index.to_numpy()
    rng = np.random.default_rng(seed)
    if n_users is not None and n_users < len(eligible):
        eligible = rng.choice(eligible, size=n_users, replace=False)
    df = df[df["user"].isin(eligible)].copy()

    # drop items with no interactions in this user subset, then reindex both axes.
    uimap = {u: i for i, u in enumerate(sorted(df["user"].unique()))}
    iimap = {it: i for i, it in enumerate(sorted(df["item"].unique()))}
    df["user"] = df["user"].map(uimap).astype(np.int32)
    df["item"] = df["item"].map(iimap).astype(np.int32)
    n_u, n_i = len(uimap), len(iimap)

    # per-user leave-one-out by timestamp: last -> test, 2nd-last -> val, rest -> train.
    df = df.sort_values(["user", "ts"], kind="stable")
    train_rows: list[np.ndarray] = []
    val = np.full((n_u, 2), -1, dtype=np.int32)
    test = np.full((n_u, 2), -1, dtype=np.int32)
    for u, g in df.groupby("user", sort=True):
        items = g["item"].to_numpy()
        test[u] = (u, items[-1])
        if len(items) >= 2:
            val[u] = (u, items[-2])
        if len(items) >= 3:
            tr = items[:-2]
        else:
            tr = items[:0]
        ui = np.empty((len(tr), 2), dtype=np.int32)
        ui[:, 0] = u
        ui[:, 1] = tr
        train_rows.append(ui)
    train = np.concatenate(train_rows, axis=0)

    item_pop = np.zeros(n_i, dtype=np.int32)
    # distinct (user,item) already; count users per item in train.
    np.add.at(item_pop, train[:, 1], 1)

    user_items = [np.array([], dtype=np.int32) for _ in range(n_u)]
    order = np.argsort(train[:, 0], kind="stable")
    t_sorted = train[order]
    # split into per-user blocks
    bounds = np.searchsorted(t_sorted[:, 0], np.arange(n_u + 1))
    for u in range(n_u):
        user_items[u] = np.sort(t_sorted[bounds[u]:bounds[u + 1], 1])

    return Dataset(n_u, n_i, train, val, test, item_pop, user_items, seed)


def summarize(ds: Dataset) -> str:
    pop = ds.item_pop
    deg = np.array([len(x) for x in ds.user_items])
    q = np.percentile(pop, [50, 90, 99]).astype(int)
    return (
        f"users={ds.n_users} items={ds.n_items} train={ds.n_train} "
        f"avg_user_deg={deg.mean():.1f} "
        f"item_pop: zero1={int((pop<=1).sum())} median={q[0]} p90={q[1]} p99={q[2]} max={int(pop.max())}"
    )


if __name__ == "__main__":
    import sys
    n = None if len(sys.argv) < 2 or sys.argv[1] == "full" else int(sys.argv[1])
    ds = preprocess(n_users=n, seed=0)
    print(summarize(ds))
