"""Download + unpack raw datasets. Run on a an HPC cluster **login node** (compute nodes have no internet,
§9). Idempotent: skips a dataset whose marker file already exists. Handles zip / gz / plain files.

    python data/scripts/download.py --datasets ml-1m ml-100k gowalla lastfm
"""
from __future__ import annotations

import argparse
import gzip
import shutil
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"

# kind: zip (extract, optionally one `member` into `into`), gz (gunzip to `out`), file (save to `out`)
SPEC: dict[str, dict] = {
    "ml-1m": {"url": "https://files.grouplens.org/datasets/movielens/ml-1m.zip",
              "kind": "zip", "marker": "ml-1m/ratings.dat"},
    "ml-100k": {"url": "https://files.grouplens.org/datasets/movielens/ml-100k.zip",
                "kind": "zip", "marker": "ml-100k/u.data"},
    "gowalla": {"url": "https://snap.stanford.edu/data/loc-gowalla_totalCheckins.txt.gz",
                "kind": "gz", "out": "gowalla/loc-gowalla_totalCheckins.txt",
                "marker": "gowalla/loc-gowalla_totalCheckins.txt"},
    "lastfm": {"url": "https://files.grouplens.org/datasets/hetrec2011/hetrec2011-lastfm-2k.zip",
               "kind": "zip", "member": "user_artists.dat", "into": "lastfm",
               "marker": "lastfm/user_artists.dat"},
    "amazon-beauty": {
        "url": "https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main/"
               "raw/review_categories/All_Beauty.jsonl",
        "kind": "file", "out": "amazon-beauty/All_Beauty.jsonl",
        "marker": "amazon-beauty/All_Beauty.jsonl"},
    # Steam (games): McAuley review dump, kept gzipped (the loader reads .gz directly). NOTE: verify
    # the URL on the login node — McAuley mirrors move; the loader's `pydict` kind matches this file.
    # Tried in order (Wang-Cheng Kang's page is the canonical SASRec Steam source); first hit wins.
    "steam": {"urls": ["https://cseweb.ucsd.edu/~wckang/steam_reviews.json.gz",
                       "http://cseweb.ucsd.edu/~wckang/steam_reviews.json.gz",
                       "https://mcauleylab.ucsd.edu/public_datasets/data/steam/steam_reviews.json.gz",
                       "https://cseweb.ucsd.edu/~jmcauley/datasets/steam/steam_reviews.json.gz"],
              "kind": "file", "out": "steam/steam_reviews.json.gz",
              "marker": "steam/steam_reviews.json.gz"},
    # Yelp (business reviews): GATED — must accept terms at https://www.yelp.com/dataset, then place
    # yelp_academic_dataset_review.json under data/raw/yelp/ (this script only prints instructions).
    "yelp": {"kind": "gated", "marker": "yelp/yelp_academic_dataset_review.json",
             "note": "Download the Yelp Open Dataset (https://www.yelp.com/dataset), extract, and "
                     "place yelp_academic_dataset_review.json at data/raw/yelp/."},
}


def _download(urls, dest: Path) -> None:
    """Download `dest` from the first working URL. `urls` may be a str or a list of mirrors tried
    in order (so a moved/404'd mirror falls through to the next)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    cands = [urls] if isinstance(urls, str) else list(urls)
    last = None
    for url in cands:
        try:
            print(f"       trying {url}")
            with urllib.request.urlopen(url, timeout=600) as r, open(dest, "wb") as f:  # noqa: S310
                shutil.copyfileobj(r, f)
            return
        except Exception as e:  # noqa: BLE001  (HTTPError/URLError/timeout -> try next mirror)
            last = e
            print(f"       .. failed ({e})")
    raise SystemExit(f"all mirrors failed for {dest.name}; last error: {last}")


def fetch(name: str) -> None:
    if name not in SPEC:
        raise SystemExit(f"unknown dataset {name}; known: {sorted(SPEC)}")
    s = SPEC[name]
    if (RAW / s["marker"]).exists():
        print(f"[skip] {name} already present")
        return
    RAW.mkdir(parents=True, exist_ok=True)
    if s["kind"] == "gated":                    # can't auto-fetch (license-gated) — instruct + stop
        raise SystemExit(f"[gated] {name}: {s['note']}")
    urls = s.get("urls", s.get("url"))          # single url or a mirror list (first hit wins)
    print(f"[get ] {name}")
    if s["kind"] == "zip":
        tmp = RAW / f"{name}.zip"; _download(urls, tmp)
        with zipfile.ZipFile(tmp) as z:
            if "member" in s:
                z.extract(s["member"], RAW / s["into"])
            else:
                z.extractall(RAW)
        tmp.unlink()
    elif s["kind"] == "gz":
        tmp = RAW / f"{name}.gz"; _download(urls, tmp)
        (RAW / s["out"]).parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(tmp, "rb") as gz, open(RAW / s["out"], "wb") as f:
            shutil.copyfileobj(gz, f)
        tmp.unlink()
    elif s["kind"] == "file":
        _download(urls, RAW / s["out"])
    print(f"[done] {RAW / s['marker']}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["ml-1m", "ml-100k", "gowalla", "lastfm"])
    for d in ap.parse_args().datasets:
        fetch(d)


if __name__ == "__main__":
    main()
