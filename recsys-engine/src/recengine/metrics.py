"""Ranking metrics, list-level diversity metrics and the bootstrap."""
from __future__ import annotations

import numpy as np


def topk(scores: np.ndarray, k: int) -> np.ndarray:
    """Indices of the k highest scores per row, best first."""
    part = np.argpartition(-scores, k, axis=1)[:, :k]
    order = np.argsort(-np.take_along_axis(scores, part, 1), axis=1, kind="stable")
    return np.take_along_axis(part, order, 1)


def per_user_metrics(top: np.ndarray, rel: np.ndarray, k: int) -> dict[str, np.ndarray]:
    """top: (n,K>=k) recommended item ids; rel: (n, n_items) 0/1 relevance. Users need >=1 relevant item."""
    gains = np.take_along_axis(rel, top[:, :k], 1)
    n_rel = rel.sum(1)
    disc = 1.0 / np.log2(np.arange(2, k + 2))
    ideal = np.array([disc[:int(min(m, k))].sum() for m in n_rel])
    return {f"recall@{k}": gains.sum(1) / n_rel, f"ndcg@{k}": (gains * disc).sum(1) / ideal,
            f"hit@{k}": (gains.sum(1) > 0).astype(float)}


def bootstrap_ci(values, n_boot=1000, seed=0):
    v = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, len(v), size=(n_boot, len(v)))].mean(1)
    return float(v.mean()), (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def paired_ci(a, b, n_boot=1000, seed=0):
    return bootstrap_ci(np.asarray(a, float) - np.asarray(b, float), n_boot, seed)


def list_metrics(top: np.ndarray, item_pop: np.ndarray, k: int) -> dict[str, float]:
    """Catalogue coverage, novelty (mean self-information of recommended items), Gini of exposure."""
    n_items = len(item_pop)
    rec = top[:, :k].ravel()
    counts = np.bincount(rec, minlength=n_items).astype(float)
    p = (item_pop + 1.0) / (item_pop.sum() + n_items)
    srt = np.sort(counts)
    n = len(srt)
    gini = float((2 * np.arange(1, n + 1) - n - 1).dot(srt) / (n * srt.sum())) if srt.sum() else 0.0
    return {"coverage": round(float((counts > 0).mean()), 4), "novelty": round(float((-np.log2(p[rec])).mean()), 3),
            "gini": round(gini, 4)}


def insert_cold(scores: np.ndarray, cold_item: np.ndarray, k: int, share: float = 0.2) -> np.ndarray:
    """Top-k lists where about `share` of the slots are reserved for brand-new items.

    Brand-new items have no history, so a ranker that trusts interaction signals buries them. Production
    systems reserve a few slots for them (exploration). The best cold items (by `scores`) are inserted at evenly
    spaced positions; the rest of the list is the best non-cold items. share=0 returns the plain top-k."""
    m = int(round(k * share))
    plain = np.where(cold_item[None, :], -np.inf, scores)
    base = topk(np.where(np.isfinite(plain), plain, -np.inf), k)
    if m == 0 or not cold_item.any():
        return topk(scores, k)
    cold_only = np.where(cold_item[None, :], scores, -np.inf)
    cold_top = topk(cold_only, min(m, int(cold_item.sum())))
    out = base.copy()
    pos = [int(round(k * (j + 1) / (cold_top.shape[1] + 1))) - 1 for j in range(cold_top.shape[1])]
    keep = [c for c in range(k) if c not in pos]
    out[:, keep] = base[:, : len(keep)]
    out[:, pos] = cold_top
    return out
