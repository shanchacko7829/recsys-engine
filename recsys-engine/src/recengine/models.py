"""Recommendation models written from scratch (numpy / scipy only).

All models expose  fit(...)  and  score(user_ids) -> (len(user_ids), n_items) float32.
Items a user has already seen are filtered by the caller, not here.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp


class Popularity:
    """Global count, optionally exponentially down-weighting old events (half_life in time units)."""
    name = "popularity"

    def __init__(self, half_life: float | None = None):
        self.half_life = half_life

    def fit(self, ev, n_users, n_items, t_now: float):
        w = np.ones(len(ev)) if not self.half_life else 0.5 ** ((t_now - ev.times) / self.half_life)
        self.pop = np.bincount(ev.items, weights=w, minlength=n_items).astype(np.float32)
        return self

    def score(self, users):
        return np.tile(self.pop, (len(users), 1))


class ItemKNN:
    """Item-item cosine similarity on the binary interaction matrix, keep top-k neighbours per item."""
    name = "item_knn"

    def __init__(self, k: int = 50, shrink: float = 10.0):
        self.k, self.shrink = k, shrink

    def fit(self, X: sp.csr_matrix):
        self.X = X
        co = (X.T @ X).toarray().astype(np.float32)               # co-occurrence counts
        norms = np.sqrt(np.maximum(np.diag(co), 1e-9))
        sim = co / (norms[:, None] * norms[None, :] + self.shrink)  # shrunk cosine: rare items matter less
        np.fill_diagonal(sim, 0.0)
        if self.k < sim.shape[0]:                                   # sparsify: keep strongest neighbours
            cut = np.partition(sim, -self.k, axis=1)[:, -self.k][:, None]
            sim = np.where(sim >= cut, sim, 0.0)
        self.sim = sim.astype(np.float32)
        return self

    def score(self, users):
        return np.asarray(self.X[users] @ self.sim, dtype=np.float32)


class ALS:
    """Implicit-feedback matrix factorisation (Hu, Koren, Volinsky 2008), alternating least squares.

    Minimises  sum_ui c_ui (p_ui - x_u.y_i)^2 + lambda(|X|^2+|Y|^2)  with p=1 on observed pairs,
    c = 1 + alpha on observed and 1 elsewhere. Each half-step solves a small ridge system per row,
    using  Y^T C Y = Y^T Y + Y^T (C-I) Y  so only observed entries are touched.
    """
    name = "als"

    def __init__(self, factors=48, reg=0.05, alpha=20.0, iters=12, seed=42):
        self.f, self.reg, self.alpha, self.iters, self.seed = factors, reg, alpha, iters, seed

    def _solve_side(self, R: sp.csr_matrix, Y: np.ndarray) -> np.ndarray:
        f = Y.shape[1]
        YtY = Y.T @ Y + self.reg * np.eye(f, dtype=np.float32)
        out = np.zeros((R.shape[0], f), dtype=np.float32)
        ptr, idx = R.indptr, R.indices
        for u in range(R.shape[0]):
            cols = idx[ptr[u]:ptr[u + 1]]
            if len(cols) == 0:
                continue
            Yu = Y[cols]
            A = YtY + self.alpha * (Yu.T @ Yu)
            b = (1.0 + self.alpha) * Yu.sum(axis=0)
            out[u] = np.linalg.solve(A, b)
        return out

    def fit(self, X: sp.csr_matrix):
        rng = np.random.default_rng(self.seed)
        n_u, n_i = X.shape
        self.U = (0.01 * rng.normal(size=(n_u, self.f))).astype(np.float32)
        self.V = (0.01 * rng.normal(size=(n_i, self.f))).astype(np.float32)
        XT = X.T.tocsr()
        for _ in range(self.iters):
            self.U = self._solve_side(X, self.V)
            self.V = self._solve_side(XT, self.U)
        return self

    def score(self, users):
        return self.U[users] @ self.V.T

    def fold_in(self, item_ids) -> np.ndarray:
        """Embed a user who was not in training from their item list (one ridge solve)."""
        Yu = self.V[item_ids]
        A = self.V.T @ self.V + self.reg * np.eye(self.f, dtype=np.float32) + self.alpha * (Yu.T @ Yu)
        return np.linalg.solve(A, (1.0 + self.alpha) * Yu.sum(axis=0))


class BPR:
    """Bayesian Personalised Ranking (Rendle et al. 2009): mini-batch SGD on  log sigmoid(s_ui - s_uj)
    with j sampled uniformly from items the user has not interacted with. Includes an item bias."""
    name = "bpr"

    def __init__(self, factors=48, lr=0.08, reg=0.002, epochs=25, batch=2048, seed=42):
        self.f, self.lr, self.reg, self.epochs, self.batch, self.seed = factors, lr, reg, epochs, batch, seed

    def fit(self, X: sp.csr_matrix):
        rng = np.random.default_rng(self.seed)
        n_u, n_i = X.shape
        coo = X.tocoo()
        pu, pi = coo.row.astype(np.int64), coo.col.astype(np.int64)
        keys = np.sort(pu * n_i + pi)
        self.U = (0.1 * rng.normal(size=(n_u, self.f))).astype(np.float32)
        self.V = (0.1 * rng.normal(size=(n_i, self.f))).astype(np.float32)
        self.b = np.zeros(n_i, dtype=np.float32)
        gU, gV, gb = (np.full_like(a, 1e-8) for a in (self.U, self.V, self.b))    # Adagrad accumulators
        n = len(pu)
        for _ in range(self.epochs):
            order = rng.permutation(n)
            for s in range(0, n, self.batch):
                sel = order[s:s + self.batch]
                u, i = pu[sel], pi[sel]
                j = rng.integers(0, n_i, size=len(sel))
                pos = np.searchsorted(keys, u * n_i + j)                          # reject sampled "negatives" that are positives
                clash = keys[np.minimum(pos, len(keys) - 1)] == u * n_i + j
                j[clash] = rng.integers(0, n_i, size=clash.sum())
                x = (self.U[u] * (self.V[i] - self.V[j])).sum(1) + self.b[i] - self.b[j]
                g = (1.0 / (1.0 + np.exp(np.clip(x, -30, 30))))[:, None].astype(np.float32)   # d loss / d x (negated)
                du = g * (self.V[i] - self.V[j]) - self.reg * self.U[u]
                di = g * self.U[u] - self.reg * self.V[i]
                dj = -g * self.U[u] - self.reg * self.V[j]
                db = g[:, 0]
                for arr, garr, idx, d in ((self.U, gU, u, du), (self.V, gV, i, di), (self.V, gV, j, dj)):
                    acc = np.zeros_like(arr); np.add.at(acc, idx, d)
                    gacc = np.zeros_like(arr); np.add.at(gacc, idx, d * d)
                    garr += gacc
                    arr += self.lr * acc / np.sqrt(garr)
                for idx, d in ((i, db), (j, -db)):
                    acc = np.zeros_like(self.b); np.add.at(acc, idx, d)
                    gb += acc * acc
                    self.b += self.lr * acc / np.sqrt(gb)
        return self

    def score(self, users):
        return self.U[users] @ self.V.T + self.b[None, :]


class ContentBased:
    """Cold-start model: a user is the mean of the (standardised) attributes of items they used;
    score = cosine(user profile, item attributes). Works for items with zero interactions."""
    name = "content"

    def fit(self, X: sp.csr_matrix, attrs: np.ndarray):
        a = (attrs - attrs.mean(0)) / (attrs.std(0) + 1e-9)
        self.a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)
        prof = X @ a
        self.prof = prof / (np.linalg.norm(prof, axis=1, keepdims=True) + 1e-9)
        return self

    def score(self, users):
        return np.asarray(self.prof[users] @ self.a.T, dtype=np.float32)
