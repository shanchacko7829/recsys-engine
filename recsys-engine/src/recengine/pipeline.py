"""Two-stage recommender: several candidate generators -> gradient-boosted ranker.

Stage 1 (recall): union of the top-N items from ALS, BPR, item-KNN, content-based and recent popularity.
Stage 2 (precision): a HistGradientBoosting classifier scores each (user, candidate) pair using every
generator's score and rank plus item/user statistics, and the final list is sorted by that score.
"""
from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from sklearn.ensemble import HistGradientBoostingClassifier

from .config import CAND_ALS, CAND_COLD, CAND_CONTENT, CAND_KNN, CAND_POP
from .metrics import insert_cold, topk
from .models import ALS, BPR, ContentBased, ItemKNN, Popularity

FEATURES = ["als", "als_rank", "bpr", "bpr_rank", "knn", "knn_rank", "content", "content_rank", "pop_rank",
            "recent_rank", "log_item_cnt", "log_recent_cnt", "item_age", "log_user_cnt"]
CONTENT_FEATURES = ["content", "content_rank"]
BPR_N = 50


def _ranks(scores: np.ndarray) -> np.ndarray:
    order = np.argsort(-scores, axis=1, kind="stable")
    r = np.empty_like(order)
    np.put_along_axis(r, order, np.arange(scores.shape[1])[None, :].repeat(scores.shape[0], 0), 1)
    return r.astype(np.float32)


@dataclass
class Params:
    als: dict
    knn_k: int = 50


class Recommender:
    def __init__(self, n_users, n_items, params: Params, seed=42):
        self.n_users, self.n_items, self.params, self.seed = n_users, n_items, params, seed
        self.ranker = None
        self.feature_idx = list(range(len(FEATURES)))

    # ------------------------------------------------------------------ fitting
    def fit_base(self, ev, attrs, t_now):
        """Fit every candidate generator on events `ev` (all earlier than t_now)."""
        X = ev.matrix(self.n_users, self.n_items)
        self.X, self.t_now, self.attrs = X, t_now, attrs
        self.als = ALS(seed=self.seed, **self.params.als).fit(X)
        self.bpr = BPR(seed=self.seed).fit(X)
        self.knn = ItemKNN(k=self.params.knn_k).fit(X)
        self.content = ContentBased().fit(X, attrs)
        self.pop = Popularity().fit(ev, self.n_users, self.n_items, t_now)
        self.recent = Popularity(half_life=0.05).fit(ev, self.n_users, self.n_items, t_now)
        cnt = np.asarray(X.sum(0)).ravel()
        first = np.full(self.n_items, np.inf, dtype=np.float32)
        np.minimum.at(first, ev.items, ev.times.astype(np.float32))
        first[~np.isfinite(first)] = np.nan
        self.item_stats = np.stack([np.log1p(cnt), np.log1p(self.recent.pop), t_now - first], 1).astype(np.float32)
        return self

    # ------------------------------------------------------------------ scoring
    def _scores(self, H: sp.csr_matrix, U_als, U_bpr):
        """Raw per-generator score matrices for users described by history H (n x items)."""
        s = {"als": U_als @ self.als.V.T,
             "bpr": (U_bpr @ self.bpr.V.T + self.bpr.b[None, :]) if U_bpr is not None
             else np.full((H.shape[0], self.n_items), np.nan, dtype=np.float32),
             "knn": np.asarray(H @ self.knn.sim, dtype=np.float32)}
        prof = H @ self.content.a
        prof = prof / (np.linalg.norm(prof, axis=1, keepdims=True) + 1e-9)
        s["content"] = np.asarray(prof @ self.content.a.T, dtype=np.float32)
        s["pop"] = np.tile(self.pop.pop, (H.shape[0], 1))
        s["recent"] = np.tile(self.recent.pop, (H.shape[0], 1))
        seen = H.toarray() > 0
        for k in s:
            s[k] = np.where(seen, -np.inf, s[k]).astype(np.float32)
        return s, seen

    def _candidates(self, s):
        mask = np.zeros(s["als"].shape, dtype=bool)
        for key, n in (("als", CAND_ALS), ("knn", CAND_KNN), ("content", CAND_CONTENT), ("recent", CAND_POP),
                       ("bpr", BPR_N)):
            if np.isnan(s[key]).all():
                continue
            top = topk(np.nan_to_num(s[key], nan=-np.inf), n)
            np.put_along_axis(mask, top, True, 1)
        cold = self.item_stats[:, 0] == 0                      # items nobody has interacted with yet
        if cold.any():
            seen_inf = np.isneginf(s["content"])
            c_scores = np.where(cold[None, :] & ~seen_inf, s["content"], -np.inf)
            top = topk(c_scores, min(CAND_COLD, int(cold.sum())))
            ok = np.isfinite(np.take_along_axis(c_scores, top, 1))
            r = np.nonzero(ok)[0]
            mask[r, top[ok]] = True
        return mask

    def _features(self, s, H, mask):
        rows, cols = np.nonzero(mask)
        rk = {k: _ranks(np.nan_to_num(s[k], nan=-np.inf)) for k in ("als", "bpr", "knn", "content", "pop", "recent")}
        if np.isnan(s["bpr"]).all():
            rk["bpr"][:] = np.nan
        ucnt = np.log1p(np.asarray(H.sum(1)).ravel())
        cols_list = []
        for name in FEATURES:
            if name in ("als", "bpr", "knn", "content"):
                v = s[name][rows, cols]
            elif name.endswith("_rank"):
                v = rk[name[:-5]][rows, cols]
            elif name == "log_item_cnt":
                v = self.item_stats[cols, 0]
            elif name == "log_recent_cnt":
                v = self.item_stats[cols, 1]
            elif name == "item_age":
                v = self.item_stats[cols, 2]
            else:
                v = ucnt[rows]
            cols_list.append(v)
        F = np.stack(cols_list, 1).astype(np.float32)
        F[~np.isfinite(F)] = np.nan
        return rows, cols, F

    def candidate_table(self, users, H=None):
        users = np.asarray(users)
        H = self.X[users] if H is None else H
        s, _ = self._scores(H, self.als.U[users], self.bpr.U[users])
        mask = self._candidates(s)
        rows, cols, F = self._features(s, H, mask)
        return s, rows, cols, F

    def fit_ranker(self, users, rel_matrix: sp.csr_matrix, drop=(), max_iter=250):
        """Train on candidate lists of `users`; label = item is in the user's future events."""
        _, rows, cols, F = self.candidate_table(users)
        y = np.asarray(rel_matrix[np.asarray(users)][rows, cols]).ravel() > 0
        self.feature_idx = [i for i, n in enumerate(FEATURES) if n not in drop]
        self.ranker = HistGradientBoostingClassifier(max_iter=max_iter, learning_rate=0.06, max_leaf_nodes=24,
                                                     l2_regularization=1.0, early_stopping=True, random_state=self.seed)
        self.ranker.fit(F[:, self.feature_idx], y)
        return {"rows": int(len(y)), "positives": int(y.sum()), "iterations": int(self.ranker.n_iter_)}

    def rank_users(self, users, mode="ranker", return_candidates=False):
        """Dense (n_users, n_items) score matrix: seen items = -inf, non-candidates = -inf under 'ranker'."""
        s, rows, cols, F = self.candidate_table(users)
        if mode == "ranker":
            out = np.full(s["als"].shape, -np.inf, dtype=np.float32)
            out[rows, cols] = self.ranker.predict_proba(F[:, self.feature_idx])[:, 1]
            if return_candidates:
                return out, rows, cols
            return out
        return s[mode]

    # ------------------------------------------------------------------ serving
    @property
    def cold_items(self):
        return self.item_stats[:, 0] == 0

    def recommend(self, history, k=10, user_idx=None, cold_share=0.2):
        """Top-k item ids for one user. `history` = item ids they used. Unknown users are folded in."""
        history = np.unique(np.asarray(history, dtype=np.int64))
        H = sp.csr_matrix((np.ones(len(history), dtype=np.float32), (np.zeros(len(history), dtype=int), history)),
                          shape=(1, self.n_items))
        if user_idx is not None:
            U_als, U_bpr = self.als.U[[user_idx]], self.bpr.U[[user_idx]]
        else:
            U_als = self.als.fold_in(history)[None, :] if len(history) else np.zeros((1, self.als.f), np.float32)
            U_bpr = None
        s, _ = self._scores(H, U_als, U_bpr)
        mask = self._candidates(s)
        rows, cols, F = self._features(s, H, mask)
        sc = self.ranker.predict_proba(F[:, self.feature_idx])[:, 1]
        dense = np.full((1, self.n_items), -np.inf, dtype=np.float32)
        dense[0, cols] = sc
        top = insert_cold(dense, self.cold_items, k, cold_share)[0]
        top = top[np.isfinite(dense[0, top])]
        return top, dense[0, top]

    def similar_items(self, item, k=10):
        v = self.als.V
        vn = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-9)
        sim = vn @ vn[item]
        sim[item] = -np.inf
        top = np.argsort(-sim)[:k]
        return top, sim[top]

    # ------------------------------------------------------------------ persistence
    def save(self, directory):
        d = Path(directory); d.mkdir(parents=True, exist_ok=True)
        sp.save_npz(d / "X.npz", self.X)
        np.savez_compressed(d / "arrays.npz", als_U=self.als.U, als_V=self.als.V, bpr_U=self.bpr.U, bpr_V=self.bpr.V,
                            bpr_b=self.bpr.b, knn_sim=self.knn.sim, content_a=self.content.a, pop=self.pop.pop,
                            recent=self.recent.pop, item_stats=self.item_stats, attrs=self.attrs)
        meta = {"n_users": self.n_users, "n_items": self.n_items, "t_now": float(self.t_now), "seed": self.seed,
                "als": self.params.als, "knn_k": self.params.knn_k, "feature_idx": self.feature_idx}
        (d / "meta.json").write_text(json.dumps(meta))
        # the ranker is a scikit-learn model and is stored with pickle: only load files you created yourself
        with open(d / "ranker.pkl", "wb") as fh:
            pickle.dump(self.ranker, fh)

    @classmethod
    def load(cls, directory):
        d = Path(directory)
        meta = json.loads((d / "meta.json").read_text())
        z = np.load(d / "arrays.npz")
        r = cls(meta["n_users"], meta["n_items"], Params(meta["als"], meta["knn_k"]), meta["seed"])
        r.X, r.t_now, r.attrs, r.feature_idx = sp.load_npz(d / "X.npz").tocsr(), meta["t_now"], z["attrs"], meta["feature_idx"]
        r.als = ALS(**meta["als"]); r.als.U, r.als.V = z["als_U"], z["als_V"]; r.als.f = r.als.V.shape[1]
        r.bpr = BPR(); r.bpr.U, r.bpr.V, r.bpr.b = z["bpr_U"], z["bpr_V"], z["bpr_b"]
        r.knn = ItemKNN(); r.knn.sim = z["knn_sim"]
        r.content = ContentBased(); r.content.a = z["content_a"]
        r.pop = Popularity(); r.pop.pop = z["pop"]
        r.recent = Popularity(0.05); r.recent.pop = z["recent"]
        r.item_stats = z["item_stats"]
        with open(d / "ranker.pkl", "rb") as fh:
            r.ranker = pickle.load(fh)
        return r
