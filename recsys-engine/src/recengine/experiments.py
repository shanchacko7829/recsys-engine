"""Full experiment: tune on validation, train the ranker on validation labels, evaluate once on test."""
from __future__ import annotations

import itertools
import time

import numpy as np

from .config import COLD_SHARE, K_EVAL, SEED, T_TRAIN, T_VAL
from .data import Events, concat, split_by_time
from .metrics import bootstrap_ci, insert_cold, list_metrics, paired_ci, per_user_metrics, topk
from .models import ALS, ItemKNN
from .pipeline import CONTENT_FEATURES, Params, Recommender

BASELINES = ("pop", "recent", "knn", "content", "als", "bpr")
LABELS = {"pop": "Popularity", "recent": "Recent popularity", "knn": "Item-KNN", "content": "Content-based",
          "als": "ALS", "bpr": "BPR", "ranker": "Two-stage ranker", "ranker_cold": "Ranker + cold-item slots", "ranker_no_content": "Ranker w/o content features",
          "oracle": "Oracle (true preferences)"}


def _ndcg_at(scores, rel, k=10):
    return per_user_metrics(topk(scores, k), rel, k)[f"ndcg@{k}"].mean()


def tune(tr: Events, va: Events, n_users, n_items, seed=SEED, log=print):
    """Small grid search on the validation window. ALS: factors x reg x alpha; KNN: neighbours."""
    X, Xv = tr.matrix(n_users, n_items), va.matrix(n_users, n_items)
    users = np.where(np.asarray(Xv.sum(1)).ravel() > 0)[0][:2000]
    rel = np.asarray(Xv[users].todense())
    seen = X[users].toarray() > 0
    best_als, best = None, -1.0
    for f, reg, alpha in itertools.product((32, 64), (0.05, 0.5), (10.0, 40.0)):
        m = ALS(f, reg, alpha, iters=10, seed=seed).fit(X)
        s = np.where(seen, -np.inf, m.score(users))
        v = _ndcg_at(s, rel)
        log(f"  tune ALS factors={f} reg={reg} alpha={alpha}: val NDCG@10={v:.4f}")
        if v > best:
            best, best_als = v, {"factors": f, "reg": reg, "alpha": alpha}
    best_knn, bk = None, -1.0
    for k in (20, 50, 100):
        m = ItemKNN(k).fit(X)
        v = _ndcg_at(np.where(seen, -np.inf, m.score(users)), rel)
        log(f"  tune KNN k={k}: val NDCG@10={v:.4f}")
        if v > bk:
            bk, best_knn = v, k
    return Params({**best_als, "iters": 12}, best_knn), {"als": best_als, "als_val_ndcg": round(best, 4), "knn_k": best_knn,
                                                          "knn_val_ndcg": round(bk, 4)}


def oracle_scores(world, users, seen_mask):
    """True probability of an interaction inside the test window - the best any model could do."""
    remaining = np.clip(1.0 - np.maximum(world.release, T_VAL), 0.0, None)[None, :]
    p = 1.0 - np.exp(-world.rate[users] * remaining)
    return np.where(seen_mask, -np.inf, p)


def run_experiment(world, n_boot=1000, seed=SEED, log=print):
    t0 = time.perf_counter()
    nu, ni = world.n_users, world.n_items
    ev = Events(world.users, world.items, world.times)
    tr, va, te = split_by_time(ev)
    log(f"events: train {len(tr):,} | val {len(va):,} | test {len(te):,}")

    params, tuning = tune(tr, va, nu, ni, seed, log)
    log(f"chosen: {tuning}")

    # --- stage 1: models fit on train, ranker learns from validation labels
    rec = Recommender(nu, ni, params, seed).fit_base(tr, world.attrs, T_TRAIN)
    Xv = va.matrix(nu, ni)
    val_users = np.where(np.asarray(Xv.sum(1)).ravel() > 0)[0]
    ranker_stats = rec.fit_ranker(val_users, Xv)
    ranker_nc = Recommender.__new__(Recommender)
    ranker_nc.__dict__.update(rec.__dict__)
    ranker_nc_stats = ranker_nc.fit_ranker(val_users, Xv, drop=CONTENT_FEATURES)
    log(f"ranker trained: {ranker_stats}")

    # --- stage 2: refit the generators on train+val, evaluate once on test
    trv = concat(tr, va)
    rec_t = Recommender(nu, ni, params, seed).fit_base(trv, world.attrs, T_VAL)
    rec_t.ranker, rec_t.feature_idx = rec.ranker, rec.feature_idx
    rec_nc = Recommender.__new__(Recommender)
    rec_nc.__dict__.update(rec_t.__dict__)
    rec_nc.ranker, rec_nc.feature_idx = ranker_nc.ranker, ranker_nc.feature_idx

    Xt = te.matrix(nu, ni)
    test_users = np.where(np.asarray(Xt.sum(1)).ravel() > 0)[0]
    rel = np.asarray(Xt[test_users].todense())
    seen = rec_t.X[test_users].toarray() > 0
    cold_item = np.asarray(rec_t.X.sum(0)).ravel() == 0
    rel_cold = rel * cold_item[None, :]
    cold_users = rel_cold.sum(1) > 0
    item_pop = np.asarray(rec_t.X.sum(0)).ravel()
    log(f"test users {len(test_users):,}; cold items {int(cold_item.sum())}; users with a cold target {int(cold_users.sum())}")

    scores, cand_recall = {}, None
    for m in BASELINES:
        scores[m] = rec_t.rank_users(test_users, m)
    scores["ranker"], rows, cols = rec_t.rank_users(test_users, "ranker", return_candidates=True)
    scores["ranker_no_content"] = rec_nc.rank_users(test_users, "ranker")
    scores["oracle"] = oracle_scores(world, test_users, seen)
    cand = np.zeros_like(rel, dtype=bool); cand[rows, cols] = True
    cand_recall = float(((cand * rel).sum(1) / rel.sum(1)).mean())

    out, per_user = {}, {}
    scores["ranker_cold"] = scores["ranker"]
    for name, S in scores.items():
        S = np.nan_to_num(S, nan=-np.inf)
        top = insert_cold(S, cold_item, max(K_EVAL), COLD_SHARE) if name == "ranker_cold" else topk(S, max(K_EVAL))
        row = {}
        for k in K_EVAL:
            for mname, vals in per_user_metrics(top, rel, k).items():
                v, (lo, hi) = bootstrap_ci(vals, n_boot)
                row[mname] = {"value": round(v, 4), "ci95": [round(lo, 4), round(hi, 4)]}
                per_user[(name, mname)] = vals
        if cold_users.any():
            cm = per_user_metrics(top[cold_users], rel_cold[cold_users], 20)["recall@20"]
            v, (lo, hi) = bootstrap_ci(cm, n_boot)
            row["cold_recall@20"] = {"value": round(v, 4), "ci95": [round(lo, 4), round(hi, 4)]}
        row["diversity@10"] = list_metrics(top, item_pop, 10)
        out[name] = row
        log(f"  {LABELS[name]:28} NDCG@10={row['ndcg@10']['value']:.4f}  Recall@20={row['recall@20']['value']:.4f}  "
            f"cold Recall@20={row.get('cold_recall@20', {}).get('value')}")

    def diff(a, b, metric="ndcg@10"):
        d, (lo, hi) = paired_ci(per_user[(a, metric)], per_user[(b, metric)], n_boot)
        return {"diff": round(d, 4), "ci95": [round(lo, 4), round(hi, 4)], "significant": bool(lo > 0 or hi < 0)}

    paired = {"ranker_cold_minus_ranker": diff("ranker_cold", "ranker"), "ranker_minus_als": diff("ranker", "als"), "ranker_minus_knn": diff("ranker", "knn"),
              "als_minus_knn": diff("als", "knn"), "als_minus_bpr": diff("als", "bpr"),
              "ranker_minus_ranker_no_content": diff("ranker", "ranker_no_content")}

    # --- serving latency of the full two-stage path, one user per request
    lat = []
    ids = test_users[:300]
    rec_t.recommend(rec_t.X[ids[0]].indices, 10, user_idx=int(ids[0]))
    for u in ids:
        t = time.perf_counter()
        rec_t.recommend(rec_t.X[u].indices, 10, user_idx=int(u))
        lat.append((time.perf_counter() - t) * 1000)
    lat = np.array(lat)

    return {"seed": seed, "data": {"users": nu, "items": ni, "events": len(ev), "train": len(tr), "val": len(va),
                                   "test": len(te), "test_users": int(len(test_users)), "cold_items": int(cold_item.sum()),
                                   "share_test_events_on_cold_items": round(float(rel_cold.sum() / rel.sum()), 3),
                                   "users_with_cold_target": int(cold_users.sum()),
                                   "simulated": True},
            "tuning": tuning, "ranker": {**ranker_stats, "candidate_recall": round(cand_recall, 4),
                                         "mean_candidates_per_user": round(float(cand.sum(1).mean()), 1)},
            "metrics": out, "paired_ndcg@10": paired,
            "latency_ms": {"p50": round(float(np.percentile(lat, 50)), 2), "p95": round(float(np.percentile(lat, 95)), 2)},
            "seconds": round(time.perf_counter() - t0, 1)}, rec_t
