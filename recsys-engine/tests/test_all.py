import tempfile
import unittest

import numpy as np
import scipy.sparse as sp
from starlette.testclient import TestClient

from recengine.data import Events, concat, split_by_time
from recengine.metrics import bootstrap_ci, insert_cold, list_metrics, per_user_metrics, topk
from recengine.models import ALS, BPR, ContentBased, ItemKNN, Popularity
from recengine.pipeline import Params, Recommender
from recengine.service import create_app
from recengine.simulate import simulate

_WORLD = None


def world():
    global _WORLD
    if _WORLD is None:
        _WORLD = simulate(n_users=400, n_items=200, mean_events=30, seed=1)
    return _WORLD


class SimulatorTests(unittest.TestCase):
    def test_deterministic(self):
        a, b = simulate(100, 60, seed=3), simulate(100, 60, seed=3)
        np.testing.assert_array_equal(a.items, b.items)

    def test_events_sorted_and_after_release(self):
        w = world()
        self.assertTrue(np.all(np.diff(w.times) >= 0))
        self.assertTrue(np.all(w.times >= w.release[w.items]))
        self.assertTrue(np.all(w.times < 1.0))

    def test_no_user_item_pair_twice(self):
        w = world()
        pairs = w.users.astype(np.int64) * w.n_items + w.items
        self.assertEqual(len(np.unique(pairs)), len(pairs))

    def test_mean_events_near_target(self):
        w = world()
        self.assertAlmostEqual(len(w.users) / w.n_users, 30, delta=4)


class SplitTests(unittest.TestCase):
    def test_split_is_temporal_and_complete(self):
        w = world()
        ev = Events(w.users, w.items, w.times)
        tr, va, te = split_by_time(ev)
        self.assertEqual(len(tr) + len(va) + len(te), len(ev))
        self.assertLess(tr.times.max(), va.times.min())
        self.assertLess(va.times.max(), te.times.min())
        self.assertEqual(len(concat(tr, va)), len(tr) + len(va))

    def test_matrix_is_binary(self):
        ev = Events(np.array([0, 0, 1]), np.array([2, 2, 1]), np.array([.1, .2, .3]))
        m = ev.matrix(2, 3)
        self.assertEqual(m.nnz, 2)
        self.assertTrue(np.all(m.data == 1.0))


def _toy():
    # two taste groups: users 0-9 like items 0-4, users 10-19 like items 5-9
    rows, cols = [], []
    rng = np.random.default_rng(0)
    for u in range(20):
        pool = range(0, 5) if u < 10 else range(5, 10)
        for i in rng.choice(list(pool), 4, replace=False):
            rows.append(u); cols.append(i)
    return sp.csr_matrix((np.ones(len(rows), np.float32), (rows, cols)), shape=(20, 10))


class ModelTests(unittest.TestCase):
    def test_als_and_bpr_recover_taste_groups(self):
        X = _toy()
        for m in (ALS(factors=4, reg=0.1, alpha=10, iters=10), BPR(factors=4, epochs=60, batch=16, lr=0.1)):
            m.fit(X)
            s = m.score(np.arange(20))
            s = np.where(X.toarray() > 0, -np.inf, s)
            best = s.argmax(1)
            ok = np.mean([(b < 5) == (u < 10) for u, b in enumerate(best)])
            self.assertGreater(ok, 0.8, m.name)

    def test_als_fold_in_matches_trained_vector(self):
        X = _toy()
        m = ALS(factors=4, reg=0.1, alpha=10, iters=10).fit(X)
        folded = m.fold_in(X[3].indices)
        cos = folded @ m.U[3] / (np.linalg.norm(folded) * np.linalg.norm(m.U[3]))
        self.assertGreater(cos, 0.95)

    def test_knn_has_zero_diagonal_and_limited_neighbours(self):
        m = ItemKNN(k=3).fit(_toy())
        self.assertTrue(np.all(np.diag(m.sim) == 0))
        self.assertLessEqual(int((m.sim > 0).sum(1).max()), 3)

    def test_content_model_scores_unseen_items(self):
        X = _toy()
        attrs = np.zeros((10, 2)); attrs[:5, 0] = 1; attrs[5:, 1] = 1
        attrs += np.random.default_rng(0).normal(0, .01, attrs.shape)
        s = ContentBased().fit(X, attrs).score(np.array([0, 15]))
        self.assertGreater(s[0, :5].mean(), s[0, 5:].mean())
        self.assertGreater(s[1, 5:].mean(), s[1, :5].mean())

    def test_popularity_time_decay_prefers_recent(self):
        ev = Events(np.array([0, 1, 2, 3]), np.array([0, 0, 1, 1]), np.array([0.0, 0.0, 0.99, 0.99]))
        p = Popularity(half_life=0.05).fit(ev, 4, 2, 1.0)
        self.assertGreater(p.pop[1], p.pop[0])


class MetricTests(unittest.TestCase):
    def test_known_values(self):
        top = np.array([[0, 1, 2]])
        rel = np.zeros((1, 5)); rel[0, [1, 4]] = 1
        m = per_user_metrics(top, rel, 3)
        self.assertAlmostEqual(m["recall@3"][0], 0.5)
        self.assertAlmostEqual(m["ndcg@3"][0], (1 / np.log2(3)) / (1 + 1 / np.log2(3)))
        self.assertEqual(m["hit@3"][0], 1.0)

    def test_perfect_ranking_has_ndcg_one(self):
        rel = np.zeros((1, 6)); rel[0, [2, 3]] = 1
        self.assertAlmostEqual(per_user_metrics(np.array([[2, 3, 0]]), rel, 3)["ndcg@3"][0], 1.0)

    def test_topk_orders_best_first(self):
        np.testing.assert_array_equal(topk(np.array([[1., 5., 3., 4.]]), 3)[0], [1, 3, 2])

    def test_bootstrap_ci_brackets_mean(self):
        v = np.random.default_rng(0).random(300)
        mean, (lo, hi) = bootstrap_ci(v, 400)
        self.assertLess(lo, mean); self.assertGreater(hi, mean)

    def test_gini_zero_for_uniform_exposure(self):
        top = np.arange(10).reshape(1, 10)
        self.assertAlmostEqual(list_metrics(top, np.ones(10), 10)["gini"], 0.0, places=3)
        self.assertEqual(list_metrics(top, np.ones(10), 10)["coverage"], 1.0)

    def test_insert_cold_reserves_slots(self):
        scores = np.arange(10, dtype=float)[None, ::-1].copy()      # item 0 best ... item 9 worst
        cold = np.zeros(10, bool); cold[[8, 9]] = True
        top = insert_cold(scores, cold, 5, share=0.4)[0]
        self.assertEqual(sorted(top[np.isin(top, [8, 9])]), [8, 9])
        self.assertEqual(len(set(top)), 5)
        np.testing.assert_array_equal(insert_cold(scores, cold, 5, 0.0), topk(scores, 5))


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        w = world()
        ev = Events(w.users, w.items, w.times)
        tr, va, _ = split_by_time(ev)
        cls.w = w
        cls.rec = Recommender(w.n_users, w.n_items, Params({"factors": 8, "reg": 0.1, "alpha": 10.0, "iters": 4}, 20))
        cls.rec.fit_base(tr, w.attrs, 0.8)
        Xv = va.matrix(w.n_users, w.n_items)
        users = np.where(np.asarray(Xv.sum(1)).ravel() > 0)[0]
        cls.rec.fit_ranker(users, Xv, max_iter=30)

    def test_seen_items_never_recommended(self):
        u = 5
        seen = set(self.rec.X[u].indices)
        items, _ = self.rec.recommend(self.rec.X[u].indices, 20, user_idx=u)
        self.assertFalse(seen & set(items.tolist()))
        self.assertEqual(len(set(items.tolist())), len(items))

    def test_new_user_via_fold_in(self):
        items, _ = self.rec.recommend([1, 2, 3], 5)
        self.assertEqual(len(items), 5)
        self.assertFalse({1, 2, 3} & set(items.tolist()))

    def test_empty_history_still_returns_items(self):
        items, _ = self.rec.recommend([], 5)
        self.assertGreater(len(items), 0)

    def test_cold_slots_present(self):
        cold = set(np.nonzero(self.rec.cold_items)[0].tolist())
        items, _ = self.rec.recommend(self.rec.X[3].indices, 10, user_idx=3, cold_share=0.2)
        self.assertGreaterEqual(len(cold & set(items.tolist())), 1)

    def test_save_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            self.rec.save(d)
            r2 = Recommender.load(d)
        a = self.rec.recommend(self.rec.X[7].indices, 8, user_idx=7)[0]
        b = r2.recommend(r2.X[7].indices, 8, user_idx=7)[0]
        np.testing.assert_array_equal(a, b)

    def test_similar_items_exclude_self(self):
        items, _ = self.rec.similar_items(4, 5)
        self.assertNotIn(4, items.tolist())


class ServiceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(create_app(PipelineTests.rec))

    def test_known_user(self):
        r = self.client.post("/v1/recommend", json={"user_id": 2, "k": 5})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["source"], "known_user")
        self.assertEqual(len(r.json()["items"]), 5)

    def test_new_user(self):
        r = self.client.post("/v1/recommend", json={"history": [1, 2, 3], "k": 4})
        self.assertEqual(r.json()["source"], "new_user_fold_in")

    def test_errors(self):
        self.assertEqual(self.client.post("/v1/recommend", json={"user_id": 10 ** 6}).status_code, 404)
        self.assertEqual(self.client.post("/v1/recommend", json={}).status_code, 422)
        self.assertEqual(self.client.post("/v1/recommend", json={"history": [10 ** 6]}).status_code, 422)
        self.assertEqual(self.client.post("/v1/recommend", json={"user_id": 1, "k": 0}).status_code, 422)
        self.assertEqual(self.client.post("/v1/recommend", content=b"xx").status_code, 400)
        self.assertEqual(self.client.get("/v1/similar/99999").status_code, 404)

    def test_similar_and_health(self):
        self.assertEqual(len(self.client.get("/v1/similar/3").json()["similar"]), 10)
        self.assertEqual(self.client.get("/health").json()["status"], "ok")


if __name__ == "__main__":
    unittest.main()
