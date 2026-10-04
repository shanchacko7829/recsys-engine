"""Synthetic implicit-feedback generator with known ground truth.

Why synthetic: public datasets (MovieLens etc.) cannot be downloaded in the build
environment, and a simulator lets us measure things real logs hide: the best achievable
score (an "oracle" that knows true preferences) and behaviour on brand-new items.
A MovieLens-format loader is provided in data.py so the same pipeline runs on real data.

Process. Users and items have hidden taste vectors. Item i is released at time r_i in [0,1].
After release, user u interacts with it after an exponential delay whose rate is
    rate_ui = c_u * (exp(p_u . q_i / tau + b_i) + eps)
(c_u = user activity, b_i = item popularity bias, eps = taste-independent noise).
So events are ordered in time, popular/high-affinity items are consumed sooner, and items
released late have no history -> a real cold-start problem.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class World:
    n_users: int
    n_items: int
    user_f: np.ndarray        # hidden taste (n_users, d)
    item_f: np.ndarray        # hidden taste (n_items, d)
    item_bias: np.ndarray
    release: np.ndarray       # item release time
    rate: np.ndarray          # (n_users, n_items) true interaction rate, float32
    attrs: np.ndarray         # observable noisy item attributes (n_items, n_attr)
    users: np.ndarray         # event arrays, sorted by time
    items: np.ndarray
    times: np.ndarray


def simulate(n_users=6000, n_items=2500, d=16, n_attr=10, mean_events=45.0, tau=3.0, eps=0.02,
             cold_frac=0.12, attr_noise=0.8, seed=42) -> World:
    rng = np.random.default_rng(seed)
    pu = rng.normal(size=(n_users, d)).astype(np.float32)
    qi = rng.normal(size=(n_items, d)).astype(np.float32)
    bias = rng.normal(0.0, 1.0, size=n_items).astype(np.float32)           # heavy-ish popularity skew
    activity = np.exp(rng.normal(0.0, 0.6, size=n_users)).astype(np.float32)
    release = rng.uniform(0.0, 0.85, size=n_items)
    cold = rng.random(n_items) < cold_frac                                   # released just before the end
    release[cold] = rng.uniform(0.88, 0.99, size=cold.sum())
    W = rng.normal(size=(d, n_attr)).astype(np.float32) / np.sqrt(d)
    attrs = qi @ W + attr_noise * rng.normal(size=(n_items, n_attr)).astype(np.float32)

    affinity = pu @ qi.T / tau / np.sqrt(d) * 4.0 + bias[None, :]
    base = np.exp(affinity - affinity.max()).astype(np.float32)
    base = base + eps * base.mean()
    remaining = (1.0 - release).astype(np.float32)[None, :]

    def expected_events(scale):
        rate = base * activity[:, None] * scale
        return float((1.0 - np.exp(-rate * remaining)).sum(1).mean())

    lo, hi = 1e-6, 1e6                                                       # bisection on the global scale
    for _ in range(60):
        mid = np.sqrt(lo * hi)
        lo, hi = (mid, hi) if expected_events(mid) < mean_events else (lo, mid)
    scale = np.sqrt(lo * hi)
    rate = (base * activity[:, None] * scale).astype(np.float32)

    users, items, times = [], [], []
    for start in range(0, n_users, 1000):
        r = rate[start:start + 1000]
        delay = rng.exponential(size=r.shape).astype(np.float32) / r
        t = release[None, :] + delay
        uu, ii = np.nonzero(t < 1.0)
        users.append(uu + start)
        items.append(ii)
        times.append(t[uu, ii])
    users, items, times = (np.concatenate(x) for x in (users, items, times))
    order = np.argsort(times, kind="stable")
    return World(n_users, n_items, pu, qi, bias, release, rate, attrs, users[order].astype(np.int32),
                 items[order].astype(np.int32), times[order])
