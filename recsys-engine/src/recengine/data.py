"""Event container, sparse matrices and the temporal split."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from .config import T_TRAIN, T_VAL


@dataclass
class Events:
    users: np.ndarray
    items: np.ndarray
    times: np.ndarray

    def __len__(self):
        return len(self.users)

    def subset(self, mask) -> "Events":
        return Events(self.users[mask], self.items[mask], self.times[mask])

    def matrix(self, n_users: int, n_items: int) -> sp.csr_matrix:
        m = sp.csr_matrix((np.ones(len(self), dtype=np.float32), (self.users, self.items)), shape=(n_users, n_items))
        m.data[:] = 1.0                      # duplicates collapse to a single implicit positive
        m.sum_duplicates()
        m.data[:] = 1.0
        return m


def split_by_time(ev: Events, t_train=T_TRAIN, t_val=T_VAL):
    """train: t < t_train, val: [t_train, t_val), test: t >= t_val."""
    return (ev.subset(ev.times < t_train), ev.subset((ev.times >= t_train) & (ev.times < t_val)),
            ev.subset(ev.times >= t_val))


def concat(a: Events, b: Events) -> Events:
    return Events(np.concatenate([a.users, b.users]), np.concatenate([a.items, b.items]),
                  np.concatenate([a.times, b.times]))
