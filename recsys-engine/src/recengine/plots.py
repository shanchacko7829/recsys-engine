from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "DejaVu Sans"
ORDER = ["pop", "recent", "content", "bpr", "knn", "als", "ranker", "ranker_cold"]
NAMES = {"pop": "Popularity", "recent": "Recent pop.", "knn": "Item-KNN", "content": "Content", "als": "ALS", "bpr": "BPR",
         "ranker": "Two-stage\nranker", "ranker_cold": "Ranker +\ncold slots"}


def _bars(res, metric, ylabel, title, out, ceiling=None):
    m = res["metrics"]
    vals = [m[k][metric]["value"] for k in ORDER]
    err = [[m[k][metric]["value"] - m[k][metric]["ci95"][0] for k in ORDER],
           [m[k][metric]["ci95"][1] - m[k][metric]["value"] for k in ORDER]]
    fig, ax = plt.subplots(figsize=(9, 4.2))
    colors = ["#b0b7c3"] * 6 + ["#1f4fbf", "#5b8def"]
    ax.bar(range(len(ORDER)), vals, yerr=err, capsize=3, color=colors)
    if ceiling is not None:
        ax.axhline(ceiling, ls="--", color="#c0392b", lw=1)
        ax.text(0.02, ceiling, " oracle (true preferences)", color="#c0392b", va="bottom", fontsize=8)
    ax.set_xticks(range(len(ORDER)))
    ax.set_xticklabels([NAMES[k] for k in ORDER], fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def make_plots(results="results/eval.json", out_dir="docs"):
    res = json.loads(Path(results).read_text())
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    _bars(res, "ndcg@10", "NDCG@10 (95% CI over users)", "Warm + cold items, temporal test split",
          Path(out_dir) / "ndcg.png", res["metrics"]["oracle"]["ndcg@10"]["value"])
    _bars(res, "cold_recall@20", "Recall@20 on brand-new items", "Items with no interactions at training time",
          Path(out_dir) / "cold_start.png")
    return ["ndcg.png", "cold_start.png"]
