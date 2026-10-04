"""python -m recengine {eval,recommend,serve,plots}"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def cmd_eval(a):
    from .experiments import run_experiment
    from .simulate import simulate
    world = simulate(a.users, a.items, seed=a.seed)
    res, rec = run_experiment(world, a.boot, a.seed)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=2))
    rec.save(a.model_dir)
    print(f"wrote {a.out} and model bundle {a.model_dir}")


def cmd_recommend(a):
    from .pipeline import Recommender
    rec = Recommender.load(a.model_dir)
    if a.user is not None:
        items, scores = rec.recommend(rec.X[a.user].indices, a.k, a.user)
    else:
        items, scores = rec.recommend([int(x) for x in a.history.split(",")], a.k)
    for i, s in zip(items, scores):
        print(f"item {int(i):>5}   score {s:.4f}")


def cmd_serve(a):
    import uvicorn
    from .pipeline import Recommender
    from .service import create_app
    uvicorn.run(create_app(Recommender.load(a.model_dir)), host=a.host, port=a.port)


def cmd_plots(a):
    from .plots import make_plots
    print("wrote", make_plots(a.results, a.out_dir))


def main(argv=None):
    p = argparse.ArgumentParser(prog="recengine")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("eval", help="simulate data, tune, train, evaluate on the test window")
    s.add_argument("--users", type=int, default=6000); s.add_argument("--items", type=int, default=2500)
    s.add_argument("--seed", type=int, default=42); s.add_argument("--boot", type=int, default=1000)
    s.add_argument("--out", default="results/eval.json"); s.add_argument("--model-dir", default="models/default")
    s.set_defaults(fn=cmd_eval)
    s = sub.add_parser("recommend"); s.add_argument("--model-dir", default="models/default")
    g = s.add_mutually_exclusive_group(required=True)
    g.add_argument("--user", type=int); g.add_argument("--history", help="comma separated item ids of a new user")
    s.add_argument("-k", type=int, default=10); s.set_defaults(fn=cmd_recommend)
    s = sub.add_parser("serve"); s.add_argument("--model-dir", default="models/default")
    s.add_argument("--host", default="127.0.0.1"); s.add_argument("--port", type=int, default=8000); s.set_defaults(fn=cmd_serve)
    s = sub.add_parser("plots"); s.add_argument("--results", default="results/eval.json")
    s.add_argument("--out-dir", default="docs"); s.set_defaults(fn=cmd_plots)
    a = p.parse_args(argv)
    a.fn(a)
    return 0
