#!/usr/bin/env python3
"""
腕の関節トルクから、手に載っている荷物の重さを推定する式（線形：m = w・τ + b）を決める。

本当の重さを知る方策（oracle）で、重さ・受け取る時刻・降ろす時刻・指令をばらばらにして歩かせ、
腕の 14 関節のトルク（actuator_force）と、本当の重さを集める。倒れた体は除く。
半分の体で式を決め、残りの半分で誤差を測る（そのままの値と、時定数 0.1 秒でならした値）。

    .venv_mjx/bin/python scripts/calib_estimator.py --name oracle

結果：models/estimator.json（w・b と誤差）
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import jax

sys.path.insert(0, str(Path(__file__).resolve().parent))
import payload_env as pe  # noqa: E402
import policy_io  # noqa: E402

REPO = Path(__file__).resolve().parent.parent


def ema(x, alpha):
    y = np.zeros_like(x)
    for i in range(len(x)):
        y[i] = (y[i - 1] if i else 0.0) + alpha * (x[i] - (y[i - 1] if i else 0.0))
    return y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="oracle")
    ap.add_argument("--n", type=int, default=1024)
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--out", default=str(REPO / "models" / "estimator.json"))
    args = ap.parse_args()
    env, conf = policy_io.load_env(args.name, push=True)
    policy = policy_io.load_policy(args.name, env, conf)
    rng = np.random.default_rng(0)
    n = args.n
    masses = rng.uniform(0.0, 3.5, n)
    onset = rng.uniform(0.0, 12.0, n)
    release = np.where(rng.random(n) < 0.4, onset + rng.uniform(2.0, 8.0, n), 1e4)
    command = np.c_[rng.uniform(-0.5, 0.8, n), rng.uniform(-0.3, 0.3, n), rng.uniform(-0.6, 0.6, n)]
    command[rng.random(n) < 0.2] = 0.0
    r = policy_io.rollout(env, policy, masses.tolist(), onset, release, command, seconds=args.seconds,
                          record=lambda s: {"tau": s.data.actuator_force[:, pe.N_ACT:]})
    tau, y, alive = r["tau"], r["m_eff"], ~r["done"]          # (T, n, 14)、(T, n)
    print(f"倒れた体：{r['done'][-1].mean() * 100:.1f}%")
    train = np.arange(n) < n // 2
    X = tau[:, train][alive[:, train]]
    Y = y[:, train][alive[:, train]]
    A = np.c_[X, np.ones(len(X))]
    lam = 1e-3 * len(X)
    sol = np.linalg.solve(A.T @ A + lam * np.diag([1.0] * 14 + [0.0]), A.T @ Y)
    w, b = sol[:14], float(sol[14])
    test = ~train
    raw = tau[:, test] @ w + b                                 # (T, n_test)
    alpha = env.dt / (env._config.payload.est_tau + env.dt)
    filt = np.stack([ema(raw[:, j], alpha) for j in range(raw.shape[1])], 1)
    a = alive[:, test]
    err_raw = np.abs(raw - y[:, test])[a].mean()
    err_filt = np.abs(filt - y[:, test])[a].mean()
    # 受け取った後 0.5 秒以上たって持ち続けている間だけの誤差（落ち着いたときの精度）
    t = r["t"][:, None]
    steady = a & (y[:, test] > 0) & (t > onset[test] + 0.5)
    err_steady = np.abs(filt - y[:, test])[steady].mean()
    print(f"推定の誤差（確かめ用の半分）：そのまま {err_raw:.3f} kg、ならした値 {err_filt:.3f} kg、持ち続けている間 {err_steady:.3f} kg")
    out = {"w": w.tolist(), "b": b, "err_raw": float(err_raw), "err_filt": float(err_filt), "err_steady": float(err_steady),
           "from": args.name, "n": n, "seconds": args.seconds}
    path = Path(args.out)
    path.write_text(json.dumps(out, indent=1))
    print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
