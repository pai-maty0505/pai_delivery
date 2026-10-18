#!/usr/bin/env python3
"""
B の環境（本物の箱）で、腕の関節トルクから手の上の荷物の重さを推定する式（m = w・τ + b）を決め直す。
持ち方を変えると腕の姿勢が変わり、腕そのものの重さによるトルクも変わるので、持ち方ごとに決める。

歩く方策（--policy）で、練習の種類（最初から持つ・受け取る・降ろす・手ぶら）と指令をばらばらにして 20 秒歩かせ、
腕の 14 関節のトルクと、本当の重さ（手に載っている間は箱の質量、それ以外は 0）を集める。倒れた・落とした体は除く。
半分の体で式を決め、残りの半分で誤差を測る（そのままの値と、時定数 0.1 秒でならした値）。

    .venv_mjx/bin/python scripts/calib_carry_estimator.py --grip tray --policy carry6 --out models/estimator_tray.json
"""
import argparse
import functools
import json
import sys
from pathlib import Path

import jax
import jax.numpy as jp
import numpy as np
from mujoco_playground._src.wrapper import BraxDomainRandomizationVmapWrapper

sys.path.insert(0, str(Path(__file__).resolve().parent))
import carry_env as ce  # noqa: E402
import policy_io  # noqa: E402

REPO = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grip", default="tray")
    ap.add_argument("--policy", default="carry6")
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    conf = json.loads((REPO / "models" / args.policy / "config.json").read_text())
    cfg = ce.default_config()
    cfg.grip = args.grip
    cfg.estimator.w, cfg.estimator.b = conf["env"]["estimator"]["w"], conf["env"]["estimator"]["b"]
    cfg.episode_length = int(args.seconds / cfg.ctrl_dt) + 10
    env = ce.CarryJoystick(cfg)
    policy = policy_io.load_policy(args.policy, env, conf)
    keys = jax.random.split(jax.random.PRNGKey(5), args.n)
    venv = BraxDomainRandomizationVmapWrapper(env, functools.partial(ce.domain_randomize, env=env, rng=keys))
    reset, step_fn = jax.jit(venv.reset), jax.jit(venv.step)
    act = jax.jit(jax.vmap(lambda o, k: policy(o, k)[0]))

    state = reset(jax.random.split(jax.random.PRNGKey(6), args.n))
    T = int(args.seconds / env.dt)
    tau = np.zeros((T, args.n, 14), np.float32)
    m = np.zeros((T, args.n), np.float32)
    alive = np.ones(args.n, bool)
    for k in range(T):
        state = step_fn(state, act(state.obs, jax.random.split(jax.random.PRNGKey(k), args.n)))
        alive &= np.array(state.done) == 0
        tau[k] = np.array(state.data.actuator_force[:, ce.N_ACT:])
        m[k] = np.array(state.info["m_eff"])
    print(f"倒れず落とさなかった体 {alive.sum()} / {args.n}", flush=True)
    idx = np.where(alive)[0]
    fit, test = idx[: len(idx) // 2], idx[len(idx) // 2:]
    skip = int(1.0 / env.dt)                                   # 最初の 1 秒は除く
    X = tau[skip:, fit].reshape(-1, 14); y = m[skip:, fit].ravel()
    A = np.hstack([X, np.ones((len(X), 1))])
    sol, *_ = np.linalg.lstsq(A, y, rcond=None)
    w, b = sol[:14], float(sol[14])
    est = tau[:, test] @ w + b
    alpha = env.dt / (cfg.payload.est_tau + env.dt)
    ema = np.zeros_like(est)
    for k in range(T):
        ema[k] = (ema[k - 1] if k else 0) + alpha * (est[k] - (ema[k - 1] if k else 0))
    err_raw = float(np.mean(np.abs(est[skip:] - m[skip:, test])))
    err_ema = float(np.mean(np.abs(ema[skip:] - m[skip:, test])))
    held = m[skip:, test] > 0
    print(f"誤差（平均の絶対値）：そのまま {err_raw:.3f} kg、ならした値 {err_ema:.3f} kg"
          f"（持っている間 {np.mean(np.abs(ema[skip:] - m[skip:, test])[held]):.3f}、手ぶら {np.mean(np.abs(ema[skip:] - m[skip:, test])[~held]):.3f}）")
    Path(args.out).write_text(json.dumps({"w": w.tolist(), "b": b, "grip": args.grip, "policy": args.policy,
                                          "err_raw_kg": err_raw, "err_ema_kg": err_ema}, indent=1))
    print(args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
