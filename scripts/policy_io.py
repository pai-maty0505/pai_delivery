#!/usr/bin/env python3
"""
学習した方策（models/<name>/params.pkl・config.json）を読み込み、決めた条件（荷物の重さ・受け取る時刻・指令）で
多数の体を同時に歩かせる（GPU、MJX）。評価（eval_payload.py）と推定の式づくり（calib_estimator.py）で使う。
"""

import json
import sys
from pathlib import Path

import jax
import jax.numpy as jp
import numpy as np
from brax.io import model as brax_model
from brax.training.acme import running_statistics
from brax.training.agents.ppo import networks as ppo_networks
from mujoco_playground._src.wrapper import BraxDomainRandomizationVmapWrapper

sys.path.insert(0, str(Path(__file__).resolve().parent))
import payload_env as pe  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
NEVER = 10 ** 6


def load_env(name, push=False, estimator=None, obs_mode=None):
    """学習したときと同じ設定の環境。push：押す力を入れるか。estimator：推定の式を差し替えるとき（json のパス）。
    obs_mode：方策に渡す重さを差し替えるとき（例：本当の重さで学習した方策に、推定した重さを渡す）。"""
    conf = json.loads((REPO / "models" / name / "config.json").read_text())
    cfg = pe.default_config()
    cfg.obs_mode = obs_mode or conf["env"]["obs_mode"]
    cfg.estimator.w = conf["env"]["estimator"]["w"]
    cfg.estimator.b = conf["env"]["estimator"]["b"]
    if estimator:
        est = json.loads(Path(estimator).read_text())
        cfg.estimator.w, cfg.estimator.b = est["w"], est["b"]
    cfg.push_config.enable = push
    return pe.PayloadJoystick(cfg), conf


def load_policy(name, env, conf, untrained=False):
    """untrained：学習前の方策（方策の重みを初めの乱数の値に戻す。入力の正規化は学習後のものを使う）。"""
    net = conf["network"]
    nets = ppo_networks.make_ppo_networks(
        env.observation_size, env.action_size, preprocess_observations_fn=running_statistics.normalize,
        policy_hidden_layer_sizes=tuple(net["policy_hidden_layer_sizes"]),
        value_hidden_layer_sizes=tuple(net["value_hidden_layer_sizes"]),
        policy_obs_key=net["policy_obs_key"], value_obs_key=net["value_obs_key"])
    params = brax_model.load_params(str(REPO / "models" / name / "params.pkl"))
    if untrained:
        params = (params[0], nets.policy_network.init(jax.random.PRNGKey(0)), params[2])
    return ppo_networks.make_inference_fn(nets)(params, deterministic=True)


def fixed_masses(env, masses, split=0.5):
    """体ごとの荷物の重さ（両手の合計）を決めたモデル。そのほかの条件は学習の平均の値。"""
    masses = jp.asarray(masses, dtype=jp.float32)

    def fn(model):
        bm = jp.tile(model.body_mass, (len(masses), 1))
        bm = bm.at[:, env._payload_body[0]].set(masses * split).at[:, env._payload_body[1]].set(masses * (1 - split))
        in_axes = jax.tree_util.tree_map(lambda x: None, model).tree_replace({"body_mass": 0})
        return model.tree_replace({"body_mass": bm}), in_axes
    return BraxDomainRandomizationVmapWrapper(env, fn)


def rollout(env, policy, masses, onset, release, command, seconds=20.0, seed=0, record=None, command_fn=None):
    """決めた条件で歩かせる。onset・release は秒（体ごとの配列でもよい。None は持たない／降ろさない）。
    command は (前後, 左右, 回る)、または体ごとの (n, 3)。
    返り値：記録（時刻ごとの配列の辞書）。record に関数を渡すと、各ステップの (state) から追加の値を記録する。"""
    n = len(masses)
    venv = fixed_masses(env, masses)
    reset = jax.jit(venv.reset)
    cmd = jp.asarray(command, dtype=jp.float32)
    cmd = cmd if cmd.ndim == 2 else jp.tile(cmd, (n, 1))

    @jax.jit
    def step(state, t):
        c = cmd if command_fn is None else command_fn(t, cmd)
        info = dict(state.info, command=c, step=jp.zeros_like(state.info["step"]))
        state = state.replace(info=info)
        act = jax.vmap(lambda o, k: policy(o, k)[0])(state.obs, jax.random.split(jax.random.PRNGKey(0), n))
        return venv.step(state, act)

    state = reset(jax.random.split(jax.random.PRNGKey(seed), n))
    def steps(sec):
        if sec is None:
            return jp.full(n, NEVER, dtype=jp.int32)
        return jp.broadcast_to(jp.round(jp.asarray(sec) / env.dt).astype(jp.int32), (n,))
    on, off = steps(onset), steps(release)
    info = dict(state.info, onset=on, release=off, m_eff=jp.where(on <= 0, jp.asarray(masses, dtype=jp.float32), 0.0))
    state = state.replace(info=info)
    out = {k: [] for k in ("t", "x", "y", "z", "vx", "vy", "wz", "done", "m_eff", "m_est", "pitch")}
    fallen = np.zeros(n, dtype=bool)
    for i in range(int(round(seconds / env.dt))):
        state = step(state, i)
        # 倒れたら（done）、そのあとは記録上ずっと倒れたままとする（自動のリセットは無い）
        fallen |= np.array(state.done) > 0
        lv = np.array(jax.vmap(lambda d: env.get_local_linvel(d, "pelvis"))(state.data))
        gy = np.array(jax.vmap(lambda d: env.get_gyro(d, "pelvis"))(state.data))
        up = np.array(jax.vmap(lambda d: env.get_gravity(d, "torso"))(state.data))
        out["t"].append((i + 1) * env.dt)
        out["x"].append(np.array(state.data.qpos[:, 0]))
        out["y"].append(np.array(state.data.qpos[:, 1]))
        out["z"].append(np.array(state.data.qpos[:, 2]))
        out["vx"].append(lv[:, 0])
        out["vy"].append(lv[:, 1])
        out["wz"].append(gy[:, 2])
        out["done"].append(fallen.copy())
        out["m_eff"].append(np.array(state.info["m_eff"]))
        out["m_est"].append(np.array(state.info["m_est"]))
        out["pitch"].append(np.arctan2(up[:, 0], up[:, 2]))
        if record is not None:
            for k, v in record(state).items():
                out.setdefault(k, []).append(np.array(v))
    return {k: np.array(v) for k, v in out.items()}
