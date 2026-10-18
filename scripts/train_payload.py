#!/usr/bin/env python3
"""
荷物を持って歩く方策を PPO で学習する（scripts/payload_env.py の環境、Brax の PPO、GPU）。

    .venv_mjx/bin/python scripts/train_payload.py --obs blind  --name blind
    .venv_mjx/bin/python scripts/train_payload.py --obs oracle --name oracle
    .venv_mjx/bin/python scripts/train_payload.py --obs estimate --estimator models/estimator.json --name estimate
    .venv_mjx/bin/python scripts/train_payload.py --env standup --name standup --steps 100e6     # 椅子から立ち上がる（A）
    .venv_mjx/bin/python scripts/train_payload.py --env carry --name carry --estimator models/estimator.json \
        --init models/estimate/params.pkl                                                          # ケースを持って歩く（B）

学習の設定は Playground の G1JoystickFlatTerrain と同じ（num_timesteps だけ変えられる）。
結果：models/<name>/params.pkl（評価のたびに上書き）、models/<name>/config.json、logs/train_<name>.jsonl（評価の記録）
"""

import argparse
import functools
import json
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jp


def _device_put_replicated(x, devices):
    """JAX 0.11 で無くなった jax.device_put_replicated の代わり（Brax 0.14.2 が使う）。先頭に機器の数の軸を足して置く。"""
    return jax.tree.map(lambda a: jax.device_put(jp.stack([a] * len(devices))), x)


try:
    jax.device_put_replicated
except AttributeError:
    jax.device_put_replicated = _device_put_replicated

from brax.io import model as brax_model  # noqa: E402
from brax.training.agents.ppo import networks as ppo_networks  # noqa: E402
from brax.training.agents.ppo import train as ppo  # noqa: E402
from mujoco_playground import wrapper  # noqa: E402
from mujoco_playground.config import locomotion_params  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import payload_env as pe  # noqa: E402

REPO = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", choices=("payload", "standup", "carry"), default="payload",
                    help="payload：荷物（質量）を持って歩く、standup：椅子から立ち上がる、carry：本物のケースを持って歩く（B）")
    ap.add_argument("--init", help="学習を始める重み（models/<name>/params.pkl）。入力の形が同じ方策から続けて学習する")
    ap.add_argument("--grip", choices=("hands", "arms", "tray"), default="hands", help="B のケースの持ち方（carry のとき）")
    ap.add_argument("--crouch", action="store_true", help="carry：方策の入力に「腰を下げる量」を入れ、体を沈めて置く練習をする")
    ap.add_argument("--crouch_file", help="carry：体を沈めて置く腕の動きと沈める量のファイル（models/ の中）")
    ap.add_argument("--crouch_weight", type=float, help="carry：腰の高さのずれの減点の重み")
    ap.add_argument("--place_yaw_weight", type=float, help="carry：置くあいだに体の向きが回った量の減点の重み")
    ap.add_argument("--mode_p", help="carry：練習の場面の割合（運ぶ・受け取る・放す・止まって放す・置く、カンマ区切り）")
    ap.add_argument("--feet_vel_weight", type=float, help="carry：止まっている間の足の速さの減点の重み")
    ap.add_argument("--crouch_lean", type=float, default=0.0, help="carry：沈める量に合わせて股関節から上体を前へ倒す角度（度）")
    ap.add_argument("--crouch_waist", action="store_true", help="carry：しゃがむ姿勢に腰の pitch を前へ足す（胴をまっすぐに）")
    ap.add_argument("--crouch_ff", action="store_true", help="carry：沈める量に合わせたしゃがむ姿勢を脚の目標に先に足す")
    ap.add_argument("--place_touch_file", help="carry：机に触れた時刻ごとの置く腕の動きの表（触れたら切り替えて開く。models/ の中）")
    ap.add_argument("--office", action="store_true", help="carry：事務所の場面で学習する（置く練習は C の道順の途中から、事務所と同じ指令で歩いて置く）")
    ap.add_argument("--office_flow", action="store_true", help="carry：事務所の場面で A の続きから B〜D を練習する（事務所で通した記録の状態から始める）")
    ap.add_argument("--stop_err_weight", type=float, help="carry：止まる位置の目標とのずれの減点の重み（事務所）")
    ap.add_argument("--stop_err_once", action="store_true", help="carry：止まる位置のずれを、置き始める時の一度だけ減点する")
    ap.add_argument("--adjust", type=int, default=0, help="carry：机の前で止まった後の、小さな踏み直しの最大の回数（事務所）")
    ap.add_argument("--yaw_shift", type=int, default=0, help="carry：上体を寄せるときに、股の向きで骨盤の向きも直す（1）")
    ap.add_argument("--drift_all", action="store_true", help="carry：止まってからの流れの減点を置き終わるまで数える")
    ap.add_argument("--desk_body", action="store_true", help="carry：置く練習の机に手・手首・胴も当てる")
    ap.add_argument("--approach", action="store_true", help="carry：置く練習で、事務所と同じ近づき方（向きを変えてゆっくり前へ）をしてから止まる")
    ap.add_argument("--stop_drift_weight", type=float, help="carry：止まってから骨盤が流れた距離の減点の重み")
    ap.add_argument("--desk_offset", type=float, help="carry：置く練習の机の位置のずれ（前後・左右、m）")
    ap.add_argument("--desk_dz", type=float, help="carry：置く練習の机の高さのずれ（m）")
    ap.add_argument("--place_edge", type=float, help="carry：止まった骨盤から机の手前の縁まで（m）")
    ap.add_argument("--stop_range", help="carry：置く練習で歩いてから止まる時刻の範囲（秒、カンマ区切り）")
    ap.add_argument("--obs", choices=("blind", "oracle", "estimate"), default="blind")
    ap.add_argument("--name", required=True)
    ap.add_argument("--steps", type=float, default=150e6)
    ap.add_argument("--envs", type=int, default=8192)
    ap.add_argument("--evals", type=int, default=15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--estimator", help="推定の式（w・b）の json。--obs estimate のとき使う")
    args = ap.parse_args()

    if args.env == "carry":
        import carry_env as ce
        cfg = ce.default_config()
        cfg.grip = args.grip
        cfg.crouch_obs = args.crouch
        cfg.crouch_ff = args.crouch_ff
        cfg.crouch_waist = args.crouch_waist
        cfg.crouch_lean = args.crouch_lean
        if args.crouch_file:
            cfg.crouch_file = args.crouch_file
        if args.crouch_weight is not None:
            cfg.reward_config.scales.crouch_height = args.crouch_weight
        if args.place_yaw_weight is not None:
            cfg.reward_config.scales.place_yaw = args.place_yaw_weight
        if args.mode_p:
            cfg.case.mode_p = [float(x) for x in args.mode_p.split(",")]
        if args.feet_vel_weight is not None:
            cfg.reward_config.scales.stand_feet_vel = args.feet_vel_weight
        if args.place_touch_file:
            cfg.place_touch_file = args.place_touch_file
        cfg.desk_body = args.desk_body
        cfg.drift_all = args.drift_all
        cfg.office = args.office
        cfg.office_flow = args.office_flow
        cfg.stop_err_once = args.stop_err_once
        cfg.adjust = args.adjust
        cfg.yaw_shift = args.yaw_shift
        if args.stop_err_weight is not None:
            cfg.reward_config.scales.stop_err = args.stop_err_weight
        if args.office:                                     # C の道順を歩いて置くまで約 25 秒（A の続きからは受け取りから置き終わるまで約 66 秒）
            cfg.episode_length = 3800 if args.office_flow else 1500
        cfg.approach = args.approach
        if args.stop_drift_weight is not None:
            cfg.reward_config.scales.stop_drift = args.stop_drift_weight
        for k in ("desk_offset", "desk_dz", "place_edge"):
            if getattr(args, k) is not None:
                cfg.case[k] = getattr(args, k)
        if args.stop_range:
            cfg.case.stop_range = [float(x) for x in args.stop_range.split(",")]
        if args.desk_body:                                  # 机との接触が増える分
            cfg.naconmax = 32 * args.envs
        if args.office:                                     # 部屋の壁・机・椅子との接触が増える分
            cfg.naconmax = 40 * args.envs
            cfg.njmax = 256
        if args.estimator:
            est = json.loads(Path(args.estimator).read_text())
            cfg.estimator.w, cfg.estimator.b = est["w"], est["b"]
        env, eval_env = ce.CarryJoystick(cfg), ce.CarryJoystick(cfg)
        randomize = functools.partial(ce.domain_randomize, env=env)
    elif args.env == "standup":
        import standup_env as se
        cfg = se.default_config()
        env, eval_env = se.StandUp(cfg), se.StandUp(cfg)
        randomize = functools.partial(se.domain_randomize, env=env)
    else:
        cfg = pe.default_config()
        cfg.obs_mode = args.obs
        if args.estimator:
            est = json.loads(Path(args.estimator).read_text())
            cfg.estimator.w, cfg.estimator.b = est["w"], est["b"]
        env = pe.PayloadJoystick(cfg)
        eval_env = pe.PayloadJoystick(cfg)
        randomize = functools.partial(pe.domain_randomize, env=env)

    p = locomotion_params.brax_ppo_config("G1JoystickFlatTerrain")
    p.num_timesteps = int(args.steps)
    p.num_envs = args.envs
    p.num_evals = args.evals
    p.episode_length = cfg.episode_length
    params = dict(p)
    net = params.pop("network_factory")
    num_eval_envs = params.pop("num_eval_envs", 128)

    out = REPO / "models" / args.name
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps({"kind": args.env, "init": args.init, "env": cfg.to_dict(), "ppo": {k: v for k, v in params.items()},
                                                 "network": dict(net), "seed": args.seed}, indent=1, default=str))
    log_path = REPO / "logs" / f"train_{args.name}.jsonl"
    log_path.parent.mkdir(exist_ok=True)
    log = open(log_path, "w")
    t0 = time.time()

    def progress(step, metrics):
        rec = {"step": int(step), "sec": round(time.time() - t0, 1)}
        rec.update({k: float(v) for k, v in metrics.items() if k.startswith("eval/")})
        log.write(json.dumps(rec) + "\n")
        log.flush()
        print(f"{step / 1e6:6.1f} M 歩  {rec['sec'] / 60:5.1f} 分  評価の報酬 {rec.get('eval/episode_reward', 0):7.2f}"
              f"  エピソードの長さ {rec.get('eval/avg_episode_length', 0):6.1f}", flush=True)

    def save(step, make_policy, policy_params):
        brax_model.save_params(str(out / "params.pkl"), policy_params)

    make_inference_fn, policy_params, _ = ppo.train(
        environment=env,
        eval_env=eval_env,
        **params,
        network_factory=functools.partial(ppo_networks.make_ppo_networks, **net),
        num_eval_envs=num_eval_envs,
        seed=args.seed,
        wrap_env_fn=wrapper.wrap_for_brax_training,
        randomization_fn=randomize,
        progress_fn=progress,
        policy_params_fn=save,
        restore_params=brax_model.load_params(args.init) if args.init else None,
    )
    brax_model.save_params(str(out / "params.pkl"), policy_params)
    print(f"終わり：{(time.time() - t0) / 60:.1f} 分。{out / 'params.pkl'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
