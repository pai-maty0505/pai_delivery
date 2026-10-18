#!/usr/bin/env python3
"""
B（本物のケースを持って歩く方策）の試験と動画。

試験：ケースの重さ（1.5〜2.5 kg）・摩擦、床の摩擦、体の質量を学習と同じ範囲でランダムにした 100 体で、20 秒動かす。
  場面 carry   ：最初からケースを持ち、前へ 0.4 m/秒で歩き、途中で向きを変える（0.5 rad/秒）
  場面 receive ：立って待ち、3 秒で受け取り、5 秒から前へ 0.4 m/秒
  場面 release ：持って歩き、12 秒で降ろす（台が持ち上げる）。そのあとも歩き続ける（荷重が急に消えても歩けるか）
  場面 stop_release：8 秒まで前へ歩いて止まり、10 秒で立ったまま箱を外す（台が持ち上げる）。20 秒まで立っている
  場面 place   ：D 机の上に置く。4 秒まで前へ歩いて止まり、全身を止めて 1.25 秒待ち、腕で置く（まっすぐ下げて、手を左右へ
                 抜く）。机の板は止まった所の前に出す。成功 ＝ 倒れず、手を抜くまで箱を落とさない（足の動きは place_test で測る）
  成功 ＝ 20 秒の間、倒れず、ケースを落とさない
動画：carry の場面を 1 体、横から写す。

    .venv_mjx/bin/python scripts/eval_carry.py --name carry --video logs/carry.mp4
"""

import argparse
import functools
import json
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
import jax  # noqa: E402
import jax.numpy as jp  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from mujoco_playground._src.wrapper import BraxDomainRandomizationVmapWrapper  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import carry_env as ce  # noqa: E402
import policy_io  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
NEVER = 10 ** 6
SCENES = {
    "carry": dict(mode=0, onset=0, release=NEVER, walk_from=0.0),
    "receive": dict(mode=1, onset=3.0, release=NEVER, walk_from=5.0),
    "release": dict(mode=2, onset=0, release=12.0, walk_from=0.0),
    "stop_release": dict(mode=2, onset=0, release=10.0, walk_from=0.0, stop_at=8.0),
    "place": dict(mode=4, onset=0, release=None, walk_from=0.0, stop_at=4.0, settle=1.25),
}


def command(t, walk_from, stop_at=1e9):
    """前へ 0.4 m/秒。10〜13 秒は回りながら（0.5 rad/秒）。walk_from までと stop_at からは止まる。"""
    moving = (t >= walk_from) & (t < stop_at)
    v = jp.where(moving, 0.4, 0.0)
    w = jp.where(moving & (t >= 10.0) & (t < 13.0), 0.5, 0.0)
    return jp.array([v, 0.0, w])


def run(env, policy, n, scene, seed=1, record=False):
    sc = SCENES[scene]
    keys = jax.random.split(jax.random.PRNGKey(seed + 100), n)
    venv = BraxDomainRandomizationVmapWrapper(env, functools.partial(ce.domain_randomize, env=env, rng=keys))
    reset, step_fn = jax.jit(venv.reset), jax.jit(venv.step)
    dt = env.dt

    @jax.jit
    def step(state, k):
        t = k * dt
        cmd = jp.tile(command(t, sc["walk_from"], sc.get("stop_at", 1e9)), (n, 1))
        info = dict(state.info, command=cmd, step=jp.zeros_like(state.info["step"]), hp_f=jp.zeros(n),
                    pl_t=jp.full(n, k, dtype=jp.int32))
        state = state.replace(info=info)
        act = jax.vmap(lambda o, kk: policy(o, kk)[0])(state.obs, jax.random.split(jax.random.PRNGKey(0), n))
        return step_fn(state, act)

    state = reset(jax.random.split(jax.random.PRNGKey(seed), n))
    on = int(round(sc["onset"] / dt)) if sc["onset"] else 0
    extra = {}
    if sc["mode"] == 4:
        t_stop = int(round(sc["stop_at"] / dt))
        arm_t0 = t_stop + int(round(sc["settle"] / dt))
        off = arm_t0 + int(round(6.0 / dt))
        extra = dict(t_stop=jp.full(n, t_stop, dtype=jp.int32), arm_t0=jp.full(n, arm_t0, dtype=jp.int32),
                     desk_off=jp.zeros((n, 2)))
    else:
        off = int(round(sc["release"] / dt)) if sc["release"] != NEVER else NEVER
        extra = dict(t_stop=jp.full(n, NEVER, dtype=jp.int32), arm_t0=jp.full(n, NEVER, dtype=jp.int32))
    info = dict(state.info, mode=jp.full(n, sc["mode"]), onset=jp.full(n, on, dtype=jp.int32),
                release=jp.full(n, off, dtype=jp.int32), **extra)
    data = jax.vmap(lambda d, i: env._place_case(d, i, i["onset"] <= 0))(state.data, info)
    state = state.replace(data=data, info=info)
    fell = np.zeros(n, bool)
    dropped = np.zeros(n, bool)
    qs = []
    for k in range(int(round(20.0 / dt))):
        state = step(state, k)
        d = np.array(state.done) > 0
        dr = np.array(state.metrics["case_dropped"]) > 0
        new = d & ~(fell | dropped)
        dropped |= new & dr
        fell |= new & ~dr
        if record:
            qs.append((np.array(state.data.qpos[0]), np.array(state.data.mocap_pos[0])))
    return fell, dropped, qs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="carry")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--video")
    args = ap.parse_args()
    conf = json.loads((REPO / "models" / args.name / "config.json").read_text())
    cfg = ce.default_config()
    cfg.estimator.w, cfg.estimator.b = conf["env"]["estimator"]["w"], conf["env"]["estimator"]["b"]
    cfg.grip = conf["env"].get("grip", "hands")
    cfg.crouch_obs = conf["env"].get("crouch_obs", False)
    cfg.crouch_file = conf["env"].get("crouch_file", "place_traj_tray_tilt_crouch.npz")
    cfg.crouch_ff = conf["env"].get("crouch_ff", False)
    cfg.crouch_waist = conf["env"].get("crouch_waist", False)
    cfg.crouch_lean = conf["env"].get("crouch_lean", 0.0)
    cfg.place_touch_file = conf["env"].get("place_touch_file", "")
    cfg.desk_body = conf["env"].get("desk_body", False)
    cfg.approach = conf["env"].get("approach", False)
    for k in ("place_edge", "desk_offset", "stop_range"):
        if k in conf["env"].get("case", {}):
            cfg.case[k] = conf["env"]["case"][k]
    cfg.episode_length = 2000
    env = ce.CarryJoystick(cfg)
    policy = policy_io.load_policy(args.name, env, conf)
    res = {}
    for scene in SCENES:
        fell, dropped, _ = run(env, policy, args.n, scene)
        ok = ~fell & ~dropped
        res[scene] = {"success": float(ok.mean()), "fell": float(fell.mean()), "dropped": float(dropped.mean())}
        print(f"{scene:8s}：成功 {ok.mean() * 100:5.1f}%  倒れた {fell.mean() * 100:5.1f}%  ケースを落とした {dropped.mean() * 100:5.1f}%",
              flush=True)
    (REPO / "logs" / f"eval_{args.name}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))

    if args.video:
        fell, dropped, qs = run(env, policy, 1, "carry", seed=7, record=True)
        m = env.mj_model
        d = mujoco.MjData(m)
        W, H = 1280, 720
        renderer = mujoco.Renderer(m, H, W)
        cam = mujoco.MjvCamera()
        cam.distance, cam.azimuth, cam.elevation = 2.4, 0.0, -15.0
        fps = int(round(1 / env.dt))
        ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                               "-r", str(fps), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
                               "-movflags", "+faststart", args.video], stdin=subprocess.PIPE)
        q0 = qs[0][0]
        yaw0 = np.degrees(np.arctan2(2 * (q0[3] * q0[6] + q0[4] * q0[5]), 1 - 2 * (q0[5] ** 2 + q0[6] ** 2)))
        for q, mp in qs:
            d.qpos[:] = q
            d.mocap_pos[:] = mp
            mujoco.mj_forward(m, d)
            cam.lookat[:] = (q[0], q[1], 0.7)
            cam.azimuth = yaw0 + 120.0                            # 体の左斜め前から
            renderer.update_scene(d, cam)
            ff.stdin.write(renderer.render().tobytes())
        ff.stdin.close()
        ff.wait()
        print(args.video, "倒れた" if fell[0] else ("ケースを落とした" if dropped[0] else "倒れず、落とさなかった"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
