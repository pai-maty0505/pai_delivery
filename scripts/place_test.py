#!/usr/bin/env python3
"""
C 置く の試験（CPU の MuJoCo、方策の計算も CPU の JAX）。

机の手前で止まって立ったまま（歩く方策に「止まれ」の指令。足は動かさない）、腕だけで ③ の箱を机に置く：
  1. 0〜1 秒   ：箱を持って立つ
  2. 1〜4 秒   ：腕を前へ伸ばしながら下げ、手が机のすぐ上に来るまで箱を水平のまま下ろす
  3. 4〜5 秒   ：さらに下げて手を机に当てる（箱は手に載ったまま）
  4. 5〜7 秒   ：手を左右の外側へ抜く（左右反対向きなので、箱を引きずる力が打ち消し合う）。箱は机に下りる
  5. 7〜9 秒   ：手を上げて体の前へ戻す
  6. 9〜12 秒  ：立って待つ
腕の関節の目標角は、手のひら（site palm）の位置と向きの目標から、立った姿勢で逆運動学で前もって求めておく（左腕を求めて右は鏡写し）。

成功（12 秒で判定）：箱が机の上に平らに載って止まっている（底が机の上面から 1 cm 以内、傾き 3° 未満、速さ 1 cm/秒 未満）、
  箱の手前の縁が机の縁より奥、ロボットが倒れていない（骨盤の高さ 0.5 m 以上）、足が床の上で 2 cm 以上動いていない。
回ごとに変える条件：机の縁までの距離 ±3 cm、左右のずれ ±3 cm、机の向き ±5°、手の上の箱の位置のずれ ±1 cm、
  箱と手の摩擦 0.8〜1.2、箱・手と机の摩擦 0.3〜0.6。

    JAX_PLATFORMS=cpu .venv_mjx/bin/python scripts/place_test.py --walk carry7 --n 25 --seed 0 --json logs/place_test.jsonl
    JAX_PLATFORMS=cpu .venv_mjx/bin/python scripts/place_test.py --walk carry7 --n 1 --video logs/place_test.mp4
"""
import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
if os.environ.get("JAX_PLATFORMS") == "cpu":
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import jax  # noqa: E402
import jax.numpy as jp  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from scipy.optimize import least_squares  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import carry_env as ce  # noqa: E402
import payload_env as pe  # noqa: E402
import policy_io  # noqa: E402
import standup_env as se  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
DT, N_SUB, N_ACT = 0.02, 10, 15
WALK_SCALE = 0.5
DESK_TOP, DESK_DEPTH, DESK_W = 0.74, 0.70, 1.2
EDGE = 0.13                 # 机の手前の縁（骨盤から前へ、m。回ごとに ±3 cm）
FORWARD = 0.085             # 箱を前へ出す量
SPREAD = 0.12               # 手を左右の外側へ動かす量（片手）
T_END = 12.0                # 腕を動かし始めてから 8 秒で置き終わる（体を沈める置き方）。1 秒立ってから始め、3 秒待つ


def ry(a):
    return np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])


def build(mu_hand, mu_desk, desk_pos, desk_yaw):
    ce.GRIP = "tray"
    pe.CARRY_LEFT.update(ce.TRAY_GRIP["arm"])
    cfg = ce.default_config()

    def decorate(spec):
        ce.decorate(spec)
        desk = spec.worldbody.add_body(name="desk_body", pos=[desk_pos[0], desk_pos[1], DESK_TOP - 0.015],
                                       quat=[math.cos(desk_yaw / 2), 0, 0, math.sin(desk_yaw / 2)])
        desk.add_geom(name="desk_top", type=mujoco.mjtGeom.mjGEOM_BOX, size=[DESK_DEPTH / 2, DESK_W / 2, 0.015],
                      rgba=[0.55, 0.4, 0.25, 1], contype=0, conaffinity=0)
        hand_geoms = []
        for s in pe.SIDES:
            for b in ("wrist_yaw_link", "wrist_pitch_link", "wrist_roll_link"):
                for k, g in enumerate(spec.body(f"{s}_{b}").geoms):
                    if g.type == mujoco.mjtGeom.mjGEOM_MESH:
                        if not g.name:
                            g.name = f"{s}_{b}_mesh{k}"
                        hand_geoms.append(g.name)
        for a in ["case_geom"] + hand_geoms:
            p = spec.add_pair(geomname1=a, geomname2="desk_top")
            p.condim = 3
            p.friction = [mu_desk, mu_desk, 0.005, 0.0001, 0.0001]

    m = pe.build_model(decorate=decorate)
    m.opt.timestep = DT / N_SUB
    m.opt.iterations, m.opt.ls_iterations = cfg.solver_iterations, cfg.solver_ls_iterations
    for i in range(m.npair):
        names = (m.geom(m.pair_geom1[i]).name, m.geom(m.pair_geom2[i]).name)
        if "case_geom" in names or "desk_top" in names:
            m.pair_solref[i] = cfg.case.contact_solref
        if "case_geom" in names and "desk_top" not in names and not any(n.startswith(("support", "grip", "floor")) for n in names):
            m.pair_friction[i, 0:2] = mu_hand
    for s in pe.SIDES:
        for j in pe.ARM_JOINTS:
            m.dof_damping[m.joint(f"{s}_{j}_joint").dofadr[0]] = cfg.arm_damping
        m.body_mass[m.body(f"{s}_payload").id] = 0.0
        m.body_inertia[m.body(f"{s}_payload").id] = 1e-8
    cb = m.body("case").id
    hx, hy, hz = ce.CASE_HALF
    m.body_mass[cb] = 1.23
    m.body_inertia[cb] = 1.23 / 3 * np.array([hy ** 2 + hz ** 2, hx ** 2 + hz ** 2, hx ** 2 + hy ** 2])
    m.geom_rgba[m.geom("case_geom").id] = [0.93, 0.93, 0.93, 1]
    return m


def arm_trajectory(m):
    """立った姿勢（骨盤は原点、+x を向く）で、手のひらの目標を時間で動かし、左腕の関節の目標角を 0.02 秒ごとに求める。"""
    d = mujoco.MjData(m)
    q0 = m.keyframe("knees_bent").qpos.copy()
    J = pe.ARM_JOINTS
    adr = [m.joint(f"left_{j}_joint").qposadr[0] for j in J]
    lo = np.array([m.joint(f"left_{j}_joint").range[0] for j in J])
    hi = np.array([m.joint(f"left_{j}_joint").range[1] for j in J])
    palm = m.site("left_palm").id
    v0 = np.array([pe.CARRY_LEFT[j] for j in J])

    def fk(v):
        d.qpos[:len(q0)] = q0
        d.qpos[adr] = v
        mujoco.mj_kinematics(m, d)
        return d.site_xpos[palm].copy(), d.site_xmat[palm].reshape(3, 3).copy()

    def hand_low(v):
        fk(v)
        z = []
        for g in range(m.ngeom):
            if m.geom_bodyid[g] in (m.body("left_wrist_yaw_link").id, m.body("left_wrist_pitch_link").id) \
                    and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
                mid = m.geom_dataid[g]
                V = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
                z.append((d.geom_xpos[g] + V @ d.geom_xmat[g].reshape(3, 3).T)[:, 2].min())
        return min(z)

    def ik(p, R, v_init):
        def res(v):
            pp, RR = fk(v)
            return np.concatenate([(pp - p) * 10.0, (RR - R).ravel() * 0.5, (v - v_init) * 0.05])
        return least_squares(res, v_init, bounds=(lo, hi), max_nfev=80).x

    P0, R0 = fk(v0)
    drop = hand_low(v0) - (DESK_TOP + 0.01)

    def pose_at(t):
        if t < 1.0:
            return P0
        if t < 4.0:
            u = (t - 1.0) / 3.0
            u = u * u * (3 - 2 * u)
            return P0 + np.array([FORWARD * u, 0, -drop * u])
        p3 = P0 + np.array([FORWARD, 0, -drop])
        if t < 5.0:
            return p3 + np.array([0, 0, -0.015 * (t - 4.0)])
        p4 = p3 + np.array([0, 0, -0.015])
        if t < 7.0:
            u = (t - 5.0) / 2.0
            u = u * u * (3 - 2 * u)
            return p4 + np.array([0, SPREAD * u, -0.005 * u])
        p5 = p4 + np.array([0, SPREAD, -0.005])
        u = min((t - 7.0) / 2.0, 1.0)
        return p5 + np.array([-0.10 * u, 0, 0.10 * u])

    traj, v = [], v0.copy()
    for k in range(int(T_END / DT) + 1):
        v = ik(pose_at(k * DT), R0, v)
        traj.append(v.copy())
    return np.array(traj)


# 体を沈めて置く（2026-10-16）：机の前で完全に止まって、脚を曲げて体を沈め、箱を水平のまま机の高さまで下ろしてから、手と腕を
# 左右へ抜く。腕は運ぶ姿勢のまま、箱を机の上へ出すために前へ少しだけ出す。時刻は腕を動かし始めてから（秒）
CROUCH_DOWN, CROUCH_PRESS, CROUCH_SPREAD, CROUCH_UP, CROUCH_END = 2.0, 2.5, 4.5, 6.5, 8.0
CROUCH_FWD = 0.07              # 腕を前へ出す量（箱の手前の縁を机の縁より奥へ）
CROUCH_RELEASE = 3.5           # 箱が手から机へ移る時刻（手を左右へ抜いている途中）


def smooth(u):
    u = min(max(u, 0.0), 1.0)
    return u * u * (3 - 2 * u)


def crouch_height(t, H):
    """腰を下げる量（m）：0〜2 秒で H まで下げ、4.5〜6.5 秒で戻す。"""
    if t < CROUCH_DOWN:
        return H * smooth(t / CROUCH_DOWN)
    if t < CROUCH_SPREAD:
        return H
    return H * (1.0 - smooth((t - CROUCH_SPREAD) / (CROUCH_UP - CROUCH_SPREAD)))


def crouch_arm_trajectory(m):
    """体を沈めて置くときの左腕の関節の目標角（0.02 秒ごと、8 秒）と、手が机に触れるのに要る腰を下げる量 H を求める。
    腕の目標は骨盤に対して決める（体が沈んでも腕は同じ形）。"""
    d = mujoco.MjData(m)
    q0 = m.keyframe("knees_bent").qpos.copy()
    J = pe.ARM_JOINTS
    adr = [m.joint(f"left_{j}_joint").qposadr[0] for j in J]
    lo = np.array([m.joint(f"left_{j}_joint").range[0] for j in J])
    hi = np.array([m.joint(f"left_{j}_joint").range[1] for j in J])
    palm = m.site("left_palm").id
    v0 = np.array([pe.CARRY_LEFT[j] for j in J])

    def fk(v):
        d.qpos[:len(q0)] = q0
        d.qpos[adr] = v
        mujoco.mj_kinematics(m, d)
        return d.site_xpos[palm].copy(), d.site_xmat[palm].reshape(3, 3).copy()

    def hand_low(v):
        fk(v)
        z = []
        for g in range(m.ngeom):
            if m.geom_bodyid[g] in (m.body("left_wrist_yaw_link").id, m.body("left_wrist_pitch_link").id) \
                    and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
                mid = m.geom_dataid[g]
                V = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
                z.append((d.geom_xpos[g] + V @ d.geom_xmat[g].reshape(3, 3).T)[:, 2].min())
        return min(z)

    def ik(p, R, v_init):
        def res(v):
            pp, RR = fk(v)
            return np.concatenate([(pp - p) * 10.0, (RR - R).ravel() * 0.5, (v - v_init) * 0.05])
        return least_squares(res, v_init, bounds=(lo, hi), max_nfev=80).x

    P0, R0 = fk(v0)
    H = hand_low(v0) - DESK_TOP                                   # 立った姿勢の手の一番低い所から、机の上面まで

    def pose_at(t):
        if t < CROUCH_DOWN:
            return P0 + np.array([CROUCH_FWD * smooth(t / CROUCH_DOWN), 0, 0])
        p1 = P0 + np.array([CROUCH_FWD, 0, 0])
        if t < CROUCH_PRESS:
            return p1 + np.array([0, 0, -0.01 * smooth((t - CROUCH_DOWN) / (CROUCH_PRESS - CROUCH_DOWN))])
        p2 = p1 + np.array([0, 0, -0.01])
        if t < CROUCH_SPREAD:
            return p2 + np.array([0, SPREAD, -0.005]) * smooth((t - CROUCH_PRESS) / (CROUCH_SPREAD - CROUCH_PRESS))
        p3 = p2 + np.array([0, SPREAD, -0.005])
        return p3 + np.array([-0.08, 0, 0.06]) * smooth((t - CROUCH_SPREAD) / (CROUCH_UP - CROUCH_SPREAD))

    traj, v = [], v0.copy()
    for k in range(int(CROUCH_END / DT) + 1):
        v = ik(pose_at(k * DT), R0, v)
        traj.append(v.copy())
    return np.array(traj), float(H)


def load_walk(name):
    conf = json.loads((REPO / "models" / name / "config.json").read_text())
    cfg = ce.default_config()
    cfg.impl = "jax"
    cfg.grip = conf["env"].get("grip", "tray")
    cfg.crouch_obs = conf["env"].get("crouch_obs", False)
    cfg.crouch_file = conf["env"].get("crouch_file", "place_traj_tray_tilt_crouch.npz")
    cfg.crouch_ff = conf["env"].get("crouch_ff", False)
    cfg.crouch_waist = conf["env"].get("crouch_waist", False)
    cfg.crouch_lean = conf["env"].get("crouch_lean", 0.0)
    cfg.place_touch_file = conf["env"].get("place_touch_file", "")
    cfg.estimator.w, cfg.estimator.b = conf["env"]["estimator"]["w"], conf["env"]["estimator"]["b"]
    env = ce.CarryJoystick(cfg)
    pol = jax.jit(policy_io.load_policy(name, env, conf))
    return pol, jp.zeros(env.observation_size["privileged_state"]), np.array(env._default_pose), cfg


def load_standup(name="standup"):
    """立ち上がる方策（A）。立ち終わった後の姿勢（参照の最後）を保つのに使う。"""
    scfg = se.default_config()
    scfg.impl = "jax"
    senv = se.StandUp(scfg)
    conf = json.loads((REPO / "models" / name / "config.json").read_text())
    pol = jax.jit(policy_io.load_policy(name, senv, conf))
    return pol, jp.zeros(senv.observation_size["privileged_state"]), senv._ref_np


def trial(m_args, traj, walk, rng, video=None, est_clip=(0.0, 1.5), legs="policy", stand=None):
    pol, priv, default, wcfg = walk
    crouch = bool(wcfg.crouch_obs)
    ff_table = ce.crouch_ff_table() if getattr(wcfg, "crouch_ff", False) else None
    if crouch:
        z = np.load(REPO / "models" / wcfg.crouch_file)
        ctraj, H = z["traj"], float(z["H"])
    p = dict(dx=rng.uniform(-0.03, 0.03), dy=rng.uniform(-0.03, 0.03), yaw=np.radians(rng.uniform(-5, 5)),
             bx=rng.uniform(-0.01, 0.01), by=rng.uniform(-0.01, 0.01), mu_hand=rng.uniform(0.8, 1.2), mu_desk=rng.uniform(0.3, 0.6))
    edge = EDGE + p["dx"]
    # 机を向き yaw だけ回す（机の手前の縁の真ん中を中心に）
    c, s = math.cos(p["yaw"]), math.sin(p["yaw"])
    off = np.array([DESK_DEPTH / 2, 0.0])
    desk_center = np.array([edge, p["dy"]]) + np.array([c * off[0] - s * off[1], s * off[0] + c * off[1]])
    m = build(p["mu_hand"], p["mu_desk"], desk_center, p["yaw"])
    d = mujoco.MjData(m)
    q0 = m.keyframe("knees_bent").qpos.copy()
    arm = np.concatenate([traj[0], traj[0] * np.array([-1.0 if j in pe.MIRROR else 1.0 for j in pe.ARM_JOINTS])])
    for i, (sd, j) in enumerate((sd, j) for sd in pe.SIDES for j in pe.ARM_JOINTS):
        q0[m.joint(f"{sd}_{j}_joint").qposadr[0]] = arm[i]
    d.qpos[:len(q0)] = q0
    cc, ca = ce.tray_case_pose()
    cq = m.joint("case_free").qposadr[0]
    d.qpos[cq:cq + 3] = cc + np.array([p["bx"], p["by"], 0.0])
    d.qpos[cq + 3:cq + 7] = [np.cos(-ca / 2), 0.0, np.sin(-ca / 2), 0.0]
    d.ctrl[:] = q0[7:7 + m.nu]
    mujoco.mj_forward(m, d)
    imu = m.site("imu_in_pelvis").id
    gyro_adr = m.sensor_adr[m.sensor("gyro_pelvis").id]
    linvel_adr = m.sensor_adr[m.sensor("local_linvel_pelvis").id]
    feet_b = [m.body(f"{sd}_ankle_roll_link").id for sd in pe.SIDES]
    feet0 = np.array([d.xpos[b][:2] for b in feet_b])
    est_w, est_b = np.array(wcfg.estimator.w), wcfg.estimator.b
    alpha = DT / (wcfg.payload.est_tau + DT)
    m_est, last = 1.23, np.zeros(N_ACT)
    phase, phase_dt = np.array([0.0, math.pi]), 2 * math.pi * DT * 1.375
    mirror = np.array([-1.0 if j in pe.MIRROR else 1.0 for j in pe.ARM_JOINTS])
    key = jax.random.PRNGKey(0)
    cb = m.body("case").id
    fell, foot_move = False, 0.0
    z_stand, z_min = None, 9.0
    yaw0 = math.atan2(2 * (d.qpos[3] * d.qpos[6] + d.qpos[4] * d.qpos[5]), 1 - 2 * (d.qpos[5] ** 2 + d.qpos[6] ** 2))
    r = frames = None
    if video:
        W, H = 960, 540
        m.vis.global_.offwidth, m.vis.global_.offheight = W, H
        r, frames = mujoco.Renderer(m, H, W), []
    for k in range(int(T_END / DT)):
        g = d.site_xmat[imu].reshape(3, 3).T @ np.array([0, 0, -1.0])
        q = d.qpos[7:7 + 29] - default
        tc = k * DT - 1.0                                       # 1 秒立ってから腕を動かし始める（体を沈める方策のとき）
        h = 0.0
        if crouch:
            h = (H * smooth(tc / ce.CROUCH_DOWN) if tc < ce.CROUCH_DOWN else H if tc < ce.CROUCH_HOLD_END
                 else H * (1 - smooth((tc - ce.CROUCH_HOLD_END) / (ce.CROUCH_UP_END - ce.CROUCH_HOLD_END)))) if tc >= 0 else 0.0
        obs = np.concatenate([d.sensordata[linvel_adr:linvel_adr + 3], d.sensordata[gyro_adr:gyro_adr + 3], g, np.zeros(3),
                              q[:N_ACT], d.qvel[6:6 + N_ACT], last, np.cos(phase), np.sin(phase), [m_est / 3.0]]
                             + ([[h / 0.1]] if crouch else []))
        v = traj[min(k + 1, len(traj) - 1)] if not crouch else ctraj[min(max(int(round(tc / DT)) + 1, 0), len(ctraj) - 1), :7]
        if legs == "standup":                                  # 立ち上がる方策で、立ち終わった姿勢を保つ（足を動かさない）
            spol, spriv, ref = stand
            q_ref = ref[-1, 7:7 + N_ACT]
            qq = d.qpos[7:7 + N_ACT]
            sobs = np.concatenate([d.sensordata[gyro_adr:gyro_adr + 3], g, qq - q_ref, d.qvel[6:6 + N_ACT] * 0.1, q_ref,
                                   np.zeros(N_ACT), last, [1.2, 1.0]])
            act = np.array(spol({"state": jp.array(sobs), "privileged_state": spriv}, key)[0])
            leg = q_ref + act * 0.6
        else:
            act = np.array(pol({"state": jp.array(obs), "privileged_state": priv}, key)[0])
            leg = default[:N_ACT] + act * WALK_SCALE if legs == "policy" else default[:N_ACT]   # pd：脚と腰は立った姿勢に保つだけ
            if crouch and wcfg.crouch_ff and legs in ("policy", "pd"):   # 沈める量に合わせたしゃがむ姿勢を先に足す（学習と同じ）
                leg = leg + ce.crouch_ff(h, ff_table, waist=getattr(wcfg, "crouch_waist", False), lean_deg=getattr(wcfg, "crouch_lean", 0.0), H=H)
        d.ctrl[:] = np.concatenate([leg, v, v * mirror])
        if os.environ.get("PT_DEBUG") and k % 5 == 0:             # 確認用：腰の高さ・指令・方策の出力の大きさ
            print(f"  {k * DT:5.2f} 秒  腰 {d.qpos[2]:.3f}  指令 {h:.3f}  出力 {np.abs(act).max():.2f}  傾き {np.degrees(np.arccos(np.clip(-g[2], -1, 1))):.1f}°")
        for _ in range(N_SUB):
            mujoco.mj_step(m, d)
        raw = float(est_w @ d.actuator_force[N_ACT:29] + est_b)
        m_est = float(np.clip(m_est + alpha * (raw - m_est), *est_clip))
        phase = np.fmod(phase + phase_dt + math.pi, 2 * math.pi) - math.pi
        phase = np.array(ce.STAND_PHASE)                       # 止まっている（指令 0）：歩く周期の時計を止める（学習と同じ）
        last = act
        fell |= d.qpos[2] < 0.5
        if k == int(0.9 / DT):
            z_stand = float(d.qpos[2])
        if k > int(0.9 / DT):
            z_min = min(z_min, float(d.qpos[2]))
        foot_move = max(foot_move, float(np.max(np.linalg.norm(np.array([d.xpos[b][:2] for b in feet_b]) - feet0, axis=1))))
        if r is not None and k % 2 == 0:
            cam = mujoco.MjvCamera()
            cam.lookat[:] = (0.25, 0.0, 0.8)
            cam.distance, cam.azimuth, cam.elevation = 1.5, 125.0, -20.0
            r.update_scene(d, cam)
            frames.append(r.render().copy())
    hx, hy, hz = ce.CASE_HALF
    R = d.xmat[cb].reshape(3, 3)
    pos = d.xpos[cb]
    bottom = pos[2] - abs(R[2, 2]) * hz - abs(R[2, 0]) * hx - abs(R[2, 1]) * hy
    tilt = math.degrees(math.acos(float(np.clip(R[2, 2], -1.0, 1.0))))
    # 箱の 4 つの下の角が、机の手前の縁より奥にあるか（机の座標で）
    Rd = np.array([[c, -s], [s, c]])
    corners = [pos[:2] + R[:2, :2] @ np.array([sx * hx, sy * hy]) for sx in (-1, 1) for sy in (-1, 1)]
    depth = min(float((Rd.T @ (cn - np.array([edge, p["dy"]])))[0]) for cn in corners)
    speed = float(np.linalg.norm(d.cvel[cb][3:]))
    res = dict(params={k_: round(float(v_), 4) for k_, v_ in p.items()}, bottom=round(float(bottom), 4), tilt=round(tilt, 2),
               depth=round(depth, 4), speed=round(speed, 4), fell=bool(fell), foot_move=round(foot_move, 4),
               sink=round((z_stand or 0) - z_min, 4),
               yaw_drift_deg=round(math.degrees((math.atan2(2 * (d.qpos[3] * d.qpos[6] + d.qpos[4] * d.qpos[5]), 1 - 2 * (d.qpos[5] ** 2 + d.qpos[6] ** 2)) - yaw0 + math.pi) % (2 * math.pi) - math.pi), 2))
    res["on_desk"] = bool(abs(bottom - DESK_TOP) < 0.01 and tilt < 3.0 and speed < 0.01 and depth > 0.0)
    res["success"] = bool(res["on_desk"] and not fell and foot_move < 0.02)
    if video:
        ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                               "-r", "25", "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
                               "-movflags", "+faststart", video], stdin=subprocess.PIPE)
        for f in frames:
            ff.stdin.write(f.tobytes())
        ff.stdin.close()
        ff.wait()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--walk", default="carry7")
    ap.add_argument("--n", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--legs", choices=("policy", "pd", "standup"), default="policy", help="立ち方：policy ＝ 歩く方策に止まれの指令、pd ＝ 脚と腰を立った姿勢に保つ")
    ap.add_argument("--json")
    ap.add_argument("--video")
    args = ap.parse_args()
    walk = load_walk(args.walk)
    stand = load_standup() if args.legs == "standup" else None
    traj = arm_trajectory(build(1.0, 0.45, (EDGE + DESK_DEPTH / 2, 0.0), 0.0))
    rng = np.random.default_rng(args.seed)
    ok = 0
    for i in range(args.n):
        res = trial(None, traj, walk, rng, video=args.video if i == 0 else None, legs=args.legs, stand=stand)
        ok += res["success"]
        print(json.dumps(res, ensure_ascii=False), flush=True)
        if args.json:
            with open(args.json, "a") as f:
                f.write(json.dumps(res) + "\n")
    print(f"成功 {ok} / {args.n}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
