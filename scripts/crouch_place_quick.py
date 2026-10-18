#!/usr/bin/env python3
"""
学習で使う「置く」腕の動き（事務所の場面の place_with_ik と同じ順番・時間配分）。体を固定した試作で、0.02 秒ごとに逆運動学で求める。
  0〜0.8 秒  ：高さを保ったまま、手と肘を一緒に前へ FWD 出す
  0.8〜2.0 秒：手先を下げながら（手のひらを前へ TILT 傾ける）机の高さまで下ろす。肘は手より高く保つ
  2.0〜2.3 秒：そのまま（指先が机に触れ、箱の奥の縁が下りる）
  2.3〜3.1 秒：手のひらを外へ開き（0.6 秒）、0.15 秒遅れて肘も同じだけ外へ開く（0.6 秒）。肘は上げない。箱は机に落ちる
  3.1〜3.5 秒：手を 10 cm 上げる
  3.5〜4.1 秒：運ぶ姿勢に戻す（腰は沈めたまま。体は 4.5〜6.5 秒で起きる）
以前の説明（前腕を平行に保って下ろす版）：

    python scripts/crouch_place_level.py <沈める量 m> <上体を前へ倒す角度 度> <手を前へ出す量 m> <出力 npz> [動画]

  0〜2 秒    ：体を H 沈め、上体を A 度前へ倒す。手のひらは、先に高さを保ったまま前へ FWD 出し（0〜0.8 秒）、それから真下へ
               「机の上面 + PALM_GAP」の高さまで下ろす（手首が机の縁の側面に当たらない）
  2〜2.5 秒  ：手を 1 cm 下げて、手の甲を机に当てる
  2.5〜4.5 秒：前腕ごと左右へ開く（肘と手を同じだけ外へ）。箱は手のひらの上から机へ落ちる
  4.5〜6.5 秒：体を起こし、上体を戻す。手は先に上げてから運ぶ位置へ戻す
どの時刻も、手のひらは水平、前腕は床と平行（肘と手首の高さをそろえる）。
"""
import math
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from scipy.optimize import least_squares  # noqa: E402

import carry_env as ce  # noqa: E402
import payload_env as pe  # noqa: E402
import place_test as pt  # noqa: E402

H, LEAN, FWD = float(sys.argv[1]), math.radians(float(sys.argv[2])), float(sys.argv[3])
TILT = math.radians(float(os.environ.get("TILT", 14.0)))
OUT = sys.argv[4]
VIDEO = sys.argv[5] if len(sys.argv) > 5 else None
PALM_GAP = float(os.environ.get("PALM_GAP", 0.025))   # 手の甲が机に触れる時の、手のひら（site）の机の上面からの高さ
SPREAD = float(os.environ.get("SPREAD", 0.16))        # 前腕を左右へ開く量（片手）
SPREAD_FWD = float(os.environ.get("SPREAD_FWD", 0.03))
FWD_T = float(os.environ.get("FWD_T", 0.8))
OPEN_MODE = os.environ.get("OPEN_MODE", "parallel")   # parallel：肘と手のひらを同じだけ外へ平行に開く           # 手を前へ出し終える時刻（秒）。それから真下へ下ろす

m = pt.build(1.0, 0.45, (pt.EDGE + pt.DESK_DEPTH / 2, 0.0), 0.0)
d, dk = mujoco.MjData(m), mujoco.MjData(m)
J = pe.ARM_JOINTS
mirror = np.array([-1.0 if j in pe.MIRROR else 1.0 for j in J])
q0 = m.keyframe("knees_bent").qpos.copy()
adrL = [m.joint(f"left_{j}_joint").qposadr[0] for j in J]
actL = [m.actuator(f"left_{j}_joint").id for j in J]
actR = [m.actuator(f"right_{j}_joint").id for j in J]
lo = np.array([m.joint(f"left_{j}_joint").range[0] for j in J])
hi = np.array([m.joint(f"left_{j}_joint").range[1] for j in J])
palm, elbow, wrist = m.site("left_palm").id, m.body("left_elbow_link").id, m.body("left_wrist_roll_link").id
v0 = np.array([pe.CARRY_LEFT[j] for j in J])
sm = lambda u: (lambda x: x * x * (3 - 2 * x))(min(max(u, 0.0), 1.0))  # noqa: E731


def base(h, a):
    """骨盤を h 下げ、上体を a 前へ倒した骨盤の位置と向き（骨盤の y 軸まわりに回す。前が下がる向きが +）。"""
    pos = q0[0:3] - np.array([0, 0, h])
    quat = np.array([math.cos(a / 2), 0, math.sin(a / 2), 0])
    return pos, quat


def fk(v, h, a):
    dk.qpos[:len(q0)] = q0
    dk.qpos[0:3], dk.qpos[3:7] = base(h, a)
    dk.qpos[adrL] = v
    mujoco.mj_kinematics(m, dk)
    return dk.site_xpos[palm].copy(), dk.site_xmat[palm].reshape(3, 3).copy(), dk.xpos[elbow].copy(), dk.xpos[wrist].copy()


def body(t):
    """時刻 t の沈める量と前へ倒す角度。"""
    if t < 2.0:
        u = sm(t / 2.0)
    elif t < 4.5:
        u = 1.0
    else:
        u = 1.0 - sm((t - 4.5) / 2.0)
    return H * u, LEAN * u


P0, R0, E0, W0 = fk(v0, 0.0, 0.0)                     # 立って運ぶ時の手のひら（世界の座標）
P_place = np.array([P0[0] + FWD, P0[1], pt.DESK_TOP + PALM_GAP])



def ry_(a):
    return np.array([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]])


E0w = fk(v0, 0.0, 0.0)[2]
traj, v = [], v0.copy()
now = None
v_up = None
for k in range(int(8.0 / 0.02) + 1):
    t = k * 0.02
    h, a = body(t)
    cap = None
    if t < 0.8:
        u = sm(t / 0.8)
        P, Rt, E = P0 + np.array([FWD * u, 0, 0]), R0, E0w + np.array([FWD * u, 0, 0])
    elif t < 2.3:
        u = sm((t - 0.8) / 1.2)
        th = TILT * sm((t - 0.8) / 0.5)
        Pz = P0[2] + u * (P_place[2] - P0[2])
        P = np.array([P0[0] + FWD, P0[1], Pz])
        lf = math.hypot(*(P - (E0w + np.array([FWD, 0, 0])))[:2])
        E = np.array([E0w[0] + FWD, E0w[1], Pz + lf * math.sin(th) + (E0w[2] - P0[2])])
        Rt = ry_(th) @ R0
    elif t < 3.1:
        if now is None:
            now = (P.copy(), E.copy())
        t3 = t - 2.3
        uh, ue = sm(t3 / 0.6), sm((t3 - 0.15) / 0.6)
        P = now[0] + np.array([0, SPREAD * uh, 0.005])
        E = now[1] + np.array([0, SPREAD * ue, 0])
        Rt = ry_(TILT * (1 - uh)) @ R0
        cap = now[1][2]
    elif t < 3.5:
        u = sm((t - 3.1) / 0.4)
        up = np.array([0, 0, 0.10 * u])
        P, E, Rt = now[0] + np.array([0, SPREAD, 0.005]) + up, now[1] + np.array([0, SPREAD, 0]) + up, R0
    else:
        if v_up is None:
            v_up = v.copy()
        u = sm((t - 3.5) / 0.6)
        v = (1 - u) * v_up + u * v0
        traj.append(v.copy())
        continue

    def res(x):
        pp, RR, ee, ww = fk(x, h, a)
        r = [(pp - P) * 10, (RR - Rt).ravel() * 0.5, (ee - E) * np.array([2.0, 10.0, 10.0]), (x - v) * 0.05]
        if cap is not None:
            r.append([max(0.0, ee[2] - cap) * 30])
        return np.concatenate(r)
    v = least_squares(res, v, bounds=(lo, hi), max_nfev=100).x
    traj.append(v.copy())
    if k in (0, 40, 100, 115, 130, 145, 155, 175):
        pp, RR, ee, ww = fk(v, h, a)
        print(f"{t:4.2f} 秒  沈み {h:.2f}  手のひら x {pp[0]:.3f} y {pp[1]:.3f} z {pp[2]:.3f}  肘 x {ee[0]:.3f} y {ee[1]:.3f} z {ee[2]:.3f}  肘 {math.degrees(v[3]):5.1f}°")
traj = np.array(traj)
np.savez(OUT, traj=np.concatenate([traj, traj * mirror], 1), H=H, lean=LEAN)
print(OUT)

# 体を固定して動かしてみる（骨盤を下げ、前へ倒す）
d.qpos[:len(q0)] = q0
for sd in pe.SIDES:
    for i, j in enumerate(J):
        d.qpos[m.joint(f"{sd}_{j}_joint").qposadr[0]] = (v0 * (mirror if sd == "right" else 1))[i]
cc, ca = ce.tray_case_pose()
hx, hy, hz = ce.CASE_HALF
cq = m.joint("case_free").qposadr[0]
d.qpos[cq:cq + 3] = cc
d.qpos[cq + 3:cq + 7] = [math.cos(-ca / 2), 0, math.sin(-ca / 2), 0]
d.ctrl[:] = q0[7:7 + m.nu]
mujoco.mj_forward(m, d)
cb = m.body("case").id
m.opt.timestep = 0.002
legs = q0[7:22].copy()
frames = []
r = None
if VIDEO:
    m.vis.global_.offwidth, m.vis.global_.offheight = 960, 540
    r = mujoco.Renderer(m, 540, 960)
for i in range(int(10.0 / 0.002)):
    tt = max(i * 0.002 - 1.0, 0.0)
    h, a = body(tt)
    k = min(int(tt / 0.02), len(traj) - 1)
    d.qpos[0:3], d.qpos[3:7] = base(h, a)
    d.qpos[7:22] = legs
    d.qvel[0:21] = 0
    d.ctrl[actL], d.ctrl[actR] = traj[k], traj[k] * mirror
    mujoco.mj_step(m, d)
    if r is not None and i % 20 == 0:
        cam = mujoco.MjvCamera()
        cam.lookat[:] = (0.25, 0, 0.75)
        cam.distance, cam.azimuth, cam.elevation = 1.4, 90, -5
        r.update_scene(d, cam)
        frames.append(r.render().copy())
R = d.xmat[cb].reshape(3, 3)
bottom = d.xpos[cb][2] - abs(R[2, 2]) * hz - abs(R[2, 0]) * hx - abs(R[2, 1]) * hy
print("最後：箱の底", round(bottom, 3), "傾き", round(math.degrees(math.acos(float(np.clip(R[2, 2], -1, 1)))), 1),
      "手前の縁（骨盤から前へ）", round(d.xpos[cb][0] - hx, 3), f"（机の縁 {pt.EDGE}）")
if VIDEO:
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "960x540", "-r", "25", "-i", "-",
                           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", "-movflags", "+faststart", VIDEO], stdin=subprocess.PIPE)
    for f in frames:
        ff.stdin.write(f.tobytes())
    ff.stdin.close()
    ff.wait()
