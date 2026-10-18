#!/usr/bin/env python3
"""しゃがむ姿勢の表 models/crouch_offsets.npy を作る：骨盤を真下へ h（0〜0.25 m、1 cm ごと）下げ、足の裏を床に平らに保つ
股・膝・足首（pitch）の角度の変化を、立った姿勢（knees_bent）から逆運動学で求める。

    CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu .venv_mjx/bin/python scripts/crouch_offsets.py
"""
import sys; sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
import numpy as np, mujoco
from scipy.optimize import least_squares
import payload_env as pe, place_test as pt
m = pt.build(1.0, 0.45, (pt.EDGE + pt.DESK_DEPTH / 2, 0.0), 0.0); d = mujoco.MjData(m)
q0 = m.keyframe("knees_bent").qpos.copy()
names = [m.joint(j).name for j in range(m.njnt)][1:16]
print(names)
def adr(n): return m.joint(n).qposadr[0]
J = {s: [adr(f"{s}_{j}_joint") for j in ("hip_pitch", "knee", "ankle_pitch")] for s in ("left", "right")}
foot = m.body("left_ankle_roll_link").id; pel = m.body("pelvis").id
def rel(q):
    d.qpos[:] = q; mujoco.mj_kinematics(m, d)
    R = d.xmat[pel].reshape(3, 3)
    p = R.T @ (d.xpos[foot] - d.xpos[pel]); Rf = R.T @ d.xmat[foot].reshape(3, 3)
    return p, Rf
p0, Rf0 = rel(q0)
print("足首（骨盤から）", p0)
out = []
x = np.zeros(3)
for h in np.arange(0, 0.2501, 0.01):
    def res(x):
        q = q0.copy()
        for s in ("left", "right"): q[J[s]] += x
        p, Rf = rel(q)
        return np.concatenate([[(p[2] - (p0[2] + h)) * 10, (p[0] - p0[0]) * 10], (Rf - Rf0)[[0, 2], 0] * 3])
    x = least_squares(res, x).x
    out.append(x.copy())
    if abs(h * 100 % 5) < 1e-6: print(f"h {h:.2f}  股 {np.degrees(x[0]):6.1f}°  膝 {np.degrees(x[1]):6.1f}°  足首 {np.degrees(x[2]):6.1f}°  残り {np.abs(res(x)).max():.4f}")
np.save(__import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "..", "models", "crouch_offsets.npy"), np.array(out))
