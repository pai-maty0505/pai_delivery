#!/usr/bin/env python3
"""
学習で使う「置く」腕の動きと沈める量を、机に触れた時刻ごとに表にする（事務所の場面の place_with_ik と同じ順番・速さ）。
学習の環境では、手のひらを下ろしている間に箱か手が机の上面に触れたら、その時刻の表に切り替えて開く（事務所と同じく
「触れたら止めて開く」）。体を固定した試作で、0.02 秒ごとに逆運動学で求める。

    python scripts/crouch_place_touch.py <沈める量の上限 m> <上体を前へ倒す角度 度> <手を前へ出す量 m> <出力 npz>

  1. 0〜0.8 秒：高さを保ったまま、手と肘を一緒に前へ FWD 出す（腰は 4 cm 沈める）
  2. 手のひらを 6 cm/秒で下げ、手先を下げる（0.5 秒で TILT 傾ける）。肘は手より高く。腰は 8 cm/秒で上限 H まで沈める。
     触れた歩 j（下ろし始めてから、1〜150 歩 ＝ 0.02〜3 秒）で止める
  3. 手のひらを外へ開き（0.6 秒）、0.15 秒遅れて肘も同じだけ外へ開く（0.6 秒）。手は 0.5 cm 浮かせ、肘は上げない
  4. 手を 10 cm 上げ（0.4 秒）、腰を沈めたまま運ぶ姿勢へ戻し（0.6 秒）、体を起こす（2 秒）
出力：tab（触れた歩 150 通り × 歩 × 15：左右の腕 14 関節と沈める量）、N1（前へ出す歩の数）
"""
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from scipy.optimize import least_squares  # noqa: E402

import payload_env as pe  # noqa: E402
import place_test as pt  # noqa: E402

H, LEAN, FWD = float(sys.argv[1]), math.radians(float(sys.argv[2])), float(sys.argv[3])
OUT = sys.argv[4]
TILT, SPREAD = math.radians(14.0), 0.12
DT, N1, N2 = 0.02, 40, 150
T_HAND, T_LAG, T_ELB = 0.6, 0.15, 0.6
N3, N_UP, N_BACK, N_RISE, N_HOLD = int((T_LAG + T_ELB + 0.2) / DT), 20, 30, 100, 50

m = pt.build(1.0, 0.45, (pt.EDGE + pt.DESK_DEPTH / 2, 0.0), 0.0)
dk = mujoco.MjData(m)
J = pe.ARM_JOINTS
mirror = np.array([-1.0 if j in pe.MIRROR else 1.0 for j in J])
q0 = m.keyframe("knees_bent").qpos.copy()
adrL = [m.joint(f"left_{j}_joint").qposadr[0] for j in J]
lo = np.array([m.joint(f"left_{j}_joint").range[0] for j in J])
hi = np.array([m.joint(f"left_{j}_joint").range[1] for j in J])
palm, elbow = m.site("left_palm").id, m.body("left_elbow_link").id
v0 = np.array([pe.CARRY_LEFT[j] for j in J])
sm = lambda u: (lambda x: x * x * (3 - 2 * x))(min(max(u, 0.0), 1.0))  # noqa: E731


def ry_(a):
    return np.array([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]])


def fk(v, h):
    """骨盤を h 沈め、上体を LEAN·h/H 前へ倒した体での、手のひらの位置・向きと肘の位置。"""
    a = LEAN * h / H
    dk.qpos[:len(q0)] = q0
    dk.qpos[0:3] = q0[0:3] - np.array([0, 0, h])
    dk.qpos[3:7] = [math.cos(a / 2), 0, math.sin(a / 2), 0]
    dk.qpos[adrL] = v
    mujoco.mj_kinematics(m, dk)
    return dk.site_xpos[palm].copy(), dk.site_xmat[palm].reshape(3, 3).copy(), dk.xpos[elbow].copy()


def solve(v, h, P, Rt, E, cap=None):
    def res(x):
        pp, RR, ee = fk(x, h)
        r = [(pp - P) * 10, (RR - Rt).ravel() * 0.5, (ee - E) * np.array([2.0, 10.0, 10.0]), (x - v) * 0.05]
        if cap is not None:
            r.append([max(0.0, ee[2] - cap) * 30])
        return np.concatenate(r)
    return least_squares(res, v, bounds=(lo, hi), max_nfev=100).x


t0 = time.time()
P0, R0, E0 = fk(v0, 0.0)
# 1・2：共通（前へ出し、下ろす）
pre, v = [], v0.copy()
for k in range(N1):
    u = sm(k / N1)
    h = 0.04 * u
    v = solve(v, h, P0 + [FWD * u, 0, 0], R0, E0 + [FWD * u, 0, 0])
    pre.append((v.copy(), h))
P, E, h = P0 + [FWD, 0, 0], E0 + [FWD, 0, 0], 0.04
lowers = []                                            # 下ろし始めて n2 歩目の（腕、沈める量、手のひらの目標、肘の目標）
for n2 in range(1, N2 + 1):
    th = TILT * sm(n2 / 25)
    h = min(H, h + 0.08 * DT)
    P = P + [0, 0, -0.06 * DT]
    lf = math.hypot(*(P - E)[:2])
    E = np.array([E[0], E[1], P[2] + lf * math.sin(th) + (E0[2] - P0[2])])
    v = solve(v, h, P, ry_(th) @ R0, E)
    lowers.append((v.copy(), h))
L = N1 + N2 + N3 + N_UP + N_BACK + N_RISE + N_HOLD
tab = np.zeros((N2, L, 8))
for j in range(1, N2 + 1):
    rows = [np.r_[a, hh] for a, hh in pre] + [np.r_[a, hh] for a, hh in lowers[:j]]
    v, hc = lowers[j - 1]
    Pn, _, En = fk(v, hc)                              # 触れた時の手のひらと肘（事務所の ik.now と同じく、実際の位置から）
    for k in range(N3):
        t3 = k * DT
        uh, ue = sm(t3 / T_HAND), sm((t3 - T_LAG) / T_ELB)
        v = solve(v, hc, Pn + [0, SPREAD * uh, 0.005], ry_(TILT * (1 - uh)) @ R0, En + [0, SPREAD * ue, 0], cap=En[2])
        rows.append(np.r_[v, hc])
    Po, _, Eo = fk(v, hc)
    for k in range(N_UP):
        up = np.array([0, 0, 0.10 * sm(k / N_UP)])
        v = solve(v, hc, Po + up, R0, Eo + up)
        rows.append(np.r_[v, hc])
    v_up = v.copy()
    for k in range(N_BACK):
        u = sm(k / N_BACK)
        rows.append(np.r_[(1 - u) * v_up + u * v0, hc])
    for k in range(N_RISE):
        rows.append(np.r_[v0, hc * (1 - sm(k / N_RISE))])
    rows += [np.r_[v0, 0.0]] * (L - len(rows))
    tab[j - 1] = np.array(rows[:L])
    if j % 25 == 0:
        print(f"触れた歩 {j}（{j * DT:.2f} 秒、沈み {hc * 100:.1f} cm、手のひらの高さ {Pn[2]:.3f}）  {time.time() - t0:.0f} 秒", flush=True)
both = np.concatenate([tab[..., :7], tab[..., :7] * mirror, tab[..., 7:]], -1)   # 左腕・右腕（左右対称）・沈める量
np.savez(OUT, tab=both.astype(np.float32), N1=N1, H=H, lean=LEAN, DT=DT)
print(OUT, both.shape)
