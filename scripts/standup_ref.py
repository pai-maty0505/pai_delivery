#!/usr/bin/env python3
"""
椅子から立ち上がる動きのお手本を IK で作る（立ち上がる方策 A の学習で、まねる目標にする）。

ロボットは +x を向き、両足は床の決まった場所に平らに着いたまま動かない。区切りの姿勢（キー）を 4 つ決め、
その間の目標（骨盤の位置・上体の傾き・重心の前後）をなめらかにつなぎ、50 Hz の各時刻で IK を解く。

  キー 0（0.0 秒）座っている     ：股 −1.5・膝 1.9・足首 −0.4（足を少し引き寄せる）。ももが座面（高さ SEAT_H）に載る
  キー 1（0.8 秒）上体を前に倒す ：骨盤はそのまま、上体（胴）を前へ LEAN だけ倒す
  キー 2（1.3 秒）お尻が離れる   ：重心を両足の真ん中の上へ運び、骨盤を 3 cm 上げる
  キー 3（2.3 秒）立つ           ：歩く方策の立ち姿勢（knees_bent）。重心は両足の上

腕は動かさない（ARMS の姿勢。座ったときに手がももに当たらないよう少し外へ開く）。

    .venv_mjx/bin/python scripts/standup_ref.py            # models/standup_ref.npz を作り、logs/standup_ref.png に描く
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import payload_env as pe  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
DT = 0.02
T_KEYS = (0.0, 0.8, 1.3, 2.3)
SEATED = dict(hip=-1.5, knee=1.9, ankle=-0.4)
LEAN = 0.75                      # 上体を前へ倒す角度（rad、約 43°）
FOOT_SOLE = 0.037                # 足の site から足の裏までの高さ（足の箱の中心 −0.029、厚さの半分 0.008）
LEG = [f"{s}_{j}_joint" for s in ("left", "right")
       for j in ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")]
WAIST = ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"]
ARMS = {"shoulder_pitch": 0.1, "shoulder_roll": 0.35, "shoulder_yaw": 0.0, "elbow": 0.9}   # 腕の姿勢（左。右は roll・yaw が逆）


def smooth(u):
    u = np.clip(u, 0.0, 1.0)
    return u * u * (3 - 2 * u)


class IK:
    def __init__(self, m):
        self.m, self.d = m, mujoco.MjData(m)
        self.adr = {m.joint(i).name: m.joint(i).qposadr[0] for i in range(1, m.njnt)}
        self.dof = {m.joint(i).name: m.joint(i).dofadr[0] for i in range(1, m.njnt)}
        self.free = list(range(6)) + [self.dof[n] for n in LEG + WAIST]     # 動かす自由度（腕は固定）
        self.feet = [m.site(f"{s}_foot").id for s in ("left", "right")]
        self.pelvis = m.body("pelvis").id
        self.torso = m.body("torso_link").id

    def forward(self, q):
        self.d.qpos[:] = q
        mujoco.mj_forward(self.m, self.d)

    def solve(self, q, foot_pos, foot_mat, pelvis=None, com_x=None, torso_pitch=None, q_nom=None, iters=80, w_pelvis=8.0):
        m, d = self.m, self.d
        q = q.copy()
        nv = m.nv
        for _ in range(iters):
            self.forward(q)
            rs, Js, ws = [], [], []
            for k, sid in enumerate(self.feet):
                jp, jr = np.zeros((3, nv)), np.zeros((3, nv))
                mujoco.mj_jacSite(m, d, jp, jr, sid)
                rs.append(foot_pos[k] - d.site_xpos[sid]); Js.append(jp); ws.append(np.full(3, 300.0))
                R = d.site_xmat[sid].reshape(3, 3)
                err = 0.5 * (np.cross(R[:, 0], foot_mat[k][:, 0]) + np.cross(R[:, 1], foot_mat[k][:, 1])
                             + np.cross(R[:, 2], foot_mat[k][:, 2]))
                rs.append(err); Js.append(jr); ws.append(np.full(3, 200.0))
            if pelvis is not None:
                jp = np.zeros((3, nv))
                mujoco.mj_jacBody(m, d, jp, None, self.pelvis)
                rs.append(pelvis - d.xpos[self.pelvis]); Js.append(jp); ws.append(np.array([0.5, 0.5, 1.0]) * w_pelvis)
            if com_x is not None:
                jc = np.zeros((3, nv))
                mujoco.mj_jacSubtreeCom(m, d, jc, self.pelvis)
                rs.append(np.array([com_x - d.subtree_com[self.pelvis][0]])); Js.append(jc[:1]); ws.append(np.array([10.0]))
            if torso_pitch is not None:
                jr = np.zeros((3, nv))
                mujoco.mj_jacBody(m, d, None, jr, self.torso)
                z = d.xmat[self.torso].reshape(3, 3)[:, 2]
                zt = np.array([np.sin(torso_pitch), 0.0, np.cos(torso_pitch)])
                rs.append(np.cross(z, zt)); Js.append(jr); ws.append(np.full(3, 6.0))
            if q_nom is not None:                               # 姿勢をなるべく目安の姿勢に近く
                for n in LEG + WAIST:
                    row = np.zeros((1, nv)); row[0, self.dof[n]] = 1.0
                    rs.append(np.array([q_nom[self.adr[n]] - q[self.adr[n]]])); Js.append(row); ws.append(np.array([0.3]))
            r = np.concatenate(rs); J = np.vstack(Js)[:, self.free]; W = np.concatenate(ws)
            JW = J.T * W
            dx = np.linalg.solve(JW @ J + 1e-3 * np.eye(len(self.free)), JW @ r)
            dv = np.zeros(nv); dv[self.free] = dx
            mujoco.mj_integratePos(m, q, dv, 1.0)
            for n in LEG + WAIST:                                # 関節の範囲
                j = m.joint(n).id
                q[self.adr[n]] = np.clip(q[self.adr[n]], *m.jnt_range[j])
            if np.linalg.norm(r * np.sqrt(W)) < 1e-4:
                break
        self.forward(q)
        return q


def seated_q(m, ik):
    """座った姿勢。足の site を (0, ±y) に、足の裏を床（z ＝ 0）に置く。"""
    q = m.keyframe("knees_bent").qpos.copy()
    q[3:7] = (1, 0, 0, 0)
    for s in ("left", "right"):
        q[ik.adr[f"{s}_hip_pitch_joint"]] = SEATED["hip"]
        q[ik.adr[f"{s}_knee_joint"]] = SEATED["knee"]
        q[ik.adr[f"{s}_ankle_pitch_joint"]] = SEATED["ankle"]
    for n in WAIST:
        q[ik.adr[n]] = 0.0
    for s, sg in (("left", 1), ("right", -1)):
        for j, v in ARMS.items():
            q[ik.adr[f"{s}_{j}_joint"]] = sg * v if j in ("shoulder_roll", "shoulder_yaw") else v
    q[2] = 1.0
    ik.forward(q)
    f = ik.d.site_xpos[ik.feet[0]]
    q[0] -= f[0]
    q[2] -= f[2] - FOOT_SOLE
    ik.forward(q)
    return q


def seat_height(m, d):
    """ももの当たり判定（カプセル）の下面の高さ ＝ 座面の高さ。"""
    zs = []
    for s in ("left", "right"):
        g = m.geom(f"{s}_thigh").id
        p, R, h, r = d.geom_xpos[g], d.geom_xmat[g].reshape(3, 3), m.geom_size[g][1], m.geom_size[g][0]
        zs += [(p + R[:, 2] * h)[2] - r, (p - R[:, 2] * h)[2] - r]
    return min(zs)


def make_reference(m):
    ik = IK(m)
    q0 = seated_q(m, ik)
    foot_pos = [ik.d.site_xpos[s].copy() for s in ik.feet]
    foot_mat = [ik.d.site_xmat[s].reshape(3, 3).copy() for s in ik.feet]
    pelvis0 = ik.d.xpos[ik.pelvis].copy()
    com0 = ik.d.subtree_com[ik.pelvis][0]
    seat = seat_height(m, ik.d)
    foot_mid = 0.5 * (foot_pos[0][0] + foot_pos[1][0]) + 0.04            # 足の裏の真ん中（箱の中心は site の 4 cm 前）
    # 立ち姿勢：knees_bent の関節で、重心を両足の真ん中の上に
    q_stand_nom = m.keyframe("knees_bent").qpos.copy()
    q_stand_nom[7 + 15:] = q0[7 + 15:]
    q_stand = ik.solve(q0, foot_pos, foot_mat, com_x=foot_mid, torso_pitch=None, q_nom=q_stand_nom, iters=200)
    # 立ち姿勢の関節をそのまま目安にし、骨盤の高さも合わせる
    pelvis3 = ik.d.xpos[ik.pelvis].copy()
    torso3 = np.arctan2(ik.d.xmat[ik.torso].reshape(3, 3)[0, 2], ik.d.xmat[ik.torso].reshape(3, 3)[2, 2])
    ik.forward(q0)
    torso0 = np.arctan2(ik.d.xmat[ik.torso].reshape(3, 3)[0, 2], ik.d.xmat[ik.torso].reshape(3, 3)[2, 2])

    n = int(round(T_KEYS[-1] / DT)) + 1
    ts = np.arange(n) * DT
    qs, info = [], []
    q = q0.copy()
    com1 = None
    for t in ts:
        if com1 is None and t > T_KEYS[1]:
            com1 = ik.d.subtree_com[ik.pelvis][0]
        if t <= T_KEYS[1]:                                      # 上体を前に倒す（骨盤は座面の上のまま）
            u = smooth(t / T_KEYS[1])
            tgt = dict(pelvis=pelvis0, torso_pitch=torso0 + u * (LEAN - torso0), q_nom=q0, w_pelvis=200.0)
        elif t <= T_KEYS[2]:                                    # 重心を足の上へ、骨盤を少し上げる
            u = smooth((t - T_KEYS[1]) / (T_KEYS[2] - T_KEYS[1]))
            tgt = dict(com_x=com1 + u * (foot_mid - com1), torso_pitch=LEAN,
                       pelvis=None, q_nom=q)
            tgt["pelvis_z"] = pelvis0[2] + 0.03 * u
        else:                                                   # 立つ
            u = smooth((t - T_KEYS[2]) / (T_KEYS[3] - T_KEYS[2]))
            tgt = dict(com_x=foot_mid, torso_pitch=LEAN + u * (torso3 - LEAN),
                       q_nom=(1 - u) * q + u * q_stand)
            tgt["pelvis_z"] = pelvis0[2] + 0.03 + u * (pelvis3[2] - pelvis0[2] - 0.03)
        pz = tgt.pop("pelvis_z", None)
        if pz is not None:
            ik.forward(q)
            p = ik.d.xpos[ik.pelvis].copy()
            p[2] = pz
            tgt["pelvis"] = p
        q = ik.solve(q, foot_pos, foot_mat, **tgt)
        qs.append(q.copy())
        info.append((t, ik.d.subtree_com[ik.pelvis][0] - foot_mid, ik.d.xpos[ik.pelvis][2], seat_height(m, ik.d) - seat,
                     max(np.linalg.norm(ik.d.site_xpos[s] - foot_pos[k]) for k, s in enumerate(ik.feet))))
    return np.array(qs), ts, dict(seat=seat, foot_pos=np.array(foot_pos), foot_mid=foot_mid, info=info,
                                  q_stand=q_stand)


def render_strip(m, qs, ts, path, times=(0.0, 0.8, 1.3, 1.8, 2.3)):
    from PIL import Image
    d = mujoco.MjData(m)
    r = mujoco.Renderer(m, 480, 400)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = (-0.1, 0.0, 0.45)
    cam.distance, cam.azimuth, cam.elevation = 2.0, 90.0, -5.0      # 横（−y 側）から
    imgs = []
    for t in times:
        i = int(np.argmin(np.abs(ts - t)))
        d.qpos[:] = qs[i]
        mujoco.mj_forward(m, d)
        r.update_scene(d, cam)
        imgs.append(r.render())
    Image.fromarray(np.hstack(imgs)).save(path)


def chair_desk(seat_h, seat_x=(-0.47, -0.10), desk_x=0.55):
    """椅子の座面と机（見た目の確認用）。足の site が x ＝ 0。"""
    def deco(spec):
        w = spec.worldbody
        x0, x1 = seat_x
        w.add_geom(name="seat", type=mujoco.mjtGeom.mjGEOM_BOX, pos=[(x0 + x1) / 2, 0, seat_h - 0.02],
                   size=[(x1 - x0) / 2, 0.22, 0.02], rgba=[0.25, 0.25, 0.28, 0.8], contype=0, conaffinity=0)
        w.add_geom(name="desk", type=mujoco.mjtGeom.mjGEOM_BOX, pos=[desk_x + 0.35, 0, 0.74 - 0.015],
                   size=[0.35, 0.6, 0.015], rgba=[0.62, 0.45, 0.30, 1], contype=0, conaffinity=0)
    return deco


def main():
    m = pe.build_model()
    qs, ts, meta = make_reference(m)
    print(f"座面の高さ {meta['seat']:.3f} m")
    d = mujoco.MjData(m)
    gap = []
    for q in qs:
        d.qpos[:] = q
        mujoco.mj_forward(m, d)
        gap.append(min(mujoco.mj_geomDistance(m, d, m.geom(f"{s}_hand_collision").id, m.geom(f"{s}_thigh").id, 1.0, None)
                       for s in ("left", "right")))
    print(f"手ともも（当たり判定）の最も近い距離 {min(gap) * 100:.1f} cm")
    for t, dcom, pz, dseat, dfoot in meta["info"][::10]:
        print(f"  {t:4.2f} 秒：重心と足の真ん中の差 {dcom:+.3f} m  骨盤の高さ {pz:.3f}  ももと座面の差 {dseat:+.3f}  足のずれ {dfoot * 1000:.1f} mm")
    out = REPO / "models" / "standup_ref.npz"
    out.parent.mkdir(exist_ok=True)
    np.savez(out, qpos=qs, t=ts, seat=meta["seat"], foot_pos=meta["foot_pos"], foot_mid=meta["foot_mid"])
    m2 = pe.build_model(decorate=chair_desk(meta["seat"]))
    render_strip(m2, qs, ts, REPO / "logs" / "standup_ref.png")
    print(out, REPO / "logs" / "standup_ref.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
