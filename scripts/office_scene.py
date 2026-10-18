#!/usr/bin/env python3
"""
事務所の場面を作り、画像にする。

- 部屋（6 × 5 m、壁の高さ 2.2 m）、東の壁に閉まったドア（幅 0.9 m、外開き、蝶番は北側）
- 西の壁ぎわにロボットの机と椅子。G1 は机に向かって椅子に座っている（立ち上がる方策 A のお手本の最初の姿勢。
  座面の高さ・机との距離も A の学習と同じ。物理では動かさない）
- ドアの外、ドアの開く範囲の外（南側）に配達員（宅配便の人。人の形のモデルで、物理では動かさない）。ノート PC のケース（40 × 30 × 6 cm、2 kg）を両手で持っている

    .venv_mjx/bin/python scripts/office_scene.py --out logs/office

結果：logs/office_top.png（真上から、名前入り）、logs/office_view.png（部屋の中を斜め上から）、
logs/office_outside.png（ドアの外から。ケースを持った配達員）
"""

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import payload_env as pe  # noqa: E402

REPO = Path(__file__).resolve().parent.parent

# 部屋（x：西 −1 〜 東 5、y：南 −2.5 〜 北 2.5）
ROOM_X, ROOM_Y, WALL_H, WALL_T = (-1.0, 5.0), (-2.5, 2.5), 2.2, 0.1
DOOR_Y, DOOR_W, DOOR_H = 1.2, 1.1, 2.0                                    # ドアの幅 1.1 m（受け取った箱を持って下がる余裕）
DESK = dict(center=(-0.62, 0.0), size=(0.35, 0.6), height=0.74)       # 机（奥行き 70・幅 120・高さ 74 cm）
# 椅子とロボットの位置は、立ち上がる方策（A）の学習と同じ関係にする：座ったロボットの足の site から机の前の縁まで 55 cm、
# 座面の高さ・前後の範囲は models/standup_ref.npz と scripts/standup_env.py の SEAT_X（足の site 基準、ロボットの後ろ側）
DESK_EDGE_X = DESK["center"][0] + DESK["size"][0]                        # 机の前の縁（東側）
FEET_X = DESK_EDGE_X + 0.55                                              # 座ったロボットの足の site（ロボットは −x を向く）
SEAT_H = float(np.load(REPO / "models" / "standup_ref.npz")["seat"])     # 座面の高さ（約 0.31 m）
SEAT_X = (FEET_X + 0.10, FEET_X + 0.47)                                  # 座面の前後の範囲
CHAIR = dict(center=((SEAT_X[0] + SEAT_X[1]) / 2, 0.0), seat=SEAT_H, half=(SEAT_X[1] - SEAT_X[0]) / 2)
DOOR_DAMPING, DOOR_FRICTION = 3.0, 0.5                                    # ドアの蝶番のダンピングと摩擦（N·m）
CORRIDOR = (5.1, 7.5, -3.0, 5.0)                                         # 廊下の床の範囲（x0, x1, y0, y1）。ドアを閉める所を写すカメラが北の壁の内側に入るよう、北へ長め
DOOR_HINGE = (5.0 + 0.1, DOOR_Y + DOOR_W / 2)                            # ドアの蝶番（北側）。外開き（廊下側へ押して開ける）
COURIER = (5.75, 0.25)                                                   # 配達員（ドアの外、ドアの開く範囲の外の南側）
COURIER_YAW = np.radians(124.0)                                          # ドアの方（北西）を向く
CASE = (0.20, 0.15, 0.03)                                                # ケースの半分の大きさ

WOOD = [0.62, 0.45, 0.30, 1]
WALL = [0.92, 0.91, 0.88, 1]
DOOR = [0.55, 0.38, 0.24, 1]
METAL = [0.35, 0.35, 0.38, 1]
UNIFORM = [0.15, 0.32, 0.62, 1]
SKIN = [0.87, 0.72, 0.58, 1]
CASE_C = [0.12, 0.12, 0.14, 1]


def box(body, pos, half, rgba, name=None, group=0):
    g = body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, pos=list(pos), size=list(half), rgba=rgba, contype=0, conaffinity=0,
                      group=group)
    if name:
        g.name = name
    return g


def decorate(spec, hinged_door=False):
    """事務所の場面を足す。hinged_door：ドアを蝶番（北側、鉛直の軸）で回る物理のドアにする（外開き、0〜100°）。"""
    w = spec.worldbody
    x0, x1 = ROOM_X
    y0, y1 = ROOM_Y
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    # 床（板張り）
    box(w, (cx, cy, -0.004), ((x1 - x0) / 2, (y1 - y0) / 2, 0.005), [0.78, 0.70, 0.58, 1])
    # 廊下（ドアの外）：部屋と同じ板張りの床と白い壁（2026-10-18。前は物理の床の灰色のタイルがむき出しで、壁もなかった）
    gx0, gx1, gy0, gy1 = CORRIDOR
    box(w, ((gx0 + gx1) / 2, (gy0 + gy1) / 2, -0.004), ((gx1 - gx0) / 2, (gy1 - gy0) / 2, 0.005), [0.78, 0.70, 0.58, 1])
    box(w, (gx1 + WALL_T / 2, (gy0 + gy1) / 2, WALL_H / 2), (WALL_T / 2, (gy1 - gy0) / 2 + WALL_T, WALL_H / 2), WALL)
    box(w, ((gx0 + gx1) / 2, gy1 + WALL_T / 2, WALL_H / 2), ((gx1 - gx0) / 2, WALL_T / 2, WALL_H / 2), WALL)
    box(w, ((gx0 + gx1) / 2, gy0 - WALL_T / 2, WALL_H / 2), ((gx1 - gx0) / 2, WALL_T / 2, WALL_H / 2), WALL)
    # 壁（北・南・西、東はドアの所を空ける）
    box(w, (cx, y1 + WALL_T / 2, WALL_H / 2), ((x1 - x0) / 2 + WALL_T, WALL_T / 2, WALL_H / 2), WALL)
    box(w, (cx, y0 - WALL_T / 2, WALL_H / 2), ((x1 - x0) / 2 + WALL_T, WALL_T / 2, WALL_H / 2), WALL)
    box(w, (x0 - WALL_T / 2, cy, WALL_H / 2), (WALL_T / 2, (y1 - y0) / 2, WALL_H / 2), WALL, group=4)   # 西の壁（斜めの図では描かない）
    d0, d1 = DOOR_Y - DOOR_W / 2, DOOR_Y + DOOR_W / 2
    box(w, (x1 + WALL_T / 2, (y0 + d0) / 2, WALL_H / 2), (WALL_T / 2, (d0 - y0) / 2, WALL_H / 2), WALL)
    box(w, (x1 + WALL_T / 2, (d1 + y1) / 2, WALL_H / 2), (WALL_T / 2, (y1 - d1) / 2, WALL_H / 2), WALL)
    box(w, (x1 + WALL_T / 2, DOOR_Y, (DOOR_H + WALL_H) / 2), (WALL_T / 2, DOOR_W / 2, (WALL_H - DOOR_H) / 2), WALL)
    # ドア（閉まっている）と取っ手（内側）
    if hinged_door:
        db = w.add_body(name="door_body", pos=[x1 + WALL_T / 2, d1 - 0.005, 0])
        j = db.add_joint(name="door_hinge", type=mujoco.mjtJoint.mjJNT_HINGE, axis=[0, 0, 1], range=[0.0, 1.75])
        j.limited = mujoco.mjtLimited.mjLIMITED_TRUE
        # 蝶番のダンピング（3）と摩擦（0.5 N·m）は、compile の後に model に入れる（DOOR_DAMPING・DOOR_FRICTION）
        g = db.add_geom(name="door", type=mujoco.mjtGeom.mjGEOM_BOX, pos=[0, -(DOOR_W / 2 - 0.005), DOOR_H / 2],
                        size=[0.02, DOOR_W / 2 - 0.01, DOOR_H / 2], rgba=DOOR, contype=0, conaffinity=0, mass=12.0)
        box(db, (-0.06, -(DOOR_W - 0.10), 1.0), (0.03, 0.06, 0.012), METAL, "door_handle")
    else:
        box(w, (x1 + WALL_T / 2, DOOR_Y, DOOR_H / 2), (0.02, DOOR_W / 2 - 0.01, DOOR_H / 2), DOOR, "door")
        box(w, (x1 + WALL_T / 2 - 0.06, d0 + 0.10, 1.0), (0.03, 0.06, 0.012), METAL, "door_handle")
    # 机（天板と 4 本の脚）
    (dx, dy), (hx, hy), dh = DESK["center"], DESK["size"], DESK["height"]
    box(w, (dx, dy, dh - 0.015), (hx, hy, 0.015), WOOD, "desk")
    for sx in (-1, 1):
        for sy in (-1, 1):
            box(w, (dx + sx * (hx - 0.04), dy + sy * (hy - 0.04), (dh - 0.03) / 2), (0.025, 0.025, (dh - 0.03) / 2), METAL)
    # 机の上の物（モニター）
    box(w, (dx - 0.18, dy + 0.25, dh + 0.22), (0.02, 0.27, 0.16), [0.1, 0.1, 0.1, 1])
    box(w, (dx - 0.18, dy + 0.25, dh + 0.04), (0.06, 0.08, 0.04), METAL)
    # 椅子（座面・背もたれ・脚）
    (chx, chy), seat = CHAIR["center"], CHAIR["seat"]
    hx = CHAIR["half"]
    box(w, (chx, chy, seat - 0.03), (hx, 0.22, 0.03), [0.2, 0.2, 0.22, 1], "chair")
    box(w, (chx + hx + 0.02, chy, seat + 0.28), (0.025, 0.21, 0.28), [0.2, 0.2, 0.22, 1])
    box(w, (chx, chy, (seat - 0.06) / 2), (0.025, 0.025, (seat - 0.06) / 2), METAL)
    # 椅子の脚の台（床の上の十字の棒）：中心から 18 cm（前は 25 cm で、立ち上がって歩き出す右足が台の棒に当たった。すべての物体に
    # 当たり判定を入れた 2026-10-18 から）
    box(w, (chx, chy, 0.02), (0.18, 0.03, 0.02), METAL)
    box(w, (chx, chy, 0.02), (0.03, 0.18, 0.02), METAL)
    # ほかの机（事務所らしく）と棚
    for ox, oy in ((1.9, -1.6), (3.2, -1.6)):
        box(w, (ox, oy, 0.725), (0.6, 0.35, 0.015), WOOD)
        for sx in (-1, 1):
            for sy in (-1, 1):
                box(w, (ox + sx * 0.55, oy + sy * 0.3, 0.355), (0.025, 0.025, 0.355), METAL)
    box(w, (2.4, 2.25, 0.9), (0.8, 0.2, 0.9), [0.85, 0.85, 0.82, 1], "shelf")
    # 配達員（宅配便の人。人の形のモデルで、物理では動かさない）とケース
    px, py = COURIER
    c = w.add_body(name="courier", pos=[px, py, 0], quat=[np.cos(COURIER_YAW / 2), 0, 0, np.sin(COURIER_YAW / 2)])   # 体の前 ＝ 体の +x

    def cap(a, b, r, rgba):
        c.add_geom(type=mujoco.mjtGeom.mjGEOM_CAPSULE, fromto=list(a) + list(b), size=[r, 0, 0], rgba=rgba,
                   contype=0, conaffinity=0)

    def ell(pos, size, rgba, quat=(1, 0, 0, 0)):
        c.add_geom(type=mujoco.mjtGeom.mjGEOM_ELLIPSOID, pos=list(pos), size=list(size), quat=list(quat), rgba=rgba,
                   contype=0, conaffinity=0)

    SHIRT, PANTS, SHOE, CAP_C, HAIR = [0.20, 0.45, 0.32, 1], [0.13, 0.15, 0.22, 1], [0.08, 0.08, 0.08, 1], \
        [0.20, 0.45, 0.32, 1], [0.10, 0.07, 0.05, 1]
    for sy in (-0.095, 0.095):
        box(c, (0.04, sy, 0.035), (0.13, 0.05, 0.035), SHOE)                       # 靴
        cap((0, sy, 0.10), (0.0, sy, 0.50), 0.062, PANTS)                          # すね
        cap((0, sy, 0.50), (0.0, sy * 0.95, 0.90), 0.078, PANTS)                   # もも
    ell((0, 0, 0.95), (0.12, 0.18, 0.11), PANTS)                                   # 腰
    ell((0, 0, 1.22), (0.13, 0.20, 0.25), SHIRT)                                   # 胴（制服）
    box(c, (0.105, -0.06, 1.30), (0.02, 0.05, 0.06), [0.9, 0.9, 0.85, 1])           # 胸の名札
    cap((0, 0, 1.42), (0, 0, 1.52), 0.05, SKIN)                                    # 首
    ell((0.01, 0, 1.62), (0.095, 0.085, 0.115), SKIN)                              # 頭
    ell((-0.015, 0, 1.66), (0.09, 0.087, 0.09), HAIR)                              # 髪
    for sy in (-0.035, 0.035):
        c.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, pos=[0.093, sy, 1.635], size=[0.011, 0, 0],
                   rgba=[0.05, 0.05, 0.05, 1], contype=0, conaffinity=0)          # 目
    c.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER, pos=[0.0, 0, 1.715], size=[0.098, 0.035, 0], rgba=CAP_C,
               contype=0, conaffinity=0)                                          # 帽子
    box(c, (0.12, 0, 1.69), (0.06, 0.08, 0.006), CAP_C)                            # 帽子のつば
    # 腕：肩から下へ、肘で前へ曲げ、手でケースを下から持つ
    hand_z = 1.08
    for sy in (-1, 1):
        sh = (0.0, sy * 0.21, 1.40)
        el = (0.02, sy * 0.23, 1.13)
        wr = (0.30, sy * 0.17, hand_z)
        cap(sh, el, 0.048, SHIRT)                                                  # 上腕
        cap(el, wr, 0.042, SHIRT)                                                  # 前腕（長袖）
        ell((wr[0] + 0.05, wr[1], wr[2] - 0.005), (0.06, 0.035, 0.022), SKIN)     # 手（手のひらを上に）
    box(c, (0.36, 0, hand_z + 0.02 + CASE[2]), (CASE[1], CASE[0], CASE[2]), CASE_C, "case")   # ケース（体の前）
    box(c, (0.36 + CASE[1] + 0.005, 0, hand_z + 0.02 + CASE[2]), (0.006, 0.07, 0.012), METAL)  # ケースの取っ手

    # 明かり
    for lx, ly in ((0.5, 0.0), (3.5, 0.0)):
        L = w.add_light(pos=[lx, ly, 3.0], dir=[0, 0, -1])
        L.diffuse = [0.28, 0.28, 0.28]
        L.castshadow = True
    # 真上から見るカメラ（正射影）
    cam = w.add_camera(name="top", pos=[cx + 0.3, cy, 8.0], quat=[1, 0, 0, 0])
    cam.proj = mujoco.mjtProjection.mjPROJ_ORTHOGRAPHIC
    cam.fovy = 6.2


def seated_qpos(m):
    """椅子に座った姿勢：立ち上がる方策（A）のお手本の最初の姿勢を、机の方（−x）を向けて FEET_X に置く。"""
    q = np.load(REPO / "models" / "standup_ref.npz")["qpos"][0].copy()
    x, y = q[0], q[1]
    q[0], q[1] = FEET_X - x, -y                                  # z 軸まわりに 180° 回す
    qw, qx, qy, qz = q[3:7]
    q[3:7] = (-qz, -qy, qx, qw)                                  # (0, 0, 0, 1) を左から掛ける
    return q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO / "logs" / "office"))
    args = ap.parse_args()
    m = pe.build_model(decorate=decorate)
    m.vis.global_.offwidth, m.vis.global_.offheight = 1600, 1200
    m.vis.headlight.ambient[:] = 0.35
    m.vis.headlight.diffuse[:] = 0.45
    d = mujoco.MjData(m)
    d.qpos[:] = seated_qpos(m)
    mujoco.mj_forward(m, d)
    print("骨盤の高さ", round(float(d.qpos[2]), 3), "股の位置", np.round(d.xpos[m.body("left_hip_pitch_link").id], 3))

    font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
    font = ImageFont.truetype(font_path, 30, index=0) if os.path.exists(font_path) else ImageFont.load_default()

    # 真上から（正射影）。画像の座標 ＝ 世界の座標の 1 次式
    W, H = 1600, 1200
    r = mujoco.Renderer(m, H, W)
    top_opt = mujoco.MjvOption()
    top_opt.geomgroup[4] = 1
    r.update_scene(d, camera="top", scene_option=top_opt)
    img = Image.fromarray(r.render())
    cam = m.camera("top")
    cx, cy = cam.pos[0], cam.pos[1]
    scale = H / m.cam_fovy[cam.id]                       # 1 m あたりの画素

    def px(x, y):
        return W / 2 + (x - cx) * scale, H / 2 - (y - cy) * scale

    dr = ImageDraw.Draw(img)

    def label(x, y, text, dx=0, dy=0):
        u, v = px(x, y)
        u, v = u + dx, v + dy
        bb = dr.textbbox((u, v), text, font=font, anchor="mm")
        dr.rectangle((bb[0] - 6, bb[1] - 4, bb[2] + 6, bb[3] + 4), fill=(255, 255, 255, 230), outline=(60, 60, 60))
        dr.text((u, v), text, fill=(20, 20, 20), font=font, anchor="mm")

    label(DESK["center"][0], DESK["center"][1] - 0.85, "ロボットの机")
    label(CHAIR["center"][0] + 0.35, CHAIR["center"][1] - 0.6, "G1（椅子に座っている）")
    label(ROOM_X[1] - 0.95, DOOR_Y + 0.75, "ドア（閉まっている、外開き）")
    label(COURIER[0] + 0.05, COURIER[1] - 0.55, "配達員")
    label(COURIER[0] + 0.05, COURIER[1] - 1.0, "ケース 2 kg")
    # ドアの開く範囲（点線の円弧）
    hu, hv = px(*DOOR_HINGE)
    rr = DOOR_W * scale
    for a0 in range(0, 90, 6):
        dr.arc((hu - rr, hv - rr, hu + rr, hv + rr), start=a0, end=a0 + 3, fill=(90, 90, 90), width=3)
    label(2.55, -1.6, "ほかの机")
    label(2.4, 2.25, "棚")
    # 歩く道すじ（矢印）
    path = [(0.4, 0.0), (1.2, 0.6), (3.5, DOOR_Y), (ROOM_X[1] - 0.5, DOOR_Y)]
    pts = [px(*p) for p in path]
    dr.line(pts, fill=(200, 40, 40), width=6)
    ex, ey = pts[-1]
    dr.polygon([(ex + 18, ey), (ex - 8, ey - 14), (ex - 8, ey + 14)], fill=(200, 40, 40))
    label(2.3, 1.75 - 0.75, "① ノックを聞いてドアへ  ② 受け取って机へ戻る")
    # 1 m の目盛り
    u0, v0 = px(-0.8, -2.25)
    dr.line([(u0, v0), (u0 + scale, v0)], fill=(20, 20, 20), width=5)
    dr.text((u0 + scale / 2, v0 - 22), "1 m", fill=(20, 20, 20), font=font, anchor="mm")
    img.save(f"{args.out}_top.png")

    # 部屋の中を斜め上から
    r2 = mujoco.Renderer(m, 900, 1600)
    vc = mujoco.MjvCamera()
    vc.lookat[:] = (1.9, 0.2, 0.3)
    vc.distance, vc.azimuth, vc.elevation = 7.2, 15.0, -38.0   # 机の側（西）の上から、ドア（東）の方を見る。西の壁は描かない
    r2.update_scene(d, vc)
    Image.fromarray(r2.render()).save(f"{args.out}_view.png")
    # ドアの外から（廊下側から、ドアの前に立つ配達員）
    vc2 = mujoco.MjvCamera()
    vc2.lookat[:] = (5.55, 0.65, 1.0)
    vc2.distance, vc2.azimuth, vc2.elevation = 3.0, 228.0, -8.0    # ドアの外の北東から（配達員を斜め前から、ドアも写す）
    r2.update_scene(d, vc2)
    Image.fromarray(r2.render()).save(f"{args.out}_outside.png")
    print(f"{args.out}_top.png", f"{args.out}_view.png", f"{args.out}_outside.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
