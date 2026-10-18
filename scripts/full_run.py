#!/usr/bin/env python3
"""
事務所の場面で、A → B → C → D を一つのシミュレーションで続けて動かす（CPU の MuJoCo、方策の計算も CPU の JAX。声はまだ入れない）。

  A：机の前の椅子に座っている → ノックの合図 → 立ち上がる方策で立つ → 歩く方策（estimate、腕を振る）で左回りに北へ出て、
     東を向いてドアまで歩く → 両手を前に出してドアを押し開ける → 少し下がって戸口の中で止まる
  B：箱を持って歩く方策（--walk、③ の持ち方）に切り替え、腕を運ぶ姿勢にして止まって待つ。ドアの外の配達員の手（左右の手で
     箱の側面を挟む）が、箱をロボットの手の上まで差し出し、そっと下げて離し、手を引く
  C：西を向いて部屋を横切り、自分の机の北側へ回って南を向き、少しずつ遅くして机の前で完全に止まる
  D：（--walk が体を沈める練習をした方策なら）体を沈め、箱を前へ出して 15° 傾け、奥の縁と指先から机に当て、手と腕を左右へ
     抜き、体を起こす。そうでなければ腕だけで置く
持ち方 ③、荷物は MacBook Air の箱（33 × 25 × 5 cm、1.23 kg）。

    JAX_PLATFORMS=cpu .venv_mjx/bin/python scripts/full_run.py --walk carry10 --video logs/full_run.mp4
"""
import argparse
import json
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
if os.environ.get("JAX_PLATFORMS") == "cpu":
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import jax  # noqa: E402
import jax.numpy as jp  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import carry_env as ce  # noqa: E402
import deploy as dp  # noqa: E402
import office_scene as office  # noqa: E402
import payload_env as pe  # noqa: E402
import place_test as pt  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
DT, N_SUB, N_ACT = dp.DT, dp.N_SUB, dp.N_ACT
ce.GRIP = "tray"
pe.CARRY_LEFT.update(ce.TRAY_GRIP["arm"])
# 配達員の手（事務所の場面だけ）：箱の縁の下にかける指を内側へ 6 cm、左右から挟む押し込みを 2 mm。部屋の外から箱を差し出すと、
# 学習の環境の手（指 3 cm、押し込み 0.5 mm）では箱が片方へ寄って指から外れ、落ちた
# 配達員（2026-10-16 19:30 の B）：箱を手のひらで下から支え、もう片方の手を箱の横に添えて、ゆっくり差し出す。
# 左右の手で強く挟むと、手を下げても箱が一緒に下りず、引っかかって傾いて落ちた。縁の下にかける指はなくす（ロボットの手と重なる）
ce.LEDGES = ()
ce.GRIP_SQUEEZE = 0.0
DESK_X, DESK_Y = office.DESK["center"]
DESK_HX, DESK_HY = office.DESK["size"]
# 置く所（2026-10-17）：机の正面（東の縁、椅子のある側）で、椅子の右隣（椅子に座って西を向いた人の右 ＝ 北）。椅子の座面は
# y = ±0.22、机の北東の脚は y = 0.56 にあるので、その間の y = 0.36 に西を向いて立つ（腰を沈めたときの膝が脚に当たらない）。
# 箱は机の手前寄りに置く（モニターは机の奥 x = −0.80）
PLACE_Y = 0.36
STOP_EDGE = 0.11                             # 机の正面の縁から 11 cm 東で、西を向いて止まる（学習の 13 ± 3 cm の範囲。20 cm 沈めると
#                                              胴（骨盤から前へ 9.5 cm）が机の天板の高さまで下がるので、それより離れて立つ）
STOP = np.array([DESK_X + DESK_HX + STOP_EDGE, PLACE_Y])
STOP_YAW = math.pi
# 置くあいだに体が左回りに回る量（机の前に立って置く試験：carry10 は 8 回で約 4.3°、carry11 は 20 回で約 5.2°）。机に向く角度を
# 先にその分だけ右へずらしておく
PLACE_YAW_BIAS_DEG = {"carry10": 4.3, "carry11": 5.2}
# 0.4 m/秒から 1 秒かけて止めるときに進む距離（事務所の場面で測った：carry10 は約 0.12 m、carry11 は約 0.20 m）
BRAKE_M = {"carry10": 0.12, "carry11": 0.20}
PLACE_YAW_BIAS = math.radians(4.3)
# 戸口で配達員の方を向く向き（南東へ 40°）。開いたドアは北東にあり、配達員は南から来るので、受け渡す所がドアから離れる
FACE_YAW = math.radians(-40.0)
# 置くとき手先を下げる角度。指先は箱の奥の縁より約 3 cm 前へ出ているので、まず指先が机に触れ、箱の奥の縁はその後に着く。
# 18° では箱が手のひらの上で前へ 13 cm 滑ったので、12°
PLACE_TILT_DEG = 14.0
# 股と腰のひねり（左右の hip yaw と waist yaw、方策が動かす 15 の関節の中の番号）。置くあいだに体が足の上で左へ 10〜24° ねじれ、前腕が
# 箱を引きずったので、置くあいだはこの 3 つを止める（--lock_yaw）
YAW_JOINTS = (2, 8, 12)
# ドアへ向かう道（y）。戸口（y = 0.65〜1.75）の中心 1.2 より少し南（ドアの先端の近くを押して大きく開ける）。下がるときに箱が
# 戸口の南の壁から離れるよう、1.0 から 1.1 へ
DOOR_LANE = 1.1
# 配達員が待つ位置：ロボットが戸口で向きを変えるあいだ、ロボットの手が配達員の持つ箱に届かない所
COURIER_WAIT = (5.95, 0.05)
SKIN = [0.87, 0.72, 0.58, 1]
# 配達員の腕と脚（office_scene の人の形と同じ寸法）：肩 (0, ±0.21, 1.40)、上腕 0.27 m・前腕 0.29 m、手首から手の中心 0.05 m。
# 股関節 (0, ±0.095, 0.90)、もも 0.40 m・すね 0.40 m、足首から靴の中心 (0.04, 0, −0.065)
ARM_L, LEG_L = (0.27, 0.29), (0.40, 0.40)
SHOULDER, HIP = (0.0, 0.21, 1.40), (0.0, 0.095, 0.90)


LABEL = True                                 # 箱の上の面に「MAC 13」を印刷する（--no_label で印刷しない）


def case_label_files():
    """箱の 6 つの面の画像（右・左・上・下・前・後の順。MuJoCo の立方体の模様は、箱の上の面に 5 番目の画像を貼る：画像の右が
    箱の +x（運ぶときのロボットの前）、画像の上が +y）。上の面に「MAC 13」。文字の上が +x、文字の右が −y なので、机の正面
    （椅子の側）から見て読める向きになる。文字は箱の長い辺（33 cm）に沿う。"""
    from PIL import Image, ImageDraw, ImageFont
    out = REPO / "models" / "case_label"
    out.mkdir(exist_ok=True)
    font = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    plain = Image.new("RGB", (1024, 1024), (255, 255, 255))   # 6 つの画像は同じ大きさの正方形
    names = ["right", "left", "up", "down", "front", "back"]
    for n in names:
        plain.save(out / f"{n}.png")
    hx, hy, _ = ce.CASE_HALF
    W, H = int(1000 * hy / hx), 1000                       # 文字を横に書く画像（横 ＝ 箱の y、縦 ＝ 箱の x）
    im = Image.new("RGB", (W, H), (255, 255, 255))
    # 箱の上の面いっぱいに 2 行（「MAC」と「13」）。それぞれの行が面の幅の 92%・高さの 46% に収まる最大の大きさ
    dr = ImageDraw.Draw(im)
    for k, line in enumerate(("MAC", "13")):
        size = H
        while True:
            f = ImageFont.truetype(font, size)
            x0, y0, x1, y1 = dr.textbbox((0, 0), line, font=f)
            if (x1 - x0) <= 0.92 * W and (y1 - y0) <= 0.46 * H:
                break
            size -= 4
        dr.text((W // 2, int(H * (0.27 + 0.48 * k))), line, fill=(35, 35, 38), font=f, anchor="mm")
    # 時計回りに 90°：文字の上が画像の右（箱の +x）。立方体の模様の画像は正方形でないといけないので、正方形に縮めて保存する
    # （箱の面に貼ると、面の縦横の比（25 × 33 cm）に戻る）
    im.rotate(-90, expand=True).resize((1024, 1024), Image.LANCZOS).save(out / "front.png")
    return [str(out / f"{n}.png") for n in names]


def full_decorate(spec):
    """事務所（ドアは蝶番で回る）＋骨盤・胴の当たり判定＋箱・手の当たり判定・配達員の手（carry_env）＋箱・手と机の当たり判定。"""
    office.decorate(spec, hinged_door=True)
    spec.body("pelvis").add_geom(name="pelvis_col", type=mujoco.mjtGeom.mjGEOM_BOX, pos=[-0.02, 0, -0.07],
                                 size=[0.07, 0.11, 0.04], contype=0, conaffinity=0, group=3)
    # 胴の当たり判定：胴のメッシュの前の端（骨盤から前へ 7〜9 cm）に合わせる（前は半径 0.11 で前の端が 13 cm と太すぎ、腰を
    # 20 cm 沈めると机の縁に当たると見なして、机から離れて立たせる必要があった）
    spec.body("torso_link").add_geom(name="torso_col", type=mujoco.mjtGeom.mjGEOM_CAPSULE, fromto=[-0.005, 0, 0.05, -0.005, 0, 0.40],
                                     size=[0.10, 0, 0], contype=0, conaffinity=0, group=3)
    # 手と机の当たり判定は手のメッシュで（手の大きな当たり判定（半径 5 cm）では、置くときに手が机の上まで下りない）
    pairs = [("pelvis_col", "chair"), ("left_thigh", "chair"), ("right_thigh", "chair"), ("left_shin", "chair"),
             ("right_shin", "chair"), ("torso_col", "desk"),
             ("left_hand_collision", "door"), ("right_hand_collision", "door"), ("torso_col", "door")]
    for a, b in pairs:
        p = spec.add_pair(geomname1=a, geomname2=b)
        p.condim = 3
        p.friction = [0.8, 0.8, 0.005, 0.0001, 0.0001]
    ce.decorate(spec)
    ce.limit_hand_hull(spec)                                 # 手のメッシュの当たり判定の形を学習（carry22 から）と同じに
    if LABEL:                                                # 箱の上の面に「MAC 13」
        t = spec.add_texture(name="case_label", type=mujoco.mjtTexture.mjTEXTURE_CUBE)
        t.cubefiles = case_label_files()
        mat = spec.add_material(name="case_label")
        mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "case_label"
        spec.geom("case_geom").material = "case_label"
    # ぶつかったら動く・落ちる（2026-10-17）：箱とロボットの体（骨盤・胴・手・もも・すね）に、部屋の動かない物（壁・ドア・机・
    # 机の脚・椅子・モニター・棚・ほかの机）との当たり判定を足す。これまでは箱が壁やドアを通り抜けても、何も起きなかった
    scene = []
    for b in (spec.worldbody, spec.body("door_body")):
        for k, g in enumerate(b.geoms):
            if g.type == mujoco.mjtGeom.mjGEOM_BOX and g.size[2] >= 0.01 and g.contype == 0:
                if not g.name:
                    g.name = f"scene_{b.name or 'world'}_{k}"
                scene.append(g.name)
    have = {tuple(sorted(x)) for x in pairs} | {("case_geom", "desk")}
    robot = ["case_geom", "pelvis_col", "torso_col", "left_hand_collision", "right_hand_collision",
             "left_thigh", "right_thigh", "left_shin", "right_shin"]
    for a in robot:
        for b in scene:
            if tuple(sorted((a, b))) in have:
                continue
            if a.endswith("hand_collision") and b == "desk":     # 手と机は手のメッシュで（手の大きな球では、置くとき手が机の上まで下りない）
                continue
            p = spec.add_pair(geomname1=a, geomname2=b)
            p.condim = 3
            p.friction = [0.6, 0.6, 0.005, 0.0001, 0.0001]
    # 配達員の手は位置だけを決めて動かす物体（モーキャップ）で、計算上の速さがないので、摩擦では箱を動かせない（摩擦は止まった
    # 面として働き、手だけが箱から抜けた）。動かしたい向きには箱の面を押す手を当てる：下は手のひら（重さで下りる）、ロボットの方は
    # 箱の配達員の側の面の真ん中を押す指、横は横に添えた手。片側だけを押すと箱が回るので、前へ押す指は真ん中に 1 つ
    hx, hy, hz = ce.CASE_HALF
    sp = spec.body("support")
    g = sp.add_geom(name="courier_push", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.006, 0.04, 0.015],
                    pos=[hx + 0.006 + 0.001, 0.0, 0.01 + hz], rgba=SKIN, contype=0, conaffinity=0)
    p = spec.add_pair(geomname1="case_geom", geomname2="courier_push")
    p.condim = 3
    p.friction = [0.2, 0.2, 0.005, 0.0001, 0.0001]
    # 配達員の体を動かせるようにし（歩み寄る）、腕を肩・肘・手の位置に合わせて描く（上腕・前腕・手。見た目だけ、当たり判定なし）
    spec.body("courier").mocap = True
    SHIRT = [0.20, 0.45, 0.32, 1]
    PANTS, SHOE = [0.13, 0.15, 0.22, 1], [0.08, 0.08, 0.08, 1]
    for side in ("l", "r"):
        for part, r_, hl, rgba in (("ua", 0.048, ARM_L[0] / 2, SHIRT), ("fa", 0.042, ARM_L[1] / 2, SHIRT),
                                   ("th", 0.078, LEG_L[0] / 2, PANTS), ("sh", 0.062, LEG_L[1] / 2, PANTS)):
            b = spec.worldbody.add_body(name=f"courier_{part}_{side}", mocap=True, pos=[0, 0, -5])
            b.add_geom(type=mujoco.mjtGeom.mjGEOM_CAPSULE, size=[r_, hl, 0], rgba=rgba, contype=0, conaffinity=0)
        b = spec.worldbody.add_body(name=f"courier_hand_{side}", mocap=True, pos=[0, 0, -5])
        b.add_geom(type=mujoco.mjtGeom.mjGEOM_ELLIPSOID, size=[0.06, 0.035, 0.022], rgba=SKIN, contype=0, conaffinity=0)
        b = spec.worldbody.add_body(name=f"courier_shoe_{side}", mocap=True, pos=[0, 0, -5])
        b.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.13, 0.05, 0.035], rgba=SHOE, contype=0, conaffinity=0)
    # 配達員がドアを閉める手：ドアの外側の面を押して閉める（押す力だけで動かす）
    dh = spec.worldbody.add_body(name="courier_door_hand", mocap=True, pos=[6.0, 0.5, -3.0])
    dh.add_geom(name="courier_door_hand_geom", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.02, 0.05, 0.06], rgba=SKIN,
                contype=0, conaffinity=0)
    p = spec.add_pair(geomname1="courier_door_hand_geom", geomname2="door")
    p.condim = 1
    hand = [g.name for s in pe.SIDES for b in ("wrist_yaw_link", "wrist_pitch_link", "wrist_roll_link")
            for g in spec.body(f"{s}_{b}").geoms if g.name.endswith(tuple(f"mesh{k}" for k in range(9)))]
    for a in ["case_geom"] + hand:
        p = spec.add_pair(geomname1=a, geomname2="desk")
        p.condim = 3
        p.friction = [0.45, 0.45, 0.005, 0.0001, 0.0001]
    if ALL_CONTACTS:
        add_all_contacts(spec)


ALL_CONTACTS = True        # すべての物体どうしに接触・衝突・摩擦を入れる（2026-10-18）。--no_all_contacts で前の組み合わせだけ
NEW_PAIRS = []             # add_all_contacts で足した組（実際に当たったかを数えるため）


def add_all_contacts(spec):
    """すべての物体どうしに接触・衝突・摩擦を入れる：ロボットのすべての部分（当たり判定のある所に加え、腕・頭・腰・膝などは見た目の
    形（メッシュ）をそのまま当たり判定に使う）と、床・部屋と廊下の物・ドア・配達員の体・箱。配達員の手は、物理の手（下の手のひら・
    横の手・押す指・ドアを閉める手）がすでにあるので、見た目の手は入れない。ぶつからない前提で、物は固定のまま。"""
    have = set()
    for p in spec.pairs:
        have.add(tuple(sorted((p.geomname1, p.geomname2))))
    # ロボット：骨盤から下のすべての体
    robot, stack = [], [spec.body("pelvis")]
    while stack:
        b = stack.pop()
        meshes = set()
        for k, g in enumerate(b.geoms):
            if g.type == mujoco.mjtGeom.mjGEOM_MESH:
                if g.meshname in meshes:                    # 同じ形が 2 つ重ねてある体は 1 つだけ
                    continue
                meshes.add(g.meshname)
                if not g.name:
                    g.name = f"{b.name}_vis{k}"
                robot.append(g.name)
            elif g.name in ("pelvis_col", "torso_col", "left_thigh", "right_thigh", "left_shin", "right_shin",
                            "left_foot", "right_foot", "left_hand_collision", "right_hand_collision"):
                robot.append(g.name)
        stack += list(b.bodies)
    # 動かない物：床・部屋と廊下の箱（壁・ドア・机・椅子・棚など）
    static = ["floor"]
    for bd in (spec.worldbody, spec.body("door_body")):
        for k, g in enumerate(bd.geoms):
            if g.type == mujoco.mjtGeom.mjGEOM_BOX and g.contype == 0 and bd.name != "courier":
                if not g.name:
                    g.name = f"scene_{bd.name or 'world'}_{k}"
                if g.size[2] >= 0.01:                       # 部屋と廊下の板張りの床（厚さ 1 cm の見た目の板）は除く。物理の床は floor
                    static.append(g.name)
    # 配達員の体・脚・腕（見た目の手と、動く腕に置き換えて消す止まった腕・脚・持っていた古いケースは除く）
    courier = []
    hidden_caps = (0.048, 0.042, 0.062, 0.078)
    for bd_name in ["courier"] + [f"courier_{p_}_{s_}" for p_ in ("ua", "fa", "th", "sh", "shoe") for s_ in ("l", "r")]:
        bd = spec.body(bd_name)
        for k, g in enumerate(bd.geoms):
            if bd_name == "courier":
                if g.name == "case" or (g.type == mujoco.mjtGeom.mjGEOM_CAPSULE and round(float(g.size[0]), 3) in hidden_caps) \
                        or (g.type == mujoco.mjtGeom.mjGEOM_ELLIPSOID and abs(g.size[0] - 0.06) < 1e-4) \
                        or (g.type == mujoco.mjtGeom.mjGEOM_BOX and (np.allclose(g.size, [0.13, 0.05, 0.035]) or np.allclose(g.size, [0.006, 0.07, 0.012]))):
                    continue
            if not g.name:
                g.name = f"{bd_name}_g{k}"
            courier.append(g.name)
    # 接触する組（2026-10-18 の仕様）：人がドアを手で開けても足はドアに当たらない、荷物を受け取るとき荷物は受け取っても手どうしは
    # 触れない、と同じ考え方で、その場面で触れるはずの所どうしだけに当たり判定を入れる。ぶつかるはずのない所どうし（見た目の形が
    # 少し重なっている所）に入れると、ありもしない接触が起きた（座った骨盤の形と椅子、足首の形と床、配達員の前腕とドア）
    arm_parts = ("shoulder", "elbow", "wrist", "hand")
    is_arm = lambda g: any(k in g for k in arm_parts)                      # noqa: E731
    is_ankle = lambda g: "ankle" in g or g.endswith(("_foot", "_shin"))   # noqa: E731  足首・すね（しゃがんでも床に着かない）
    furniture = [b for b in static if b not in ("floor", "door", "door_handle", "chair")]
    cour_body = [g for g in courier if not g.startswith(("courier_fa_", "courier_ua_"))]   # 配達員の体・脚（腕は除く）
    combos = []
    for a in robot:
        combos += [(a, b) for b in furniture]                              # 机・壁・棚・モニターなど：体・腕どれでも
        if not is_ankle(a):
            combos.append((a, "floor"))                                    # 倒れたときの体（足は足の裏の当たり判定で）
        if not is_arm(a):
            combos += [(a, b) for b in cour_body]                          # 配達員の体・脚とロボットの体・脚（手・腕どうしは触れない）
    combos += [("case_geom", b) for b in furniture]
    n = 0
    for a, b in combos:
        key = tuple(sorted((a, b)))
        if a == b or key in have:
            continue
        if (a.endswith("hand_collision") and b == "desk") or (a.endswith("_foot") and b == "floor"):
            continue                                        # 手と机は手のメッシュで、足と床は前からの組で
        p = spec.add_pair(geomname1=a, geomname2=b)
        p.condim = 3
        p.friction = [0.6, 0.6, 0.005, 0.0001, 0.0001]
        have.add(key)
        NEW_PAIRS.append(key)
        n += 1
    print(f"すべての物体どうしの接触：ロボット {len(robot)}・動かない物 {len(static)}・配達員 {len(courier)}、足した組 {n}", flush=True)


class Walker:
    """箱を持って歩く方策（carry_env と同じ観測）。止まっている（指令 0）間は歩く周期の時計を止める。"""

    def __init__(self, rb, name):
        self.rb = rb
        self.pol, self.priv, self.default, self.cfg = pt.load_walk(name)
        self.crouch = bool(self.cfg.crouch_obs)
        self.ff_table = ce.crouch_ff_table() if getattr(self.cfg, "crouch_ff", False) else None
        self.crouch_H = float(np.load(REPO / "models" / self.cfg.crouch_file)["H"]) if self.crouch else 0.20
        m = rb.m
        self.est_w, self.est_b = np.array(self.cfg.estimator.w), self.cfg.estimator.b
        self.alpha = DT / (self.cfg.payload.est_tau + DT)
        self.m_est, self.last = 0.0, np.zeros(N_ACT)
        self.phase, self.phase_dt = np.array([0.0, math.pi]), 2 * math.pi * DT * 1.375
        self.key = jax.random.PRNGKey(0)
        self.hooks = []
        self.lock_yaw, self.yaw_hold = False, None
        self.shift = np.zeros(2)                             # 寄せた上体（両足を着けたまま骨盤を前後・左右へ、体の座標、m）
        self.shift_yaw = 0.0                                 # 骨盤の向きを股の向きで直した量（rad）
        self.shift_table = None

    def step(self, cmd, arm, h_cmd=0.0, before=None):
        rb, d = self.rb, self.rb.d
        q = d.qpos[7:7 + 29] - self.default
        obs = np.concatenate([d.sensordata[rb.linvel_adr:rb.linvel_adr + 3], rb.gyro(), rb.gravity(), cmd, q[:N_ACT],
                              d.qvel[6:6 + N_ACT], self.last, np.cos(self.phase), np.sin(self.phase), [self.m_est / 3.0]]
                             + ([[h_cmd / 0.1]] if self.crouch else []))
        act = np.array(self.pol({"state": jp.array(obs), "privileged_state": self.priv}, self.key)[0])
        if before is not None:
            before()
        for h in self.hooks:
            h()
        leg = self.default[:N_ACT] + act * pt.WALK_SCALE
        if self.lock_yaw:                                    # 置くあいだ、股と腰のひねりを止める（止め始めた時の目標のまま）
            if self.yaw_hold is None:
                self.yaw_hold = leg[list(YAW_JOINTS)].copy()
            leg[list(YAW_JOINTS)] = self.yaw_hold
        else:
            self.yaw_hold = None
        if self.shift_table is not None:                    # 寄せた上体：脚の角度を先に足す（学習と同じ）
            leg = leg.copy()
            leg[:12] += ce.shift_ff(self.shift, self.shift_table)
            leg[list(ce.HIP_YAW_IDX)] += ce.YAW_SIGN * self.shift_yaw
        if self.crouch and self.cfg.crouch_ff:              # 沈める量に合わせたしゃがむ姿勢を先に足す（学習と同じ）
            leg = leg + ce.crouch_ff(h_cmd, self.ff_table, waist=getattr(self.cfg, "crouch_waist", False),
                                     lean_deg=getattr(self.cfg, "crouch_lean", 0.0), H=self.crouch_H)
        rb.step_physics(np.concatenate([leg, arm]))
        raw = float(self.est_w @ d.actuator_force[N_ACT:29] + self.est_b)
        self.m_est = float(np.clip(self.m_est + self.alpha * (raw - self.m_est), 0.0, 1.5))
        self.phase = np.fmod(self.phase + self.phase_dt + math.pi, 2 * math.pi) - math.pi
        if np.linalg.norm(cmd) < 0.01:
            self.phase = np.array(ce.STAND_PHASE)
        self.last = act


class ArmIK:
    """置くあいだ、いまの体の姿勢から腕の関節の目標角をその場で求める（逆運動学）。手のひらの位置・向きと肘の位置を、
    置き始めた時の体の向きの座標（x：机の方、y：左、z：上）で決める。左右の腕を別々に解く。"""

    def __init__(self, m, d):
        from scipy.optimize import least_squares
        self.ls = least_squares
        self.m, self.d, self.dk = m, d, mujoco.MjData(m)
        J = pe.ARM_JOINTS
        self.adr = {sd: [m.joint(f"{sd}_{j}_joint").qposadr[0] for j in J] for sd in pe.SIDES}
        self.lo = {sd: np.array([m.joint(f"{sd}_{j}_joint").range[0] for j in J]) for sd in pe.SIDES}
        self.hi = {sd: np.array([m.joint(f"{sd}_{j}_joint").range[1] for j in J]) for sd in pe.SIDES}
        self.palm = {sd: m.site(f"{sd}_palm").id for sd in pe.SIDES}
        self.elbow = {sd: m.body(f"{sd}_elbow_link").id for sd in pe.SIDES}
        self.wrist = {sd: m.body(f"{sd}_wrist_roll_link").id for sd in pe.SIDES}
        x, y = d.qpos[0], d.qpos[1]
        R = d.xmat[m.body("pelvis").id].reshape(3, 3)
        yaw = math.atan2(R[1, 0], R[0, 0])
        self.o = np.array([x, y, 0.0])
        self.F = rotz(yaw)                                    # 置き始めた時の体の向き
        self.v = {sd: d.qpos[self.adr[sd]].copy() for sd in pe.SIDES}

    def fk(self, sd, v):
        dk = self.dk
        dk.qpos[:] = self.d.qpos
        dk.qpos[self.adr[sd]] = v
        mujoco.mj_kinematics(self.m, dk)
        to = lambda p: self.F.T @ (p - self.o)                # noqa: E731
        return (to(dk.site_xpos[self.palm[sd]]), self.F.T @ dk.site_xmat[self.palm[sd]].reshape(3, 3),
                to(dk.xpos[self.elbow[sd]]), to(dk.xpos[self.wrist[sd]]))

    def now(self, sd):
        return self.fk(sd, self.d.qpos[self.adr[sd]].copy())

    def solve(self, sd, P, Rp, E, w_e=(2.0, 10.0, 10.0), e_cap=None):
        """手のひら P・向き Rp、肘 E（x・y・z の重み w_e）。e_cap：肘の高さの上限（これより上がらない）。"""
        v0 = self.v[sd]

        def res(x):
            pp, RR, ee, _ = self.fk(sd, x)
            r = [(pp - P) * 10, (RR - Rp).ravel() * 0.5, (ee - E) * np.array(w_e), (x - v0) * 0.05]
            if e_cap is not None:
                r.append([max(0.0, ee[2] - e_cap) * 30])
            return np.concatenate(r)
        self.v[sd] = self.ls(res, v0, bounds=(self.lo[sd], self.hi[sd]), max_nfev=40).x
        return self.v[sd]


def rotz(a):
    return np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1.0]])


def quat_of(R):
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(R, float).ravel())
    return q


def nlerp(q0, q1, u):
    q1 = q1 if q0 @ q1 >= 0 else -q1
    q = (1 - u) * q0 + u * q1
    return q / np.linalg.norm(q)


class Courier:
    """配達員を描く：体の位置と向き（床の上を動く）、腕は肩から手の位置まで（2 本の棒の逆運動学、肘は下・外へ）、脚は歩く速さに
    合わせて振る。見た目だけで、物理の計算には入らない。持っている箱（box）は、手と一緒に決めた位置へ動かす。"""

    def __init__(self, m, d):
        self.m, self.d = m, d
        self.mid = lambda n: m.body(n).mocapid[0]
        self.body = self.mid("courier")
        self.xy, self.yaw = np.array(COURIER_WAIT, float), float(office.COURIER_YAW)
        self.hands = {"l": None, "r": None}                 # 手の中心の位置（世界の座標）。None なら下ろした腕
        self.palm = {"l": "up", "r": "up"}                  # 手のひらの向き："up"（上）か、ベクトル（押す向き）
        self.hang_w = {"l": 0.0, "r": 0.0}                  # 手の向きを、垂らした手の向きへ寄せる割合（腕を下ろす途中）
        self.box = None                                     # (位置, 四元数)：箱を持っている間
        self.cq = m.joint("case_free").qposadr[0]
        self.cv = m.joint("case_free").dofadr[0]
        self.gait, self.amp, self.prev = 0.0, 0.0, None
        self.crouch = 0.0                                   # 膝を曲げて体を下げる量（m）
        self.short, self.short_at, self.n = 0.0, None, 0   # 手が届かなかった距離の最大と、その時（確認用）

    def rest_hand(self, side):
        sy = 1 if side == "l" else -1
        return self.world(np.array([0.06, sy * 0.24, 0.91]))

    def world(self, p, dz=0.0):
        return np.concatenate([self.xy, [dz]]) + rotz(self.yaw) @ p

    def seg(self, name, a, b):
        i = self.mid(name)
        self.d.mocap_pos[i] = (a + b) / 2
        q = np.zeros(4)
        v = b - a
        mujoco.mju_quatZ2Vec(q, v / max(np.linalg.norm(v), 1e-9))
        self.d.mocap_quat[i] = q

    def update(self, dt):
        d = self.d
        self.n += 1
        # 歩く速さから脚の振り（片足 1 歩 = 周期の半分、0.9 Hz、歩幅 = 4 × 0.9 × 0.8 × sin(振り) で速さに合わせる）
        if self.prev is None:
            self.prev = (self.xy.copy(), self.yaw)
        v = np.linalg.norm(self.xy - self.prev[0]) / dt + 0.25 * abs(dp.wrap(self.yaw - self.prev[1])) / dt
        self.prev = (self.xy.copy(), self.yaw)
        amp_goal = math.asin(min(v / 2.88, 0.4)) if v > 0.03 else 0.0
        self.amp += min(dt / 0.25, 1.0) * (amp_goal - self.amp)
        if self.amp > 0.005 or abs(math.sin(self.gait)) > 0.05:
            self.gait += 2 * math.pi * 0.9 * dt
        legs = {}
        thc = math.acos(min((LEG_L[0] + LEG_L[1] - self.crouch) / (LEG_L[0] + LEG_L[1]), 1.0))   # 膝を曲げて体を下げる
        for side, sy, ph in (("l", 1, 0.0), ("r", -1, math.pi)):
            g = self.gait + ph
            th = thc + self.amp * math.sin(g)
            kn = 2 * thc + 2.2 * self.amp * max(0.0, math.cos(g))   # 脚を前へ振る間は膝を曲げる
            hip = np.array([HIP[0], sy * HIP[1], HIP[2]])
            knee = hip + LEG_L[0] * np.array([math.sin(th), 0, -math.cos(th)])
            ank = knee + LEG_L[1] * np.array([math.sin(th - kn), 0, -math.cos(th - kn)])
            legs[side] = (hip, knee, ank)
        dz = 0.10 - min(legs[s][2][2] for s in legs)           # 低い方の足首が床の上（0.10 m）にくるよう体を上下させる
        d.mocap_pos[self.body] = [self.xy[0], self.xy[1], dz]
        d.mocap_quat[self.body] = [math.cos(self.yaw / 2), 0, 0, math.sin(self.yaw / 2)]
        for side, (hip, knee, ank) in legs.items():
            hip_w, knee_w, ank_w = self.world(hip, dz), self.world(knee, dz), self.world(ank, dz)
            self.seg(f"courier_th_{side}", hip_w, knee_w)
            self.seg(f"courier_sh_{side}", knee_w, ank_w)
            i = self.mid(f"courier_shoe_{side}")
            d.mocap_pos[i] = ank_w + rotz(self.yaw) @ np.array([0.04, 0, -0.065])
            d.mocap_quat[i] = [math.cos(self.yaw / 2), 0, 0, math.sin(self.yaw / 2)]
        # 腕：手の中心 → 手首（手の向きの後ろ 5 cm）→ 肩と手首から肘を決める
        for side, sy in (("l", 1), ("r", -1)):
            sh = self.world(np.array([SHOULDER[0], sy * SHOULDER[1], SHOULDER[2]]), dz)
            hang = self.hands[side] is None                  # 下ろした腕：手は前腕の延長に垂らす
            hand = self.hands[side] if not hang else self.rest_hand(side) + [0, 0, dz]
            fwd = hand - sh
            fwd[2] = 0.0
            fwd = fwd / np.linalg.norm(fwd) if np.linalg.norm(fwd) > 0.03 else rotz(self.yaw)[:, 0]
            if isinstance(self.palm[side], str):
                hx_ = fwd
                hz_ = np.array([0, 0, 1.0])
            else:                                            # 手のひらを押す相手の方へ（指は上）
                hz_ = np.asarray(self.palm[side], float)
                hx_ = np.array([0, 0, 1.0])
            wr = hand - hx_ * 0.05
            D = wr - sh
            L1, L2 = ARM_L
            dist = float(np.linalg.norm(D))
            if dist > L1 + L2 - 1e-3:                        # 届かない（記録しておく）
                if dist - (L1 + L2 - 1e-3) > self.short:
                    self.short, self.short_at = dist - (L1 + L2 - 1e-3), (self.n, side, self.hands[side] is None)
                D = D / dist * (L1 + L2 - 1e-3)
                dist = L1 + L2 - 1e-3
                wr = sh + D
            u = D / dist
            a = (L1 ** 2 - L2 ** 2 + dist ** 2) / (2 * dist)
            h = math.sqrt(max(L1 ** 2 - a ** 2, 0.0))
            pole = rotz(self.yaw) @ np.array([-0.3, sy * 0.6, 0.0]) + np.array([0, 0, -1.0])
            pole = pole - (pole @ u) * u
            el = sh + a * u + h * pole / max(np.linalg.norm(pole), 1e-9)
            self.seg(f"courier_ua_{side}", sh, el)
            self.seg(f"courier_fa_{side}", el, wr)
            i = self.mid(f"courier_hand_{side}")
            w_h = 1.0 if hang else self.hang_w[side]
            if w_h > 0:
                gx = (wr - el) / np.linalg.norm(wr - el)
                gz = rotz(self.yaw) @ np.array([0, -sy, 0.0])             # 手のひらは体の方へ
                hx_ = (1 - w_h) * hx_ + w_h * gx
                hx_ /= np.linalg.norm(hx_)
                hz_ = (1 - w_h) * hz_ + w_h * gz
                hz_ = hz_ - (hz_ @ hx_) * hx_
                hz_ /= np.linalg.norm(hz_)
            hy_ = np.cross(hz_, hx_)
            d.mocap_pos[i] = wr + hx_ * 0.05
            d.mocap_quat[i] = quat_of(np.column_stack([hx_, hy_ / np.linalg.norm(hy_), hz_]))
        if self.box is not None:                             # 箱は配達員の手と一緒に動く
            d.qpos[self.cq:self.cq + 3] = self.box[0]
            d.qpos[self.cq + 3:self.cq + 7] = self.box[1]
            d.qvel[self.cv:self.cv + 6] = 0.0


def place_with_ik(m, d, wk, z, H, marks):
    """机に置く（腕の角度をその場で求める）。人が箱を机に置くように：
      1. 高さを保ったまま、手と肘を一緒に机の上まで前へ出す（0.8 秒）
      2. 手先を下げながら（手のひらを前へ PLACE_TILT_DEG 傾ける）下ろし、腰も沈める。箱の奥の縁が机に着いたら沈むのを止める
      3. 肘と手を同じだけ外へ平行に開く（2.5 秒）。肘は決して上げず、少しずつ下げる（箱の手前側が静かに机へ下りる）。届かない分は
         腰を少し深く沈めて補う
      4. 手を上げ、体を起こし（2 秒）、運ぶ姿勢に戻る（1.5 秒）
    手のひら・肘の目標は、置き始めた時の体の向きの座標で決める（体が少し回っても、手は机に対して決めた所へ行く）。"""
    ik = ArmIK(m, d)
    desk_top = office.DESK["height"]
    desk_g = m.geom("desk").id
    case_g = m.geom("case_geom").id
    hand_g = {g for g in range(m.ngeom) if m.geom_bodyid[g] in {m.body(f"{sd}_{b}").id for sd in pe.SIDES
                                                               for b in ("elbow_link", "wrist_roll_link", "wrist_pitch_link", "wrist_yaw_link")}}

    def on_desk(gs):
        for ci in range(d.ncon):
            g1, g2 = d.contact[ci].geom1, d.contact[ci].geom2
            if (g1 == desk_g and g2 in gs) or (g2 == desk_g and g1 in gs):
                if d.contact[ci].pos[2] > desk_top - 0.008 and d.contact[ci].pos[0] < DESK_X + DESK_HX - 0.01:
                    return True
        return False
    sgn = {"left": 1.0, "right": -1.0}
    start = {sd: ik.now(sd) for sd in pe.SIDES}               # 手のひら・向き・肘・手首（置き始め）
    v_carry = {sd: d.qpos[ik.adr[sd]].copy() for sd in pe.SIDES}
    FWD, TILT, SPREAD, DROP_E = 0.08, math.radians(PLACE_TILT_DEG), 0.12, 0.03

    def ry_(a):
        return np.array([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]])

    def arms(targets):
        out = []
        for sd in pe.SIDES:
            P, Rp, E, cap = targets[sd]
            out.append(ik.solve(sd, P, Rp, E, e_cap=cap))
        return np.concatenate(out)

    def step(targets, h):
        wk.step(np.zeros(3), arms(targets), h_cmd=h)

    log = lambda s: print("    " + s, flush=True)              # noqa: E731
    # 1. 前へ（0.8 秒）
    n1 = 40
    for k in range(n1):
        u = sm_(k / n1)
        tg = {}
        for sd in pe.SIDES:
            P0, R0, E0, _ = start[sd]
            tg[sd] = (P0 + np.array([FWD * u, 0, 0]), R0, E0 + np.array([FWD * u, 0, 0]), None)
        step(tg, 0.04 * u)
    # 2. 手先から下ろす（腰も沈める）。箱の奥の縁が机に着いたら止める
    h, n2, touched = 0.04, 0, None
    cur = {sd: (start[sd][0] + np.array([FWD, 0, 0]), start[sd][2] + np.array([FWD, 0, 0])) for sd in pe.SIDES}
    while n2 < int(3.0 / DT):
        n2 += 1
        th = TILT * sm_(n2 / 25)
        h = min(H, h + 0.08 * DT)                             # 腰は 8 cm/秒で沈める
        tg = {}
        for sd in pe.SIDES:
            P, E = cur[sd]
            P = P + np.array([0, 0, -0.06 * DT])               # 手のひらは 6 cm/秒で下げる
            lf = math.hypot(*(P - E)[:2])
            E = np.array([E[0], E[1], P[2] + lf * math.sin(th) + (start[sd][2][2] - start[sd][0][2])])   # 肘は手より高く（手先が下がる）
            cur[sd] = (P, E)
            tg[sd] = (P, ry_(th) @ start[sd][1], E, None)
        step(tg, h)
        if on_desk({case_g}):
            touched = "箱"
        elif on_desk(hand_g):
            touched = "手（指先・手首）"
        if touched:
            break
    marks["touch"] = rb_t(wk)
    log(f"机に触れた：{touched}、沈み {h * 100:.1f} cm（目安 {H * 100:.0f} cm）、下ろし始めて {n2 * DT:.2f} 秒")
    hc = h
    now = {sd: ik.now(sd) for sd in pe.SIDES}
    # 3. 手を開き、続けてすぐ肘を開く。手が机に触れて箱の奥の縁が下りたら、手のひらを外へ開き（0.6 秒）、手が開き始めて 0.15 秒後
    #    から肘も同じだけ外へ開く（0.6 秒）。手の開き → 肘の開きを続けて行い、すぐに箱を机に落とす。手は 0.5 cm 浮かせて机を
    #    押さない（押し続けると、その反動で体が後ろへ押されて倒れた）。肘は上げない。開くあいだ腰は沈めない
    T_HAND, T_LAG, T_ELB = 0.6, 0.15, 0.6
    n3 = int((T_LAG + T_ELB + 0.2) / DT)
    DH = 0.0
    first_case = None
    for k in range(n3):
        t3 = k * DT
        uh, ue = sm_(t3 / T_HAND), sm_((t3 - T_LAG) / T_ELB)
        if first_case is None:                                 # 箱が机に最初に着いた所（奥の縁か、手前か）
            for ci in range(d.ncon):
                g1, g2 = d.contact[ci].geom1, d.contact[ci].geom2
                if {g1, g2} == {desk_g, case_g}:
                    Rb = d.xmat[m.body("case").id].reshape(3, 3)
                    loc = ik.F.T @ (d.contact[ci].pos - d.xpos[m.body("case").id])
                    first_case = (t3, "奥の縁" if loc[0] > 0.05 else ("手前の縁" if loc[0] < -0.05 else "真ん中"))
                    break
        tg = {}
        for sd in pe.SIDES:
            P, Rn, E, _ = now[sd]
            th = TILT * (1 - uh)
            tg[sd] = (P + np.array([0, sgn[sd] * SPREAD * uh, 0.005]), ry_(th) @ start[sd][1],
                      E + np.array([0, sgn[sd] * SPREAD * ue, 0]), E[2])
        step(tg, hc)
        if os.environ.get("PLACE_DEBUG") and k % 5 == 0:
            Rp_ = d.xmat[m.body("pelvis").id].reshape(3, 3)
            log(f"開く {t3:4.2f} 秒  骨盤の高さ {d.qpos[2]:.3f}  骨盤の傾き {math.degrees(math.acos(np.clip(Rp_[2, 2], -1, 1))):5.1f}°  "
                f"左手のひら {np.round(ik.now('left')[0], 3)}  左肘 {np.round(ik.now('left')[2], 3)}")
    hd = min(H, hc + DH)
    opened = {sd: ik.now(sd) for sd in pe.SIDES}
    log(f"箱が最初に机に着いた所：{first_case[1] + '（開き始めて ' + format(first_case[0], '.2f') + ' 秒）' if first_case else '開いているあいだには着かなかった'}")
    for sd in pe.SIDES:                                       # 確認用：開き終わった時の手のひらと肘（置き始めからの動き）
        P_, _, E_, _ = opened[sd]
        log(f"{sd}：手のひら 横 {(P_ - now[sd][0])[1] * 100:+.1f} cm・高さ {(P_ - now[sd][0])[2] * 100:+.1f} cm、"
            f"肘 横 {(E_ - now[sd][2])[1] * 100:+.1f} cm・高さ {(E_ - now[sd][2])[2] * 100:+.1f} cm（目標 横 {sgn[sd] * SPREAD * 100:+.0f} cm）")
    # 4. 手を 10 cm 上げ（0.4 秒）、腰を沈めたまま腕を運ぶ姿勢に戻し（0.6 秒）、それから体を起こす（2 秒）。腕を左右に開いたまま
    #    体を起こすと、体が後ろへ傾いて倒れた（方策は運ぶ姿勢で体を起こすことを学習している）
    for k in range(20):
        u = sm_(k / 20)
        tg = {}
        for sd in pe.SIDES:
            P, Rn, E, _ = opened[sd]
            up = np.array([0, 0, 0.10 * u])
            tg[sd] = (P + up, start[sd][1], E + up, None)
        step(tg, hd)
    v_up = {sd: ik.v[sd].copy() for sd in pe.SIDES}
    for k in range(30):
        u = sm_(k / 30)
        wk.step(np.zeros(3), np.concatenate([(1 - u) * v_up[sd] + u * v_carry[sd] for sd in pe.SIDES]), h_cmd=hd)
    if os.environ.get("PLACE_DEBUG"):
        log(f"腕を運ぶ姿勢に戻した：骨盤の高さ {d.qpos[2]:.3f}")
    carry_arms = np.concatenate([v_carry[sd] for sd in pe.SIDES])
    rest = [(carry_arms, hd * (1 - sm_(k / 100))) for k in range(100)] + [(carry_arms, 0.0)] * int(2.0 / DT)
    return [(k, a_, h_) for k, (a_, h_) in enumerate(rest)]


def desk_touched(m, d):
    """学習（carry_env の _desk_touched）と同じ決め方：箱の底の角か、手・手首の円柱の下の端が、机の上面（＋0.3 cm）まで下りて、
    机の上（手前の縁より奥）にあるか。何が触れたかを返す（触れていなければ None）。"""
    top = office.DESK["height"]

    def hit(p, r):
        return p[2] - r <= top + 0.003 and p[0] <= DESK_X + DESK_HX and abs(p[1] - DESK_Y) <= DESK_HY
    for sd in pe.SIDES:
        for b, (a0, a1, r) in ce.HAND_CAPS.items():
            bid = m.body(f"{sd}_{b}").id
            R = d.xmat[bid].reshape(3, 3)
            sg = np.array([1, 1 if sd == "left" else -1, 1])
            for a in (a0, a1):
                if hit(d.xpos[bid] + R @ (np.array(a) * sg), r):
                    return "手（指先・手首）"
    cb = m.body("case").id
    Rc = d.xmat[cb].reshape(3, 3)
    hx, hy, hz = ce.CASE_HALF
    for sx in (-1, 1):
        for sy in (-1, 1):
            for sz in (-1, 1):
                if hit(d.xpos[cb] + Rc @ np.array([sx * hx, sy * hy, sz * hz]), 0.0):
                    return "箱"
    return None


def place_with_tab(m, d, wk, marks):
    """机に置く（学習 carry22 と同じ表と同じ決め方）。表は scripts/crouch_place_touch.py が作った、机に触れた歩ごとの腕の
    動きと沈める量：前へ出す（0.8 秒）→ 手先を下げながら下ろす（触れるまで、最長 3 秒）→ 触れたら手を開き、0.15 秒遅れて肘も
    開く → 手を上げる → 運ぶ姿勢に戻す → 体を起こす。触れたかは desk_touched（学習と同じ）で調べる。"""
    zt = np.load(REPO / "models" / wk.cfg.place_touch_file)
    tab, n1 = zt["tab"], int(zt["N1"])
    n2, L = tab.shape[0], tab.shape[1]
    log = lambda s: print("    " + s, flush=True)              # noqa: E731
    j = None
    case_g, desk_g = m.geom("case_geom").id, m.geom("desk").id
    first_case = None
    for k in range(L):
        row = tab[(j if j is not None else n2) - 1, k]
        wk.step(np.zeros(3), row[:14], h_cmd=float(row[14]))
        k_low = k - n1 + 1
        if j is None and k_low >= 1:
            who = desk_touched(m, d)
            if who or k_low >= n2:
                j = min(k_low, n2)
                marks["touch"] = rb_t(wk)
                log(f"机に触れた：{who or '（3 秒たった）'}、沈み {row[14] * 100:.1f} cm、下ろし始めて {k_low * DT:.2f} 秒")
        if j is not None and first_case is None:
            for ci in range(d.ncon):
                if {d.contact[ci].geom1, d.contact[ci].geom2} == {desk_g, case_g}:
                    loc = d.xmat[m.body("pelvis").id].reshape(3, 3).T @ (d.contact[ci].pos - d.xpos[m.body("case").id])
                    first_case = (k - n1 - j, "奥の縁" if loc[0] > 0.05 else ("手前の縁" if loc[0] < -0.05 else "真ん中"))
                    log(f"箱が最初に机に着いた所：{first_case[1]}（開き始めて {first_case[0] * DT:.2f} 秒）")
                    break
    return [(k, tab[j - 1, -1, :14], 0.0) for k in range(int(1.0 / DT))]


def sm_(u):
    u = min(max(u, 0.0), 1.0)
    return u * u * (3 - 2 * u)


def rb_t(wk):
    return wk.rb.t


def render_still(rb, path, t, cam, W=640, H=480):
    """時刻 t の 1 コマを写す（確認用）。"""
    from PIL import Image
    m, d = rb.m, mujoco.MjData(rb.m)
    r = mujoco.Renderer(m, H, W)
    opt = mujoco.MjvOption()
    opt.geomgroup[4] = 1                                    # 近くから写すので、西の壁も描く
    m.geom_rgba[m.geom("floor").id, 3] = 0.0                # 物理の床（灰色のタイル）は描かない
    i = min(int(round(t / DT)), len(rb.frames) - 1)
    n0 = len(rb.frames) - len(rb.mframes)
    d.qpos[:] = rb.frames[i]
    mp, mq = rb.mframes[min(max(i - n0, 0), len(rb.mframes) - 1)]
    d.mocap_pos[:], d.mocap_quat[:] = mp, mq
    mujoco.mj_forward(m, d)
    r.update_scene(d, cam, scene_option=opt)
    Image.fromarray(r.render()).save(path)
    r.close()


def render(rb, path, cam_fn, fps=50, hold=2.0, fade=None, room_fn=None):
    """deploy.render と同じ。配達員の手（モーキャップ）の位置も写す。room_fn(t) が True の間は、西の壁を描き、物理の床（灰色の
    タイル）を描かない（部屋の板張りの床が物理の床の 1 mm 上にあり、遠くでタイルが透けて見えた。西の壁は部屋全体を写すときは描かない）。"""
    import subprocess
    m, d = rb.m, mujoco.MjData(rb.m)
    W, H = 1280, 720
    m.vis.global_.offwidth, m.vis.global_.offheight = W, H
    m.vis.headlight.ambient[:] = 0.35
    m.vis.headlight.diffuse[:] = 0.45
    r = mujoco.Renderer(m, H, W)
    opt = mujoco.MjvOption()
    opt.geomgroup[4] = 0
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                           "-r", str(fps), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
                           "-movflags", "+faststart", path], stdin=subprocess.PIPE)
    n0 = len(rb.frames) - len(rb.mframes)                    # 記録を始める前の歩（座って立ち上がるまでは手は動かない）
    frames = rb.frames + [rb.frames[-1]] * int(hold * fps)
    wall_s = [g for g in range(m.ngeom) if abs(m.geom_pos[g][0] - 5.05) < 0.01 and abs(m.geom_pos[g][1] + 0.875) < 0.01]
    floor_g = m.geom("floor").id
    for i, q in enumerate(frames):
        room = room_fn is not None and room_fn(i / fps)
        opt.geomgroup[4] = int(room)
        m.geom_rgba[floor_g, 3] = 0.0                       # 物理の床（灰色のタイル）はどの場面でも描かない。見えるのはオフィスと廊下の床だけ
        if fade is not None:                                 # 受け渡しを写す間は、ドアの南側の壁を半透明に
            for g in wall_s:
                m.geom_rgba[g, 3] = fade(i / fps)
        d.qpos[:] = q
        mp, mq = rb.mframes[min(max(i - n0, 0), len(rb.mframes) - 1)]
        d.mocap_pos[:] = mp
        d.mocap_quat[:] = mq
        mujoco.mj_forward(m, d)
        r.update_scene(d, cam_fn(q, i / fps), scene_option=opt)
        ff.stdin.write(r.render().tobytes())
    ff.stdin.close()
    ff.wait()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--walk", default="carry10", help="B〜D の歩く方策（③ の持ち方で箱を持って歩く）")
    ap.add_argument("--place_walk", default=None, help="D 机の上に置く で使う方策（なしなら --walk と同じ）。机の前で全身を止めている間に切り替える")
    ap.add_argument("--go", type=float, default=2.0, help="ノックの合図の時刻（秒）")
    ap.add_argument("--video")
    ap.add_argument("--seed", type=int, default=None, help="条件をばらつかせる（摩擦、机の前で止まる位置など）。なしなら決まった条件")
    ap.add_argument("--json", help="結果を 1 行の JSON で書き足すファイル")
    ap.add_argument("--dump_states", help="通しの途中の状態（A の終わり・C の始め・C の道順の途中）を 1 行の JSON で書き足すファイル。"
                    "事務所の場面で B〜D を学習するとき、練習をこの状態から始める")
    ap.add_argument("--events", help="声と音を入れる時刻（JSON）を書き出すファイル")
    ap.add_argument("--place_tab", help="置く腕の動きの表（models/ の中）。方策の設定に place_touch_file がなくても、学習と同じ表で置く（確かめる用）")
    ap.add_argument("--place_ik", type=int, default=1, help="置くとき腕の角度をその場で求める（1）：手先から下ろし、奥の縁が着いたら肘と手を平行に外へ離す")
    ap.add_argument("--stop_edge", type=float, help="机の正面の縁から骨盤までの距離（m、確認用）")
    ap.add_argument("--torsion", type=float, default=0.0, help="足と床のねじり摩擦（m）。0 ならなし")
    ap.add_argument("--lock_yaw", type=int, default=0, help="置くあいだ股と腰のひねりを止める（1）")
    ap.add_argument("--yaw_bias", type=float, help="机に向く角度を先に右へずらす量（度）。なしなら方策ごとの値")
    ap.add_argument("--yaw_shift", type=int, default=0, help="調整で上体を寄せるとき、股の向きで骨盤の向きも直し、踏み直すときも上体を一緒に運ぶ（1）")
    ap.add_argument("--no_all_contacts", action="store_true", help="すべての物体どうしの接触を入れない（前の組み合わせだけ）")
    ap.add_argument("--adjust", type=int, default=0, help="机の前で止まった後の、小さな踏み直しの最大の回数（0 ならしない）")
    ap.add_argument("--lateral", type=int, default=0, help="机の前へ進むとき横のずれも直す（1）か、まっすぐ進むだけ（0）。横の指令を出すと、止まって待つ間に体が約 10° 回り、置くときに箱を引きずった")
    ap.add_argument("--no_label", action="store_true", help="箱の上の面に「MAC 13」を印刷しない（印刷するのが標準）")
    ap.add_argument("--stills", help="受け渡しを近くから写した静止画を書き出すフォルダ（確認用）")
    args = ap.parse_args()
    global PLACE_YAW_BIAS, LABEL, ALL_CONTACTS
    LABEL = not args.no_label
    ALL_CONTACTS = not args.no_all_contacts
    place_walk = args.place_walk or args.walk
    PLACE_YAW_BIAS = math.radians(args.yaw_bias if args.yaw_bias is not None else PLACE_YAW_BIAS_DEG.get(place_walk, 4.3))
    dp.office_decorate = full_decorate                       # deploy.Robot が使う事務所の場面を差し替える
    rb = dp.Robot(walk_name="estimate")
    m, d = rb.m, rb.d
    rb.mframes = []                                          # 動画のため、配達員の手（モーキャップ）の位置も記録する
    _step = rb.step_physics

    new_ids = {tuple(sorted((m.geom(a).id, m.geom(b).id))): (a, b) for a, b in NEW_PAIRS} if ALL_CONTACTS else {}
    new_hits = {}                                            # 新しく足した組で実際に当たった回数（歩の数）と最初の時刻

    def step_physics(target):
        cour.update(DT)                                      # 配達員の体・腕・脚を描き、持っている箱を手と一緒に動かす
        _step(target)
        rb.mframes.append((d.mocap_pos.copy(), d.mocap_quat.copy()))
        if new_ids:
            seen = set()
            for ci in range(d.ncon):
                key = tuple(sorted((d.contact[ci].geom1, d.contact[ci].geom2)))
                if key in new_ids and d.contact[ci].dist < 0 and key not in seen:
                    seen.add(key)
                    nm = "-".join(new_ids[key])
                    c_, t_ = new_hits.get(nm, (0, rb.t))
                    new_hits[nm] = (c_ + 1, t_)
    rb.step_physics = step_physics
    cour = Courier(m, d)
    m.opt.iterations, m.opt.ls_iterations = 10, 10
    cfg = ce.default_config()
    rng = np.random.default_rng(args.seed) if args.seed is not None else None
    cond = dict(mu_hand=1.0, mu_desk=0.45, mu_floor=0.6, stop_dx=0.0, stop_dy=0.0)
    if rng is not None:                                      # 条件をばらつかせる
        cond = dict(mu_hand=rng.uniform(0.8, 1.2), mu_desk=rng.uniform(0.3, 0.6), mu_floor=rng.uniform(0.5, 1.0),
                    stop_dx=rng.uniform(-0.03, 0.03), stop_dy=rng.uniform(-0.02, 0.02))
    global STOP
    if args.stop_edge is not None:
        STOP = np.array([DESK_X + DESK_HX + args.stop_edge, PLACE_Y])
    STOP = STOP + np.array([cond["stop_dy"], cond["stop_dx"]])  # 机の前で止まる位置（置く位置）を左右（y）±3 cm、前後（x）±2 cm
    for i in range(m.npair):
        names = (m.geom(m.pair_geom1[i]).name, m.geom(m.pair_geom2[i]).name)
        if "case_geom" in names and any(n.endswith("_pad") for n in names):
            m.pair_friction[i, 0:2] = cond["mu_hand"]
        if "case_geom" in names and "desk" in names:
            m.pair_friction[i, 0:2] = cond["mu_desk"]
        if "floor" in names and any("foot" in n for n in names):
            m.pair_friction[i, 0:2] = cond["mu_floor"]
            if args.torsion > 0:                             # 足の裏のねじり摩擦（ゴムの足裏が床の上でその場で回りにくい）
                m.pair_dim[i] = 4
                m.pair_friction[i, 2] = args.torsion
    for i in range(m.npair):
        names = (m.geom(m.pair_geom1[i]).name, m.geom(m.pair_geom2[i]).name)
        if "case_geom" in names or ("desk" in names and any("mesh" in n for n in names)):
            m.pair_solref[i] = cfg.case.contact_solref
    cb = m.body("case").id
    hx, hy, hz = ce.CASE_HALF
    m.body_mass[cb] = 1.23
    m.body_inertia[cb] = 1.23 / 3 * np.array([hy ** 2 + hz ** 2, hx ** 2 + hz ** 2, hx ** 2 + hy ** 2])
    m.geom_rgba[m.geom("case_geom").id] = [0.93, 0.93, 0.93, 1]
    for s in pe.SIDES:
        for j in pe.ARM_JOINTS:
            m.dof_damping[m.joint(f"{s}_{j}_joint").dofadr[0]] = cfg.arm_damping
    # 配達員が持っていた古いケース（形だけ）を消し、配達員の手（左右の板）を肌の色で見えるようにする
    courier = m.body("courier").id
    for g in range(m.ngeom):
        if m.geom_bodyid[g] == courier and (m.geom(g).name == "case" or (m.geom_size[g] == [0.006, 0.07, 0.012]).all()):
            m.geom_rgba[g, 3] = 0.0
    grips = [m.body(f"grip_{s}").mocapid[0] for s in ("l", "r")]
    sg = m.geom("support_geom").id
    courier_b = m.body("courier").id
    for g in range(m.ngeom):                                 # 配達員の止まった腕と脚は消して、動く腕と脚を描く
        if m.geom_bodyid[g] == courier_b and (
                (m.geom_type[g] == mujoco.mjtGeom.mjGEOM_CAPSULE and round(float(m.geom_size[g][0]), 3) in (0.048, 0.042, 0.062, 0.078))
                or (m.geom_type[g] == mujoco.mjtGeom.mjGEOM_ELLIPSOID and abs(m.geom_size[g][0] - 0.06) < 1e-4)
                or (m.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX and np.allclose(m.geom_size[g], [0.13, 0.05, 0.035]))):
            m.geom_rgba[g, 3] = 0.0
    for name in ("support_geom", "grip_l_geom", "grip_r_geom", "courier_push", "courier_door_hand_geom"):
        m.geom_group[m.geom(name).id] = 3                    # 物理の板は描かない（手は腕の先に描く）
    m.geom_size[sg] = [0.07, 0.05, 0.01]                                 # 配達員の手のひら（箱の底の真ん中を支える）
    # 配達員の手（手のひら・押す指・横の手）と箱の間は摩擦なし（押す力だけで動かす）。手は位置だけを決めて動かす物体で速さが
    # ないので、摩擦があると箱のその側だけがその場に引き留められ、箱が回ってひっくり返った
    for i in range(m.npair):
        names = {m.geom(m.pair_geom1[i]).name, m.geom(m.pair_geom2[i]).name}
        if "case_geom" in names and names & {"grip_r_geom", "support_geom", "courier_push"}:
            m.pair_dim[i] = 1
    support = m.body("support").mocapid[0]
    cq = m.joint("case_free").qposadr[0]
    # 箱は最初、配達員の体の前で両手の上に載せて持つ（向きは、受け渡すときにロボットの持つ向きになるよう、配達員に対して決める）。
    # 配達員が持っている間は、箱を配達員の手と一緒に決めた位置へ動かす（配達員の腕の中の動き）
    ca0 = ce.tray_case_pose()[1]
    Ry = lambda a: np.array([[math.cos(a), 0, -math.sin(a)], [0, 1, 0], [math.sin(a), 0, math.cos(a)]])  # noqa: E731
    LBOX = np.array([0.38, 0.0, 1.06])                      # 持っている箱の中心（配達員から見た位置）

    def held_box():
        return cour.world(LBOX), rotz(cour.yaw - math.pi) @ Ry(ca0)

    def hands_under(c, R, drop=0.0):
        """配達員の両手：箱の底の、配達員の側の半分を下から支える（手のひらは上）。drop だけ下げる。"""
        z = np.array([0, 0, drop])
        return {"l": c + R @ np.array([0.08, -0.07, -(hz + 0.023)]) - z, "r": c + R @ np.array([0.08, 0.07, -(hz + 0.023)]) - z}

    start, Rb = held_box()
    def set_grips(center, R, lower=0.0, open_=0.0, palm_down=0.0, palm_back=0.0):
        """配達員の手：左右の手で箱の側面に添え、手のひら（台）で箱の底の真ん中を下から支える。"""
        off = hy + ce.GRIP_HALF[1] - ce.GRIP_SQUEEZE + open_
        base = center + R[:, 0] * ce.GRIP_X - R[:, 2] * lower
        q = np.zeros(4)
        mujoco.mju_mat2Quat(q, R.ravel())
        d.mocap_pos[grips[0]] = [0, 2.0, -5.0]                            # 片方の手は手のひら（下の台）なので、横の手は 1 つ
        d.mocap_pos[grips[1]] = base - R[:, 1] * off
        d.mocap_quat[grips[1]] = q
        d.mocap_pos[support] = center - R[:, 2] * (hz + 0.01 + lower + palm_down) + R[:, 0] * palm_back
        d.mocap_quat[support] = q

    rb.place_seated()
    d.qpos[cq:cq + 3] = start
    d.qpos[cq + 3:cq + 7] = quat_of(Rb)
    cour.box = (start, quat_of(Rb))
    cour.hands = hands_under(start, Rb)
    set_grips(start, Rb)
    mujoco.mj_forward(m, d)
    log, marks = [], {}

    def note(name):
        x, y, th = rb.pose()
        log.append((name, rb.t, x, y, th, float(d.qpos[2])))
        print(f"{rb.t:6.1f} 秒  {name:30s} 位置 ({x:+.2f}, {y:+.2f})  向き {math.degrees(th):+6.1f}°  骨盤の高さ {d.qpos[2]:.2f}", flush=True)

    # ---------------- A
    marks["A"] = rb.t
    rb.stand_up(args.go)
    note("A 立ち上がった")
    rb.walk_init()
    rb.hold(0.6)
    x0, y0, _ = rb.pose()
    # 向きを変えるときは、目標の向きの近くまで速く回る（比例の強さ 2 → 4）。近づくと遅くなって、その場の足踏みが長かった
    rb.walk_to(x0, y0, yaw=math.pi / 2, tol=0.3, timeout=12.0, yaw_gain=4.0, yaw_tol=0.08)   # 左（北）を向く
    rb.walk_to(rb.pose()[0], DOOR_LANE, yaw=None, timeout=15.0)             # 北へ出る
    rb.walk_to(rb.pose()[0], rb.pose()[1], yaw=0.0, tol=0.3, timeout=12.0, yaw_gain=4.0, yaw_tol=0.08)  # 東（ドア）を向く
    rb.walk_to(4.3, DOOR_LANE, yaw=0.0, timeout=40.0)
    note("A ドアの前に着いた")
    marks["door"] = rb.t
    rb.set_arms("carry")
    rb.hold(1.2)
    ang = rb.push_door(x_max=5.25)
    note(f"A ドアを開けた（{math.degrees(ang):.0f}°）")
    # ドアを開けた所（戸口）にとどまり、配達員（南東）の方を向いて待つ。部屋へは、受け取ってから戻る
    rb.hold(0.5)
    x_, y_, _ = rb.pose()
    rb.walk_to(x_, y_, yaw=FACE_YAW, tol=0.3, timeout=10.0, yaw_gain=4.0, yaw_tol=0.08)
    note("A 配達員の方を向いた")
    t_C = [None]

    def dump(stage, wk_=None):
        """通しの途中の状態を書き出す（学習の練習の始めに使う）。"""
        if not args.dump_states:
            return
        rec = dict(stage=stage, seed=args.seed, t=rb.t, qpos=d.qpos[:36].tolist(), qvel=d.qvel[:35].tolist(),
                   case_q=d.qpos[cq:cq + 7].tolist(), case_v=d.qvel[m.joint("case_free").dofadr[0]:m.joint("case_free").dofadr[0] + 6].tolist(),
                   door=float(d.qpos[rb.door_q]), t_since_C=(rb.t - t_C[0]) if t_C[0] is not None else None,
                   last=(wk_.last.tolist() if wk_ is not None else None), phase=(wk_.phase.tolist() if wk_ is not None else None),
                   m_est=(float(wk_.m_est) if wk_ is not None else None), pelvis_z_min=float(min(q[2] for q in rb.frames[-500:])))
        with open(args.dump_states, "a") as f:
            f.write(json.dumps(rec) + "\n")
    dump("B")
    n_stand = int(round((args.go + 3.3) / DT))                              # 立ち上がった後に骨盤が下がったか
    A_ok = ang > math.radians(45) and min(q[2] for q in rb.frames[n_stand:]) > 0.5
    # ---------------- B
    marks["B"] = rb.t
    marks["courier_says"] = rb.t + 0.5                       # 配達員「荷物が届いています」
    wk = Walker(rb, args.walk)
    arm_from = d.qpos[7 + N_ACT:7 + 29].copy()
    carry = wk.default[N_ACT:].copy()
    sm = lambda u: (lambda x: x * x * (3 - 2 * x))(min(max(u, 0.0), 1.0))  # noqa: E731
    for k in range(int(1.5 / DT)):                                          # 腕を運ぶ姿勢へ（止まって）
        u = sm(k * DT / 1.5)
        wk.step(np.zeros(3), (1 - u) * arm_from + u * carry)
    # 受け取る位置：運ぶ姿勢での箱のあるべき位置（体の向き）＋手の 5 mm 上
    cc, ca = ce.tray_case_pose()
    kb = m.keyframe("knees_bent").qpos

    def case_home():
        x, y, th = rb.pose()
        off = cc - kb[0:3]
        pos = d.xpos[rb.pelvis] + np.array([math.cos(th) * off[0] - math.sin(th) * off[1], math.sin(th) * off[0] + math.cos(th) * off[1], off[2]])
        R = np.array([[math.cos(th), -math.sin(th), 0], [math.sin(th), math.cos(th), 0], [0, 0, 1]]) @ \
            np.array([[math.cos(ca), 0, -math.sin(ca)], [0, 1, 0], [math.sin(ca), 0, math.cos(ca)]])
        return pos, R

    # 配達員（2026-10-17）：箱を両手の上に載せて持ったまま、ロボットの正面まで歩いて近づき（T_W）、箱をロボットの腕の間（手のひらの
    # 3 cm 上）へ差し出し（T2）、真下へ下ろす（T3）。ロボットの手のひらの 3 mm 上で箱を放し、そこからは物理で計算する（箱は重さで
    # 手のひらに載る）。配達員の手は 2 cm 下げてから手前へ引き、腕を下ろす（T_BACK）。下ろし始めるまでは、ロボットの今の向きと
    # 位置に合わせて差し出す位置を直す
    T_W, T2, T3, T_BACK = 3.5, 2.0, 1.5, 2.0
    above = 0.03
    home, Rh = case_home()
    tgt = home + Rh[:, 2] * above
    th_r = rb.pose()[2]
    fwd_r = np.array([math.cos(th_r), math.sin(th_r)])
    yaw1 = th_r + math.pi                                                  # ロボットと向き合う
    # 受け渡す位置：配達員の手首が肩から 0.50 m ほどで届く距離（体の前と箱の間は空く）
    dz_ = SHOULDER[2] - (tgt[2] - hz - 0.023)
    HX = float(np.clip(math.sqrt(max(0.50 ** 2 - (SHOULDER[1] - 0.07) ** 2 - dz_ ** 2, 0.0)) + 0.05, 0.24, 0.42))
    c1 = tgt[:2] + fwd_r * (HX + 0.08)
    xy0, yaw0 = cour.xy.copy(), cour.yaw
    print(f"    受け渡し：箱の目標の高さ {tgt[2]:.3f} m、配達員の手の前後 {HX:.2f} m、立つ位置 ({c1[0]:.2f}, {c1[1]:.2f})", flush=True)
    for side in pe.SIDES:                                                   # 確認用：ロボットの手の位置（箱から見て）
        pw = d.xpos[m.body(f"{side}_wrist_yaw_link").id]
        print(f"      ロボットの{side}手首（箱から見て）{np.round(Rh.T @ (pw - home), 3)}", flush=True)
    held, c, R_ = True, None, None
    for k in range(int((T_W + T2 + T3 + T_BACK + 1.0) / DT)):
        t = k * DT
        if t < T_W:                                                        # 歩いて近づく（箱は両手の上）
            u = sm(t / T_W)
            cour.xy = (1 - u) * xy0 + u * c1
            cour.yaw = yaw0 + u * dp.wrap(yaw1 - yaw0)
            c, R_ = held_box()
            c_hold, q_hold = c.copy(), quat_of(R_)
        if t < T_W + T2:
            h_, R2 = case_home()
            a = DT / (0.3 + DT)
            tgt = tgt + a * ((h_ + R2[:, 2] * above) - tgt)
            Rh = R2
        if T_W <= t < T_W + T2:                                            # 差し出す（膝を少し曲げて体を下げる）
            u = sm((t - T_W) / T2)
            cour.crouch = 0.06 * u
            c = (1 - u) * c_hold + u * tgt
            q_ = nlerp(q_hold, quat_of(Rh), u)
            R_ = np.zeros(9)
            mujoco.mju_quat2Mat(R_, q_)
            R_ = R_.reshape(3, 3)
        elif T_W + T2 <= t < T_W + T2 + T3:                                # 真下へ下ろす
            u = sm((t - T_W - T2) / T3)
            c, R_ = tgt - Rh[:, 2] * (above + 0.02) * u, Rh
        elif t >= T_W + T2 + T3:                                           # 手を手前へ引いて、腕を下ろす
            c, R_ = tgt - Rh[:, 2] * (above + 0.02), Rh
        if held and t >= T_W + T2 and (c - home)[2] <= 0.003:
            held = False                                                   # 手のひらの 3 mm 上で放す
        if held and t >= T_W + T2 + T3:
            held = False
        if not held and "release" not in marks:
            marks["release"] = rb.t
        hands = hands_under(c, R_)
        if t >= T_W + T2 + T3:
            u1 = sm((t - T_W - T2 - T3) / (0.5 * T_BACK))
            u2 = sm((t - T_W - T2 - T3 - 0.5 * T_BACK) / (0.5 * T_BACK))
            for sd in hands:
                p_ = hands[sd] + Rh[:, 0] * 0.10 * u1 - np.array([0, 0, 0.02 * u1])
                hands[sd] = (1 - u2) * p_ + u2 * cour.rest_hand(sd)
                cour.hang_w[sd] = u2
            cour.crouch = 0.06 * (1 - u2)
        cour.hands = hands
        cour.box = (c, quat_of(R_)) if held else None
        set_grips(c, R_, open_=0.03 if not held else 0.0)
        wk.step(np.zeros(3), carry)
    cour.hands, cour.hang_w = {"l": None, "r": None}, {"l": 0.0, "r": 0.0}
    for b_ in grips + [support]:                                            # 物理の板は遠くへ
        d.mocap_pos[b_] = [0, 2.0, -5.0]
    home, _ = case_home()
    B_ok = bool(np.linalg.norm(d.xpos[cb] - home) < 0.03)
    note(f"B 受け取った（箱のずれ {np.linalg.norm(d.xpos[cb] - home) * 100:.1f} cm）")
    t_C[0] = rb.t
    if B_ok:
        dump("C0", wk)
    # ---------------- C
    marks["C"] = rb.t
    dropped = False
    # 配達員がドアを閉める（ロボットが部屋へ下がり始めて 2.5 秒後から）。ドアは外開きで、開いたドアの外側の面は北東を向く。受け渡した
    # 所（ドアの内側の面の側）から押すと開いてしまうので、配達員はドアの先端の外を歩いて回り、外側の面の前に立って右手で押して閉める。
    # 手は配達員の右手そのもの：ドアを押す物理の手（見えない板）を右手と同じ位置に置く
    dhand = m.body("courier_door_hand").mocapid[0]
    hinge = np.array(office.DOOR_HINGE[:2]) + np.array([-0.05, -0.005])
    t_close = rb.t + 2.5
    door_q = rb.door_q
    th_start = float(d.qpos[door_q])
    marks["door_close"] = t_close
    TW_D, TR_D, TP_D, TB_D = 4.5, 1.2, 3.5, 1.2                             # 回り込む・手を当てる・押す・手を下ろす
    dirv = lambda th: np.array([math.sin(th), -math.cos(th)])              # noqa: E731  蝶番からドアの先端への向き
    nv = lambda th: np.array([math.cos(th), math.sin(th)])                 # noqa: E731  ドアの外側の面の向き

    def door_pt(th, along, off):
        return np.concatenate([hinge + along * dirv(th) + nv(th) * off, [1.10]])

    def stand(th):
        """ドアの外側の面の前で、右手がドアの押す所の正面にくる立つ位置（ドアの方を向く）。"""
        yaw = th + math.pi
        return door_pt(th, 0.6, 0.042)[:2] + nv(th) * 0.36 + np.array([-math.sin(yaw), math.cos(yaw)]) * 0.15

    tip = hinge + office.DOOR_W * dirv(th_start)
    path = None

    def close_door():
        nonlocal path
        t = rb.t - t_close
        d.mocap_pos[dhand] = [6.0, 0.5, -3.0]
        if t < 0:
            return
        if path is None:                                      # 道すじ：今の位置 → ドアの先端の南 → 先端の東 → 外側の面の前
            pts = [cour.xy.copy(), tip + dirv(th_start) * 0.30 - nv(th_start) * 0.35,
                   tip + dirv(th_start) * 0.35 + nv(th_start) * 0.40, stand(th_start)]
            seg_l = [np.linalg.norm(pts[i + 1] - pts[i]) for i in range(3)]
            path = (pts, np.concatenate([[0], np.cumsum(seg_l)]))
        pts, cum = path
        if t < TW_D:                                          # 歩いて回り込む
            sdist = sm(t / TW_D) * cum[-1]
            i = min(int(np.searchsorted(cum, sdist, side="right")) - 1, 2)
            w_ = (sdist - cum[i]) / max(cum[i + 1] - cum[i], 1e-9)
            cour.xy = (1 - w_) * pts[i] + w_ * pts[i + 1]
            head = math.atan2(*(pts[i + 1] - pts[i])[::-1])
            goal = head if cum[-1] - sdist > 0.25 else th_start + math.pi
            cour.yaw += float(np.clip(dp.wrap(goal - cour.yaw), -2.5 * DT, 2.5 * DT))
            return
        if t < TW_D + TR_D:                                   # 右手をドアの外側の面に当てる
            cour.yaw += float(np.clip(dp.wrap(th_start + math.pi - cour.yaw), -2.5 * DT, 2.5 * DT))
            u = sm((t - TW_D) / TR_D)
            a_ = cour.rest_hand("r")
            pos = (1 - u) * a_ + u * door_pt(th_start, 0.6, 0.042) if u < 1 else door_pt(th_start, 0.6, 0.042)
            cour.hang_w["r"] = 1.0 - u
            th = th_start
        elif t < TW_D + TR_D + TP_D:                          # 押して閉める（体もドアについて動く）
            th = th_start * (1.0 - sm((t - TW_D - TR_D) / TP_D))
            cour.xy, cour.yaw = stand(th), th + math.pi
            pos = door_pt(th, 0.6, 0.042)
        elif t < TW_D + TR_D + TP_D + TB_D:                   # 手を下ろす
            th = 0.0
            u = sm((t - TW_D - TR_D - TP_D) / TB_D)
            pos = (1 - u) * door_pt(0.0, 0.6, 0.042) + u * cour.rest_hand("r")
            cour.hang_w["r"] = u
        else:
            cour.hands["r"], cour.palm["r"], cour.hang_w["r"] = None, "up", 0.0
            return
        cour.hands["r"] = pos
        cour.palm["r"] = np.concatenate([-nv(th), [0.0]])
        if TW_D + TR_D * 0.7 <= t < TW_D + TR_D + TP_D + 0.2:   # 物理の手（見えない板）も右手と同じ所に置いて押す
            d.mocap_pos[dhand] = pos
            d.mocap_quat[dhand] = [math.cos(th / 2), 0, 0, math.sin(th / 2)]
    wk.hooks.append(close_door)

    def drive(target, yaw_goal, face, tol, yaw_tol, vmax=0.4, timeout=40.0, vmin=0.12, forward_only=False):
        nonlocal dropped
        end = rb.t + timeout
        while rb.t < end:
            x, y, th = rb.pose()
            e = target - np.array([x, y])
            dist = float(np.linalg.norm(e))
            if dist < tol and (yaw_goal is None or abs(dp.wrap(yaw_goal - th)) < yaw_tol):
                return True
            if forward_only:                                             # 前へだけ（横向きの指令は使わない）
                fwd = math.cos(th) * e[0] + math.sin(th) * e[1]
                w = 0.0 if yaw_goal is None else float(np.clip(2.0 * dp.wrap(yaw_goal - th), -0.4, 0.4))
                v = float(np.sign(fwd) * np.clip(abs(fwd) * 0.8, vmin, vmax))
                if abs(fwd) < tol:
                    return True
                wk.step(np.array([v, 0.0, w]), carry)
                continue
            if face and dist > tol:
                head = math.atan2(e[1], e[0])
                err = dp.wrap(head - th)
                # 遅すぎる指令では足を出さずに立ってしまうので、残りが少なくても vmin 以上で進む
                v = float(np.clip(0.8 * dist, vmin, vmax)) * max(0.0, math.cos(err))
                cmd = np.array([v, 0.0, float(np.clip(1.5 * err, -0.8, 0.8))])
            else:
                c_, s_ = math.cos(th), math.sin(th)
                w = 0.0 if yaw_goal is None else float(np.clip(2.0 * dp.wrap(yaw_goal - th), -0.6, 0.6))
                cmd = np.array([float(np.clip(c_ * e[0] + s_ * e[1], -0.25, 0.3)), float(np.clip(-s_ * e[0] + c_ * e[1], -0.2, 0.2)), w])
                if np.linalg.norm(cmd) < 0.05:                                  # 小さすぎる指令では止まってしまうので、少し大きく
                    cmd = cmd / max(np.linalg.norm(cmd), 1e-6) * 0.05
            wk.step(cmd, carry)
            h, _ = case_home()
            gap = d.xpos[cb] - h
            if gap[2] < -0.25 or np.linalg.norm(gap[:2]) > 0.40 or d.qpos[2] < 0.5:
                dropped = True
                return False
        return False

    # 指令は学習したとおりに出す：一定の速さで進み、残りが減速距離になったら 1 秒かけて指令を 0 にする（carry10 が練習した
    # 止まり方）。残りの距離に比例して遅くすると、0.1 m/秒より遅い指令で方策が足を止め、決めた位置に届かなかった
    def guard():
        nonlocal dropped
        h, _ = case_home()
        gap = d.xpos[cb] - h
        if gap[2] < -0.25 or np.linalg.norm(gap[:2]) > 0.40 or d.qpos[2] < 0.5:
            dropped = True
        return not dropped

    def back_up(dist, v=0.2, brake=0.08):
        """今の向きのまま、後ろへ dist 下がる（一定の速さ v、残りが brake になったら 1 秒かけて止める）。"""
        x0_, y0_, th0 = rb.pose()
        k_b = None
        for k in range(int(20 / DT)):
            x, y, th = rb.pose()
            moved = -((x - x0_) * math.cos(th0) + (y - y0_) * math.sin(th0))
            if k_b is None and moved >= dist - brake:
                k_b = k
            u = 1.0 if k_b is None else max(0.0, 1.0 - (k - k_b) * DT / 1.0)
            if u <= 0.0:
                break
            w = float(np.clip(1.5 * dp.wrap(th0 - th), -0.3, 0.3))
            wk.step(np.array([-v * u, 0.0, w * u]), carry)
            if not guard():
                return False
        x, y, th = rb.pose()
        print(f"    後ろへ下がった：{-((x - x0_) * math.cos(th0) + (y - y0_) * math.sin(th0)):.2f} m（目標 {dist:.2f} m）、位置 ({x:.2f}, {y:.2f})", flush=True)
        return True

    def turn(yaw_goal, w=0.8, brake=0.22, cw=None):
        """その場で一定の速さ w で回り、残りの角度が brake になったら 1 秒かけて止める（0.5 rad/秒から 1 秒で止めると約 0.13 rad 回る）。
        足踏みの時間を短くするため、学習した範囲の上限に近い 0.8 rad/秒で回る。
        cw：True なら時計回り、False なら反時計回りに決める（None なら近い方）。"""
        k_b = None
        for k in range(int(20 / DT)):
            err = dp.wrap(yaw_goal - rb.pose()[2])
            if cw is not None and abs(err) > 0.3:
                err = -abs(err) if cw else abs(err)
            if k_b is None and abs(err) <= brake:
                k_b = k
            u = 1.0 if k_b is None else max(0.0, 1.0 - (k - k_b) * DT / 1.0)
            if u <= 0.0:
                break
            wk.step(np.array([0.0, 0.0, math.copysign(w, err) * u]), carry)
            if not guard():
                return False
        print(f"    回って止まった：目標の向きとの差 {math.degrees(dp.wrap(yaw_goal - rb.pose()[2])):+.1f}°", flush=True)
        return True

    def walk(target, v=0.4, brake=0.2, yaw_hold=None, lateral=False, through=False):
        x0_, y0_, _ = rb.pose()
        """target の方を向いたまま一定の速さ v で進み、残りが brake になったら 1 秒かけて止める（向きのずれは小さく直す）。"""
        k_b = None
        for k in range(int(40 / DT)):
            x, y, th = rb.pose()
            e = target - np.array([x, y])
            fwd = math.cos(th) * e[0] + math.sin(th) * e[1]
            if yaw_hold is not None:                              # 向きを保つ（短い前進で向きがずれるため）
                w = float(np.clip(1.5 * dp.wrap(yaw_hold - th), -0.3, 0.3))
            else:
                w = float(np.clip(1.0 * dp.wrap(math.atan2(e[1], e[0]) - th), -0.3, 0.3)) if np.linalg.norm(e) > 0.3 else 0.0
            if through and fwd <= 0.35:                           # 止まらずに次の目標へ（通り過ぎる点）
                return True
            if k_b is None and fwd <= brake:
                k_b = k
            u = 1.0 if k_b is None else max(0.0, 1.0 - (k - k_b) * DT / 1.0)
            if u <= 0.0:
                break
            vy = 0.0
            if lateral:                                           # 横のずれも直す（0.1 m/秒より遅い横の指令では足が出ないので、0.1 以上で）
                side = -math.sin(th) * e[0] + math.cos(th) * e[1]
                if abs(side) > 0.015:
                    vy = float(np.clip(1.5 * side, -0.15, 0.15))
                    vy = math.copysign(max(abs(vy), 0.1), vy)
            wk.step(np.array([v * u, vy * u, w * u]), carry)
            if not guard():
                return False
        if k_b is not None:
            x, y, th = rb.pose()
            print(f"    歩いて止まった：減速を始めてから進んだ距離の残り {brake:.2f} m に対し、残り {math.cos(th) * (target[0] - x) + math.sin(th) * (target[1] - y):+.3f} m で止まった", flush=True)
        return True

    # 戸口から、配達員の方を向いたまま後ろへ 0.9 m 下がって部屋の中へ入り、時計回り（南回り）に回って西を向く。戸口のまま回ると、
    # 体の前の箱が壁やドアを通り抜ける（箱は骨盤から最大 0.37 m 先まで届く）
    ok = back_up(0.9, v=0.25)
    ok = ok and turn(math.atan2(1.2 - rb.pose()[1], 1.6 - rb.pose()[0]), cw=True)   # その場で西を向く（時計回り）
    if ok:
        dump("C1", wk)
    # 机の正面へは、椅子の北を通って斜めに近づく（椅子の東から西へまっすぐ進むと、椅子に当たる）。机の正面の 0.4 m 手前で止まり、
    # 西を向いてから、まっすぐ前へ進んで机の前で止まる
    wp = np.array([1.6, 1.2])
    ok = ok and walk(wp, through=True)                                      # 西へ部屋を横切る（椅子の北東は止まらずに通る）
    note("C 椅子の北東を通った")
    if ok:
        dump("C2", wk)
    pre = STOP + np.array([0.40, 0.0])
    ok = ok and walk(pre, brake=BRAKE_M.get(args.walk, 0.12))               # 歩きながら机の正面の手前へ向きを変える
    note("C 机の正面の手前に着いた")
    if ok:
        dump("C3", wk)
    for _ in range(int(0.5 / DT)):
        wk.step(np.zeros(3), carry)
    ok = ok and turn(STOP_YAW - PLACE_YAW_BIAS)                             # その場で西（机）を向く（置くあいだに回る分だけ右へ）
    note("C 西を向いた")
    ok = ok and walk(STOP, v=0.2, brake=BRAKE_M.get(args.walk, 0.12) / 2, yaw_hold=STOP_YAW - PLACE_YAW_BIAS, lateral=bool(args.lateral))   # 前へ進んで机の前へ
    note(f"C 机の前（止まる前）：ずれ {np.linalg.norm(np.array(rb.pose()[:2]) - STOP) * 100:.1f} cm")
    # 机の前で止まった後の調整（学習の環境 carry_env.py の段階 7 と同じ決め方）：体が静止する（骨盤の速さ 1 cm/秒未満が 0.3 秒続く。
    # 最長 1.5 秒）まで待ってずれを測る。2 cm 未満なら終わり。上体を寄せて直せる（寄せる量の合計が前後・左右 ±6 cm 以内）なら、両足を
    # 着けたまま 0.5 秒で上体（骨盤）を寄せる。それを超えるときだけ、ずれの方向へ 0.1 m/秒で踏み直す（最長 0.6 秒）。最大 args.adjust 回
    if args.adjust > 0:
        wk.shift_table = ce.shift_table()
    goal_yaw = STOP_YAW - PLACE_YAW_BIAS
    for a_ in range(args.adjust + 1):
        still, k_ = 0, 0
        while True:                                                         # 静止するまで待つ
            wk.step(np.zeros(3), carry)
            k_ += 1
            still = still + 1 if np.linalg.norm(d.qvel[0:2]) < 0.01 else 0
            if (still >= 15 and k_ >= 15) or k_ >= 75:
                break
        x, y, th = rb.pose()
        e = STOP - np.array([x, y])
        fwd, side = math.cos(th) * e[0] + math.sin(th) * e[1], -math.sin(th) * e[0] + math.cos(th) * e[1]
        if a_ >= args.adjust or (math.hypot(fwd, side) < 0.02 and not args.yaw_shift):
            break
        tgt = wk.shift + np.array([fwd, side])
        ey = dp.wrap(goal_yaw - th)
        if math.hypot(fwd, side) < 0.02 and (not args.yaw_shift or abs(ey) < ce.YAW_TOL):
            break
        if np.all(np.abs(tgt) <= ce.SHIFT_MAX + 1e-6):                      # 上体を寄せる（向きも股の向きで直す）
            sh0, y0_ = wk.shift.copy(), wk.shift_yaw
            y1_ = float(np.clip(y0_ + ey, -ce.YAW_SHIFT_MAX, ce.YAW_SHIFT_MAX)) if args.yaw_shift else y0_
            for k in range(25):
                u = sm_((k + 1) / 25)
                wk.shift = sh0 + u * (tgt - sh0)
                wk.shift_yaw = y0_ + u * (y1_ - y0_)
                wk.step(np.zeros(3), carry)
            how = f"上体を寄せた（合計 前後 {wk.shift[0] * 100:+.1f}・左右 {wk.shift[1] * 100:+.1f} cm、向き {math.degrees(wk.shift_yaw):+.1f}°）"
        else:                                                               # 踏み直す（上体も踏み出す方へ一緒に運ぶ）
            vx = math.copysign(0.1, fwd) if abs(fwd) > 0.015 else 0.0
            vy = math.copysign(0.1, side) if abs(side) > 0.015 else 0.0
            sh0 = wk.shift.copy()
            sh1 = np.clip(sh0 + np.clip(0.5 * np.array([fwd, side]), -0.03, 0.03), -ce.SHIFT_MAX, ce.SHIFT_MAX) if args.yaw_shift else sh0
            for k in range(int(round(min(max(abs(fwd), abs(side)) / 0.1, 0.6) / DT))):
                wk.shift = sh0 + sm_((k + 1) / 25) * (sh1 - sh0)
                wk.step(np.array([vx, vy, float(np.clip(1.5 * dp.wrap(goal_yaw - rb.pose()[2]), -0.3, 0.3))]), carry)
            how = "踏み直した（上体も一緒に）"
        note(f"C 調整 {a_ + 1}：前後 {fwd * 100:+.1f} cm・左右 {side * 100:+.1f} cm → {how}、ずれ {np.linalg.norm(np.array(rb.pose()[:2]) - STOP) * 100:.1f} cm")
    for _ in range(int(2.0 / DT)):                                          # 完全に止まる（向きが落ち着くまで）
        wk.step(np.zeros(3), carry)
    # 置く前に向きを合わせ直す（0.3 rad/秒でその場で回る）と、足踏みしながら後ろへ 12〜17 cm 下がったので、合わせ直さない
    note("C 全身を止めて待った")
    yaw_D0 = rb.pose()[2]
    z_D0 = float(d.qpos[2])
    C_ok = bool(ok and not dropped and d.qpos[2] > 0.5 and np.linalg.norm(np.array(rb.pose()[:2]) - STOP) < 0.06)
    note(f"C 机の前で止まった（目標からのずれ {np.linalg.norm(np.array(rb.pose()[:2]) - STOP) * 100:.1f} cm）")
    # ---------------- D
    marks["D"] = rb.t
    if place_walk != args.walk:                              # 置く方策に切り替える（全身を止めている間に）
        wk2 = Walker(rb, place_walk)
        wk2.m_est, wk2.last, wk2.phase, wk2.hooks = wk.m_est, wk.last, wk.phase, wk.hooks
        wk2.shift, wk2.shift_table, wk2.shift_yaw = wk.shift, wk.shift_table, wk.shift_yaw   # 寄せた上体はそのまま
        wk = wk2
    wk.lock_yaw = bool(args.lock_yaw)
    if args.place_tab:
        wk.cfg.place_touch_file = args.place_tab
    feet_b = [m.body(f"{s}_ankle_roll_link").id for s in pe.SIDES]
    feet0 = np.array([d.xpos[b][:2] for b in feet_b])
    fyaw = lambda b: math.atan2(d.xmat[b].reshape(3, 3)[1, 0], d.xmat[b].reshape(3, 3)[0, 0])   # noqa: E731
    feet_yaw0 = [fyaw(b) for b in feet_b]
    if wk.crouch:
        z = np.load(REPO / "models" / wk.cfg.crouch_file)
        traj, H = z["traj"], float(z["H"])
    else:
        traj, H = np.load(REPO / "models" / "place_traj_tray.npy"), 0.0
    touch_mode = wk.crouch and "lean" in z.files
    if getattr(wk.cfg, "place_touch_file", ""):                 # 学習と同じ表と同じ決め方（carry22 から）
        steps = place_with_tab(m, d, wk, marks)
    elif touch_mode and args.place_ik:
        steps = place_with_ik(m, d, wk, z, H, marks)
    elif touch_mode:
        # 前腕を床と平行に保ったまま下ろし、手・前腕・箱が机に触れたら、沈むのをそこで止めて前腕ごと左右へ開く（20 cm は目安）。
        # 腕の動きは学習と同じ表（0〜2 秒：下ろす、2.5〜4.5 秒：開く、4.5〜6.5 秒：起こす）を、触れた時の分だけずらしてなぞる
        desk_g = m.geom("desk").id
        touch_g = {g for g in range(m.ngeom) if m.geom(g).name == "case_geom"
                   or (m.geom_bodyid[g] in {m.body(f"{sd}_{b}").id for sd in pe.SIDES for b in ("elbow_link", "wrist_roll_link", "wrist_pitch_link", "wrist_yaw_link")})}

        def touching():
            """手・前腕・箱が机の上面に触れたか（触れた点が上面の高さで、机の縁より奥）。縁の側面に当たったのは数えない。"""
            for ci in range(d.ncon):
                g1, g2 = d.contact[ci].geom1, d.contact[ci].geom2
                if (g1 == desk_g and g2 in touch_g) or (g2 == desk_g and g1 in touch_g):
                    p_ = d.contact[ci].pos
                    if p_[2] > office.DESK["height"] - 0.008 and p_[0] < DESK_X + DESK_HX - 0.01:
                        return m.geom(g2 if g1 == desk_g else g1).name or m.body(m.geom_bodyid[g2 if g1 == desk_g else g1]).name
            return None
        n_down = int(round(ce.CROUCH_DOWN / DT))
        plan = []                                             # (腕の目標, 沈める量) を 1 歩ずつ
        kc, hc, who = n_down, H, None
        for k in range(n_down):
            h = H * sm(k / n_down)
            wk.step(np.zeros(3), traj[k], h_cmd=h)
            who = touching()
            if who:
                kc, hc = k, h
                break
        marks["touch"] = rb.t
        print(f"    机に触れた：沈み {hc * 100:.1f} cm（目安 {H * 100:.0f} cm）、下ろし始めて {kc * DT:.2f} 秒、触れたのは {who}", flush=True)
        off = traj[kc] - traj[n_down + 25]                   # 触れた時の腕と、表の「開き始め」の腕の差
        for j in range(25):                                  # 0.5 秒そのまま（手の甲を机に当てたまま）
            plan.append((traj[kc], hc))
        for j in range(100):                                 # 2 秒で前腕ごと左右へ開く
            plan.append((traj[n_down + 25 + j] + off, hc))
        for j in range(100):                                 # 2 秒で体を起こす（ずれは少しずつ 0 に）
            u = sm(j / 100)
            plan.append((traj[min(n_down + 125 + j, len(traj) - 1)] + off * (1 - u), hc * (1 - u)))
        for j in range(len(traj) - (n_down + 225) + int(2.0 / DT)):
            plan.append((traj[min(n_down + 225 + j, len(traj) - 1)], 0.0))
        steps = [(k_ + kc + 1, arm_, h_) for k_, (arm_, h_) in enumerate(plan)]
    else:
        steps = []
        for k in range(len(traj) + int(2.0 / DT)):
            tc = k * DT
            h = (H * sm(tc / ce.CROUCH_DOWN) if tc < ce.CROUCH_DOWN else H if tc < ce.CROUCH_HOLD_END
                 else H * (1 - sm((tc - ce.CROUCH_HOLD_END) / (ce.CROUCH_UP_END - ce.CROUCH_HOLD_END))))
            steps.append((k, traj[min(k, len(traj) - 1)], h))
    for k, arm_k, h in steps:
        tc = k * DT
        wk.step(np.zeros(3), arm_k, h_cmd=h)
        if k % 25 == 0:                                      # 確認用：0.5 秒ごとの体と左足の向き
            Rb_ = d.xmat[cb].reshape(3, 3)
            near = d.xpos[cb][0] + max(abs(Rb_[0, 0]) * hx + abs(Rb_[0, 1]) * hy, 0) * 1.0      # 箱の東の端（机の正面の側）
            Rt_ = d.xmat[m.body("torso_link").id].reshape(3, 3)
            Rp_ = d.xmat[m.body("pelvis").id].reshape(3, 3)
            con = set()
            for ci in range(d.ncon):
                n1, n2 = m.geom(d.contact[ci].geom1).name, m.geom(d.contact[ci].geom2).name
                if "desk" in (n1, n2) or "case_geom" in (n1, n2):
                    con.add(f"{n1}-{n2}")
            print(f"      D {tc:4.1f} 秒  当たっている：{sorted(con)}", flush=True)
            print(f"      D {tc:4.1f} 秒  胴の前への傾き {math.degrees(math.asin(np.clip(-Rt_[2, 0], -1, 1))):+5.1f}°  骨盤 {math.degrees(math.asin(np.clip(-Rp_[2, 0], -1, 1))):+5.1f}°"
                  f"  腰の pitch {math.degrees(d.qpos[7 + 14] - wk.default[14]):+5.1f}°", flush=True)
            print(f"      D {tc:4.1f} 秒  体の向き {math.degrees(dp.wrap(rb.pose()[2] - yaw_D0)):+5.1f}°  左足 {math.degrees(dp.wrap(fyaw(feet_b[0]) - feet_yaw0[0])):+5.1f}°  沈み 指令 {h:.2f}・実際 {z_D0 - d.qpos[2]:.3f}"
                  f"  箱の底 {d.xpos[cb][2] - hz:.3f}  箱の東の端が机の縁から {((DESK_X + DESK_HX) - near) * 100:+5.1f} cm 内側  箱の向き {math.degrees(math.atan2(Rb_[1, 0], Rb_[0, 0])):+6.1f}°", flush=True)
    foot_move = float(np.max(np.linalg.norm(np.array([d.xpos[b][:2] for b in feet_b]) - feet0, axis=1)))
    R = d.xmat[cb].reshape(3, 3)
    pos = d.xpos[cb]
    bottom = pos[2] - abs(R[2, 2]) * hz - abs(R[2, 0]) * hx - abs(R[2, 1]) * hy
    tilt = math.degrees(math.acos(float(np.clip(R[2, 2], -1, 1))))
    corners = [pos[:2] + R[:2, :2] @ np.array([sx * hx, sy * hy]) for sx in (-1, 1) for sy in (-1, 1)]
    inside = min(min(c[0] - (DESK_X - DESK_HX), (DESK_X + DESK_HX) - c[0], c[1] - (DESK_Y - DESK_HY), (DESK_Y + DESK_HY) - c[1]) for c in corners)
    D_ok = bool(abs(bottom - office.DESK["height"]) < 0.01 and tilt < 3 and inside > 0 and foot_move < 0.02 and d.qpos[2] > 0.5)
    print("    D のあいだに足の向きが回った量：" + "、".join(f"{sd} {math.degrees(dp.wrap(fyaw(b) - y0)):+.1f}°" for sd, b, y0 in zip(pe.SIDES, feet_b, feet_yaw0)), flush=True)
    print(f"    D のあいだに体が回った量：{math.degrees(dp.wrap(rb.pose()[2] - yaw_D0)):+.1f}°（左回りが +）", flush=True)
    note(f"D 置いた（箱の底 {bottom:.3f}、傾き {tilt:.1f}°、机の縁から {inside * 100:.1f} cm 内側、足の動き {foot_move * 100:.1f} cm）")
    res = dict(A=bool(A_ok), B=B_ok, C=C_ok, D=D_ok, 通し=bool(A_ok and B_ok and C_ok and D_ok), walk=args.walk,
               place_walk=place_walk, seed=args.seed, cond={k: round(float(v), 4) for k, v in cond.items()}, door_closed_deg=round(math.degrees(float(d.qpos[rb.door_q])), 1),
               stop_err_cm=round(float(np.linalg.norm(np.array(rb.pose()[:2]) - STOP)) * 100, 1), foot_move_cm=round(foot_move * 100, 2),
               box_inside_cm=round(inside * 100, 1), box_tilt_deg=round(tilt, 1),
               box_xy=[round(float(pos[0]), 3), round(float(pos[1]), 3)], box_yaw_deg=round(math.degrees(math.atan2(R[1, 0], R[0, 0])), 1),
               robot_yaw_deg=round(math.degrees(rb.pose()[2]), 1),
               edges_cm={k: round(float(v) * 100, 1) for k, v in dict(
                   west=min(c[0] for c in corners) - (DESK_X - DESK_HX), east=(DESK_X + DESK_HX) - max(c[0] for c in corners),
                   south=min(c[1] for c in corners) - (DESK_Y - DESK_HY), north=(DESK_Y + DESK_HY) - max(c[1] for c in corners)).items()},
               new_contacts={k: [v[0], round(v[1], 1)] for k, v in sorted(new_hits.items(), key=lambda kv: -kv[1][0])})
    print("新しく足した組で当たったもの（歩の数、最初の時刻）：", res["new_contacts"] or "なし", flush=True)
    # 確かめ：足踏み（体はほとんど動かないのに足を上げている）の時間。0.5 秒ごとに、骨盤の水平の速さが 0.05 m/秒より遅く、
    # どちらかの足首が一番低い時より 2 cm 以上上がっていれば足踏み
    dk = mujoco.MjData(m)
    ank = [m.body(f"{sd}_ankle_roll_link").id for sd in pe.SIDES]
    zs, ps = [], []
    for i in range(len(rb.frames)):
        dk.qpos[:] = rb.frames[i]
        mujoco.mj_kinematics(m, dk)
        zs.append([dk.xpos[b][2] for b in ank])
        ps.append(dk.qpos[:2].copy())
    zs, ps = np.array(zs), np.array(ps)
    zmin = zs.min(axis=0)
    W = int(0.5 / DT)
    stepping = []
    for i in range(0, len(zs) - W, W):
        v = np.linalg.norm(ps[i + W] - ps[i]) / (W * DT)
        lift = (zs[i:i + W] - zmin).max() > 0.02
        stepping.append(bool(v < 0.05 and lift))
    segs, start = [], None
    for k, st in enumerate(stepping + [False]):
        if st and start is None:
            start = k
        if not st and start is not None:
            segs.append((start * W * DT, k * W * DT))
            start = None
    print("足踏みの区間（秒）：" + "、".join(f"{a:.1f}〜{b:.1f}" for a, b in segs) + f"　合計 {sum(b - a for a, b in segs):.1f} 秒", flush=True)
    print("区切りの時刻：", {k: round(v, 1) for k, v in marks.items()}, flush=True)
    # 確かめ：箱が壁・ドアを通り抜けていないか（箱と壁・ドアの形の間の距離の最小。負なら重なっている）
    dd = mujoco.MjData(m)
    walls = [g for g in range(m.ngeom) if m.geom_bodyid[g] in (0, m.body("door_body").id) and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX
             and m.geom_size[g][2] > 0.5]
    cg = m.geom("case_geom").id
    worst = (1.0, None, None)
    for i in range(0, len(rb.frames), 5):
        dd.qpos[:] = rb.frames[i]
        mujoco.mj_kinematics(m, dd)
        for g in walls:
            dist = mujoco.mj_geomDistance(m, dd, cg, g, 0.2, None)
            if dist < worst[0]:
                worst = (dist, i * DT, (m.geom(g).name or f"{g}（{np.round(m.geom_pos[g][:2], 2)}）")
                         + f"、箱 {np.round(dd.xpos[cb][:2], 2)}・ロボット {np.round(dd.qpos[:2], 2)}")
    print(f"箱と壁・ドアの距離の最小：{worst[0] * 100:.1f} cm（{worst[1]} 秒、{worst[2]}）")
    res["wall_clear_cm"] = round(worst[0] * 100, 1)
    print(f"配達員の手が届かなかった距離の最大：{cour.short * 100:.1f} cm（{cour.short_at}、記録の歩 {len(rb.frames) - len(rb.mframes)}）",
          {k: round(v, 1) for k, v in marks.items()})
    print(json.dumps(res, ensure_ascii=False))
    if args.json:
        with open(args.json, "a") as f:
            f.write(json.dumps(res, ensure_ascii=False) + "\n")
    if args.events:
        ev = dict(knock=args.go, hai=args.go + 0.9, courier=marks["courier_says"], thanks=marks.get("release", marks["C"]) + 0.6,
                  door_close=marks["door_close"], end=rb.t)
        Path(args.events).write_text(json.dumps(ev, indent=1))
    if args.stills:                                          # 受け渡しを近くから（確認用）
        Path(args.stills).mkdir(parents=True, exist_ok=True)
        tB = marks["B"]
        cams = {"side": ((5.45, 0.82, 1.0), 1.4, 230.0, -10.0), "top": ((5.45, 0.82, 0.95), 1.0, 230.0, -60.0),
                "wide": ((5.2, 0.9, 0.9), 3.0, 200.0, -35.0), "door": ((5.45, 1.15, 1.0), 3.4, 250.0, -14.0)}
        for name, (la, dist, az, el) in cams.items():
            for ts in (3.0, 5.0, 6.5, 7.5, 8.5, 9.5, 10.5, 12.0):
                if name == "door":
                    ts = marks["door_close"] - tB + ts - 2.0
                cam = mujoco.MjvCamera()
                cam.lookat[:], cam.distance, cam.azimuth, cam.elevation = la, dist, az, el
                render_still(rb, f"{args.stills}/{name}_{ts:04.1f}.png", tB + ts, cam)
    if args.stills:                                          # 置くところを真横（北）から、0.5 秒ごと
        for k in range(18):
            cam = mujoco.MjvCamera()
            cam.lookat[:], cam.distance, cam.azimuth, cam.elevation = (STOP[0] - 0.2, STOP[1], 0.75), 1.6, -90.0, -5.0
            render_still(rb, f"{args.stills}/dside_{k:02d}.png", marks["D"] + 0.5 * k, cam)
            cam = mujoco.MjvCamera()                         # 真上から（手が箱の下から抜けたか）
            cam.lookat[:], cam.distance, cam.azimuth, cam.elevation = (STOP[0] - 0.25, STOP[1], 0.75), 1.1, 180.0, -85.0
            render_still(rb, f"{args.stills}/dtop_{k:02d}.png", marks["D"] + 0.5 * k, cam)
    if args.stills:                                          # 置いた箱を、机の正面（椅子の側）の上から
        for nm, (az, el, dist) in {"desk_end": (180.0, -55.0, 0.9), "desk_end_wide": (200.0, -35.0, 2.0), "desk_top": (180.0, -89.0, 2.6)}.items():
            cam = mujoco.MjvCamera()
            cam.lookat[:], cam.distance, cam.azimuth, cam.elevation = d.xpos[cb], dist, az, el
            render_still(rb, f"{args.stills}/{nm}.png", (len(rb.frames) - 1) * DT, cam)
    if args.video:
        tB, tC, tD = marks["B"], marks["C"], marks["D"]

        t_end = rb.t

        def cam_fn(q, t):
            c = mujoco.MjvCamera()
            tc = marks["door_close"]
            if t > t_end + 0.5:                                            # 最後：置いた箱を真上から（「MAC 13」が読める向き）
                c.lookat[:] = (q[cq], q[cq + 1], 0.75)
                c.distance, c.azimuth, c.elevation = 1.1, 180.0, -89.0
                return c
            if t < marks["door"] - 1.0:                                    # A：部屋全体を斜め上から
                c.lookat[:] = (2.0, 0.1, 0.3)
                c.distance, c.azimuth, c.elevation = 7.0, 15.0, -40.0
            elif t < tB:                                                   # A：部屋の中から、ドアを押すところ
                c.lookat[:] = (4.9, 1.0, 0.9)
                c.distance, c.azimuth, c.elevation = 2.8, 30.0, -18.0
            elif t < tC:                                                   # B：廊下の南から、戸口での受け渡しを横から
                c.lookat[:] = (5.45, 0.8, 1.0)
                c.distance, c.azimuth, c.elevation = 2.6, 100.0, -12.0
            elif tc + 3.0 <= t < tc + 10.5:                                # 配達員がドアの外へ回って押して閉めるところ（廊下の北東から）
                c.lookat[:] = (5.45, 1.15, 1.0)
                c.distance, c.azimuth, c.elevation = 3.4, 250.0, -14.0
            elif t < tD - 1.0:                                             # C：ロボットを斜め上から追う
                c.lookat[:] = (q[0], q[1], 0.6)
                c.distance, c.azimuth, c.elevation = 3.5, 60.0, -30.0
            else:                                                          # D：机の北東から（ロボットの左斜め後ろ）
                c.lookat[:] = (STOP[0] - 0.25, STOP[1], 0.8)
                c.distance, c.azimuth, c.elevation = 2.0, 240.0, -22.0
            return c
        render(rb, args.video, cam_fn, hold=3.5, room_fn=lambda t: tD - 1.0 <= t <= t_end + 0.5)   # D の場面（机の北東から）
        print(args.video)
    return 0


if __name__ == "__main__":
    sys.exit(main())
