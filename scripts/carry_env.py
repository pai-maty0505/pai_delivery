#!/usr/bin/env python3
"""
B 歩く方策（作り直し）の強化学習の環境：本物のノート PC のケース（40 × 30 × 6 cm、1.5〜2.5 kg）を両手の 4 本の指に
載せて、落とさずに歩く（MuJoCo Playground の G1 を土台にする。scripts/payload_env.py を引き継ぐ）。

- 手：両手の手首と 4 本の指に、ケースを支える面の当たり判定（水平な薄い板）を付ける（片手 2 か所、前後に約 24 cm の範囲で支える）
- ケース：自由に動く箱。指の面との摩擦だけで載る。胴（の当たり判定）にも当たる
- 練習の種類（エピソードごとに選ぶ）。受け取るは、配達員の左右の手がケースの側面を挟んで差し出し、受け取る時刻に下げて離す
    最初から持つ／途中で受け取る（配達員から）／途中で降ろす（机に置く）／手ぶら
  受け取る・降ろすは、ケースの下の見えない台（モーキャップの体。ケースとだけ当たる）で作る：
    受け取る前は、台がケースを指の 1 cm 上に支え、ロボットの手に付いて動く。受け取る時刻に台を下げると、ケースが指に載る
    降ろす時刻に台を下から上げると、ケースが指から持ち上がる（その後は手から離れた所で支える）
    手ぶらの練習では、ケースは離れた所の台の上にある
- ドアを押す反動：ときどき両手に後ろ向きの力（5〜25 N、1〜2 秒）をかける
- ケースを落とす（ケースが指の面より 25 cm 以上下がる、または手から 40 cm 以上離れる）と、練習をやり直す（大きな罰）
- 方策の入力：③（腕で測った重さで学習）と同じ形（62）。③ の重みから学習を始められる
- 1 回の練習を打ち切って初めに戻るときも、時間（受け取る・降ろす時刻を数える）を数え直す
  （payload_env では倒れたときしか数え直していなかった）

ロボットの qpos は先頭 36（体 7＋関節 29）、その後ろにケースの 7。
"""

from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
import mujoco
import numpy as np
from pathlib import Path
from ml_collections import config_dict
from mujoco import mjx
from mujoco.mjx._src import math
from mujoco_playground._src import mjx_env
from mujoco_playground._src.locomotion.g1 import base as g1_base
from mujoco_playground._src.locomotion.g1 import g1_constants as consts
from mujoco_playground._src.locomotion.g1 import joystick

import payload_env as pe

N_ACT = pe.N_ACT
NQ_ROBOT, NV_ROBOT = 36, 35
CASE_HALF = (0.125, 0.165, 0.025)       # 箱の半分の大きさ：MacBook Air 13 インチの純正箱（前後 25・左右 33・厚さ 5 cm、1.23 kg）
# 手でケースを支える面の当たり判定（水平な薄い板）。片手に 2 か所：後ろ ＝ 手首（wrist_pitch_link、運ぶ姿勢で x 0.13〜0.23 m、
# 上面 0.866 m）、前 ＝ 4 本の指（wrist_yaw_link、x 0.28〜0.38 m、上面 0.867 m）。手のひらは 2 cm くぼみ、前腕は 1.2 cm 高い
# ので使わない。傾いた板では平らなケースの底が板の縁（線）にしか触れず跳ねた。位置と向きは pad_frames() で腕のメッシュから求める
PADS = {                                  # 名前：(付ける体、メッシュ、前後の範囲（世界の x）、左右の幅の半分、厚さの半分)
    "finger": ("wrist_yaw_link", "rubber_hand", (0.275, 0.38), 0.045, 0.008),
    "wrist": ("wrist_pitch_link", "wrist_pitch_link", (0.13, 0.232), 0.030, 0.008),
}
CASE_X = 0.29                             # ケースの中心の前後（運ぶ姿勢の世界の x、骨盤が 0）。前後 0.14〜0.44 m

# 持ち方 "arms"（腕も使う。② 2026-10-16）：肘を 0.08 rad 上げ（−0.08）、前腕・手首・指の 3 か所でケースを受ける。
# 前腕（x 0.10 m）から指先（x 0.36 m）まで、指先側が 1.4 cm 高い面（胸側が低い、約 3°）になるので、支えの面もこの傾きに
# そろえる。ケースの後ろの縁は胸の前面（x 0.088 m）に当て、重さで胸へ寄って胸で止まる
ARMS_GRIP = dict(
    elbow=-0.08,
    plane=((0.10, 0.886), (0.36, 0.900)),  # 支えの面が通る 2 点（世界の x, 高さ）
    pads={"forearm": ("elbow_link", (0.085, 0.125), 0.030, 0.008),
          "wrist": ("wrist_pitch_link", (0.14, 0.232), 0.030, 0.008),
          "finger": ("wrist_yaw_link", (0.28, 0.38), 0.045, 0.008)},
    chest_front=0.088,                     # 胸の前面（世界の x、高さ 0.85〜0.95 m）
    case_x=0.088 + CASE_HALF[0] + 0.004,
)
# 持ち方 "tray"（2026-10-16）：手のひらを上に向け、前腕・手首・手でケースを下から支え、手の外側を少し上げて、指先をケースの
# 側面の外側に沿って立てる（横の止め）。人が箱を下から受け取る形。G1 の指は曲がらないので、指先を底へ回り込ませる持ち方は
# できなかった。腕の姿勢は、支える面（ケースの底、水平）と指先の止めが両方できるように探した（logs/tray_pose_all.png）
TRAY_GRIP = dict(
    arm=dict(shoulder_pitch=-0.2937, shoulder_roll=-0.1622, shoulder_yaw=-0.1679, elbow=0.2008, wrist_roll=-1.1442, wrist_pitch=-1.0414, wrist_yaw=0.0375),
    bodies=("elbow_link", "wrist_roll_link", "wrist_pitch_link", "wrist_yaw_link"),
    case_x=0.088 + CASE_HALF[0] + 0.004,   # ケースの後ろの縁を胸の前面のすぐ前に
    reach=0.008,                           # 底から下へこの深さまでの腕の点を「支える面」にする
    tilt_deg=5.0,                          # 箱を胸側へ下げる角度。歩くとき体が前へ傾いても箱が前へ滑らない
)
TRAY_BOTTOM = None                        # pad_frames が求めるケースの底の高さ（運ぶ姿勢の世界の z）
GRIP = "hands"                            # CarryJoystick が config の grip で書き換える（decorate が使う）
DESK_BODY = False                         # 置く練習の机・椅子・当たり判定・摩擦を事務所の場面と同じにする（config の desk_body で書き換える。carry22 から）
# 机に触れたかを調べる円柱（体の座標の端 2 つ、半径。当たり判定ではなく、調べるためだけ。事務所の場面でも同じ決め方で調べる）。
# 事務所の場面の手のメッシュ（手首の yaw の体で x −0.018〜0.173、手のひらの向き y −0.045〜0.03（右は逆）、z −0.043〜0.063）と、
# 手首（roll から yaw まで 8.4 cm）に合わせた
HAND_CAPS = {"wrist_yaw_link": ((0.0, -0.0075, 0.01), (0.135, -0.0075, 0.01), 0.0375),
             "wrist_roll_link": ((0.0, 0.0, 0.0), (0.084, 0.0, 0.0), 0.03)}
# 配達員の左右の手（受け取る練習）：ケースの左右の側面の前寄りを挟む薄い板。ケースの下を空けておくので、ロボットの前腕・手首・指が
# ケースの下の正しい位置に入る（2026-10-16 まで、ケースの真下から台で支えていた）
GRIP_HALF = (0.05, 0.01, 0.03)           # 前後・厚さ・高さの半分
GRIP_X = 0.06                             # ケースの中心から前へ
GRIP_SQUEEZE = 0.0005                     # 側面に押し込む深さ（左右に動かないように軽く押さえる）
# 指：ケースの底の左右の縁の下にかける（側面を押さえる摩擦だけでは、ケースが毎秒約 2.4 cm ずり落ちた。前の角だけでは
# 後ろが回って下がった）。前後 2 か所：ロボットの手のひらのくぼみの上（支えの面のない所）と、前の角
LEDGE_HALF = (0.02, 0.015, 0.005)         # 前後（下の LEDGES で決める）・左右・厚さの半分
LEDGES = ((0.0, 0.02), (0.117, 0.008))  # （ケースの中心から前への位置, 前後の長さの半分）
GRIP_LOWER, GRIP_LOWER_T, GRIP_OPEN, GRIP_OPEN_T = 0.012, 0.2, 0.04, 0.3   # 受け取る時刻から：0.2 秒で 1.2 cm 下げ、0.3 秒で 4 cm 開く
FAR = jp.array([0.0, 0.0, -5.0])        # 使わないときの台の場所（床の下、ケースとも離れる）


def jnp_atan2v(v):
    return jp.arctan2(v[1], v[0])


def wrap_j(a):
    return jp.arctan2(jp.sin(a), jp.cos(a))


def jnp_far2():
    return jp.stack([FAR + jp.array([0.0, 1.0, 0.0]), FAR - jp.array([0.0, 1.0, 0.0])])

MODES = ("carry", "receive", "release", "empty", "place")   # 最初から持つ／受け取る／降ろす／手ぶら／止まって机に置く
STAND_PHASE = (0.0, float(np.pi))           # 止まっている間の歩く周期の時計（決まった値に止める）
PLACE_EDGE = 0.13                         # 置く練習：止まった骨盤から机の手前の縁まで（m）
DESK_TOP = 0.74                           # 机の上面の高さ（事務所の場面と同じ）
DESK_HALF = (0.35, 0.6, 0.015)            # 机の板の半分の大きさ（奥行き 70・幅 120・厚さ 3 cm）
# 置き方（2026-10-16 18:30）：机の前で少しずつ遅くして完全に止まり、体を沈め（腰を下げる）、箱を前へ出して 15° 傾け、奥の縁と
# 指先から机に当て、手と腕を左右へ抜き、体を起こす。時刻は腕を動かし始めてから（秒）。腕の動きと沈める量は
# models/place_traj_tray_tilt_crouch.npz（体を固定した試作で決めた：沈める 6 cm、腕で 7.4 cm 下げ、前へ 12 cm、15°）
CROUCH_DOWN, CROUCH_HOLD_END, CROUCH_UP_END = 2.0, 4.5, 6.5
PLACE_RELEASE = 4.0                       # 箱が手から机へ移り終わる時刻
DECEL_T = 1.0                             # 止まる前に指令を少しずつ 0 へ下げる時間（急に止まると倒れやすかった）


def default_config() -> config_dict.ConfigDict:
    cfg = pe.default_config()
    cfg.obs_mode = "estimate"
    cfg.crouch_obs = False                    # 方策の入力に「腰を下げる量」を入れる（体を沈めて置く。carry10 から）
    cfg.crouch_file = "place_traj_tray_tilt_crouch.npz"   # 体を沈めて置く腕の動きと沈める量（carry10：6 cm、carry11：8 cm）
    cfg.crouch_lean = 0.0                     # 沈める量に合わせて股関節から上体を前へ倒す角度（度、沈める量 H のとき。carry19 から）
    cfg.crouch_waist = False                  # しゃがむ姿勢に腰の pitch を前へ足す（carry18 から）
    cfg.crouch_ff = False                     # 沈める量に合わせたしゃがむ姿勢（股・膝・足首）を脚の目標に先に足す（carry13 から）
    cfg.place_touch_file = ""                 # 机に触れた時刻ごとの、置く腕の動きと沈める量の表（scripts/crouch_place_touch.py）。
    #                                           あれば事務所と同じく「下ろしていて机に触れたら、その表に切り替えて開く」（carry22 から）
    cfg.office = False                        # 事務所の場面で学習する（机・椅子・壁は最初からその場所。置く練習は C の道順の途中から、
    #                                           事務所と同じ歩き方の指令で机の前まで歩いて止まり、置く。carry24 から）
    cfg.yaw_shift = 0                         # 上体を寄せるときに、股の向きで骨盤の向きも直す（1。carry29 から）
    cfg.adjust = 0                            # 机の前で止まった後の、小さな踏み直しの最大の回数（full_run.py の --adjust。carry27 から）
    cfg.stop_err_once = False                 # 止まる位置のずれを、置き始める時の一度だけ数える（carry26 から）
    cfg.office_flow = False                   # 事務所の場面で、A の続き（戸口でドアを開けた所）から B〜D を練習する。練習は事務所で通した
    #                                           記録の状態（A の終わり・C の始め・C の道順の途中）から始める（carry25 から）
    cfg.flow_file = "flow/flow_states.jsonl"
    cfg.flow_p = [0.4, 0.15, 0.15, 0.15, 0.15]  # 始める所：A の終わり（B から）・C の始め・C の道順（3 か所）の割合
    cfg.drift_all = False                     # 止まってからの流れの減点を、置き終わるまで数える（carry23 から）
    cfg.desk_body = False                     # 置く練習の机に手・手首・胴も当てる（carry22 から）
    cfg.approach = False                      # 置く練習で、止まる前に事務所と同じ近づき方（止まる → その場で向きを変える → ゆっくり前へ
    #                                           進む（横へ直すこともある）→ 止まる）をする（carry22 から）
    cfg.grip = "hands"                        # ケースの持ち方："hands"（手首と指）、"arms"（前腕・手首・指＋胸。②）
    cfg.njmax = 192
    cfg.solver_iterations = 10
    cfg.solver_ls_iterations = 10
    cfg.arm_damping = 2.0                      # 腕の関節のダンピング（元の G1 は 2）
    cfg.naconmax = 24 * 8192                  # 配達員の手・下の支えを足して 16 ではあふれた
    cfg.lin_vel_x = [-0.3, 0.6]               # ケースを持って歩く速さ（後ろ 0.3・前 0.6 m/秒）
    cfg.lin_vel_y = [-0.2, 0.2]
    cfg.ang_vel_yaw = [-0.8, 0.8]
    cfg.push_config.magnitude_range = [0.1, 1.0]
    cfg.case = config_dict.create(
        mass_range=[1.23, 1.23],               # 荷物は MacBook Air 13 インチの純正箱に固定（1.23 kg）
        friction_range=[0.8, 1.2],             # ケースと手の摩擦（ゴムの指と段ボールで約 1.0）。0.6〜1.0 では 95% に届かなかった
        mode_p=[0.25, 0.3, 0.1, 0.1, 0.25],    # 最初から持つ／受け取る／降ろす／手ぶら／止まって机に置く の割合
        stop_range=[1.0, 4.0],                 # 置く練習：歩いてから止まる時刻（秒）
        settle_range=[1.0, 1.5],               # 置く練習：止まってから腕を動かし始めるまで（秒）。全身を止めてから下ろす
        desk_offset=0.03,                      # 置く練習：机の位置のずれ（前後・左右、m）
        desk_dz=0.0,                           # 置く練習：机の高さのずれ（m）
        place_edge=0.13,                       # 置く練習：止まった骨盤から机の手前の縁まで（m。事務所の場面は 0.11）
        desk_fwd=[0.0, 0.06],                  # desk_body：机の手前の縁までのずれ（m。事務所で止まる位置のずれ：手前で止まることが多い。
        #                                        近い方へは出さない：carry22 は −2 cm からで、出た時に胴が机の縁に重なることがあった）
        place_y=0.36,                          # desk_body：机の天板の中心がロボットの左へ（m。事務所の場面の PLACE_Y）
        desk_yaw=0.07,                         # desk_body：机の向きのずれ（rad、±4°）
        onset_range=[1.0, 10.0],               # 受け取る時刻（秒）
        release_range=[3.0, 15.0],             # 降ろす時刻（秒）
        contact_solref=[0.005, 1.0],            # 箱の接触の固さ（時定数・減衰比）。0.02 では摩擦の方向もたわみ、揺れのたびに箱が少しずつずれて落ちた
        support_gap=0.005,                     # 受け取る前の、台が支えるケースと指の間のすき間
        drop_speed=0.3,                        # 受け取るとき台を下げる速さ（m/秒）。1.0 では乱暴で、受け取る場面の 6 割で落とした
        lift=0.06,                             # 降ろすとき台がケースを持ち上げる高さ
        hand_push=config_dict.create(p=0.3, force_range=[5.0, 25.0], time_range=[2.0, 15.0], dur_range=[1.0, 2.0]),
    )
    rs = cfg.reward_config.scales
    rs.case_drop = -100.0                      # ケースを落とす（練習をやり直す）
    rs.case_slip = -3.0                        # ケースが手の上でずれる
    # 止まっている間（指令 0）：足が床の上で動く速さ、床から離れている足の数、体が動く速さ（全身を止める）
    rs.stand_feet_vel = -4.0
    rs.stand_feet_air = -2.0
    rs.stand_base_vel = -2.0
    rs.place_yaw = 0.0                         # 置く練習で腕を動かし始めてから、体の向きがその時の向きから回った量（rad）。carry13 は
    #                                           置くあいだに約 12° 回り、前腕が箱の角を押して箱を回し、手前へ引きずった
    rs.stop_err = 0.0                          # 事務所：止まってから置き終わるまでの、止まる位置の目標とのずれ（m）
    rs.stop_drift = 0.0                        # 置く練習で止まってから（指令 0 になってから）、骨盤が止まった所から動いた距離（m）。carry21 は
    #                                           止まってから 2 秒で左へ 4〜5 cm 流れ、事務所の机の前で止まる位置が 6 cm 以上ずれた
    rs.crouch_height = -20.0                   # 体を沈めている間（carry11 は −40：carry10 は指示 6 cm に 4.5 cm しか沈まなかった）、腰の高さが指示（止まった時の高さ − 下げる量）からずれた量（m）
    return cfg


def _carry_state():
    """運ぶ姿勢の model と data（当たり判定を付ける前）。"""
    m0 = pe.build_model()
    d0 = mujoco.MjData(m0)
    d0.qpos[:] = m0.keyframe("knees_bent").qpos
    arm = pe.carry_pose()
    for i, (s, j) in enumerate((s, j) for s in pe.SIDES for j in pe.ARM_JOINTS):
        d0.qpos[m0.joint(f"{s}_{j}_joint").qposadr[0]] = arm[i]
    mujoco.mj_forward(m0, d0)
    return m0, d0


def pad_frames():
    """運ぶ姿勢で、ケースを支える面の当たり判定（薄い板）の位置と向きを、付ける体の座標で求める。
    "hands"：腕のメッシュの上面の高さ（90%）に水平な板。"arms"：ARMS_GRIP の傾いた面に上面をそろえた板。
    返り値：{(手, 部分): (付ける体の名前, 体の座標での位置, 向きの四元数, 大きさの半分)}。"""
    m0, d0 = _carry_state()
    out = {}
    if GRIP == "tray":
        return _tray_frames(m0, d0)
    if GRIP == "arms":
        (xa, za), (xb, zb) = ARMS_GRIP["plane"]
        slope = (zb - za) / (xb - xa)
        ang = np.arctan(slope)
        Rw = np.array([[np.cos(ang), 0, -np.sin(ang)], [0, 1, 0], [np.sin(ang), 0, np.cos(ang)]])   # x を前上がりに
        for s in pe.SIDES:
            sg = 1.0 if s == "left" else -1.0
            for part, (body, (x0, x1), half_w, half_t) in ARMS_GRIP["pads"].items():
                b = m0.body(f"{s}_{body}").id
                Rb, pb = d0.xmat[b].reshape(3, 3), d0.xpos[b]
                xc = (x0 + x1) / 2
                top = za + slope * (xc - xa)
                center = np.array([xc, sg * 0.168, top]) - Rw[:, 2] * half_t
                quat = np.zeros(4)
                mujoco.mju_mat2Quat(quat, (Rb.T @ Rw).copy().ravel())
                out[(s, part)] = (f"{s}_{body}", Rb.T @ (center - pb), quat, ((x1 - x0) / 2, half_w, half_t))
        return out
    for s in pe.SIDES:
        for part, (body, mesh, (x0, x1), half_w, half_t) in PADS.items():
            mid = m0.mesh(f"{s}_{mesh}").id
            gid = [g for g in range(m0.ngeom) if m0.geom_dataid[g] == mid and m0.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH][0]
            V = m0.mesh_vert[m0.mesh_vertadr[mid]:m0.mesh_vertadr[mid] + m0.mesh_vertnum[mid]]
            W = d0.geom_xpos[gid] + V @ d0.geom_xmat[gid].reshape(3, 3).T
            W = W[(W[:, 0] >= x0) & (W[:, 0] <= x1)]
            b = m0.body(f"{s}_{body}").id
            Rb, pb = d0.xmat[b].reshape(3, 3), d0.xpos[b]
            center = np.array([(x0 + x1) / 2, W[:, 1].mean(), np.percentile(W[:, 2], 90) - half_t])
            local = Rb.T @ (center - pb)
            quat = np.zeros(4)
            mujoco.mju_mat2Quat(quat, Rb.T.copy().ravel())      # 世界の向き（単位行列）を体の座標で表す
            out[(s, part)] = (f"{s}_{body}", local, quat, ((x1 - x0) / 2, half_w, half_t))
    return out


def _mesh_world(m0, d0, body):
    out = []
    for g in range(m0.ngeom):
        if m0.geom_bodyid[g] == m0.body(body).id and m0.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
            mid = m0.geom_dataid[g]
            V = m0.mesh_vert[m0.mesh_vertadr[mid]:m0.mesh_vertadr[mid] + m0.mesh_vertnum[mid]]
            out.append(d0.geom_xpos[g] + V @ d0.geom_xmat[g].reshape(3, 3).T)
    return np.vstack(out)


TILT_Z = 0.90                             # ③ の箱を傾ける中心の高さ（箱の底のあたり）


def _tilt_R(a):
    """前が a（rad）だけ高くなる向き（y 軸まわり）。列が箱の x・y・z 軸。"""
    return np.array([[np.cos(a), 0, -np.sin(a)], [0, 1, 0], [np.sin(a), 0, np.cos(a)]])


def _tray_frames(m0, d0):
    """持ち方 "tray"（③）の当たり判定。箱は前が tilt だけ高い向き（胸側が低く、ずれても胸の方へ戻る）。
    箱の座標で、箱の下に入る腕の点の最も高い所を箱の底にする。支える面：底から reach の深さまでの点を覆う板（体ごと、箱と同じ
    向き）。横の止め：底より上に出た手の点（箱の側面の外）を覆う板（内側の面を側面に合わせる）。"""
    global TRAY_BOTTOM
    cx = TRAY_GRIP["case_x"]
    Rt = _tilt_R(np.radians(TRAY_GRIP.get("tilt_deg", 0.0)))
    c0 = np.array([cx, 0.0, TILT_Z])
    local = lambda W: (W - c0) @ Rt                            # 箱の座標（原点は箱の中心の真下、高さ TILT_Z）
    inside = lambda L: L[(np.abs(L[:, 0]) < CASE_HALF[0]) & (np.abs(L[:, 1]) < CASE_HALF[1])]
    pts = {(s, b): local(_mesh_world(m0, d0, f"{s}_{b}")) for s in pe.SIDES for b in TRAY_GRIP["bodies"]}
    bot = max(inside(L)[:, 2].max() for L in pts.values() if len(inside(L)))
    TRAY_BOTTOM = float(bot)
    out = {}
    for (s, b), L in pts.items():
        sg = 1.0 if s == "left" else -1.0
        body = m0.body(f"{s}_{b}").id
        Rb, pb = d0.xmat[body].reshape(3, 3), d0.xpos[body]
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, (Rb.T @ Rt).copy().ravel())   # 箱と同じ向きの板
        U = inside(L)
        U = U[U[:, 2] > bot - TRAY_GRIP["reach"]]
        if len(U) >= 10:
            half = np.array([max(np.ptp(U[:, 0]) / 2, 0.01), max(np.ptp(U[:, 1]) / 2, 0.01), 0.008])
            cl = np.array([(U[:, 0].min() + U[:, 0].max()) / 2, (U[:, 1].min() + U[:, 1].max()) / 2, bot - half[2]])
            out[(s, f"{b}_up")] = (f"{s}_{b}", Rb.T @ (c0 + Rt @ cl - pb), quat.copy(), tuple(half))
        if b == "wrist_yaw_link":
            S = L[(np.abs(L[:, 0]) < CASE_HALF[0]) & (sg * L[:, 1] >= CASE_HALF[1]) & (L[:, 2] > bot + 0.004)]
            if len(S) >= 10:
                top = min(S[:, 2].max(), bot + 2 * CASE_HALF[2])
                half = np.array([np.ptp(S[:, 0]) / 2, 0.004, (top - bot) / 2])
                cl = np.array([(S[:, 0].min() + S[:, 0].max()) / 2, sg * (CASE_HALF[1] + 0.0005 + half[1]), (top + bot) / 2])
                out[(s, "side_stop")] = (f"{s}_{b}", Rb.T @ (c0 + Rt @ cl - pb), quat.copy(), tuple(half))
    return out


def tray_case_pose():
    """③ の箱の中心（運ぶ姿勢の世界の座標）と傾き（rad）。pad_frames のあとで使う。"""
    a = np.radians(TRAY_GRIP.get("tilt_deg", 0.0))
    c = np.array([TRAY_GRIP["case_x"], 0.0, TILT_Z]) + _tilt_R(a) @ np.array([0.0, 0.0, TRAY_BOTTOM + CASE_HALF[2]])
    return c, a


def place_trajectory():
    """置く腕の動き（左右 14 関節 × 0.02 秒ごと、12 秒）。scripts/place_test.py の arm_trajectory と同じ（まっすぐ下げて、
    手を左右へ抜く）。一度求めたら models/place_traj_tray.npy に保存して使う。"""
    path = Path(__file__).resolve().parent.parent / "models" / "place_traj_tray.npy"
    if path.exists():
        return np.load(path)
    import place_test as pt
    left = pt.arm_trajectory(pt.build(1.0, 0.45, (pt.EDGE + pt.DESK_DEPTH / 2, 0.0), 0.0))
    mirror = np.array([-1.0 if j in pe.MIRROR else 1.0 for j in pe.ARM_JOINTS])
    traj = np.concatenate([left, left * mirror], axis=1)
    path.parent.mkdir(exist_ok=True)
    np.save(path, traj)
    return traj


def chest_frame():
    """胸の前面の当たり判定（箱）：運ぶ姿勢で前面が x ＝ chest_front、高さ 0.83〜1.05 m、幅 ±0.10 m。torso_link の座標で返す。"""
    m0, d0 = _carry_state()
    b = m0.body("torso_link").id
    Rb, pb = d0.xmat[b].reshape(3, 3), d0.xpos[b]
    half = np.array([0.03, 0.10, 0.11])
    center = np.array([ARMS_GRIP["chest_front"] - half[0], 0.0, 0.94])
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, Rb.T.copy().ravel())
    return Rb.T @ (center - pb), quat, half


def pad_names():
    if GRIP == "tray":
        return [f"{s}_{part}_pad" for (s, part) in _tray_frames(*_carry_state())]
    parts = ARMS_GRIP["pads"] if GRIP == "arms" else PADS
    return [f"{s}_{part}_pad" for s in ("left", "right") for part in parts]


# しゃがむ姿勢（2026-10-17）：骨盤を真下へ h 下げ、足の裏を床に平らに保つ股・膝・足首（pitch）の角度の変化を、
# h = 0〜0.25 m（1 cm ごと）について立った姿勢（knees_bent）から求めた表（scratch の計算、models/crouch_offsets.npy）。
# 方策の出力の幅は ±0.5 rad（±29°）だけで、20 cm 沈めるのに要る膝の 64° に届かず、17 cm で出力が上限に張り付いて倒れた。
# この角度を脚の目標に先に足し、方策はそこからの直しだけを受け持つ
CROUCH_FF_DH = 0.01
CROUCH_FF_IDX = (0, 3, 4, 6, 9, 10)          # 左右の股（pitch）・膝・足首（pitch）（方策が動かす 15 の関節の中の番号）
# 腰の pitch（15 の関節の中の 14 番）：沈める量に比例して前へ（20 cm で 10°）。carry17 は深くしゃがむと胴を後ろへ約 10° 反らせ、
# 箱が机より 4.5 cm 高く、手前に残ったまま前腕を開いて箱を落とした。腕の動きは胴がまっすぐな形で作ってある
WAIST_PITCH_IDX = 14
WAIST_FF_PER_M = float(np.radians(10.0)) / 0.20


YAW_SHIFT_MAX = float(np.radians(10.0))   # 骨盤の向きを股の向きで直す量の上限（rad）
YAW_TOL = float(np.radians(2.0))          # 向きのずれがこれより小さければ直さない
HIP_YAW_IDX = (2, 8)
YAW_SIGN = -1.0                           # 股の向きを −a 回すと骨盤が +a 回る（足は床に着いたまま）
SHIFT_MAX = 0.06                          # 上体を寄せる量の上限（前後・左右、m）。models/shift_offsets.npy の範囲


def shift_table():
    """両足を着けたまま骨盤を前後 dx・左右 dy（体の座標、−6〜6 cm、1 cm ごと）寄せる脚 12 関節の角度の変化（scripts/shift_offsets.py）。"""
    return np.load(Path(__file__).resolve().parent.parent / "models" / "shift_offsets.npy")


def shift_ff_j(sh, table):
    """寄せる量 sh（前後, 左右）の脚 12 関節の角度（JAX、表を双線形で補間）。"""
    x = jp.clip((sh + SHIFT_MAX) / 0.01, 0.0, table.shape[0] - 1.0)
    i = jp.minimum(jp.floor(x).astype(jp.int32), table.shape[0] - 2)
    f = x - i
    a = table[i[0], i[1]] * (1 - f[1]) + table[i[0], i[1] + 1] * f[1]
    b = table[i[0] + 1, i[1]] * (1 - f[1]) + table[i[0] + 1, i[1] + 1] * f[1]
    return a * (1 - f[0]) + b * f[0]


def shift_ff(sh, table):
    """shift_ff_j の numpy 版（事務所の場面で使う）。"""
    sh = np.asarray(sh, float)
    x = np.clip((sh + SHIFT_MAX) / 0.01, 0.0, table.shape[0] - 1.0)
    i = np.minimum(np.floor(x).astype(int), table.shape[0] - 2)
    f = x - i
    a = table[i[0], i[1]] * (1 - f[1]) + table[i[0], i[1] + 1] * f[1]
    b = table[i[0] + 1, i[1]] * (1 - f[1]) + table[i[0] + 1, i[1] + 1] * f[1]
    return a * (1 - f[0]) + b * f[0]


def crouch_ff_table():
    return np.load(Path(__file__).resolve().parent.parent / "models" / "crouch_offsets.npy")


HIP_PITCH_IDX = (0, 6)


def crouch_ff(h, table, waist=False, lean_deg=0.0, H=0.20):
    """沈める量 h（m）のしゃがむ姿勢：15 の関節に足す角度（numpy）。"""
    x = np.clip(h / CROUCH_FF_DH, 0, len(table) - 1)
    i = min(int(x), len(table) - 2)
    v = table[i] + (x - i) * (table[i + 1] - table[i])
    out = np.zeros(N_ACT)
    out[list(CROUCH_FF_IDX)] = np.concatenate([v, v])
    if waist:
        out[WAIST_PITCH_IDX] = WAIST_FF_PER_M * h
    if lean_deg:                                              # 股関節を曲げて上体を前へ倒す（股の pitch を負の向きへ）
        out[list(HIP_PITCH_IDX)] -= np.radians(lean_deg) * h / H
    return out


# 事務所の場面で学習する（carry24 から）。事務所の場面（full_run.py）の C の道順と止まる位置
OFFICE_WP = (1.6, 1.2)                    # 椅子の北東を止まらずに通る点
OFFICE_STOP_EDGE, OFFICE_PLACE_Y = 0.11, 0.36
OFFICE_YAW_BIAS = float(np.radians(4.3))  # 置くあいだに回る分だけ右へ向けて止まる（full_run.py の PLACE_YAW_BIAS の既定）
OFFICE_BRAKE = 0.12                       # 減速を始める残りの距離（full_run.py の BRAKE_M の既定）


def office_stop():
    import office_scene as office
    return (office.DESK["center"][0] + office.DESK["size"][0] + OFFICE_STOP_EDGE, OFFICE_PLACE_Y)


FLOW = False                              # A の続きから B〜D を練習する（config の office_flow で書き換える。carry25 から）
# 事務所の場面（full_run.py）の受け渡しとドアを閉める動き
COURIER_WAIT = (5.95, 0.05)
SHOULDER_C = (0.0, 0.21, 1.40)
LBOX = (0.38, 0.0, 1.06)
T_ARM, T_W, T2, T3, T_BACK = 1.5, 3.5, 2.0, 1.5, 2.0


def office_train_decorate(spec):
    """事務所の場面（full_run.py の full_decorate と同じ部屋・当たり判定・摩擦）。FLOW のときは、ドアは蝶番で回り、配達員の
    手渡しの手（下の手のひら・横の手・箱を押す指）とドアを閉める手も事務所と同じ。FLOW でなければドアは閉まった板。"""
    import office_scene as office
    office.decorate(spec, hinged_door=FLOW)
    spec.body("pelvis").add_geom(name="pelvis_col", type=mujoco.mjtGeom.mjGEOM_BOX, pos=[-0.02, 0, -0.07],
                                 size=[0.07, 0.11, 0.04], contype=0, conaffinity=0, group=3)
    spec.body("torso_link").add_geom(name="torso_col", type=mujoco.mjtGeom.mjGEOM_CAPSULE, fromto=[-0.005, 0, 0.05, -0.005, 0, 0.40],
                                     size=[0.10, 0, 0], contype=0, conaffinity=0, group=3)
    pairs = [("pelvis_col", "chair"), ("left_thigh", "chair"), ("right_thigh", "chair"), ("left_shin", "chair"),
             ("right_shin", "chair"), ("torso_col", "desk"),
             ("left_hand_collision", "door"), ("right_hand_collision", "door"), ("torso_col", "door")]
    for a, b in pairs:
        p = spec.add_pair(geomname1=a, geomname2=b)
        p.condim = 3
        p.friction = [0.8, 0.8, 0.005, 0.0001, 0.0001]
    decorate(spec)
    limit_hand_hull(spec)
    if FLOW:                                           # 配達員の箱を押す指（下の手のひらの体）とドアを閉める手
        hx, hy, hz = CASE_HALF
        g = spec.body("support").add_geom(name="courier_push", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.006, 0.04, 0.015],
                                          pos=[hx + 0.006 + 0.001, 0.0, 0.01 + hz], contype=0, conaffinity=0, group=3)
        p = spec.add_pair(geomname1="case_geom", geomname2="courier_push")
        p.condim = 3
        p.friction = [0.2, 0.2, 0.005, 0.0001, 0.0001]
        dh = spec.worldbody.add_body(name="courier_door_hand", mocap=True, pos=[6.0, 0.5, -3.0])
        dh.add_geom(name="courier_door_hand_geom", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.02, 0.05, 0.06],
                    contype=0, conaffinity=0, group=3)
        p = spec.add_pair(geomname1="courier_door_hand_geom", geomname2="door")
        p.condim = 1
    scene = []
    bodies = [spec.worldbody] + ([spec.body("door_body")] if FLOW else [])
    for bd in bodies:
        for k, g in enumerate(bd.geoms):
            if g.type == mujoco.mjtGeom.mjGEOM_BOX and g.size[2] >= 0.01 and g.contype == 0:
                if not g.name:
                    g.name = f"scene_{bd.name or 'world'}_{k}"
                scene.append(g.name)
    have = {tuple(sorted(x)) for x in pairs}
    robot = ["case_geom", "pelvis_col", "torso_col", "left_hand_collision", "right_hand_collision",
             "left_thigh", "right_thigh", "left_shin", "right_shin"]
    for a in robot:
        for b in scene:
            if tuple(sorted((a, b))) in have or (a.endswith("hand_collision") and b == "desk"):
                continue
            p = spec.add_pair(geomname1=a, geomname2=b)
            p.condim = 3
            mu = 0.45 if (a, b) == ("case_geom", "desk") else 0.6
            p.friction = [mu, mu, 0.005, 0.0001, 0.0001]
    hand = [g.name for sd in ("left", "right") for b in ("wrist_yaw_link", "wrist_pitch_link", "wrist_roll_link")
            for g in spec.body(f"{sd}_{b}").geoms if g.name.endswith(tuple(f"mesh{k}" for k in range(9)))]
    for a in hand:
        p = spec.add_pair(geomname1=a, geomname2="desk")
        p.condim = 3
        p.friction = [0.45, 0.45, 0.005, 0.0001, 0.0001]


HAND_HULL = 16                            # 手のメッシュの当たり判定に使う形（凸包）の頂点の数の上限。そのままでは GPU の計算で接触を
#                                           求めるメモリが足りず学習が止まった。事務所の場面（full_run.py）でも同じ上限にする


def limit_hand_hull(spec):
    """手首・手のメッシュの当たり判定の形（凸包）の頂点を HAND_HULL までにする（見た目は元のまま）。"""
    for s in ("left", "right"):
        for n in ("wrist_roll_link", "wrist_pitch_link", "wrist_yaw_link", "rubber_hand"):
            spec.mesh(f"{s}_{n}").maxhullvert = HAND_HULL


def office_furniture():
    """事務所の机の脚・モニター・椅子（名前、机の天板の中心からの位置、半分の大きさ）。ロボットは事務所で西（−x）を向いて机の前に
    止まるので、事務所の座標を 180° 回したものが机の座標（+x がロボットの前、+y がロボットの左）。"""
    import office_scene as office
    (dx, dy), (hx, hy), dh = office.DESK["center"], office.DESK["size"], office.DESK["height"]
    top = (dx, dy, dh - 0.015)

    def rel(w):
        return (-(w[0] - top[0]), -(w[1] - top[1]), w[2] - top[2])
    out = []
    for k, (sx, sy) in enumerate(((-1, -1), (-1, 1), (1, -1), (1, 1))):
        out.append((f"place_desk_leg{k}", rel((dx + sx * (hx - 0.04), dy + sy * (hy - 0.04), (dh - 0.03) / 2)),
                    (0.025, 0.025, (dh - 0.03) / 2)))
    out.append(("place_monitor", rel((dx - 0.18, dy + 0.25, dh + 0.22)), (0.02, 0.27, 0.16)))
    out.append(("place_monitor_base", rel((dx - 0.18, dy + 0.25, dh + 0.04)), (0.06, 0.08, 0.04)))
    (chx, chy), seat, chh = office.CHAIR["center"], office.CHAIR["seat"], office.CHAIR["half"]
    out.append(("place_chair", rel((chx, chy, seat - 0.03)), (chh, 0.22, 0.03)))
    out.append(("place_chair_back", rel((chx + chh + 0.02, chy, seat + 0.28)), (0.025, 0.21, 0.28)))
    out.append(("place_chair_post", rel((chx, chy, (seat - 0.06) / 2)), (0.025, 0.025, (seat - 0.06) / 2)))
    out.append(("place_chair_base0", rel((chx, chy, 0.02)), (0.25, 0.03, 0.02)))
    out.append(("place_chair_base1", rel((chx, chy, 0.02)), (0.03, 0.25, 0.02)))
    return out


def decorate(spec):
    """指の面・胴の当たり判定、ケース、台（モーキャップ）、接触の組を足す。"""
    for (s, part), (body, pos, quat, half) in pad_frames().items():
        spec.body(body).add_geom(name=f"{s}_{part}_pad", type=mujoco.mjtGeom.mjGEOM_BOX, pos=list(pos), quat=list(quat),
                                 size=list(half), rgba=[0.2, 0.6, 0.2, 0.4], contype=0, conaffinity=0, group=3)
    tor = spec.body("torso_link")
    if "torso_col" not in [g.name for g in tor.geoms]:          # 事務所の場面（deploy.py）では先に足してある
        # desk_body のときは事務所の場面（full_run.py）と同じ大きさ（胴のメッシュの前の端に合わせた）
        x_, r_ = (-0.005, 0.10) if DESK_BODY else (0.02, 0.11)
        tor.add_geom(name="torso_col", type=mujoco.mjtGeom.mjGEOM_CAPSULE, fromto=[x_, 0, 0.05, x_, 0, 0.40],
                     size=[r_, 0, 0], rgba=[0.2, 0.6, 0.2, 0.3], contype=0, conaffinity=0, group=3)
    if DESK_BODY:
        limit_hand_hull(spec)
    if DESK_BODY and "pelvis_col" not in [g.name for g in spec.body("pelvis").geoms]:   # 事務所の場面と同じ骨盤の当たり判定
        spec.body("pelvis").add_geom(name="pelvis_col", type=mujoco.mjtGeom.mjGEOM_BOX, pos=[-0.02, 0, -0.07],
                                     size=[0.07, 0.11, 0.04], contype=0, conaffinity=0, group=3)
    case = spec.worldbody.add_body(name="case", pos=[0.3, 0, 0.9])
    case.add_freejoint(name="case_free")
    case.add_geom(name="case_geom", type=mujoco.mjtGeom.mjGEOM_BOX, size=list(CASE_HALF), mass=1.23,
                  rgba=[0.12, 0.12, 0.14, 1], contype=0, conaffinity=0)
    for side, sg in (("l", 1.0), ("r", -1.0)):                    # 配達員の左右の手（側面を押さえ、前の角の下に指をかける）
        g = spec.worldbody.add_body(name=f"grip_{side}", mocap=True, pos=[0, 0, -5.0])
        g.add_geom(name=f"grip_{side}_geom", type=mujoco.mjtGeom.mjGEOM_BOX, size=list(GRIP_HALF),
                   rgba=[0.9, 0.6, 0.3, 0.5], contype=0, conaffinity=0, group=3)
        # 指：ケースの底の左右の縁の下、前後 2 か所（ロボットの支えの面と重ならない所）。体の座標で、内側へ、ケースの底の下
        for k, (lx, lhx) in enumerate(LEDGES):
            g.add_geom(name=f"grip_{side}_ledge{k}", type=mujoco.mjtGeom.mjGEOM_BOX, size=[lhx, LEDGE_HALF[1], LEDGE_HALF[2]],
                       pos=[lx - GRIP_X, -sg * (GRIP_HALF[1] + LEDGE_HALF[1]), -(CASE_HALF[2] + LEDGE_HALF[2])],
                       rgba=[0.9, 0.6, 0.3, 0.5], contype=0, conaffinity=0, group=3)
    desk = spec.worldbody.add_body(name="place_desk", mocap=True, pos=[0, 0, -5.0])
    desk.add_geom(name="place_desk_top", type=mujoco.mjtGeom.mjGEOM_BOX, size=list(DESK_HALF),
                  rgba=[0.55, 0.4, 0.25, 1], contype=0, conaffinity=0)
    if DESK_BODY:                                       # 事務所の机の脚・モニター・椅子（机の天板の中心からの位置。机の座標の +x はロボットの前）
        for name, pos, half in office_furniture():
            desk.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_BOX, pos=list(pos), size=list(half),
                          rgba=[0.3, 0.3, 0.32, 1], contype=0, conaffinity=0)
    sup = spec.worldbody.add_body(name="support", mocap=True, pos=[0, 0, -5.0])
    sup.add_geom(name="support_geom", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.14, 0.12, 0.01],
                 rgba=[0.9, 0.3, 0.3, 0.3], contype=0, conaffinity=0, group=3)
    pads = pad_names()
    body_stop = "torso_col"
    if GRIP in ("arms", "tray"):                                           # 胸の前面（ケースの後ろの止め）
        pos, quat, half = chest_frame()
        tor.add_geom(name="chest_col", type=mujoco.mjtGeom.mjGEOM_BOX, pos=list(pos), quat=list(quat), size=list(half),
                     rgba=[0.2, 0.6, 0.2, 0.3], contype=0, conaffinity=0, group=3)
        body_stop = "chest_col"
    holders = (["support_geom", "grip_l_geom", "grip_r_geom"]
               + [f"grip_{sd}_ledge{k}" for sd in ("l", "r") for k in range(len(LEDGES))])
    hand_geoms = []
    for s in ("left", "right"):
        for b in ("wrist_yaw_link", "wrist_pitch_link", "wrist_roll_link"):
            for k, g in enumerate(spec.body(f"{s}_{b}").geoms):
                if g.type == mujoco.mjtGeom.mjGEOM_MESH:
                    if not g.name:
                        g.name = f"{s}_{b}_mesh{k}"
                    hand_geoms.append(g.name)
    # 置く練習の机は箱だけが当たる（手のメッシュと机の当たり判定を入れると、GPU の計算のメモリが足りず学習が止まった）
    if not DESK_BODY:
        p = spec.add_pair(geomname1="case_geom", geomname2="place_desk_top")
        p.condim = 3
        p.friction = [0.45, 0.45, 0.005, 0.0001, 0.0001]
    else:
        # 事務所の場面（full_run.py の full_decorate）と同じ当たり判定と摩擦：箱・骨盤・胴・手（大きな球）・もも・すねと、机の天板・
        # 脚・モニター・椅子（0.6。胴と天板、骨盤・もも・すねと座面は 0.8、箱と天板は 0.45）。手の大きな球と天板の代わりに、
        # 手のメッシュと天板（0.45）
        scene = ["place_desk_top"] + [n for n, _, _ in office_furniture()]
        robot = ["case_geom", "pelvis_col", "torso_col", "left_hand_collision", "right_hand_collision",
                 "left_thigh", "right_thigh", "left_shin", "right_shin"]
        strong = {("torso_col", "place_desk_top")} | {(a, "place_chair") for a in ("pelvis_col", "left_thigh", "right_thigh",
                                                                                  "left_shin", "right_shin")}
        for a in robot:
            for b in scene:
                if a.endswith("hand_collision") and b == "place_desk_top":
                    continue
                p = spec.add_pair(geomname1=a, geomname2=b)
                p.condim = 3
                mu = 0.45 if (a, b) == ("case_geom", "place_desk_top") else (0.8 if (a, b) in strong else 0.6)
                p.friction = [mu, mu, 0.005, 0.0001, 0.0001]
        for a in hand_geoms:
            p = spec.add_pair(geomname1=a, geomname2="place_desk_top")
            p.condim = 3
            p.friction = [0.45, 0.45, 0.005, 0.0001, 0.0001]
    for a, b in [("case_geom", p) for p in pads + holders + ["floor", body_stop]]:
        p = spec.add_pair(geomname1=a, geomname2=b)
        p.condim = 3
        mu = 1.5 if b.startswith(("support", "grip")) else 0.8   # 配達員の手はケースをしっかり持つ
        p.friction = [mu, mu, 0.005, 0.0001, 0.0001]


class CarryJoystick(joystick.Joystick):
    def __init__(self, config: config_dict.ConfigDict = default_config(),
                 config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None):
        mjx_env.MjxEnv.__init__(self, config, config_overrides)
        self._model_assets = g1_base.get_assets()
        global GRIP, DESK_BODY, FLOW
        GRIP = self._config.grip
        DESK_BODY = bool(self._config.desk_body)
        FLOW = bool(self._config.office and self._config.office_flow)
        if GRIP == "arms":                                       # 肘を少し上げた運ぶ姿勢
            pe.CARRY_LEFT["elbow"] = ARMS_GRIP["elbow"]
        elif GRIP == "tray":
            pe.CARRY_LEFT.update(TRAY_GRIP["arm"])
        m = pe.build_model(decorate=office_train_decorate if self._config.office else decorate)
        for s in pe.SIDES:                                      # payload_env の荷物の体は使わない
            b = m.body(f"{s}_payload").id
            m.body_mass[b] = 0.0
            m.body_inertia[b] = 1e-8
        m.opt.timestep = self.sim_dt
        # 接触の力を求める繰り返しの回数。Playground の G1 は 3 回（足と床だけなら足りる）。3 回ではケースと指の摩擦が
        # 正しく求まらず、ケースが滑り落ちた
        m.opt.iterations = self._config.solver_iterations
        m.opt.ls_iterations = self._config.solver_ls_iterations
        # ケースと指の接触を柔らかく（ゴムの指）、腕の関節の揺れを抑える（軽い手がばねのように跳ね返り、ケースが弾んだ）
        for i in range(m.npair):
            nm = {m.geom(m.pair_geom1[i]).name, m.geom(m.pair_geom2[i]).name}
            if self._config.office:                             # 事務所の場面（full_run.py）と同じ
                if "case_geom" in nm or ("desk" in nm and any("mesh" in n for n in nm)):
                    m.pair_solref[i] = self._config.case.contact_solref
            elif DESK_BODY:                                       # 事務所の場面と同じ：箱の接触と、手のメッシュと机の天板の接触
                if "case_geom" in nm or ("place_desk_top" in nm and any("mesh" in n for n in nm)):
                    m.pair_solref[i] = self._config.case.contact_solref
            elif {"case_geom", "place_desk_top"} & nm:
                m.pair_solref[i] = self._config.case.contact_solref
        for s_ in pe.SIDES:
            for j in pe.ARM_JOINTS:
                m.dof_damping[m.joint(f"{s_}_{j}_joint").dofadr[0]] = self._config.arm_damping
        if FLOW:                                            # 事務所の場面（full_run.py）と同じ：ドアの蝶番、配達員の手のひらの大きさ、手と箱は摩擦なし
            import office_scene as office
            dh_ = m.joint("door_hinge").dofadr[0]
            m.dof_damping[dh_], m.dof_frictionloss[dh_] = office.DOOR_DAMPING, office.DOOR_FRICTION
            m.geom_size[m.geom("support_geom").id] = [0.07, 0.05, 0.01]
            for i in range(m.npair):
                nm = {m.geom(m.pair_geom1[i]).name, m.geom(m.pair_geom2[i]).name}
                if "case_geom" in nm and nm & {"grip_r_geom", "support_geom", "courier_push"}:
                    m.pair_dim[i] = 1
        m.vis.global_.offwidth, m.vis.global_.offheight = 1920, 1080
        self._mj_model = m
        self._mjx_model = mjx.put_model(m, impl=self._config.impl)
        self._xml_path = consts.FEET_ONLY_FLAT_TERRAIN_XML.as_posix()
        self._post_init()
        arm = jp.array(pe.carry_pose())
        self._init_q = self._init_q.at[7 + N_ACT:NQ_ROBOT].set(arm)
        self._default_pose = self._default_pose[:29].at[N_ACT:].set(arm)
        self._lowers, self._uppers = self._lowers[:29], self._uppers[:29]
        self._soft_lowers, self._soft_uppers = self._soft_lowers[:29], self._soft_uppers[:29]
        self._est_w = jp.array(self._config.estimator.w)
        self._est_b = self._config.estimator.b
        self._desk = m.body("place_desk").mocapid[0]
        if self._config.crouch_obs:
            z = np.load(Path(__file__).resolve().parent.parent / "models" / self._config.crouch_file)
            self._place_traj, self._crouch_H = jp.array(z["traj"]), float(z["H"])
            self._ff_table = jp.array(crouch_ff_table()) if self._config.crouch_ff else None
            self._touch_tab = None
            if self._config.place_touch_file:                       # 触れたら切り替える表（触れた歩 × 歩 × 腕 14・沈める量）
                zt = np.load(Path(__file__).resolve().parent.parent / "models" / self._config.place_touch_file)
                self._touch_tab, self._touch_n1, self._crouch_H = jp.array(zt["tab"]), int(zt["N1"]), float(zt["H"])
                self._touch_len = int(zt["tab"].shape[1])
        else:
            self._place_traj = jp.array(place_trajectory()) if GRIP == "tray" else jp.tile(jp.array(pe.carry_pose()), (601, 1))
            self._crouch_H = 0.0
        self._case_body = m.body("case").id
        if self._config.office:
            import office_scene as office
            self._office_desk = jp.array([office.DESK["center"][0], office.DESK["center"][1], office.DESK["height"] - DESK_HALF[2]])
            self._office_stop = jp.array(office_stop())
        if FLOW:
            self._load_flow(m)
        self._shift_table = jp.array(shift_table()) if (self._config.office and self._config.adjust > 0) else None
        self._leg_local = [np.array(p_[:2]) for n_, p_, _ in office_furniture() if n_.startswith("place_desk_leg")] if self._config.desk_body else []
        # 机に触れたかを調べる円柱（体、端 2 つ、半径）：手・手首
        self._caps = [(m.body(f"{s}_{b}").id, np.array(a0) * [1, 1 if s == "left" else -1, 1],
                       np.array(a1) * [1, 1 if s == "left" else -1, 1], r) for s in pe.SIDES for b, (a0, a1, r) in HAND_CAPS.items()]
        self._case_geom = m.geom("case_geom").id
        self._case_q = m.joint("case_free").qposadr[0]
        self._case_v = m.joint("case_free").dofadr[0]
        self._support = m.body("support").mocapid[0]
        self._grips = np.array([m.body(f"grip_{s_}").mocapid[0] for s_ in ("l", "r")])
        self._pads = np.array([m.geom(n).id for n in pad_names()])
        self._wrists = np.array([m.body(f"{s}_wrist_yaw_link").id for s in pe.SIDES])
        holders = {m.geom(i).id for i in range(m.ngeom) if m.geom(i).name.startswith(("support", "grip"))}
        pn = [(m.geom(m.pair_geom1[i]).name, m.geom(m.pair_geom2[i]).name) for i in range(m.npair)]
        self._pad_pair = np.array([i for i, n in enumerate(pn) if "case_geom" in n and any(x.endswith("_pad") for x in n)])
        self._desk_case_pair = np.array([i for i, n in enumerate(pn) if set(n) in ({"case_geom", "place_desk_top"}, {"case_geom", "desk"})])
        self._floor_pair = np.array([i for i, n in enumerate(pn) if "floor" in n and any("foot" in x for x in n)])
        self._case_pair = np.array([i for i in range(m.npair) if (m.pair_geom1[i] == self._case_geom
                                    or m.pair_geom2[i] == self._case_geom) and not ({m.pair_geom1[i], m.pair_geom2[i]} & holders)])
        # ケースを指に載せた位置（骨盤の座標）：運ぶ姿勢で、左右の指の面の上面の真ん中にケースの底を置く
        d = mujoco.MjData(m)
        q = np.array(self._init_q)
        q[0:3], q[3:7] = (0, 0, q[2]), (1, 0, 0, 0)
        d.qpos[:] = q
        mujoco.mj_forward(m, d)
        top = []
        for g in self._pads:
            R = d.geom_xmat[g].reshape(3, 3)
            top.append(d.geom_xpos[g] + R[:, 2] * m.geom_size[g][2])   # 板の上面（板の z が上向き）
        top_z = max(t[2] for t in top)
        pel = d.xpos[m.body("pelvis").id]
        if GRIP == "arms":
            # ② ケースも支えの面と同じ傾き（胸側が低い）で置く。ケースの底を面に合わせ、後ろの縁を胸の前面のすぐ前に
            (xa, za), (xb, zb) = ARMS_GRIP["plane"]
            slope = (zb - za) / (xb - xa)
            self._case_tilt = float(np.arctan(slope))
            n = np.array([-np.sin(self._case_tilt), 0.0, np.cos(self._case_tilt)])          # 面の法線（後ろへ少し傾く）
            cx = ARMS_GRIP["case_x"]
            center = np.array([cx, 0.0, za + slope * (cx - xa)]) + n * CASE_HALF[2]
        elif GRIP == "tray":
            center, self._case_tilt = tray_case_pose()
        else:
            self._case_tilt = 0.0
            center = np.array([CASE_X, 0.0, top_z + CASE_HALF[2]])
        self._case_local = jp.array(center - pel)                  # 骨盤（向きは体の向き）からの位置
        print("ケースの中心（骨盤の座標）", np.round(np.array(self._case_local), 3), "指の上面の高さ", np.round([t[2] for t in top], 3))

    @property
    def action_size(self) -> int:
        return N_ACT

    def _touch_mode(self):
        return bool(self._config.crouch_obs and self._config.place_touch_file)

    def _desk_touched(self, data, info):
        """箱の底の角か、手・手首の円柱の下の端が、机の上面（＋0.3 cm）まで下りて、机の上（手前の縁より奥）にあるか。
        事務所の場面の「机の上面に触れたら止める」と同じ（接触の点が上面の高さで、縁より奥）。"""
        dpos, dq = info["desk_pos"], info["desk_quat"]
        dyaw = 2.0 * jp.arctan2(dq[3], dq[0])
        c, s_ = jp.cos(dyaw), jp.sin(dyaw)
        top = dpos[2] + DESK_HALF[2]

        def hit(p, r):
            lx = c * (p[0] - dpos[0]) + s_ * (p[1] - dpos[1])
            ly = -s_ * (p[0] - dpos[0]) + c * (p[1] - dpos[1])
            return (p[2] - r <= top + 0.003) & (lx >= -DESK_HALF[0]) & (jp.abs(ly) <= DESK_HALF[1])
        out = jp.array(False)
        for b, a0, a1, r in self._caps:
            R = data.xmat[b].reshape(3, 3)
            for a in (a0, a1):
                out = out | hit(data.xpos[b] + R @ jp.array(a), r)
        Rc = data.xmat[self._case_body].reshape(3, 3)
        hx, hy, hz = CASE_HALF
        for sx in (-1, 1):
            for sy in (-1, 1):
                for sz in (-1, 1):
                    out = out | hit(data.xpos[self._case_body] + Rc @ jp.array([sx * hx, sy * hy, sz * hz]), 0.0)
        return out

    # ------------------------------------------------------------ 補助
    def _robot(self, data):
        """ロボットの部分だけの qpos・qvel・qacc にした data（Playground の報酬の関数は qpos[7:] を関節として使うため）。"""
        return data.replace(qpos=data.qpos[:NQ_ROBOT], qvel=data.qvel[:NV_ROBOT], qacc=data.qacc[:NV_ROBOT])

    def _case_home(self, data, with_normal=False, pose=None):
        """支えの面の上の、ケースのあるべき位置と向き（体の向きだけ使う）。with_normal：面の法線（世界の向き）も返す。
        ② の持ち方では、ケースは面と同じく胸側が低い向きに _case_tilt だけ傾く。pose：（骨盤の位置, 向き）を直接与える"""
        if pose is None:
            pel = data.xpos[self._mj_model.body("pelvis").id]
            R = data.xmat[self._mj_model.body("pelvis").id].reshape(3, 3)
            yaw = jp.arctan2(R[1, 0], R[0, 0])
        else:
            pel, yaw = pose
        c, s = jp.cos(yaw), jp.sin(yaw)
        off = self._case_local
        pos = pel + jp.array([c * off[0] - s * off[1], s * off[0] + c * off[1], off[2]])
        a = self._case_tilt
        q_yaw = jp.array([jp.cos(yaw / 2), 0.0, 0.0, jp.sin(yaw / 2)])
        q_tilt = jp.array([np.cos(-a / 2), 0.0, np.sin(-a / 2), 0.0])
        quat = math.quat_mul(q_yaw, q_tilt)
        normal = jp.array([-np.sin(a) * c, -np.sin(a) * s, np.cos(a)])
        return (pos, quat, normal) if with_normal else (pos, quat)

    def _sample_episode(self, rng):
        c = self._config.case
        never = 10 ** 6
        k = jax.random.split(rng, 9)
        mode = jax.random.choice(k[0], len(c.mode_p), p=jp.array(c.mode_p))
        t_on = jp.round(jax.random.uniform(k[1], minval=c.onset_range[0], maxval=c.onset_range[1]) / self.dt)
        t_off = jp.round(jax.random.uniform(k[2], minval=c.release_range[0], maxval=c.release_range[1]) / self.dt)
        onset = jp.select([mode == 0, mode == 1, mode == 2, mode == 4], [0, t_on, 0, 0], never).astype(jp.int32)
        # 置く練習：t_stop で止まり、settle 待って腕を動かし始め（arm_t0）、手を左右へ抜き始める 6 秒後に箱は机へ移る
        k2 = jax.random.split(k[7], 4)
        t_stop = jp.round(jax.random.uniform(k2[0], minval=c.stop_range[0], maxval=c.stop_range[1]) / self.dt)
        settle = jp.round(jax.random.uniform(k2[1], minval=c.settle_range[0], maxval=c.settle_range[1]) / self.dt)
        arm_t0 = jp.where(mode == 4, t_stop + settle, never).astype(jp.int32)
        t_stop = jp.where(mode == 4, t_stop, never).astype(jp.int32)
        desk_off = jax.random.uniform(k2[2], (2,), minval=-c.desk_offset, maxval=c.desk_offset)
        desk_dz = jax.random.uniform(k2[3], minval=-c.desk_dz, maxval=c.desk_dz)
        if self._config.desk_body:                       # 事務所：机の手前の縁までのずれ、左右のずれ、向きのずれ
            kd = jax.random.split(k2[2], 3)
            desk_off = jp.array([jax.random.uniform(kd[0], minval=c.desk_fwd[0], maxval=c.desk_fwd[1]),
                                 c.place_y + jax.random.uniform(kd[1], minval=-c.desk_offset, maxval=c.desk_offset)])
            desk_dz = jax.random.uniform(kd[2], minval=-c.desk_yaw, maxval=c.desk_yaw)   # desk_body では向きのずれに使う
        t_rel = PLACE_RELEASE if self._config.crouch_obs else 6.0
        # 触れたら切り替える表のときは、箱が手を離れる時刻は触れた時に決める（それまでは手の上）
        rel4 = never if self._touch_mode() else arm_t0 + int(round(t_rel / self.dt))
        release = jp.select([mode == 2, mode == 4], [t_off, rel4], never).astype(jp.int32)
        # 事務所と同じ近づき方（止まる前）：止まる（0.5 秒）→ その場で向きを変える（0.6〜2 秒、±0.3〜0.9 rad/秒）→ 0.2 m/秒で前へ
        # （1.5〜3 秒。半分は横へ ±0.1〜0.15 m/秒で直しながら）→ 止まる。t_stop は近づき方の分だけ後ろへずらす
        k3 = jax.random.split(k[8], 6)
        ap_turn = jp.round(jax.random.uniform(k3[0], minval=0.6, maxval=2.0) / self.dt)
        ap_w = jax.random.uniform(k3[1], minval=0.3, maxval=0.9) * jp.where(jax.random.bernoulli(k3[2]), 1.0, -1.0)
        ap_fwd = jp.round(jax.random.uniform(k3[3], minval=1.5, maxval=3.0) / self.dt)
        ap_vy = jax.random.uniform(k3[4], minval=0.1, maxval=0.15) * jax.random.choice(k3[5], jp.array([-1.0, 0.0, 0.0, 1.0]))
        ap_pause = int(round(0.5 / self.dt))
        if self._config.approach:
            shift = jp.where(mode == 4, ap_pause + ap_turn + ap_fwd, 0).astype(jp.int32)
            t_stop, arm_t0 = t_stop + shift, arm_t0 + shift
            release = jp.where((mode == 4) & (release < never), release + shift, release)
        hp = c.hand_push
        push_on = jax.random.bernoulli(k[3], hp.p)
        push_t0 = jp.round(jax.random.uniform(k[4], minval=hp.time_range[0], maxval=hp.time_range[1]) / self.dt)
        push_t1 = push_t0 + jp.round(jax.random.uniform(k[5], minval=hp.dur_range[0], maxval=hp.dur_range[1]) / self.dt)
        push_f = jp.where(push_on, jax.random.uniform(k[6], minval=hp.force_range[0], maxval=hp.force_range[1]), 0.0)
        ep = dict(mode=mode, onset=onset, release=release, hp_t0=push_t0.astype(jp.int32),
                  hp_t1=push_t1.astype(jp.int32), hp_f=push_f, t_stop=t_stop, arm_t0=arm_t0, desk_off=desk_off,
                  desk_dz=desk_dz, ap_t0=(t_stop - ap_fwd - ap_turn - ap_pause).astype(jp.int32),
                  ap_turn=ap_turn.astype(jp.int32), ap_w=ap_w, ap_vy=ap_vy, touch_k=jp.int32(never))
        if self._config.office:
            ep.update(self._office_episode(jax.random.fold_in(rng, 7), mode, ep))
        return ep

    def _load_flow(self, m):
        """事務所で通した記録の状態（scripts/full_run.py --dump_states）を読む。"""
        import json
        import office_scene as office
        rs = [json.loads(l) for l in open(Path(__file__).resolve().parent.parent / "models" / self._config.flow_file)]
        th0 = {r["seed"]: r["door"] for r in rs if r["stage"] == "C0"}           # C の始めのドアの角度（閉める動きの始め）
        rs = [r for r in rs if r["seed"] in th0 and r["pelvis_z_min"] > 0.6]
        names = ["B", "C0", "C1", "C2", "C3"]
        self._fl_stage = jp.array([names.index(r["stage"]) for r in rs], dtype=jp.int32)
        self._fl_q = jp.array([r["qpos"] for r in rs])
        self._fl_v = jp.array([r["qvel"] for r in rs])
        self._fl_cq = jp.array([r["case_q"] for r in rs])
        self._fl_cv = jp.array([r["case_v"] for r in rs])
        self._fl_door = jp.array([r["door"] for r in rs])
        self._fl_th0 = jp.array([th0[r["seed"]] for r in rs])
        self._fl_tc = jp.array([r["t_since_C"] or 0.0 for r in rs])
        self._fl_last = jp.array([r["last"] or [0.0] * N_ACT for r in rs])
        self._fl_phase = jp.array([r["phase"] or [0.0, float(np.pi)] for r in rs])
        self._fl_mest = jp.array([r["m_est"] or 0.0 for r in rs])
        self._fl_idx = [np.where(np.array([names.index(r["stage"]) for r in rs]) == k)[0] for k in range(5)]
        self._fl_idx_pad = jp.array([np.resize(ix, 64) for ix in self._fl_idx])      # 始める所ごとの番号（64 に揃える）
        self._door_q = m.joint("door_hinge").qposadr[0]
        self._door_v = m.joint("door_hinge").dofadr[0]
        self._door_hand = m.body("courier_door_hand").mocapid[0]
        self._hinge = np.array(office.DOOR_HINGE[:2]) + np.array([-0.05, -0.005])
        self._courier_yaw0 = float(office.COURIER_YAW)
        print("事務所で通した記録の状態：", {n: len(ix) for n, ix in zip(names, self._fl_idx)})

    def _flow_handover(self, data, info):
        """受け渡し（full_run.py の B と同じ）：配達員が箱を両手に載せて歩いて近づき（3.5 秒）、ロボットの腕の間（手のひらの 3 cm 上）へ
        差し出し（2 秒）、真下へ下ろし（1.5 秒）、手のひらの 3 mm 上で放す。手を引く（2 秒）。配達員が持っている間は箱を手と一緒に動かす。
        配達員の物理の手：下の手のひら（support）・箱を押す指・横の手（grip_r）。"""
        hx, hy, hz = CASE_HALF
        dt = self.dt
        # 腕を運ぶ姿勢へ移す間（10）も、配達員は箱を持って待っている（歩き出す前の所）
        on = (info["mode"] == 4) & ((info["ph"] == 13) | (info["ph"] == 10))
        t = jp.where(info["ph"] == 13, info["ph_k"] * dt, 0.0)
        start = on & (info["ph"] == 13) & (info["ph_k"] == 0)      # 受け渡しを始める時（腕を運ぶ姿勢へ移した後）の体の位置から決める
        home, qh = self._case_home(data)
        Rh_now = math.quat_to_mat(qh)
        tgt0 = home + Rh_now[:, 2] * 0.03
        Rp = data.xmat[self._mj_model.body("pelvis").id].reshape(3, 3)
        th_r = jp.arctan2(Rp[1, 0], Rp[0, 0])
        fwd_r = jp.array([jp.cos(th_r), jp.sin(th_r)])
        dz_ = SHOULDER_C[2] - (tgt0[2] - hz - 0.023)
        HX = jp.clip(jp.sqrt(jp.maximum(0.50 ** 2 - (SHOULDER_C[1] - 0.07) ** 2 - dz_ ** 2, 0.0)) + 0.05, 0.24, 0.42)
        info["b_tgt"] = jp.where(start, tgt0, info["b_tgt"])
        info["b_home"] = jp.where(start, home, info["b_home"])
        info["b_c1"] = jp.where(start, tgt0[:2] + fwd_r * (HX + 0.08), info["b_c1"])
        info["b_yaw1"] = jp.where(start, th_r + jp.pi, info["b_yaw1"])
        sm = lambda u: (lambda x: x * x * (3 - 2 * x))(jp.clip(u, 0.0, 1.0))  # noqa: E731
        # 配達員が歩いて近づく間：箱は配達員の両手の上
        u = sm(t / T_W)
        y0 = self._courier_yaw0
        cxy = (1 - u) * jp.array(COURIER_WAIT) + u * info["b_c1"]
        cyaw = y0 + u * wrap_j(info["b_yaw1"] - y0)
        cz, sz = jp.cos(cyaw), jp.sin(cyaw)
        c_walk = jp.array([cxy[0] + cz * LBOX[0] - sz * LBOX[1], cxy[1] + sz * LBOX[0] + cz * LBOX[1], LBOX[2]])
        a_ = self._case_tilt
        q_walk = math.quat_mul(jp.array([jp.cos((cyaw - jp.pi) / 2), 0.0, 0.0, jp.sin((cyaw - jp.pi) / 2)]),
                               jp.array([np.cos(-a_ / 2), 0.0, np.sin(-a_ / 2), 0.0]))
        walking = t < T_W
        info["b_chold"] = jp.where(on & walking, c_walk, info["b_chold"])
        info["b_qhold"] = jp.where(on & walking, q_walk, info["b_qhold"])
        # 下ろし始めるまでは、ロボットの今の位置と向きに合わせて差し出す位置を直す（0.3 秒で追いつく）
        a = dt / (0.3 + dt)
        track = on & (t < T_W + T2) & ~start
        info["b_tgt"] = jp.where(track, info["b_tgt"] + a * ((home + Rh_now[:, 2] * 0.03) - info["b_tgt"]), info["b_tgt"])
        info["b_qRh"] = jp.where(on & (t < T_W + T2), qh, info["b_qRh"])
        Rh = math.quat_to_mat(info["b_qRh"])
        tgt = info["b_tgt"]
        u2 = sm((t - T_W) / T2)
        qa, qb = info["b_qhold"], info["b_qRh"]
        qb = jp.where(jp.dot(qa, qb) >= 0, qb, -qb)
        q_off = (1 - u2) * qa + u2 * qb
        q_off = q_off / jp.linalg.norm(q_off)
        u3 = sm((t - T_W - T2) / T3)
        c = jp.where(walking, c_walk, jp.where(t < T_W + T2, (1 - u2) * info["b_chold"] + u2 * tgt,
                                              tgt - Rh[:, 2] * 0.05 * jp.where(t < T_W + T2 + T3, u3, 1.0)))
        q = jp.where(walking, q_walk, jp.where(t < T_W + T2, q_off, qb))
        rel = info["b_held"] & (((t >= T_W + T2) & ((c - info["b_home"])[2] <= 0.003)) | (t >= T_W + T2 + T3))
        released = on & rel
        info["b_held"] = jp.where(released, False, info["b_held"])
        info["onset"] = jp.where(released & (info["onset"] >= 10 ** 6), info["pl_t"], info["onset"]).astype(jp.int32)
        held = on & info["b_held"]
        # 配達員が持っている間は、箱を手と一緒に動かす
        qpos = data.qpos.at[self._case_q:self._case_q + 3].set(jp.where(held, c, data.qpos[self._case_q:self._case_q + 3]))
        qpos = qpos.at[self._case_q + 3:self._case_q + 7].set(jp.where(held, q, qpos[self._case_q + 3:self._case_q + 7]))
        qvel = data.qvel.at[self._case_v:self._case_v + 6].set(jp.where(held, 0.0, data.qvel[self._case_v:self._case_v + 6]))
        # 配達員の物理の手：横の手（右）と下の手のひら（押す指つき）
        R_ = math.quat_to_mat(q)
        open_ = jp.where(info["b_held"], 0.0, 0.03)
        off = hy + GRIP_HALF[1] - GRIP_SQUEEZE + open_
        gpos = c + R_[:, 0] * GRIP_X - R_[:, 1] * off
        spos = c - R_[:, 2] * (hz + 0.01)
        mp = data.mocap_pos
        mq = data.mocap_quat
        mp = mp.at[self._grips[1]].set(jp.where(on, gpos, mp[self._grips[1]])).at[self._grips[0]].set(
            jp.where(on, FAR + jp.array([0.0, 2.0, 0.0]), mp[self._grips[0]]))
        mp = mp.at[self._support].set(jp.where(on, spos, mp[self._support]))
        mq = mq.at[self._grips[1]].set(jp.where(on, q, mq[self._grips[1]])).at[self._support].set(jp.where(on, q, mq[self._support]))
        return data.replace(qpos=qpos, qvel=qvel, mocap_pos=mp, mocap_quat=mq), info

    def _flow_door_hand(self, data, info):
        """配達員がドアを閉める手（full_run.py の close_door と同じ時刻・位置）。C の始めから 2.5 秒後に回り込み始め、7.84〜11.9 秒の間、
        ドアの外側の面に手を当てて押す（ドアの角度の予定：8.2 秒まで開いたまま、そこから 3.5 秒で閉める）。"""
        tdc = (info["pl_t"] - info["c_t0"]) * self.dt - 2.5
        sm = lambda u: (lambda x: x * x * (3 - 2 * x))(jp.clip(u, 0.0, 1.0))  # noqa: E731
        th0 = info["door_th0"]
        th = jp.where(tdc < 5.7, th0, th0 * (1.0 - sm((tdc - 5.7) / 3.5)))
        dirv = jp.array([jp.sin(th), -jp.cos(th)])
        nv = jp.array([jp.cos(th), jp.sin(th)])
        pos2 = jp.array(self._hinge) + 0.6 * dirv + nv * 0.042
        act = (info["mode"] == 4) & (info["c_t0"] < 10 ** 6) & (tdc >= 4.5 + 1.2 * 0.7) & (tdc < 4.5 + 1.2 + 3.5 + 0.2)
        pos = jp.where(act, jp.array([pos2[0], pos2[1], 1.10]), jp.array([6.0, 0.5, -3.0]))
        quat = jp.array([jp.cos(th / 2), 0.0, 0.0, jp.sin(th / 2)])
        return data.replace(mocap_pos=data.mocap_pos.at[self._door_hand].set(pos),
                            mocap_quat=data.mocap_quat.at[self._door_hand].set(quat))

    def _office_drive(self, info, xy, th, vxy=None):
        """事務所の場面（full_run.py の C）と同じ歩き方の指令を、置く練習（mode 4）で 1 歩ずつ出す。
          0：椅子の北東の点へ 0.4 m/秒（体の前の残りが 0.35 m で次へ。止まらない）
          1：机の手前 0.4 m の点へ 0.4 m/秒、残り 0.12 m から 1 秒かけて止める
          2：0.5 秒止まる　3：その場で 0.8 rad/秒で西（−4.3°）を向く（残り 0.22 rad から 1 秒かけて止める）
          4：向きを保ち 0.2 m/秒で前へ、横のずれも直す（残り 0.06 m から 1 秒かけて止める）
          5：2 秒止まる（止まった時刻 t_stop）　6：置く（置き始めの時刻 arm_t0）"""
        dt = self.dt
        ph, k, kb = info["ph"], info["ph_k"], info["kb"]
        stop = info["stop_tgt"]
        pre = stop + jp.array([0.40, 0.0])
        goal = jp.pi - OFFICE_YAW_BIAS
        c, s_ = jp.cos(th), jp.sin(th)
        fwd = lambda e: c * e[0] + s_ * e[1]                      # noqa: E731
        head = lambda e: jp.where(jp.linalg.norm(e) > 0.3, jp.clip(1.0 * wrap_j(jnp_atan2v(e) - th), -0.3, 0.3), 0.0)  # noqa: E731

        def brake(cond):
            kb_ = jp.where((kb < 0) & cond, k, kb)
            u = jp.where(kb_ < 0, 1.0, jp.maximum(0.0, 1.0 - (k - kb_) * dt / 1.0))
            return kb_, u
        e0 = jp.array(OFFICE_WP) - xy
        c0, d0 = jp.array([0.4, 0.0, head(e0)]), fwd(e0) <= 0.35
        e1 = pre - xy
        kb1, u1 = brake(fwd(e1) <= OFFICE_BRAKE)
        c1, d1 = jp.array([0.4, 0.0, head(e1)]) * u1, (kb1 >= 0) & (u1 <= 0.0)
        c2, d2 = jp.zeros(3), k >= int(round(0.5 / dt))
        err = wrap_j(goal - th)
        kb3, u3 = brake(jp.abs(err) <= 0.22)
        c3, d3 = jp.array([0.0, 0.0, jp.sign(err) * 0.8]) * u3, (kb3 >= 0) & (u3 <= 0.0)
        e4 = stop - xy
        kb4, u4 = brake(fwd(e4) <= OFFICE_BRAKE / 2)
        side = -s_ * e4[0] + c * e4[1]
        vy = jp.clip(1.5 * side, -0.15, 0.15)
        vy = jp.where(jp.abs(side) > 0.015, jp.sign(vy) * jp.maximum(jp.abs(vy), 0.1), 0.0)
        c4 = jp.array([0.2, vy, jp.clip(1.5 * wrap_j(goal - th), -0.3, 0.3)]) * u4
        d4 = (kb4 >= 0) & (u4 <= 0.0)
        c5, d5 = jp.zeros(3), k >= int(round(2.0 / dt))
        # 7：机の前で止まった後の調整（carry28 から。full_run.py の adjust_stop と同じ決め方）。体が静止する（骨盤の速さ 1 cm/秒未満が
        # 0.3 秒続く。最長 1.5 秒）まで待ってずれを測る。2 cm 未満なら終わり。上体を寄せて直せる（寄せる量の合計が前後・左右 ±6 cm 以内）
        # なら、両足を着けたまま 0.5 秒で上体（骨盤）を寄せる。それを超えるときだけ、ずれの方向へ 0.1 m/秒で踏み直す（最長 0.6 秒）。
        # これを最大 cfg.adjust 回。寄せた上体は置き終わるまでそのまま
        on7 = ph == 7
        sub, ak = info["a_sub"], info["a_k"]
        e7 = stop - xy
        f7, s7 = c * e7[0] + s_ * e7[1], -s_ * e7[0] + c * e7[1]
        still = jp.where(jp.linalg.norm(vxy) < 0.01, info["a_still"] + 1, 0)
        settled = on7 & (sub == 0) & (((still >= 15) & (ak >= 15)) | (ak >= 75))
        ey7 = wrap_j(goal - th)                                  # 向きのずれ（carry29 から、上体を寄せるときに股の向きで直す）
        small = (jp.hypot(f7, s7) < 0.02) & ((jp.abs(ey7) < YAW_TOL) | (self._config.yaw_shift == 0))
        tgt_sh = info["sh"] + jp.array([f7, s7])
        can_shift = jp.all(jp.abs(tgt_sh) <= SHIFT_MAX + 1e-6)
        tgt_y = jp.clip(info["shy"] + ey7, -YAW_SHIFT_MAX, YAW_SHIFT_MAX) if self._config.yaw_shift else info["shy"]
        d7 = settled & (small | (info["adj_n"] >= self._config.adjust))
        go_sh = settled & ~d7 & can_shift
        go_st = settled & ~d7 & ~can_shift
        ramp_end = on7 & (sub == 1) & (ak >= 25)
        step_end = on7 & (sub == 2) & (ak >= info["adj_T"])
        ur = jp.clip(ak / 25.0, 0.0, 1.0)
        ur = ur * ur * (3 - 2 * ur)
        # 上体を寄せる（1）と、踏み直す（2）とき：踏み直すときも、上体を踏み出す方へ一緒に運ぶ（ずれの半分、3 cm まで。carry29 から）
        ramp = on7 & ((sub == 1) | ((sub == 2) & (self._config.yaw_shift > 0)))
        info["sh"] = jp.where(ramp, info["sh_from"] + ur * (info["sh_to"] - info["sh_from"]), info["sh"])
        info["shy"] = jp.where(on7 & (sub == 1), info["shy_from"] + ur * (info["shy_to"] - info["shy_from"]), info["shy"])
        info["shy_from"] = jp.where(go_sh, info["shy"], info["shy_from"])
        info["shy_to"] = jp.where(go_sh, tgt_y, info["shy_to"])
        lead = jp.clip(0.5 * jp.array([f7, s7]), -0.03, 0.03)
        info["sh_from"] = jp.where(go_sh | go_st, info["sh"], info["sh_from"])
        info["sh_to"] = jp.where(go_sh, jp.clip(tgt_sh, -SHIFT_MAX, SHIFT_MAX),
                                 jp.where(go_st, jp.clip(info["sh"] + lead * (self._config.yaw_shift > 0), -SHIFT_MAX, SHIFT_MAX),
                                          info["sh_to"]))
        info["adj_v"] = jp.where(go_st, jp.array([jp.where(jp.abs(f7) > 0.015, jp.sign(f7) * 0.1, 0.0),
                                                  jp.where(jp.abs(s7) > 0.015, jp.sign(s7) * 0.1, 0.0)]), info["adj_v"])
        info["adj_T"] = jp.where(go_st, jp.round(jp.minimum(jp.maximum(jp.abs(f7), jp.abs(s7)) / 0.1, 0.6) / dt), info["adj_T"]).astype(jp.int32)
        w7 = jp.clip(1.5 * wrap_j(goal - th), -0.3, 0.3)
        c7 = jp.where(on7 & (sub == 2), jp.array([info["adj_v"][0], info["adj_v"][1], w7]), jp.zeros(3))
        info["adj_n"] = (info["adj_n"] + (go_sh | go_st)).astype(jp.int32)
        new_sub = jp.where(go_sh, 1, jp.where(go_st, 2, jp.where(ramp_end | step_end, 0, sub)))
        info["a_k"] = jp.where(on7 & (new_sub == sub), ak + 1, 0).astype(jp.int32)
        info["a_still"] = jp.where(on7 & (new_sub == 0) & (sub == 0), still, 0).astype(jp.int32)
        info["a_sub"] = jp.where(on7, new_sub, 0).astype(jp.int32)
        cyc7 = jp.array(False)
        # A の続き（FLOW）：10 腕を運ぶ姿勢へ（1.5 秒止まる）→ 13 受け渡し（10 秒止まる）→ 11 配達員の方を向いたまま後ろへ 0.9 m 下がる
        # （0.25 m/秒、残り 0.08 m から 1 秒かけて止める）→ 12 その場で時計回りに椅子の北東の点の方を向く → 0 から
        c10, d10 = jp.zeros(3), k >= int(round(T_ARM / dt))
        c13, d13 = jp.zeros(3), k >= int(round((T_W + T2 + T3 + T_BACK + 1.0) / dt))
        info["bu_xy0"] = jp.where((ph == 11) & (k == 0), xy, info["bu_xy0"])
        info["bu_th0"] = jp.where((ph == 11) & (k == 0), th, info["bu_th0"])
        moved = -((xy[0] - info["bu_xy0"][0]) * jp.cos(info["bu_th0"]) + (xy[1] - info["bu_xy0"][1]) * jp.sin(info["bu_th0"]))
        kb11, u11 = brake(moved >= 0.9 - 0.08)
        c11 = jp.array([-0.25, 0.0, jp.clip(1.5 * wrap_j(info["bu_th0"] - th), -0.3, 0.3)]) * u11
        d11 = (kb11 >= 0) & (u11 <= 0.0)
        info["tgoal"] = jp.where((ph == 12) & (k == 0), jnp_atan2v(jp.array(OFFICE_WP) - xy), info["tgoal"])
        e12 = wrap_j(info["tgoal"] - th)
        e12 = jp.where(jp.abs(e12) > 0.3, -jp.abs(e12), e12)            # 時計回り
        kb12, u12 = brake(jp.abs(e12) <= 0.22)
        c12, d12 = jp.array([0.0, 0.0, jp.sign(e12) * 0.8]) * u12, (kb12 >= 0) & (u12 <= 0.0)
        sel = lambda xs: jp.select([ph == 0, ph == 1, ph == 2, ph == 3, ph == 4, ph == 5, ph == 10, ph == 11, ph == 12, ph == 13, ph == 7],
                                   xs[:11], xs[11])  # noqa: E731
        cmd = sel([c0, c1, c2, c3, c4, c5, c10, c11, c12, c13, c7, jp.zeros(3)])
        done = sel([d0, d1, d2, d3, d4, d5, d10, d11, d12, d13, d7, jp.array(False)]) & (ph >= 0)
        kb_new = sel([kb, kb1, kb, kb3, kb4, kb, kb, kb11, kb12, kb, kb, kb])
        nxt = jp.select([ph == 10, ph == 13, ph == 11, ph == 12, ph == 4, ph == 7],
                        [13, 11, 12, 0, 7 if self._config.adjust > 0 else 5, 5], ph + 1)
        info["adj_n"] = jp.where(done & (ph == 4), 0, jp.where(cyc7, info["adj_n"] + 1, info["adj_n"])).astype(jp.int32)
        info["ap_cmd"] = cmd
        info["c_t0"] = jp.where(done & (ph == 13), info["pl_t"], info["c_t0"]).astype(jp.int32)   # C の始め（ドアを閉める動きの時計）
        info["ph"] = jp.where(done, nxt, ph).astype(jp.int32)
        info["ph_k"] = jp.where(done | cyc7, 0, k + 1).astype(jp.int32)
        info["kb"] = jp.where(done, -1, kb_new).astype(jp.int32)
        never = 10 ** 6
        info["t_stop"] = jp.where((info["ph"] == 5) & (info["t_stop"] >= never), info["pl_t"], info["t_stop"]).astype(jp.int32)
        info["arm_t0"] = jp.where((info["ph"] == 6) & (info["arm_t0"] >= never), info["pl_t"], info["arm_t0"]).astype(jp.int32)
        return info

    def _office_episode(self, rng, mode, ep):
        """事務所の場面の練習の始め。置く練習（mode 4）：C の道順の途中（戸口から椅子の北東の点へ 20%、椅子の北東から机の手前へ
        50%、机の手前で向きを変える所 30%）から、箱を持って始める。止まる位置は事務所の試験と同じく前後 ±2 cm・左右 ±3 cm ずらす。
        止まる時刻・置き始める時刻は、歩き方の指令が止まった時に決まる。押す力はかけない（事務所にはない）。
        ほかの練習：部屋の空いている所（x 1.2〜4.2 m、y −0.7〜1.5 m）から、でたらめな向きで始める。"""
        never = 10 ** 6
        k = jax.random.split(rng, 10)
        stop = self._office_stop + jp.array([jax.random.uniform(k[0], minval=-0.02, maxval=0.02),
                                             jax.random.uniform(k[1], minval=-0.03, maxval=0.03)])
        wp = jp.array(OFFICE_WP)
        pre = stop + jp.array([0.40, 0.0])
        stage = jax.random.choice(k[2], 3, p=jp.array([0.2, 0.5, 0.3]))
        u = jax.random.uniform(k[3])
        a0, a1 = jp.array([4.5, 1.5]), jp.array([2.4, 1.25])
        p0 = a0 + u * (a1 - a0)                                           # 戸口から椅子の北東の点へ
        p1 = wp + jp.array([jax.random.uniform(k[4], minval=0.05, maxval=0.35), jax.random.uniform(k[5], minval=-0.1, maxval=0.1)])
        p2 = pre + jax.random.uniform(k[6], (2,), minval=-0.05, maxval=0.05)
        tgt = jp.select([stage == 0, stage == 1], [wp, pre], pre)
        sp = jp.select([stage == 0, stage == 1], [p0, p1], p2)
        head = jnp_atan2v(tgt - sp)
        yaw_tbl = jp.where(stage == 2, jnp_atan2v(pre - wp), head)        # 向きを変える所：机の手前へ歩いてきた向き
        yaw = yaw_tbl + jax.random.uniform(k[7], minval=-0.15, maxval=0.15)
        free = jp.array([jax.random.uniform(k[8], minval=1.2, maxval=4.2), jax.random.uniform(k[9], minval=-0.7, maxval=1.5)])
        yaw_free = jax.random.uniform(jax.random.fold_in(k[9], 1), minval=-jp.pi, maxval=jp.pi)
        place = mode == 4
        ph0 = jp.select([stage == 0, stage == 1], [0, 1], 2)
        extra = {}
        if FLOW:                                         # A の続き：事務所で通した記録の状態から始める
            kf = jax.random.split(jax.random.fold_in(rng, 11), 2)
            st = jax.random.choice(kf[0], 5, p=jp.array(self._config.flow_p))
            fi = self._fl_idx_pad[st, jax.random.randint(kf[1], (), 0, 64)]
            ph0 = jp.select([st == 0, st == 1, st == 2, st == 3], [10, 11, 0, 1], 2)
            c_t0 = jp.where(st == 0, 10 ** 6, -jp.round(self._fl_tc[fi] / self.dt)).astype(jp.int32)
            # 箱が手に載る時刻：B から始めるときは配達員が放した時、C から始めるときは初めから
            extra = dict(fl_i=fi.astype(jp.int32), c_t0=c_t0, door_th0=self._fl_th0[fi],
                         onset=jp.where(place, jp.where(st == 0, never, 0), ep["onset"]).astype(jp.int32))
        return dict(sp_xy=jp.where(place, sp, free), sp_yaw=jp.where(place, yaw, yaw_free), stop_tgt=stop, **extra,
                    ph=jp.where(place, ph0, -1).astype(jp.int32),
                    ph_k=jp.int32(0), kb=jp.int32(-1),
                    t_stop=jp.where(place, never, never).astype(jp.int32), arm_t0=jp.int32(never),
                    release=jp.where(place, never, ep["release"]).astype(jp.int32), hp_f=jp.where(place, 0.0, ep["hp_f"]))

    def _grip_poses(self, center, quat, lower=0.0, open_=0.0):
        """配達員の左右の手の位置：ケースの中心 center・向き quat から（下げる量 lower、開く量 open_）。"""
        Rc = math.quat_to_mat(quat)
        ex, ey, ez = Rc[:, 0], Rc[:, 1], Rc[:, 2]
        off = CASE_HALF[1] + GRIP_HALF[1] - GRIP_SQUEEZE + open_
        base = center + ex * GRIP_X - ez * lower
        return jp.stack([base + ey * off, base - ey * off])

    def _place_case(self, data, ep, on_hands, pose=None):
        """練習の始めに、ケースを支えの面の上（持つ）、配達員の手の間（受け取る前）、離れた所の台の上（手ぶら）に置く。"""
        pos, quat, n = self._case_home(data, with_normal=True, pose=pose)
        gap = self._config.case.support_gap
        far = pos + jp.array([3.0, 3.0, 0.0])                    # 手ぶら：離れた所
        if self._config.office:                                  # 事務所では部屋の外の遠い所
            far = jp.array([20.0, 20.0, pos[2]])
        receive = (ep["mode"] == 1) & ~on_hands
        cpos = jp.where(ep["mode"] == 3, far, jp.where(on_hands, pos + n * 0.002, pos + n * gap))
        qpos = data.qpos.at[self._case_q:self._case_q + 3].set(cpos).at[self._case_q + 3:self._case_q + 7].set(quat)
        qvel = data.qvel.at[self._case_v:self._case_v + 6].set(0.0)
        sup = jp.where((ep["mode"] == 3), cpos - n * (CASE_HALF[2] + 0.01), FAR)   # 下からの台は手ぶらのときだけ
        grips = jp.where(receive, self._grip_poses(cpos, quat), jnp_far2())
        mocap = data.mocap_pos.at[self._support].set(sup).at[self._grips].set(grips)
        mq = data.mocap_quat.at[self._support].set(quat).at[self._grips].set(jp.stack([quat, quat]))
        return data.replace(qpos=qpos, qvel=qvel, mocap_pos=mocap, mocap_quat=mq)

    def _support_anchor(self, data, pose=None):
        """受け取る前に、配達員がケースを差し出したまま止めている場所（ケースの中心）：練習の始めの、支えの面の上のケースの
        位置から面に垂直に少し上。② の持ち方では、ケースの後ろの縁が胸の前面のすぐ前に来る。"""
        pos, _, n = self._case_home(data, with_normal=True, pose=pose)
        return pos + n * self._config.case.support_gap

    def _update_support(self, data, info):
        """配達員の手と台の動き。
        受け取る（mode 1）：受け取る前は、配達員の左右の手がケースを挟んで決まった場所（anchor）に止まる。受け取る時刻から
          0.2 秒で 1.2 cm 下げてケースを腕に載せ、0.3 秒で左右に 4 cm 開いて離し、そのあと手を離れた所へ引く
        降ろす（mode 2）：降ろす時刻から、下からの台でケースを持ち上げ、手の上に保つ（手からなくなる）
        手ぶら（mode 3）：ケースは離れた所の台の上"""
        c = self._config.case
        t = info["pl_t"]
        pos, quat, n = self._case_home(data, with_normal=True)
        mode = info["mode"]
        # 配達員の左右の手（受け取る）
        dt_on = jp.maximum(t - info["onset"], 0) * self.dt
        lower = jp.clip(dt_on / GRIP_LOWER_T, 0.0, 1.0) * GRIP_LOWER
        open_ = jp.clip((dt_on - GRIP_LOWER_T) / GRIP_OPEN_T, 0.0, 1.0) * GRIP_OPEN
        grips = self._grip_poses(info["anchor"], info["anchor_quat"], lower, open_)
        gone = dt_on > GRIP_LOWER_T + GRIP_OPEN_T + 0.1
        grips = jp.where((mode == 1) & ~gone, grips, jnp_far2())
        # 下からの台（降ろす・手ぶら）
        under = pos - n * (CASE_HALF[2] + 0.01)
        lift = jp.clip((t - info["release"]) * self.dt * 0.3, 0.0, c.lift)      # 0.3 m/秒で lift まで上げる
        p = jp.where((mode == 2) & (t >= info["release"]), under + jp.array([0.0, 0.0, 1.0]) * lift, FAR)
        p = jp.where(mode == 3, data.qpos[self._case_q:self._case_q + 3] * jp.array([1, 1, 0]) + jp.array([0, 0, 0.85]), p)
        mocap = data.mocap_pos.at[self._support].set(p).at[self._grips].set(grips)
        mq = data.mocap_quat.at[self._support].set(quat).at[self._grips].set(jp.stack([info["anchor_quat"]] * 2))
        return data.replace(mocap_pos=mocap, mocap_quat=mq)

    def _on_hands(self, info):
        return (info["pl_t"] >= info["onset"]) & (info["pl_t"] < info["release"]) & (info["mode"] != 3)

    def _estimate_raw(self, data):
        return jp.dot(self._est_w, data.actuator_force[N_ACT:]) + self._est_b

    # ------------------------------------------------------------ 始める・進める
    def reset(self, rng: jax.Array) -> mjx_env.State:
        qpos = self._init_q
        qvel = jp.zeros(self.mjx_model.nv)
        rng, key = jax.random.split(rng)
        qpos = qpos.at[0:2].add(jax.random.uniform(key, (2,), minval=-0.5, maxval=0.5))
        rng, key = jax.random.split(rng)
        yaw = jax.random.uniform(key, (1,), minval=-3.14, maxval=3.14)
        qpos = qpos.at[3:7].set(math.quat_mul(qpos[3:7], math.axis_angle_to_quat(jp.array([0, 0, 1]), yaw)))
        rng, k1, k2 = jax.random.split(rng, 3)
        # 脚と腰の初めの姿勢のずれ（元の ×U(0.5, 1.5) はケースが落ちやすいので ×U(0.8, 1.2)）
        qpos = qpos.at[7:7 + N_ACT].set(qpos[7:7 + N_ACT] * jax.random.uniform(k1, (N_ACT,), minval=0.8, maxval=1.2))
        qpos = qpos.at[7 + N_ACT:NQ_ROBOT].add(jax.random.uniform(k2, (29 - N_ACT,), minval=-0.03, maxval=0.03))
        rng, key = jax.random.split(rng)
        qvel = qvel.at[0:6].set(jax.random.uniform(key, (6,), minval=-0.2, maxval=0.2))
        data = mjx_env.make_data(self.mj_model, qpos=qpos, qvel=qvel, ctrl=qpos[7:NQ_ROBOT], impl=self.mjx_model.impl.value,
                                 naconmax=self._config.naconmax, njmax=self._config.njmax)
        data = mjx.forward(self.mjx_model, data)
        rng, ep_rng = jax.random.split(rng)
        ep = self._sample_episode(ep_rng)
        data = self._place_case(data, ep, ep["onset"] <= 0)
        data = mjx.forward(self.mjx_model, data)

        rng, key = jax.random.split(rng)
        gait_freq = jax.random.uniform(key, (1,), minval=1.25, maxval=1.5)
        rng, cmd_rng, push_rng = jax.random.split(rng, 3)
        push_interval = jax.random.uniform(push_rng, minval=self._config.push_config.interval_range[0],
                                           maxval=self._config.push_config.interval_range[1])
        info = {
            "rng": rng, "step": 0, "command": self.sample_command(cmd_rng),
            "last_act": jp.zeros(N_ACT), "last_last_act": jp.zeros(N_ACT), "motor_targets": jp.zeros(self.mjx_model.nu),
            "feet_air_time": jp.zeros(2), "last_contact": jp.zeros(2, dtype=bool), "swing_peak": jp.zeros(2),
            "phase_dt": 2 * jp.pi * self.dt * gait_freq, "phase": jp.array([0, jp.pi]),
            "push": jp.array([0.0, 0.0]), "push_step": 0,
            "push_interval_steps": jp.round(push_interval / self.dt).astype(jp.int32),
            "pl_t": 0, "m_eff": 0.0, "m_est": 0.0, "h_cmd": 0.0, "z_ref": 0.755, "desk_pos": FAR, "desk_quat": jp.array([1.0, 0, 0, 0]), "stop_xy": jp.zeros(2), "stop_yaw": 0.0, "ap_cmd": jp.zeros(3),
            "b_tgt": jp.zeros(3), "b_home": jp.zeros(3), "b_c1": jp.zeros(2), "b_yaw1": 0.0, "b_held": jp.array(False),
            "b_chold": jp.zeros(3), "b_qhold": jp.array([1.0, 0, 0, 0]), "b_qRh": jp.array([1.0, 0, 0, 0]),
            "bu_xy0": jp.zeros(2), "bu_th0": 0.0, "tgoal": 0.0, "adj_n": jp.int32(0), "adj_T": jp.int32(0), "adj_v": jp.zeros(2),
            "a_sub": jp.int32(0), "a_k": jp.int32(0), "a_still": jp.int32(0), "sh": jp.zeros(2), "sh_from": jp.zeros(2), "sh_to": jp.zeros(2),
            "shy": 0.0, "shy_from": 0.0, "shy_to": 0.0, **ep,
        }
        info["m_eff"] = jp.where(self._on_hands(info), self.mjx_model.body_mass[self._case_body], 0.0)
        info["anchor"] = self._support_anchor(data)
        info["anchor_quat"] = self._case_home(data)[1]
        metrics = {f"reward/{k}": jp.zeros(()) for k in self._config.reward_config.scales.keys()}
        metrics.update(swing_peak=jp.zeros(()), payload=jp.zeros(()), payload_est_err=jp.zeros(()), case_dropped=jp.zeros(()))
        contact = jp.array([data.sensordata[self._mj_model.sensor_adr[s]] > 0 for s in self._feet_floor_found_sensor])
        obs = self._get_obs(data, info, contact)
        return mjx_env.State(data, obs, jp.zeros(()), jp.zeros(()), metrics, info)

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        info = state.info
        info["rng"], push1_rng, push2_rng, ep_rng = jax.random.split(info["rng"], 4)
        push_theta = jax.random.uniform(push1_rng, maxval=2 * jp.pi)
        push_magnitude = jax.random.uniform(push2_rng, minval=self._config.push_config.magnitude_range[0],
                                            maxval=self._config.push_config.magnitude_range[1])
        push = jp.array([jp.cos(push_theta), jp.sin(push_theta)])
        push *= jp.mod(info["push_step"] + 1, info["push_interval_steps"]) == 0
        push *= self._config.push_config.enable
        # 止まっている間（指令 0）は押さない（止まっているときに押されて立て直すことは、ここでは考えない。2026-10-16）
        push *= jp.linalg.norm(info["command"]) >= 0.01
        data = state.data
        # 練習の始め（pl_t ＝ 0）には、練習の種類に合わせてケースを置き直し、台の止まる場所を決める。自動のリセットは
        # 体とケースを「最初の練習の始めの状態」に戻すだけなので、選び直した練習の種類とケースの置き方が合わなかった
        fresh = info["pl_t"] == 0
        pose = None
        if self._config.office:                            # 事務所：練習の始めの場所・向きへ体を置き直す
            yw = info["sp_yaw"]
            root = jp.array([info["sp_xy"][0], info["sp_xy"][1], self._init_q[2]])
            qt = data.qpos.at[0:3].set(root).at[3:7].set(jp.array([jp.cos(yw / 2), 0.0, 0.0, jp.sin(yw / 2)]))
            data_t = data.replace(qpos=qt, qvel=data.qvel.at[0:6].set(0.0))
            pose = (root, yw)
        else:
            data_t = data
        placed = self._place_case(data_t, info, info["onset"] <= 0, pose=pose)
        data = data.replace(qpos=jp.where(fresh, placed.qpos, data.qpos), qvel=jp.where(fresh, placed.qvel, data.qvel),
                            mocap_pos=jp.where(fresh, placed.mocap_pos, data.mocap_pos))
        info["anchor"] = jp.where(fresh, self._support_anchor(data, pose=pose), info["anchor"])
        info["anchor_quat"] = jp.where(fresh, self._case_home(data, pose=pose)[1], info["anchor_quat"])
        if FLOW:                                           # A の続き：事務所で通した記録の状態（体・箱・ドア）から始める
            fi = info["fl_i"]
            f4 = fresh & (info["mode"] == 4)
            q = data.qpos.at[0:NQ_ROBOT].set(jp.where(f4, self._fl_q[fi], data.qpos[0:NQ_ROBOT]))
            q = q.at[self._case_q:self._case_q + 7].set(jp.where(f4, self._fl_cq[fi], q[self._case_q:self._case_q + 7]))
            q = q.at[self._door_q].set(jp.where(f4, self._fl_door[fi], jp.where(fresh, 0.0, q[self._door_q])))
            v = data.qvel.at[0:NV_ROBOT].set(jp.where(f4, self._fl_v[fi], data.qvel[0:NV_ROBOT]))
            v = v.at[self._case_v:self._case_v + 6].set(jp.where(f4, self._fl_cv[fi], v[self._case_v:self._case_v + 6]))
            v = v.at[self._door_v].set(jp.where(fresh, 0.0, v[self._door_v]))
            data = data.replace(qpos=q, qvel=v)
            info["last_act"] = jp.where(f4, self._fl_last[fi], info["last_act"])
            info["phase"] = jp.where(f4, self._fl_phase[fi], info["phase"])
            info["m_est"] = jp.where(f4, self._fl_mest[fi], info["m_est"])
            info["b_held"] = jp.where(f4, info["ph"] == 10, info["b_held"])   # B から始めるときは、配達員が箱を持っている
        if self._config.office:                            # 事務所の C の歩き方の指令（置く練習）
            qq = data.qpos[3:7]
            th = jp.arctan2(2 * (qq[0] * qq[3] + qq[1] * qq[2]), 1 - 2 * (qq[2] ** 2 + qq[3] ** 2))
            info["sh"] = jp.where(fresh, 0.0, info["sh"])           # 寄せた上体は練習の始めに戻す
            info["shy"] = jp.where(fresh, 0.0, info["shy"])
            info = self._office_drive(info, data.qpos[0:2], th, data.qvel[0:2])
            push *= info["mode"] != 4
        # 受け取る前は、配達員がロボットの腕の位置に合わせて、ケースを差し出す位置を直す（0.05 秒で追いつく）。
        # 止まって待つ間もロボットは足踏みで少しずつ動く（3 秒で約 15 cm）ので、手を止めたままでは受け取る時にずれた。
        # 0.3 秒で追いつく速さでは、約 3 cm 遅れた
        follow = (info["mode"] == 1) & (info["pl_t"] < info["onset"]) & ~fresh
        a = self.dt / (0.05 + self.dt)
        tgt = self._support_anchor(data)
        info["anchor"] = jp.where(follow, info["anchor"] + a * (tgt - info["anchor"]), info["anchor"])
        info["anchor_quat"] = jp.where(follow, self._case_home(data)[1], info["anchor_quat"])
        # 受け取る練習では、受け取る前から受け取った 1 秒後まで、止まって待つ
        wait = (info["mode"] == 1) & (info["pl_t"] < info["onset"] + int(round(1.0 / self.dt)))
        info["command"] = jp.where(wait, jp.zeros(3), info["command"])
        if self._config.office:
            info["command"] = jp.where(info["mode"] == 4, info["ap_cmd"], info["command"])
        # 置く練習：t_stop から止まる（指令 0）。全身を止めて待ち終わった時の骨盤の位置と向きから、前に机を出す
        stopped = info["pl_t"] >= info["t_stop"]
        if self._config.approach:                          # 事務所と同じ近づき方：止まる → その場で向きを変える → ゆっくり前へ
            t_turn = info["ap_t0"] + int(round(0.5 / self.dt))
            t_fwd = t_turn + info["ap_turn"]
            cmd_ap = jp.where(info["pl_t"] < t_turn, jp.zeros(3),
                              jp.where(info["pl_t"] < t_fwd, jp.array([0.0, 0.0, 1.0]) * info["ap_w"],
                                       jp.array([0.2, 0.0, 0.0]) + jp.array([0.0, 1.0, 0.0]) * info["ap_vy"]))
            info["command"] = jp.where((info["mode"] == 4) & (info["pl_t"] >= info["ap_t0"]), cmd_ap, info["command"])
        decel = jp.clip((info["t_stop"] - info["pl_t"]) * self.dt / DECEL_T, 0.0, 1.0)     # 止まる前の 1 秒で少しずつ遅く
        info["command"] = jp.where(stopped, jp.zeros(3), info["command"] * jp.where(info["mode"] == 4, decel, 1.0))
        # 体を沈める量（腕を動かし始めてから：2 秒で下げ、4.5〜6.5 秒で戻す）
        tc = (info["pl_t"] - info["arm_t0"]) * self.dt
        sm = lambda u: (lambda x: x * x * (3 - 2 * x))(jp.clip(u, 0.0, 1.0))
        h = jp.where(tc < CROUCH_DOWN, sm(tc / CROUCH_DOWN),
                     jp.where(tc < CROUCH_HOLD_END, 1.0, 1.0 - sm((tc - CROUCH_HOLD_END) / (CROUCH_UP_END - CROUCH_HOLD_END))))
        info["h_cmd"] = jp.where(tc >= 0.0, h * self._crouch_H, 0.0)
        if self._touch_mode():                             # 触れたら切り替える表：触れる前は 3 秒下ろし続ける表（どの表も触れるまでは同じ）
            k_pl = jp.clip(info["pl_t"] - info["arm_t0"], 0, self._touch_len - 1)
            j_tab = jp.where(info["touch_k"] < 10 ** 6, info["touch_k"], self._touch_tab.shape[0]) - 1
            row = self._touch_tab[j_tab, k_pl]
            info["h_cmd"] = jp.where(tc >= 0.0, row[14], 0.0)
        info["z_ref"] = jp.where(info["pl_t"] == info["arm_t0"], data.qpos[2], info["z_ref"])
        pel = data.xpos[self._mj_model.body("pelvis").id]
        Rp = data.xmat[self._mj_model.body("pelvis").id].reshape(3, 3)
        yaw = jp.arctan2(Rp[1, 0], Rp[0, 0])
        cy, sy = jp.cos(yaw), jp.sin(yaw)
        lx, ly = PLACE_EDGE + DESK_HALF[0] + info["desk_off"][0], info["desk_off"][1]
        lx = lx - PLACE_EDGE + self._config.case.place_edge
        if self._config.desk_body:                       # 高さは事務所の机と同じ。desk_dz は向きのずれ
            dyaw = yaw + info["desk_dz"]
            # 机の脚が足と重なって出ないように（carry22 では、止まった時に机の脚が右のすね・ももと重なって出て、体がはじかれた）、
            # 脚と足の間が 10 cm より近ければ、その分だけ机を前へずらす（4 cm まで）
            cd, sd = jp.cos(dyaw), jp.sin(dyaw)
            base = jp.array([pel[0] + cy * lx - sy * ly, pel[1] + sy * lx + cy * ly])
            feet = data.site_xpos[self._feet_site_id][:, :2]
            gap = jp.array(10.0)
            for lp in self._leg_local:
                w = base + jp.array([cd * lp[0] - sd * lp[1], sd * lp[0] + cd * lp[1]])
                gap = jp.minimum(gap, jp.min(jp.linalg.norm(feet - w, axis=1)))
            push_ = jp.clip(0.10 - gap, 0.0, 0.04)
            base = base + push_ * jp.array([cy, sy])
            dpos = jp.array([base[0], base[1], DESK_TOP - DESK_HALF[2]])
        else:
            dpos = jp.array([pel[0] + cy * lx - sy * ly, pel[1] + sy * lx + cy * ly, DESK_TOP - DESK_HALF[2] + info["desk_dz"]])
            dyaw = yaw
        # 止まって待ち終わった所の前に机を出す。触れたら切り替える表のときは、止まった時（指令が 0 になった時）に出す（事務所の
        # 机は動かないので、止まってから体が流れると机との位置もずれる）
        at_stop = info["pl_t"] == (info["t_stop"] if self._touch_mode() else info["arm_t0"])
        info["stop_xy"] = jp.where(info["pl_t"] == info["t_stop"], pel[:2], info["stop_xy"])
        info["stop_yaw"] = jp.where(info["pl_t"] == info["t_stop"], yaw, info["stop_yaw"])
        if self._config.office:                            # 事務所の机は最初からその場所（西を向いて置くので、机の座標の +x は西）
            info["desk_pos"] = self._office_desk
            info["desk_quat"] = jp.array([0.0, 0.0, 0.0, 1.0])
        else:
            info["desk_pos"] = jp.where(at_stop, dpos, jp.where(fresh, FAR, info["desk_pos"]))
            info["desk_quat"] = jp.where(at_stop, jp.array([jp.cos(dyaw / 2), 0.0, 0.0, jp.sin(dyaw / 2)]), info["desk_quat"])
            data = data.replace(mocap_pos=data.mocap_pos.at[self._desk].set(info["desk_pos"]),
                                mocap_quat=data.mocap_quat.at[self._desk].set(info["desk_quat"]))
        data = data.replace(qvel=data.qvel.at[:2].set(push * push_magnitude + data.qvel[:2]))
        data = self._update_support(data, info)
        if FLOW:                                           # 配達員の手渡しとドアを閉める手（事務所の場面と同じ）
            data, info = self._flow_handover(data, info)
            data = self._flow_door_hand(data, info)
        # ドアを押す反動：両手に、体の後ろ向きの力
        R = data.xmat[self._mj_model.body("pelvis").id].reshape(3, 3)
        back = -jp.array([R[0, 0], R[1, 0], 0.0])
        hp_on = (info["pl_t"] >= info["hp_t0"]) & (info["pl_t"] < info["hp_t1"])
        f = back * info["hp_f"] * hp_on
        data = data.replace(xfrc_applied=data.xfrc_applied.at[self._wrists, 0:3].set(jp.stack([f, f])))

        motor_targets = self._default_pose.at[:N_ACT].add(action * self._config.action_scale)
        if self._config.crouch_obs and self._config.crouch_ff:  # 沈める量に合わせたしゃがむ姿勢を先に足す
            x = jp.clip(info["h_cmd"] / CROUCH_FF_DH, 0.0, self._ff_table.shape[0] - 1.0)
            i = jp.minimum(jp.floor(x).astype(jp.int32), self._ff_table.shape[0] - 2)
            v = self._ff_table[i] + (x - i) * (self._ff_table[i + 1] - self._ff_table[i])
            motor_targets = motor_targets.at[jp.array(CROUCH_FF_IDX)].add(jp.concatenate([v, v]))
            if self._config.crouch_lean:                      # 股関節から上体を前へ倒す
                motor_targets = motor_targets.at[jp.array(HIP_PITCH_IDX)].add(
                    -float(np.radians(self._config.crouch_lean)) * info["h_cmd"] / max(self._crouch_H, 1e-6))
            if self._config.crouch_waist:                     # 腰の pitch を前へ（胴をまっすぐに）
                motor_targets = motor_targets.at[WAIST_PITCH_IDX].add(WAIST_FF_PER_M * info["h_cmd"])
        if self._config.office and self._config.adjust > 0:   # 寄せた上体（両足を着けたまま骨盤を前後・左右へ）：脚の角度を先に足す
            motor_targets = motor_targets.at[:12].add(shift_ff_j(info["sh"], self._shift_table))
            # 骨盤の向き：両足を着けたまま、左右の股の向き（hip yaw）を同じだけ逆へ回すと、骨盤が回る
            motor_targets = motor_targets.at[jp.array(HIP_YAW_IDX)].add(YAW_SIGN * info["shy"])
        # 置く練習：全身を止めて settle 待ってから、腕で置く動き（まっすぐ下げて、手を左右へ抜く）をなぞる
        k_arm = jp.clip(info["pl_t"] - info["arm_t0"], 0, self._place_traj.shape[0] - 1)
        placing = info["pl_t"] >= info["arm_t0"]
        arm_rows = row[:14] if self._touch_mode() else self._place_traj[k_arm]
        motor_targets = jp.where(placing, motor_targets.at[N_ACT:].set(arm_rows), motor_targets)
        if FLOW:                                           # 受け渡しの前に、ドアを押した腕から運ぶ姿勢へ 1.5 秒で移す
            ua = jp.clip(info["ph_k"] * self.dt / T_ARM, 0.0, 1.0)
            ua = ua * ua * (3 - 2 * ua)
            arm_from = self._fl_q[info["fl_i"]][7 + N_ACT:NQ_ROBOT]
            blend = (1 - ua) * arm_from + ua * self._default_pose[N_ACT:]
            motor_targets = jp.where((info["mode"] == 4) & (info["ph"] == 10), motor_targets.at[N_ACT:].set(blend), motor_targets)
        data = mjx_env.step(self.mjx_model, data, motor_targets, self.n_substeps)
        info["motor_targets"] = motor_targets
        rd = self._robot(data)
        if self._touch_mode():                             # 下ろしている間に机に触れたら（3 秒たったら）、次の歩から開く
            k_low = info["pl_t"] - info["arm_t0"] - self._touch_n1 + 1      # 下ろし始めて何歩目か（1〜150）
            lowering = (info["touch_k"] >= 10 ** 6) & (k_low >= 1) & (info["mode"] == 4)
            hit = lowering & (self._desk_touched(data, info) | (k_low >= self._touch_tab.shape[0]))
            info["touch_k"] = jp.where(hit, jp.clip(k_low, 1, self._touch_tab.shape[0]), info["touch_k"]).astype(jp.int32)
            # 箱は開き始めて約 0.4 秒で机へ移る（それからは手の上のずれ・落としたかを数えない）
            info["release"] = jp.where(hit, info["pl_t"] + int(round(0.3 / self.dt)), info["release"]).astype(jp.int32)

        contact = jp.array([data.sensordata[self._mj_model.sensor_adr[s]] > 0 for s in self._feet_floor_found_sensor])
        contact_filt = contact | info["last_contact"]
        first_contact = (info["feet_air_time"] > 0.0) * contact_filt
        info["feet_air_time"] += self.dt
        info["swing_peak"] = jp.maximum(info["swing_peak"], data.site_xpos[self._feet_site_id][..., -1])

        on = self._on_hands(info)
        info["m_eff"] = jp.where(on, self.mjx_model.body_mass[self._case_body], 0.0)
        alpha = self.dt / (self._config.payload.est_tau + self.dt)
        info["m_est"] = jp.clip(info["m_est"] + alpha * (self._estimate_raw(data) - info["m_est"]), 0.0, 1.5)

        # ケースを落としたか（手に載っているはずの間）
        home, _ = self._case_home(data)
        cpos = data.qpos[self._case_q:self._case_q + 3]
        gap = cpos - home
        dropped = on & ((gap[2] < -0.25) | (jp.linalg.norm(gap[:2]) > 0.40))
        slip = jp.where(on, jp.sum(jp.square(gap)), 0.0)

        obs = self._get_obs(data, info, contact)
        fell = self._get_termination(rd)
        # 物理の計算が壊れた体（NaN）は練習をやり直させ、その歩の観測と報酬を 0 にする。8192 体で長く学習すると、ごくまれに
        # 1 体の計算が壊れ、その NaN が学習全体に広がって方策が壊れた（carry7 の最初の学習、2026-10-16）
        broken = ~(jp.isfinite(data.qpos).all() & jp.isfinite(data.qvel).all()
                   & jp.all(jp.array([jp.isfinite(v).all() for v in jax.tree.leaves(obs)])))
        obs = jax.tree.map(lambda v: jp.where(broken, jp.zeros_like(v), v), obs)
        done = fell | dropped | broken
        rewards = self._get_reward(rd, action, info, state.metrics, fell, first_contact, contact)
        rewards["case_drop"] = dropped.astype(jp.float32)
        # 止まっている間（指令 0）は全身を止める
        stand = jp.linalg.norm(info["command"]) < 0.01
        feet_v = data.sensordata[self._foot_linvel_sensor_adr]
        rewards["stand_feet_vel"] = stand * jp.sum(jp.linalg.norm(feet_v[..., :2], axis=-1))
        rewards["stand_feet_air"] = stand * jp.sum(~contact)
        rewards["stand_base_vel"] = stand * jp.linalg.norm(self.get_local_linvel(data, "pelvis")[:2])
        crouching = info["h_cmd"] > 0.002
        rewards["crouch_height"] = crouching * jp.abs(data.qpos[2] - (info["z_ref"] - info["h_cmd"]))
        yaw_ref = 2.0 * jp.arctan2(info["desk_quat"][3], info["desk_quat"][0])   # 腕を動かし始めた時の向き（机の向き）
        if self._config.desk_body:                       # 机の向きはずらしてあるので、止まった時の体の向き
            yaw_ref = yaw_ref - info["desk_dz"]
        if self._config.office:                          # 止まった時の体の向き
            yaw_ref = info["stop_yaw"]
        yaw_t0 = info["t_stop"] if self._touch_mode() else info["arm_t0"]
        rewards["place_yaw"] = (info["pl_t"] >= yaw_t0) * jp.abs(jp.arctan2(jp.sin(yaw - yaw_ref), jp.cos(yaw - yaw_ref)))
        # 止まってから骨盤が流れた距離。drift_all のときは置き終わるまで数える（carry22 は置くあいだに机の方へ足を出し、胴を机に
        # 寄せた）
        drift_end = info["arm_t0"] + (self._touch_len if (self._config.drift_all and self._touch_mode()) else 0)
        rewards["stop_drift"] = ((info["pl_t"] >= info["t_stop"]) & (info["pl_t"] < drift_end)) * jp.linalg.norm(
            pel[:2] - info["stop_xy"])
        # 事務所：止まる位置の目標とのずれ。carry25 は止まってから置き終わるまで毎歩数えたので、置きながら体を動かしてずれを
        # 直そうとして倒れる回が増えた。carry26 からは、止まって 2 秒待ち終わった時（置き始める時。事務所の C の成功を決める時）の
        # 一度だけ数える（1 歩の報酬は dt 倍されるので、dt で割って「ずれ（m）× 重み」がそのまま 1 回の減点になるようにする）
        if self._config.office and self._config.stop_err_once:
            ey_ = jp.abs(wrap_j(yaw - (jp.pi - OFFICE_YAW_BIAS))) * (0.5 if self._config.yaw_shift else 0.0)   # 向き 0.1 rad を 5 cm と同じに
            rewards["stop_err"] = ((info["mode"] == 4) & (info["pl_t"] == info["arm_t0"])) * (jp.linalg.norm(
                pel[:2] - info["stop_tgt"]) + ey_) / self.dt
        elif self._config.office:
            rewards["stop_err"] = ((info["mode"] == 4) & (info["ph"] >= 5) & (info["ph"] <= 6)) * jp.linalg.norm(pel[:2] - info["stop_tgt"])
        else:
            rewards["stop_err"] = jp.array(0.0)
        for k in ("stand_still", "pose", "joint_deviation_knee", "joint_deviation_hip"):       # 膝・股を曲げることは減点しない
            if k in rewards:
                rewards[k] = rewards[k] * (1.0 - crouching)
        # 手の上のずれ（ずれの 2 乗）。5 cm のずれで 1（それ以上は 1 のまま）。前は 20 cm で 1 だったので、小さなずれに効かず、
        # 歩いているうちにケースが少しずつずれて、終わりの方で落ちた
        rewards["case_slip"] = jp.minimum(slip / 0.0025, 1.0)
        rewards = {k: v * self._config.reward_config.scales[k] for k, v in rewards.items()}
        reward = sum(rewards.values()) * self.dt
        reward = jp.where(broken | ~jp.isfinite(reward), 0.0, reward)

        info["push"] = push
        info["step"] += 1
        info["push_step"] += 1
        phase_tp1 = info["phase"] + info["phase_dt"]
        info["phase"] = jp.fmod(phase_tp1 + jp.pi, 2 * jp.pi) - jp.pi
        # 止まっている間（指令 0）は、歩く周期の時計を決まった値に止める（足踏みを続けさせない）
        info["phase"] = jp.where(jp.linalg.norm(info["command"]) < 0.01, jp.array(STAND_PHASE), info["phase"])
        info["last_last_act"] = info["last_act"]
        info["last_act"] = action
        info["rng"], cmd_rng = jax.random.split(info["rng"])
        info["command"] = jp.where(info["step"] > 500, self.sample_command(cmd_rng), info["command"])
        info["step"] = jp.where(done | (info["step"] > 500), 0, info["step"])
        info["feet_air_time"] *= ~contact
        info["last_contact"] = contact
        info["swing_peak"] *= ~contact
        # 練習をやり直すとき（倒れた・落とした・打ち切り）は、時間と練習の種類を決め直す
        restart = done | (info["pl_t"] + 1 >= self._config.episode_length)
        ep = self._sample_episode(ep_rng)
        info["pl_t"] = jp.where(restart, 0, info["pl_t"] + 1)
        for k in ep:
            info[k] = jp.where(restart, ep[k], info[k])
        info["m_est"] = jp.where(restart, 0.0, info["m_est"])
        for k, v in rewards.items():
            state.metrics[f"reward/{k}"] = v
        state.metrics.update(swing_peak=jp.mean(info["swing_peak"]), payload=info["m_eff"],
                             payload_est_err=jp.abs(info["m_est"] - info["m_eff"]), case_dropped=dropped.astype(jp.float32))
        return state.replace(data=data, obs=obs, reward=reward, done=done.astype(reward.dtype), info=info)

    # ------------------------------------------------------------ 観測（③ と同じ形）・報酬
    def _get_obs(self, data, info, contact):
        nz = self._config.noise_config
        info["rng"], *keys = jax.random.split(info["rng"], 6)

        def noisy(x, scale, key):
            return x + (2 * jax.random.uniform(key, shape=x.shape) - 1) * nz.level * scale

        gyro = self.get_gyro(data, "pelvis")
        gravity = data.site_xmat[self._pelvis_imu_site_id].T @ jp.array([0, 0, -1])
        q = data.qpos[7:NQ_ROBOT] - self._default_pose
        dq = data.qvel[6:NV_ROBOT]
        linvel = self.get_local_linvel(data, "pelvis")
        phase = jp.concatenate([jp.cos(info["phase"]), jp.sin(info["phase"])])
        state = jp.hstack([
            noisy(linvel, nz.scales.linvel, keys[0]), noisy(gyro, nz.scales.gyro, keys[1]),
            noisy(gravity, nz.scales.gravity, keys[2]), info["command"],
            noisy(q[:N_ACT], nz.scales.joint_pos, keys[3]), noisy(dq[:N_ACT], nz.scales.joint_vel, keys[4]),
            info["last_act"], phase, jp.array([info["m_est"] / 3.0]),
        ] + ([jp.array([info["h_cmd"] / 0.1])] if self._config.crouch_obs else []))
        privileged = jp.hstack([
            state, gyro, self.get_accelerometer(data, "pelvis"), gravity, linvel,
            self.get_global_angvel(data, "pelvis"), q, dq, data.qpos[2], data.actuator_force, contact,
            data.sensordata[self._foot_linvel_sensor_adr].ravel(), info["feet_air_time"], info["m_eff"] / 3.0,
        ])
        return {"state": state, "privileged_state": privileged}

    def _cost_pose(self, qpos):
        return jp.sum(jp.square(qpos[:N_ACT] - self._default_pose[:N_ACT]))

    def _cost_stand_still(self, commands, qpos):
        return jp.sum(jp.abs(qpos[:N_ACT] - self._default_pose[:N_ACT])) * (jp.linalg.norm(commands) < 0.01)


def domain_randomize(model: mjx.Model, rng: jax.Array, env: CarryJoystick):
    """床の摩擦・関節の摩擦・リンクの質量・胴の質量に、ケースの重さ（1.5〜2.5 kg）とケースの摩擦を加える。"""
    c = env._config.case
    torso = env.mj_model.body("torso_link").id
    cb = env._case_body
    cpairs = jp.array(env._case_pair)

    @jax.vmap
    def rand(rng):
        rng, k = jax.random.split(rng)
        friction = jax.random.uniform(k, minval=0.4, maxval=1.0)
        pair_friction = model.pair_friction.at[0:2, 0:2].set(friction)
        rng, k = jax.random.split(rng)
        cf = jax.random.uniform(k, minval=c.friction_range[0], maxval=c.friction_range[1])
        if env._config.desk_body or env._config.office:  # 事務所の場面の試験と同じ範囲：床 0.5〜1.0、箱と手 0.8〜1.2、箱と机 0.3〜0.6
            rng, k1, k2 = jax.random.split(rng, 3)
            pair_friction = model.pair_friction.at[jp.array(env._floor_pair), 0:2].set(jax.random.uniform(k1, minval=0.5, maxval=1.0))
            pair_friction = pair_friction.at[jp.array(env._pad_pair), 0:2].set(cf)
            pair_friction = pair_friction.at[jp.array(env._desk_case_pair), 0:2].set(jax.random.uniform(k2, minval=0.3, maxval=0.6))
        else:
            pair_friction = pair_friction.at[cpairs, 0:2].set(cf)
        rng, k = jax.random.split(rng)
        dof_frictionloss = model.dof_frictionloss.at[6:NV_ROBOT].set(
            model.dof_frictionloss[6:NV_ROBOT] * jax.random.uniform(k, shape=(29,), minval=0.5, maxval=2.0))
        rng, k = jax.random.split(rng)
        body_mass = model.body_mass * jax.random.uniform(k, shape=(model.nbody,), minval=0.9, maxval=1.1)
        rng, k = jax.random.split(rng)
        body_mass = body_mass.at[torso].add(jax.random.uniform(k, minval=-1.0, maxval=1.0))
        rng, k = jax.random.split(rng)
        cm = jax.random.uniform(k, minval=c.mass_range[0], maxval=c.mass_range[1])
        body_mass = body_mass.at[cb].set(cm)
        scale = cm / model.body_mass[cb]
        body_inertia = model.body_inertia.at[cb].set(model.body_inertia[cb] * scale)
        return pair_friction, dof_frictionloss, body_mass, body_inertia

    pair_friction, frictionloss, body_mass, body_inertia = rand(rng)
    in_axes = jax.tree_util.tree_map(lambda x: None, model)
    in_axes = in_axes.tree_replace({"pair_friction": 0, "dof_frictionloss": 0, "body_mass": 0, "body_inertia": 0})
    model = model.tree_replace({"pair_friction": pair_friction, "dof_frictionloss": frictionloss,
                                "body_mass": body_mass, "body_inertia": body_inertia})
    return model, in_axes
