#!/usr/bin/env python3
"""
事務所の場面で、学習した方策を順に切り替えて通しで動かす（CPU の MuJoCo。方策の計算も CPU の JAX）。

いまの流れ（段階 A の続き）：
  机に向かって椅子に座っている → ノックの合図（GO 秒）→ A で立ち上がる → 歩く方策に切り替え、腕を運ぶ姿勢へ →
  その場で左（北、机のない広い側）を向く → 北へ歩いて椅子から離れる → ドアの方（東）を向く → 腕を振ってまっすぐドアの前まで歩く →
  両手を前に出し、ゆっくり前へ歩いてドアを押し開ける → 少し下がって戸口の中で止まる

- 物理：事務所の場面（scripts/office_scene.py）。ドアは蝶番で回る（外開き）。当たり判定は、椅子の座面 ↔ もも・すね・骨盤、
  机 ↔ 手・胴、ドア ↔ 手・胴、足 ↔ 床。学習と同じく、ほかの部分どうしは当たらない
- 方策の入力は、それぞれの学習の環境（standup_env・payload_env）と同じ作り方（雑音は入れない）
- 歩く方策：B ができるまでは ③（models/estimate、腕で測った重さを入力）を使う

    JAX_PLATFORMS=cpu .venv_mjx/bin/python scripts/deploy.py --video logs/to_door.mp4
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
    # CPU で動かすときは GPU を使わない（物理計算の部品 Warp が GPU のメモリを確保し、GPU の学習がメモリ不足で止まった）
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import jax  # noqa: E402
import jax.numpy as jp  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import office_scene as office  # noqa: E402
import payload_env as pe  # noqa: E402
import policy_io  # noqa: E402
import standup_env as se  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
DT, N_SUB, N_ACT = 0.02, 10, 15
WALK_SCALE = 0.5                     # 歩く方策の出力の大きさ（Playground の G1 と同じ）
# 何も持たないで歩くときの腕：体の横に下ろし（左腕。右は roll・yaw が逆）、反対側の脚の前後の動きに合わせて振る
ARM_REST = {"shoulder_pitch": 0.15, "shoulder_roll": 0.15, "shoulder_yaw": 0.0, "elbow": 0.5,
            "wrist_roll": 0.0, "wrist_pitch": 0.0, "wrist_yaw": 0.0}
SWING_GAIN, SWING_MAX = 0.9, 0.45     # 腕の振り ＝ 反対側の股の前後の角度 × SWING_GAIN（±SWING_MAX rad まで）
GAIT_FREQ = 1.375                    # 歩く周期の速さ（学習では 1.25〜1.5 のランダム）


def office_decorate(spec):
    """事務所（ドアは蝶番で回る）＋骨盤・胴の当たり判定＋接触の組。"""
    office.decorate(spec, hinged_door=True)
    spec.body("pelvis").add_geom(name="pelvis_col", type=mujoco.mjtGeom.mjGEOM_BOX, pos=[-0.02, 0, -0.07],
                                 size=[0.07, 0.11, 0.04], contype=0, conaffinity=0, group=3)
    spec.body("torso_link").add_geom(name="torso_col", type=mujoco.mjtGeom.mjGEOM_CAPSULE, fromto=[0.02, 0, 0.05, 0.02, 0, 0.40],
                                     size=[0.11, 0, 0], contype=0, conaffinity=0, group=3)
    pairs = [("pelvis_col", "chair"), ("left_thigh", "chair"), ("right_thigh", "chair"), ("left_shin", "chair"),
             ("right_shin", "chair"), ("left_hand_collision", "desk"), ("right_hand_collision", "desk"), ("torso_col", "desk"),
             ("left_hand_collision", "door"), ("right_hand_collision", "door"), ("torso_col", "door")]
    for a, b in pairs:
        p = spec.add_pair(geomname1=a, geomname2=b)
        p.condim = 3
        p.friction = [0.8, 0.8, 0.005, 0.0001, 0.0001]


def yaw_of(q):
    return math.atan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2))


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


class Robot:
    def __init__(self, walk_name="estimate", standup_name="standup"):
        self.m = pe.build_model(decorate=office_decorate)
        for s in pe.SIDES:                                        # payload_env の荷物の体は使わない
            b = self.m.body(f"{s}_payload").id
            self.m.body_mass[b] = 0.0
            self.m.body_inertia[b] = 1e-8
        self.m.opt.timestep = DT / N_SUB
        dh = self.m.joint("door_hinge").dofadr[0]
        self.m.dof_damping[dh], self.m.dof_frictionloss[dh] = office.DOOR_DAMPING, office.DOOR_FRICTION
        self.d = mujoco.MjData(self.m)
        m = self.m
        self.pelvis = m.body("pelvis").id
        self.imu = m.site("imu_in_pelvis").id
        self.gyro_adr = m.sensor_adr[m.sensor("gyro_pelvis").id]
        self.linvel_adr = m.sensor_adr[m.sensor("local_linvel_pelvis").id]
        self.door_q = m.joint("door_hinge").qposadr[0]
        # A（立ち上がる）
        scfg = se.default_config()
        scfg.impl = "jax"
        senv = se.StandUp(scfg)
        sconf = json.loads((REPO / "models" / standup_name / "config.json").read_text())
        self.pol_a = jax.jit(policy_io.load_policy(standup_name, senv, sconf))
        self.priv_a = jp.zeros(senv.observation_size["privileged_state"])
        self.ref = senv._ref_np
        self.arm_a = self.ref[0, 7 + N_ACT:]
        # 歩く方策
        wconf = json.loads((REPO / "models" / walk_name / "config.json").read_text())
        wcfg = pe.default_config()
        wcfg.impl = "jax"
        wcfg.obs_mode = wconf["env"]["obs_mode"]
        wcfg.estimator.w, wcfg.estimator.b = wconf["env"]["estimator"]["w"], wconf["env"]["estimator"]["b"]
        wenv = pe.PayloadJoystick(wcfg)
        self.pol_w = jax.jit(policy_io.load_policy(walk_name, wenv, wconf))
        self.priv_w = jp.zeros(wenv.observation_size["privileged_state"])
        self.default = np.array(wenv._default_pose)               # 脚・腰は knees_bent、腕は運ぶ姿勢
        self.est_w, self.est_b = np.array(wcfg.estimator.w), wcfg.estimator.b
        self.est_alpha = DT / (wcfg.payload.est_tau + DT)
        self.key = jax.random.PRNGKey(0)
        self.t = 0.0
        self.frames = []

    # ------------------------------------------------------------ 共通
    def place_seated(self):
        q = self.ref[0].copy()
        x, y = q[0], q[1]
        q[0], q[1] = office.FEET_X - x, -y
        qw, qx, qy, qz = q[3:7]
        q[3:7] = (-qz, -qy, qx, qw)
        self.d.qpos[:len(q)] = q
        self.d.ctrl[:] = q[7:7 + 29]
        mujoco.mj_forward(self.m, self.d)

    def gravity(self):
        return self.d.site_xmat[self.imu].reshape(3, 3).T @ np.array([0, 0, -1.0])

    def gyro(self):
        return self.d.sensordata[self.gyro_adr:self.gyro_adr + 3].copy()

    def pose(self):
        q = self.d.qpos
        return q[0], q[1], yaw_of(q[3:7])

    def step_physics(self, target):
        self.d.ctrl[:] = target
        for _ in range(N_SUB):
            mujoco.mj_step(self.m, self.d)
        self.t += DT
        self.frames.append(self.d.qpos.copy())

    # ------------------------------------------------------------ A 立ち上がる
    def stand_up(self, go_sec, after=1.0):
        """go_sec 秒まで座って待ち、合図で立ち上がる。立ち終わって after 秒たつまで A を使う。"""
        T = len(self.ref)
        go = int(round(go_sec / DT))
        last = np.zeros(N_ACT)
        n_steps = go + T + int(round(after / DT))
        for k in range(n_steps):
            i = min(max(k - go, 0), T - 1)
            q_ref = self.ref[i, 7:7 + N_ACT]
            q_next = self.ref[min(i + 5, T - 1), 7:7 + N_ACT]
            t_since = max(k - go, -50) * DT
            phase = np.clip(t_since / (T * DT), -0.5, 1.2)
            q = self.d.qpos[7:7 + N_ACT]
            obs = np.concatenate([self.gyro(), self.gravity(), q - q_ref, self.d.qvel[6:6 + N_ACT] * 0.1, q_ref,
                                  q_next - q_ref, last, [phase, float(k >= go)]])
            act = np.array(self.pol_a({"state": jp.array(obs), "privileged_state": self.priv_a}, self.key)[0])
            legs = q_ref + (act * 0.6 if k >= go else 0.0)
            # 学習の環境と同じく、行動は「次の 1 歩」で使う（step の中で obs を作るのは行動の後）
            self.step_physics(np.concatenate([legs, self.arm_a]))
            last = act
        return self.d.qpos[2]

    # ------------------------------------------------------------ 歩く
    def walk_init(self):
        self.phase = np.array([0.0, math.pi])
        self.phase_dt = 2 * math.pi * DT * GAIT_FREQ
        self.last_w = np.zeros(N_ACT)
        self.m_est = 0.0
        self.arm_from = self.d.qpos[7 + N_ACT:7 + 29].copy()
        self.arm_t = 0.0
        self.arm_mode = "swing"                                   # swing：下ろして振る、carry：前に出す（運ぶ・押す）
        self.swing = np.zeros(2)
        rest = []
        for s_ in pe.SIDES:
            for j in pe.ARM_JOINTS:
                v = ARM_REST[j]
                rest.append(-v if (s_ == "right" and j in pe.MIRROR) else v)
        self.arm_rest = np.array(rest)
        self.hip = [self.m.joint(f"{s_}_hip_pitch_joint").qposadr[0] for s_ in pe.SIDES]

    def set_arms(self, mode):
        """腕の姿勢を切り替える（1 秒かけて移す）。"""
        if mode != self.arm_mode:
            self.arm_from = self.d.qpos[7 + N_ACT:7 + 29].copy()
            self.arm_t = 0.0
            self.arm_mode = mode

    def arm_target(self, moving):
        if self.arm_mode == "carry":
            goal = self.default[N_ACT:].copy()
        else:
            goal = self.arm_rest.copy()
            # 腕の振り：左腕は右の股、右腕は左の股の前後の角度（立ち姿勢から）に合わせる（ゆっくり追う）
            hip_dev = np.array([self.d.qpos[self.hip[1]], self.d.qpos[self.hip[0]]]) - self.default[[6, 0]]
            want = np.clip(SWING_GAIN * hip_dev, -SWING_MAX, SWING_MAX) * float(moving)
            self.swing += 0.25 * (want - self.swing)
            goal[0] += self.swing[0]
            goal[7] += self.swing[1]
        self.arm_t = min(self.arm_t + DT, 1.0)
        u = self.arm_t * self.arm_t * (3 - 2 * self.arm_t)
        return (1 - u) * self.arm_from + u * goal

    def walk_step(self, cmd, arm_blend=1.0):
        q = self.d.qpos[7:7 + 29] - self.default
        linvel = self.d.sensordata[self.linvel_adr:self.linvel_adr + 3].copy()
        ph = np.concatenate([np.cos(self.phase), np.sin(self.phase)])
        obs = np.concatenate([linvel, self.gyro(), self.gravity(), cmd, q[:N_ACT], self.d.qvel[6:6 + N_ACT], self.last_w, ph,
                              [self.m_est / 3.0]])
        act = np.array(self.pol_w({"state": jp.array(obs), "privileged_state": self.priv_w}, self.key)[0])
        arm = self.arm_target(moving=np.linalg.norm(cmd) > 0.05)
        target = np.concatenate([self.default[:N_ACT] + act * WALK_SCALE, arm])
        self.step_physics(target)
        raw = float(self.est_w @ self.d.actuator_force[N_ACT:29] + self.est_b)
        self.m_est += self.est_alpha * (raw - self.m_est)
        self.phase = np.fmod(self.phase + self.phase_dt + math.pi, 2 * math.pi) - math.pi
        self.last_w = act

    def hold(self, seconds):
        for _ in range(int(round(seconds / DT))):
            self.walk_step(np.zeros(3))

    def walk_to(self, x, y, yaw=None, speed=0.4, tol=0.08, yaw_tol=0.06, timeout=40.0, face_target=True, yaw_gain=2.0):
        """目標の位置へ歩く。face_target：目標の方を向いて歩く（False なら今の向きのまま横・後ろへも歩く）。
        着いたら yaw へ向きを変える。"""
        end = self.t + timeout
        while self.t < end:
            px, py, th = self.pose()
            dx, dy = x - px, y - py
            dist = math.hypot(dx, dy)
            fwd_b = dx * math.cos(th) + dy * math.sin(th)       # 体の座標での目標の位置
            side_b = -dx * math.sin(th) + dy * math.cos(th)
            if dist > tol:
                if face_target and dist > 0.3:
                    err = wrap(math.atan2(dy, dx) - th)
                    cmd = (speed * max(0.0, math.cos(err)) * min(1.0, dist / 0.4), 0.0, 1.5 * err)
                else:
                    err = 0.0 if yaw is None else wrap(yaw - th)
                    cmd = (1.2 * fwd_b, 1.2 * side_b, 1.2 * err)
            else:
                if yaw is None or abs(wrap(yaw - th)) < yaw_tol:
                    break
                cmd = (0.0, 0.0, yaw_gain * wrap(yaw - th))
            cmd = np.array([np.clip(cmd[0], -0.3, speed), np.clip(cmd[1], -0.3, 0.3), np.clip(cmd[2], -0.9, 0.9)])
            self.walk_step(cmd)
        return math.hypot(x - self.pose()[0], y - self.pose()[1])

    def push_door(self, speed=0.15, open_angle=1.2, x_max=5.25, timeout=20.0):
        """前へゆっくり歩き、前に出した両手でドアを押す。ドアが open_angle（rad）開くか、骨盤が x_max まで来たら止まる。"""
        end = self.t + timeout
        while self.t < end:
            px, py, th = self.pose()
            if self.d.qpos[self.door_q] >= open_angle or px >= x_max:
                break
            self.walk_step(np.array([speed, 0.0, np.clip(1.5 * wrap(0.0 - th), -0.4, 0.4)]))
        return float(self.d.qpos[self.door_q])

    def back_off(self, x_to, speed=0.15, timeout=8.0):
        """後ろへ下がり、骨盤が x_to まで戻ったら止まる（戸口の中で止まるため）。"""
        end = self.t + timeout
        while self.t < end and self.pose()[0] > x_to:
            th = self.pose()[2]
            self.walk_step(np.array([-speed, 0.0, np.clip(1.5 * wrap(0.0 - th), -0.4, 0.4)]))


def render(robot, path, cam_fn, fps=50, hold=2.0, west_wall=True):
    m, d = robot.m, mujoco.MjData(robot.m)
    W, H = 1280, 720
    m.vis.global_.offwidth, m.vis.global_.offheight = W, H
    m.vis.headlight.ambient[:] = 0.35
    m.vis.headlight.diffuse[:] = 0.45
    r = mujoco.Renderer(m, H, W)
    opt = mujoco.MjvOption()
    opt.geomgroup[4] = int(west_wall)
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                           "-r", str(fps), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
                           "-movflags", "+faststart", path], stdin=subprocess.PIPE)
    frames = robot.frames + [robot.frames[-1]] * int(hold * fps)
    for i, q in enumerate(frames):
        d.qpos[:] = q
        mujoco.mj_forward(m, d)
        cam = cam_fn(q, i / fps)
        r.update_scene(d, cam, scene_option=opt)
        ff.stdin.write(r.render().tobytes())
    ff.stdin.close()
    ff.wait()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--walk", default="estimate")
    ap.add_argument("--go", type=float, default=2.0)
    ap.add_argument("--video")
    args = ap.parse_args()
    rb = Robot(walk_name=args.walk)
    rb.place_seated()
    log = []
    h = rb.stand_up(args.go)
    log.append(("立ち上がった", rb.t, rb.pose(), float(h)))
    rb.walk_init()
    rb.hold(1.2)
    log.append(("歩く方策に切り替え、腕を下ろした", rb.t, rb.pose(), float(rb.d.qpos[2])))
    x0, y0, th0 = rb.pose()
    rb.walk_to(x0, y0, yaw=math.pi / 2, tol=0.3, timeout=12.0)                     # ① その場で左（北、机のない広い側）を向く
    log.append(("左（北）を向いた", rb.t, rb.pose(), float(rb.d.qpos[2])))
    e = rb.walk_to(rb.pose()[0], 1.0, yaw=None, timeout=15.0)                      # ② 北へ歩いて椅子から離れる
    log.append((f"椅子から離れた（ずれ {e:.2f} m）", rb.t, rb.pose(), float(rb.d.qpos[2])))
    rb.walk_to(rb.pose()[0], rb.pose()[1], yaw=0.0, tol=0.3, timeout=12.0)         # ③ ドアの方（東）を向く
    log.append(("ドアの方（東）を向いた", rb.t, rb.pose(), float(rb.d.qpos[2])))
    e = rb.walk_to(4.3, 1.0, yaw=0.0, timeout=40.0)                                # まっすぐドアの前へ（腕を振って歩く）
    log.append((f"ドアの前に着いた（ずれ {e:.2f} m）", rb.t, rb.pose(), float(rb.d.qpos[2])))
    rb.set_arms("carry")                                                           # ドアを押すため、両手を前に出す
    rb.hold(1.2)
    ang = rb.push_door(x_max=5.25)
    log.append((f"ドアを押した（ドアの角度 {math.degrees(ang):.0f}°）", rb.t, rb.pose(), float(rb.d.qpos[2])))
    rb.back_off(4.85)
    rb.set_arms("swing")
    rb.hold(2.0)
    log.append((f"止まった（ドアの角度 {math.degrees(rb.d.qpos[rb.door_q]):.0f}°）", rb.t, rb.pose(), float(rb.d.qpos[2])))
    for name, t, (x, y, th), z in log:
        print(f"{t:6.1f} 秒  {name:28s} 位置 ({x:+.2f}, {y:+.2f})  向き {math.degrees(th):+6.1f}°  骨盤の高さ {z:.2f}", flush=True)
    n_stand = int(round((args.go + 2.3 + 1.0) / DT))
    fell = min(q[2] for q in rb.frames[n_stand:]) < 0.5                                # 立ち上がった後に骨盤が下がったか
    print("倒れた" if fell else "倒れなかった")
    if args.video:
        t_door = next(t for name, t, _, _ in log if name.startswith("ドアの前に着いた"))

        def cam_fn(q, t):
            c = mujoco.MjvCamera()
            if t < t_door - 1.0:                                 # 部屋全体を斜め上から（西の壁は描かない）
                c.lookat[:] = (2.0, 0.1, 0.3)
                c.distance, c.azimuth, c.elevation = 7.0, 15.0, -40.0
            else:                                                # 部屋の中から、ドアを押すところ
                c.lookat[:] = (4.9, 1.0, 0.9)
                c.distance, c.azimuth, c.elevation = 2.8, 30.0, -18.0
            return c
        render(rb, args.video, cam_fn, west_wall=False)
        print(args.video)
    return 0


if __name__ == "__main__":
    sys.exit(main())
