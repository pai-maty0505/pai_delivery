#!/usr/bin/env python3
"""
荷物（0〜3.5 kg）を両手で持って歩く G1 の強化学習の環境（MuJoCo Playground の G1JoystickFlatTerrain を土台にする）。

- ロボット：Playground の G1（29 関節、ゴムの手）。方策が動かすのは脚 12＋腰 3 の 15 関節。腕は運ぶ姿勢（CARRY）に保つ
- 荷物：両手（wrist_yaw_link）に子の体を付け、箱の重さを半分ずつ持たせる。位置は、運ぶ姿勢で箱の底の左右の端が
  4 本の指に載るときの、箱の中心の高さ。重さは環境ごとに変える（randomize）
- B（途中で荷物が載る・降ろす）：載る前と降ろした後は、荷物の重さを打ち消す上向きの力（xfrc_applied）をかける。
  受け取る瞬間・置く瞬間は、エピソードごとにランダムに決める
- A（重さを入力する）：obs_mode で方策への入力を変える
    blind     ：荷物の重さを知らない
    oracle    ：本当の重さ（いま手に載っている重さ）を知る
    estimate  ：腕の関節トルクから推定した重さを知る（推定の式は estimator の w・b。oracle の方策で集めたデータで決める）
  価値関数（critic）は、どの場合も本当の重さを知る
"""

from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
import mujoco
import numpy as np
from ml_collections import config_dict
from mujoco import mjx
from mujoco.mjx._src import math
from mujoco_playground._src import mjx_env
from mujoco_playground._src.locomotion.g1 import base as g1_base
from mujoco_playground._src.locomotion.g1 import g1_constants as consts
from mujoco_playground._src.locomotion.g1 import joystick

N_ACT = 15                       # 方策が動かす関節（脚 12＋腰 3）。qpos[7:] の先頭 15
G = 9.81
SIDES = ("left", "right")
ARM_JOINTS = ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw")
# 運ぶ姿勢（左手。右手は roll・yaw の向きが逆）。前腕を水平に前へ、手首を回して 4 本の指の上面を上へ、外側が少し高い向きに
CARRY_LEFT = {"shoulder_pitch": 0.0, "shoulder_roll": 0.1, "shoulder_yaw": 0.0, "elbow": 0.0,
              "wrist_roll": -1.35, "wrist_pitch": 0.0, "wrist_yaw": 0.0}
MIRROR = {"shoulder_roll", "shoulder_yaw", "wrist_roll", "wrist_yaw"}
WRIST_KP = 20.0                  # 手首の位置制御の強さ（元の 2 では荷物で大きく傾く）
BOX_CENTER = (0.03, 0.20, 0.92)  # 運ぶ姿勢での、荷物の半分を置く位置（x は手のひらの位置から前へ、y は左右の中心から、z は高さ）


def carry_pose():
    """腕 14 関節の運ぶ姿勢（左腕 7、右腕 7 の順）。"""
    out = []
    for s in SIDES:
        for j in ARM_JOINTS:
            v = CARRY_LEFT[j]
            out.append(-v if (s == "right" and j in MIRROR) else v)
    return np.array(out)


def default_config() -> config_dict.ConfigDict:
    cfg = joystick.default_config()
    cfg.obs_mode = "blind"
    cfg.njmax = 128               # 元の 90 では拘束の数があふれた（関節の範囲・摩擦・足の接触）
    cfg.payload = config_dict.create(
        mass_range=[0.0, 3.5],    # 箱の重さ（両手の合計）
        split_range=[0.4, 0.6],   # 左手が持つ割合
        pos_jitter=0.03,          # 荷物の位置のずれ（各方向）
        p_from_start=0.25,        # 最初から持っている割合
        p_never=0.1,              # 持たない割合
        onset_range=[1.0, 12.0],  # 受け取る時刻（秒）
        p_release=0.4,            # 途中で降ろす割合
        hold_range=[2.0, 8.0],    # 持っている時間（秒）
        est_tau=0.1,              # 推定した重さをならす時定数（秒）
    )
    cfg.estimator = config_dict.create(w=[0.0] * 14, b=0.0)
    return cfg


def build_model(decorate=None) -> mujoco.MjModel:
    """Playground の G1 の場面に、両手の荷物の体を足す。手首の制御を強くする。
    decorate：compile の前に MjSpec を書き足す関数（動画用の見た目の箱など。関節を足してはいけない）。"""
    assets = g1_base.get_assets()
    xml = consts.FEET_ONLY_FLAT_TERRAIN_XML.read_text()
    includes = {k: v for k, v in assets.items() if k.endswith(".xml")}
    m0 = mujoco.MjModel.from_xml_string(xml, assets=assets)
    d0 = mujoco.MjData(m0)
    d0.qpos[:] = m0.keyframe("knees_bent").qpos
    arm = carry_pose()
    for i, (s, j) in enumerate((s, j) for s in SIDES for j in ARM_JOINTS):
        d0.qpos[m0.joint(f"{s}_{j}_joint").qposadr[0]] = arm[i]
    mujoco.mj_forward(m0, d0)

    spec = mujoco.MjSpec.from_string(xml, include=includes, assets=assets)
    for s, sg in zip(SIDES, (1.0, -1.0)):
        b = m0.body(f"{s}_wrist_yaw_link").id
        palm = d0.site_xpos[m0.site(f"{s}_palm").id]
        world = np.array([palm[0] + BOX_CENTER[0], sg * BOX_CENTER[1], BOX_CENTER[2]])
        local = d0.xmat[b].reshape(3, 3).T @ (world - d0.xpos[b])
        pb = spec.body(f"{s}_wrist_yaw_link").add_body(name=f"{s}_payload", pos=local)
        pb.explicitinertial = True
        pb.mass = 0.75
        pb.inertia = [1e-4, 1e-4, 1e-4]
    if decorate is not None:
        decorate(spec)
    m = spec.compile()
    for s in SIDES:
        for j in ("wrist_roll", "wrist_pitch", "wrist_yaw"):
            a = m.actuator(f"{s}_{j}_joint").id
            m.actuator_gainprm[a, 0] = WRIST_KP
            m.actuator_biasprm[a, 1] = -WRIST_KP
    return m


class PayloadJoystick(joystick.Joystick):
    """荷物を持って、指令の速さで歩く。"""

    def __init__(
        self,
        config: config_dict.ConfigDict = default_config(),
        config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
    ):
        mjx_env.MjxEnv.__init__(self, config, config_overrides)
        self._model_assets = g1_base.get_assets()
        self._mj_model = build_model()
        self._mj_model.opt.timestep = self.sim_dt
        self._mj_model.vis.global_.offwidth = 1920
        self._mj_model.vis.global_.offheight = 1080
        self._mjx_model = mjx.put_model(self._mj_model, impl=self._config.impl)
        self._xml_path = consts.FEET_ONLY_FLAT_TERRAIN_XML.as_posix()
        self._post_init()
        arm = jp.array(carry_pose())
        self._init_q = self._init_q.at[7 + N_ACT:].set(arm)
        self._default_pose = self._default_pose.at[N_ACT:].set(arm)
        self._payload_body = np.array([self._mj_model.body(f"{s}_payload").id for s in SIDES])
        self._est_w = jp.array(self._config.estimator.w)
        self._est_b = self._config.estimator.b
        self._obs_mode = self._config.obs_mode

    @property
    def action_size(self) -> int:
        return N_ACT

    # ------------------------------------------------------------ 荷物
    def _sample_schedule(self, rng):
        """受け取る・降ろすステップ（制御の周期で数える）。持たないときは大きな数。"""
        c = self._config.payload
        never = 10 ** 6
        k1, k2, k3, k4 = jax.random.split(rng, 4)
        u = jax.random.uniform(k1)
        t_on = jax.random.uniform(k2, minval=c.onset_range[0], maxval=c.onset_range[1])
        onset = jp.where(u < c.p_from_start, 0, jp.where(u < c.p_from_start + c.p_never, never, jp.round(t_on / self.dt)))
        hold = jax.random.uniform(k3, minval=c.hold_range[0], maxval=c.hold_range[1])
        release = jp.where(jax.random.bernoulli(k4, c.p_release), onset + jp.round(hold / self.dt), never)
        return onset.astype(jp.int32), release.astype(jp.int32)

    def _payload_masses(self):
        return self.mjx_model.body_mass[self._payload_body]

    def _estimate_raw(self, data):
        return jp.dot(self._est_w, data.actuator_force[N_ACT:]) + self._est_b

    def _payload_obs(self, info):
        if self._obs_mode == "oracle":
            return jp.array([info["m_eff"] / 3.0])
        if self._obs_mode == "estimate":
            return jp.array([info["m_est"] / 3.0])
        return jp.zeros(0)

    # ------------------------------------------------------------ 始める・進める
    def reset(self, rng: jax.Array) -> mjx_env.State:
        qpos = self._init_q
        qvel = jp.zeros(self.mjx_model.nv)
        rng, key = jax.random.split(rng)
        dxy = jax.random.uniform(key, (2,), minval=-0.5, maxval=0.5)
        qpos = qpos.at[0:2].set(qpos[0:2] + dxy)
        rng, key = jax.random.split(rng)
        yaw = jax.random.uniform(key, (1,), minval=-3.14, maxval=3.14)
        qpos = qpos.at[3:7].set(math.quat_mul(qpos[3:7], math.axis_angle_to_quat(jp.array([0, 0, 1]), yaw)))
        # 脚と腰は元の環境と同じく ×U(0.5, 1.5)。腕は運ぶ姿勢に小さなずれ
        rng, k1, k2 = jax.random.split(rng, 3)
        qpos = qpos.at[7:7 + N_ACT].set(qpos[7:7 + N_ACT] * jax.random.uniform(k1, (N_ACT,), minval=0.5, maxval=1.5))
        qpos = qpos.at[7 + N_ACT:].add(jax.random.uniform(k2, (29 - N_ACT,), minval=-0.05, maxval=0.05))
        rng, key = jax.random.split(rng)
        qvel = qvel.at[0:6].set(jax.random.uniform(key, (6,), minval=-0.5, maxval=0.5))

        data = mjx_env.make_data(self.mj_model, qpos=qpos, qvel=qvel, ctrl=qpos[7:], impl=self.mjx_model.impl.value,
                                 naconmax=self._config.naconmax, njmax=self._config.njmax)
        data = mjx.forward(self.mjx_model, data)

        rng, key = jax.random.split(rng)
        gait_freq = jax.random.uniform(key, (1,), minval=1.25, maxval=1.5)
        rng, cmd_rng, push_rng, pl_rng = jax.random.split(rng, 4)
        push_interval = jax.random.uniform(push_rng, minval=self._config.push_config.interval_range[0],
                                           maxval=self._config.push_config.interval_range[1])
        onset, release = self._sample_schedule(pl_rng)
        info = {
            "rng": rng,
            "step": 0,
            "command": self.sample_command(cmd_rng),
            "last_act": jp.zeros(N_ACT),
            "last_last_act": jp.zeros(N_ACT),
            "motor_targets": jp.zeros(self.mjx_model.nu),
            "feet_air_time": jp.zeros(2),
            "last_contact": jp.zeros(2, dtype=bool),
            "swing_peak": jp.zeros(2),
            "phase_dt": 2 * jp.pi * self.dt * gait_freq,
            "phase": jp.array([0, jp.pi]),
            "push": jp.array([0.0, 0.0]),
            "push_step": 0,
            "push_interval_steps": jp.round(push_interval / self.dt).astype(jp.int32),
            # 荷物
            "pl_t": 0,
            "onset": onset,
            "release": release,
            "m_eff": jp.where(onset <= 0, jp.sum(self._payload_masses()), 0.0),
            "m_est": 0.0,
        }
        metrics = {f"reward/{k}": jp.zeros(()) for k in self._config.reward_config.scales.keys()}
        metrics["swing_peak"] = jp.zeros(())
        metrics["payload"] = jp.zeros(())
        metrics["payload_est_err"] = jp.zeros(())
        contact = jp.array([data.sensordata[self._mj_model.sensor_adr[s]] > 0 for s in self._feet_floor_found_sensor])
        obs = self._get_obs(data, info, contact)
        reward, done = jp.zeros(2)
        return mjx_env.State(data, obs, reward, done, metrics, info)

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        info = state.info
        info["rng"], push1_rng, push2_rng, pl_rng = jax.random.split(info["rng"], 4)
        push_theta = jax.random.uniform(push1_rng, maxval=2 * jp.pi)
        push_magnitude = jax.random.uniform(push2_rng, minval=self._config.push_config.magnitude_range[0],
                                            maxval=self._config.push_config.magnitude_range[1])
        push = jp.array([jp.cos(push_theta), jp.sin(push_theta)])
        push *= jp.mod(info["push_step"] + 1, info["push_interval_steps"]) == 0
        push *= self._config.push_config.enable
        data = state.data
        data = data.replace(qvel=data.qvel.at[:2].set(push * push_magnitude + data.qvel[:2]))

        # 荷物：持っていないときは、重さを打ち消す上向きの力
        masses = self._payload_masses()
        on = (info["pl_t"] >= info["onset"]) & (info["pl_t"] < info["release"])
        lift = jp.where(on, 0.0, masses * G)
        data = data.replace(xfrc_applied=data.xfrc_applied.at[self._payload_body, 2].set(lift))

        motor_targets = self._default_pose.at[:N_ACT].add(action * self._config.action_scale)
        data = mjx_env.step(self.mjx_model, data, motor_targets, self.n_substeps)
        info["motor_targets"] = motor_targets

        contact = jp.array([data.sensordata[self._mj_model.sensor_adr[s]] > 0 for s in self._feet_floor_found_sensor])
        contact_filt = contact | info["last_contact"]
        first_contact = (info["feet_air_time"] > 0.0) * contact_filt
        info["feet_air_time"] += self.dt
        p_fz = data.site_xpos[self._feet_site_id][..., -1]
        info["swing_peak"] = jp.maximum(info["swing_peak"], p_fz)

        # 荷物の重さ（本当の値と、腕のトルクからの推定をならした値）
        info["m_eff"] = jp.where(on, jp.sum(masses), 0.0)
        alpha = self.dt / (self._config.payload.est_tau + self.dt)
        info["m_est"] = info["m_est"] + alpha * (self._estimate_raw(data) - info["m_est"])

        obs = self._get_obs(data, info, contact)
        done = self._get_termination(data)
        rewards = self._get_reward(data, action, info, state.metrics, done, first_contact, contact)
        rewards = {k: v * self._config.reward_config.scales[k] for k, v in rewards.items()}
        reward = sum(rewards.values()) * self.dt

        info["push"] = push
        info["step"] += 1
        info["push_step"] += 1
        phase_tp1 = info["phase"] + info["phase_dt"]
        info["phase"] = jp.fmod(phase_tp1 + jp.pi, 2 * jp.pi) - jp.pi
        info["last_last_act"] = info["last_act"]
        info["last_act"] = action
        info["rng"], cmd_rng = jax.random.split(info["rng"])
        info["command"] = jp.where(info["step"] > 500, self.sample_command(cmd_rng), info["command"])
        info["step"] = jp.where(done | (info["step"] > 500), 0, info["step"])
        info["feet_air_time"] *= ~contact
        info["last_contact"] = contact
        info["swing_peak"] *= ~contact
        # エピソードが終わったら、次の受け取り・降ろす時刻を決め直す（自動のリセットは info を戻さないため）。
        # 倒れたときだけでなく、episode_length で打ち切って初めに戻るときも数え直す（2026-10-16 まで、打ち切りのときに
        # 数え直しておらず、学習の途中から受け取る・降ろす瞬間の練習が少なくなっていた）
        restart = done | (info["pl_t"] + 1 >= self._config.episode_length)
        onset, release = self._sample_schedule(pl_rng)
        info["pl_t"] = jp.where(restart, 0, info["pl_t"] + 1)
        info["onset"] = jp.where(restart, onset, info["onset"])
        info["release"] = jp.where(restart, release, info["release"])
        info["m_est"] = jp.where(restart, 0.0, info["m_est"])
        for k, v in rewards.items():
            state.metrics[f"reward/{k}"] = v
        state.metrics["swing_peak"] = jp.mean(info["swing_peak"])
        state.metrics["payload"] = info["m_eff"]
        state.metrics["payload_est_err"] = jp.abs(info["m_est"] - info["m_eff"])

        done = done.astype(reward.dtype)
        return state.replace(data=data, obs=obs, reward=reward, done=done, info=info)

    # ------------------------------------------------------------ 観測・報酬
    def _get_obs(self, data: mjx.Data, info: dict[str, Any], contact: jax.Array) -> mjx_env.Observation:
        nz = self._config.noise_config
        info["rng"], *keys = jax.random.split(info["rng"], 6)

        def noisy(x, scale, key):
            return x + (2 * jax.random.uniform(key, shape=x.shape) - 1) * nz.level * scale

        gyro = self.get_gyro(data, "pelvis")
        gravity = data.site_xmat[self._pelvis_imu_site_id].T @ jp.array([0, 0, -1])
        q = data.qpos[7:] - self._default_pose
        dq = data.qvel[6:]
        linvel = self.get_local_linvel(data, "pelvis")
        phase = jp.concatenate([jp.cos(info["phase"]), jp.sin(info["phase"])])
        state = jp.hstack([
            noisy(linvel, nz.scales.linvel, keys[0]),
            noisy(gyro, nz.scales.gyro, keys[1]),
            noisy(gravity, nz.scales.gravity, keys[2]),
            info["command"],
            noisy(q[:N_ACT], nz.scales.joint_pos, keys[3]),
            noisy(dq[:N_ACT], nz.scales.joint_vel, keys[4]),
            info["last_act"],
            phase,
            self._payload_obs(info),
        ])
        privileged_state = jp.hstack([
            state,
            gyro,
            self.get_accelerometer(data, "pelvis"),
            gravity,
            linvel,
            self.get_global_angvel(data, "pelvis"),
            q,
            dq,
            data.qpos[2],
            data.actuator_force,
            contact,
            data.sensordata[self._foot_linvel_sensor_adr].ravel(),
            info["feet_air_time"],
            info["m_eff"] / 3.0,
        ])
        return {"state": state, "privileged_state": privileged_state}

    def _cost_pose(self, qpos: jax.Array) -> jax.Array:
        return jp.sum(jp.square(qpos[:N_ACT] - self._default_pose[:N_ACT]))

    def _cost_stand_still(self, commands: jax.Array, qpos: jax.Array) -> jax.Array:
        cost = jp.sum(jp.abs(qpos[:N_ACT] - self._default_pose[:N_ACT]))
        return cost * (jp.linalg.norm(commands) < 0.01)


def domain_randomize(model: mjx.Model, rng: jax.Array, env: PayloadJoystick):
    """元の G1 の条件（床の摩擦・関節の摩擦・リンクの質量・胴の質量）に、荷物の重さ・左右の割合・位置のずれを加える。"""
    c = env._config.payload
    torso = env.mj_model.body("torso_link").id
    pbody = env._payload_body

    @jax.vmap
    def rand(rng):
        rng, k = jax.random.split(rng)
        friction = jax.random.uniform(k, minval=0.4, maxval=1.0)
        pair_friction = model.pair_friction.at[0:2, 0:2].set(friction)
        rng, k = jax.random.split(rng)
        dof_frictionloss = model.dof_frictionloss.at[6:].set(
            model.dof_frictionloss[6:] * jax.random.uniform(k, shape=(29,), minval=0.5, maxval=2.0))
        rng, k = jax.random.split(rng)
        dof_armature = model.dof_armature.at[6:].set(
            model.dof_armature[6:] * jax.random.uniform(k, shape=(29,), minval=1.0, maxval=1.05))
        rng, k = jax.random.split(rng)
        body_mass = model.body_mass * jax.random.uniform(k, shape=(model.nbody,), minval=0.9, maxval=1.1)
        rng, k = jax.random.split(rng)
        body_mass = body_mass.at[torso].add(jax.random.uniform(k, minval=-1.0, maxval=1.0))
        # 荷物
        rng, k1, k2, k3 = jax.random.split(rng, 4)
        total = jax.random.uniform(k1, minval=c.mass_range[0], maxval=c.mass_range[1])
        split = jax.random.uniform(k2, minval=c.split_range[0], maxval=c.split_range[1])
        body_mass = body_mass.at[pbody].set(jp.array([total * split, total * (1 - split)]))
        body_pos = model.body_pos.at[pbody].add(jax.random.uniform(k3, (2, 3), minval=-c.pos_jitter, maxval=c.pos_jitter))
        return pair_friction, dof_frictionloss, dof_armature, body_mass, body_pos

    pair_friction, frictionloss, armature, body_mass, body_pos = rand(rng)
    in_axes = jax.tree_util.tree_map(lambda x: None, model)
    in_axes = in_axes.tree_replace({"pair_friction": 0, "dof_frictionloss": 0, "dof_armature": 0,
                                    "body_mass": 0, "body_pos": 0})
    model = model.tree_replace({"pair_friction": pair_friction, "dof_frictionloss": frictionloss,
                                "dof_armature": armature, "body_mass": body_mass, "body_pos": body_pos})
    return model, in_axes
