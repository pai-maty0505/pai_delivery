#!/usr/bin/env python3
"""
椅子から立ち上がる方策（A）の強化学習の環境（MuJoCo Playground の G1 を土台にする）。

- 場面：G1 は +x を向いて椅子に座っている。両足の site は x ＝ 0。椅子の座面と机に当たり判定を付ける
  （座面 ↔ もも・骨盤、机 ↔ 手・胴・もも）。机は前に STAND_CLEAR だけ離す（立つだけではぶつからない）
- お手本：scripts/standup_ref.py の関節の角度の列（座る → 上体を倒す → お尻が離れる → 立つ、2.3 秒）
- 方策が動かすのは脚 12＋腰 3。出力は、お手本の角度からのずれ（残差）。腕はお手本の姿勢に保つ
- 合図：エピソードの初めの 0〜1 秒は座ったまま待ち、合図（ノックの音にあたる）で立ち上がり始める。方策には
  「合図からの時間」を渡す
- 報酬：お手本の関節の角度・骨盤の高さ・胴の傾きに近いほど高い。足が滑る・浮く、机に当たる、倒れる、は罰。
  立ち終わった後は、立ち姿勢を保つ
"""

from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
import mujoco
import numpy as np
from ml_collections import config_dict
from mujoco import mjx
from mujoco_playground._src import mjx_env
from mujoco_playground._src.locomotion.g1 import base as g1_base
from mujoco_playground._src.locomotion.g1 import g1_constants as consts

import payload_env as pe
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
N_ACT = 15
STAND_CLEAR = 0.55            # 足の site から机の前の縁まで（m）
SEAT_X = (-0.47, -0.10)       # 座面の前後の範囲（足の site 基準）
DESK_TOP = 0.74


def default_config() -> config_dict.ConfigDict:
    return config_dict.create(
        ctrl_dt=0.02,
        sim_dt=0.002,
        episode_length=250,           # 5 秒（待つ 0〜1 秒 ＋ 立つ 2.3 秒 ＋ 立ったまま保つ）
        action_repeat=1,
        action_scale=0.6,            # 方策が加える修正の上限（rad）。0.3 では立ち上がるのに必要なずれ（約 0.45 rad）を出せなかった
        wait_range=[0.0, 1.0],        # 合図までの待ち時間
        impl="warp",
        naconmax=16 * 8192,
        njmax=160,
        noise_config=config_dict.create(level=1.0, scales=config_dict.create(
            joint_pos=0.03, joint_vel=1.5, gravity=0.05, gyro=0.2)),
        reward_config=config_dict.create(
            scales=config_dict.create(
                track_q=2.0,          # お手本の関節の角度
                track_height=2.0,     # お手本の骨盤の高さ
                track_torso=1.0,      # お手本の胴の傾き
                feet_slip=-2.0,       # 足の滑り
                feet_air=-2.0,        # 足が浮く
                desk_contact=-5.0,    # 机に当たる
                action_rate=-0.02,
                torques=-1e-4,
                termination=-100.0,
                alive=0.5,
            ),
            sigma_q=0.15, sigma_h=0.003, sigma_torso=0.05,
        ),
    )


def decorate(seat_h):
    """座面と机（当たり判定あり）、骨盤と胴の当たり判定、接触の組を足す。"""
    def deco(spec):
        w = spec.worldbody
        x0, x1 = SEAT_X
        w.add_geom(name="seat", type=mujoco.mjtGeom.mjGEOM_BOX, pos=[(x0 + x1) / 2, 0, seat_h - 0.02],
                   size=[(x1 - x0) / 2, 0.22, 0.02], rgba=[0.25, 0.25, 0.28, 1], contype=0, conaffinity=0)
        w.add_geom(name="desk", type=mujoco.mjtGeom.mjGEOM_BOX, pos=[STAND_CLEAR + 0.35, 0, DESK_TOP - 0.015],
                   size=[0.35, 0.6, 0.015], rgba=[0.62, 0.45, 0.30, 1], contype=0, conaffinity=0)
        pel = spec.body("pelvis")
        pel.add_geom(name="pelvis_col", type=mujoco.mjtGeom.mjGEOM_BOX, pos=[-0.02, 0, -0.07], size=[0.07, 0.11, 0.04],
                     rgba=[0.2, 0.6, 0.2, 0.3], contype=0, conaffinity=0, group=3)
        tor = spec.body("torso_link")
        tor.add_geom(name="torso_col", type=mujoco.mjtGeom.mjGEOM_CAPSULE, fromto=[0.02, 0, 0.05, 0.02, 0, 0.40],
                     size=[0.11, 0, 0], rgba=[0.2, 0.6, 0.2, 0.3], contype=0, conaffinity=0, group=3)
        pairs = [("pelvis_col", "seat"), ("left_thigh", "seat"), ("right_thigh", "seat"),
                 ("left_hand_collision", "desk"), ("right_hand_collision", "desk"), ("torso_col", "desk"),
                 ("left_shin", "seat"), ("right_shin", "seat")]
        for a, b in pairs:
            p = spec.add_pair(geomname1=a, geomname2=b)
            p.condim = 3
            p.friction = [0.8, 0.8, 0.005, 0.0001, 0.0001]
    return deco


def load_reference():
    z = np.load(REPO / "models" / "standup_ref.npz")
    return z["qpos"], float(z["seat"]), z["foot_pos"]


class StandUp(mjx_env.MjxEnv):
    def __init__(self, config: config_dict.ConfigDict = default_config(),
                 config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None):
        super().__init__(config, config_overrides)
        ref, seat, foot_pos = load_reference()
        self._mj_model = pe.build_model(decorate=decorate(seat))
        self._mj_model.opt.timestep = self.sim_dt
        for side in ("left", "right"):                             # 荷物の体（payload_env）は使わないので質量を 0 に
            b = self._mj_model.body(f"{side}_payload").id
            self._mj_model.body_mass[b] = 0.0
            self._mj_model.body_inertia[b] = 1e-8
        self._mj_model.vis.global_.offwidth, self._mj_model.vis.global_.offheight = 1920, 1080
        self._mjx_model = mjx.put_model(self._mj_model, impl=self._config.impl)
        self._xml_path = consts.FEET_ONLY_FLAT_TERRAIN_XML.as_posix()
        m = self._mj_model
        self._ref = jp.array(ref)                                  # (T, nq)
        self._ref_len = ref.shape[0]
        self._ref_np = ref
        self._seat_h = seat
        self._foot_pos = jp.array(foot_pos)
        self._arm_pose = jp.array(ref[0, 7 + N_ACT:])
        self._feet_site = np.array([m.site(f"{s}_foot").id for s in ("left", "right")])
        self._pelvis = m.body("pelvis").id
        self._torso = m.body("torso_link").id
        self._feet_floor_found_sensor = [m.sensor(f"{g}_floor_found").id for g in ("left_foot", "right_foot")]
        self._desk = m.geom("desk").id
        self._desk_geoms = np.array([m.geom(n).id for n in ("left_hand_collision", "right_hand_collision", "torso_col")])
        self._desk_half = jp.array(m.geom_size[self._desk_geoms, 1])
        self._desk_rad = jp.array(m.geom_size[self._desk_geoms, 0])
        self._lowers, self._uppers = m.jnt_range[1:].T
        # 参照の骨盤の高さ・胴の傾き（前後）
        d = mujoco.MjData(m)
        h, tp = [], []
        for q in ref:
            d.qpos[:] = q
            mujoco.mj_forward(m, d)
            h.append(d.xpos[self._pelvis][2])
            R = d.xmat[self._torso].reshape(3, 3)
            tp.append(np.arctan2(R[0, 2], R[2, 2]))
        self._ref_h = jp.array(h)
        self._ref_tp = jp.array(tp)

    @property
    def action_size(self) -> int:
        return N_ACT

    @property
    def xml_path(self) -> str:
        return self._xml_path

    @property
    def mj_model(self) -> mujoco.MjModel:
        return self._mj_model

    @property
    def mjx_model(self) -> mjx.Model:
        return self._mjx_model

    # ------------------------------------------------------------
    def _ref_index(self, info):
        """合図からの経過ステップ → お手本の番号（合図の前は 0、最後は最後の姿勢のまま）。"""
        k = jp.maximum(info["step"] - info["go"], 0)
        return jp.minimum(k, self._ref_len - 1)

    def reset(self, rng: jax.Array) -> mjx_env.State:
        rng, k1, k2, k3 = jax.random.split(rng, 4)
        q = self._ref[0]
        q = q.at[7:7 + N_ACT].add(jax.random.uniform(k1, (N_ACT,), minval=-0.03, maxval=0.03))
        q = q.at[2].add(0.005)                                  # 座面の少し上から落とす
        data = mjx_env.make_data(self.mj_model, qpos=q, qvel=jp.zeros(self.mjx_model.nv), ctrl=q[7:],
                                 impl=self.mjx_model.impl.value, naconmax=self._config.naconmax, njmax=self._config.njmax)
        data = mjx.forward(self.mjx_model, data)
        wait = jax.random.uniform(k2, minval=self._config.wait_range[0], maxval=self._config.wait_range[1])
        info = {"rng": rng, "step": 0, "go": jp.round(wait / self.dt).astype(jp.int32),
                "last_act": jp.zeros(N_ACT), "last_last_act": jp.zeros(N_ACT)}
        metrics = {f"reward/{k}": jp.zeros(()) for k in self._config.reward_config.scales.keys()}
        metrics["height"] = jp.zeros(())
        obs = self._get_obs(data, info)
        return mjx_env.State(data, obs, jp.zeros(()), jp.zeros(()), metrics, info)

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        info = state.info
        i = self._ref_index(info)
        q_ref = self._ref[i]
        go = info["step"] >= info["go"]
        # 合図の前は座ったまま（方策の出力は使わない）。合図の後は、お手本の角度 ＋ 方策の出力
        legs = q_ref[7:7 + N_ACT] + jp.where(go, action * self._config.action_scale, 0.0)
        target = jp.concatenate([legs, self._arm_pose])
        data = mjx_env.step(self.mjx_model, state.data, target, self.n_substeps)
        info["step"] = info["step"] + 1
        obs = self._get_obs(data, info)
        done = self._get_termination(data)
        rewards = self._get_reward(data, action, info, done)
        rewards = {k: v * self._config.reward_config.scales[k] for k, v in rewards.items()}
        reward = sum(rewards.values()) * self.dt
        info["last_last_act"] = info["last_act"]
        info["last_act"] = action
        # エピソードが終わったら、時間と合図の時刻を初めからにし直す（自動のリセットは体の状態だけを戻し、info は戻さない）
        info["rng"], k = jax.random.split(info["rng"])
        wait = jax.random.uniform(k, minval=self._config.wait_range[0], maxval=self._config.wait_range[1])
        # 1 回の練習の打ち切り（episode_length 歩）のときも同じ（学習の仕組みが座った姿勢に戻すため）
        restart = done | (info["step"] >= self._config.episode_length)
        info["step"] = jp.where(restart, 0, info["step"])
        info["go"] = jp.where(restart, jp.round(wait / self.dt).astype(jp.int32), info["go"])
        info["last_act"] = jp.where(restart, jp.zeros(N_ACT), info["last_act"])
        info["last_last_act"] = jp.where(restart, jp.zeros(N_ACT), info["last_last_act"])
        for k, v in rewards.items():
            state.metrics[f"reward/{k}"] = v
        state.metrics["height"] = data.xpos[self._pelvis, 2]
        return state.replace(data=data, obs=obs, reward=reward, done=done.astype(reward.dtype), info=info)

    def _get_termination(self, data):
        up = data.xmat[self._torso].reshape(3, 3)[2, 2]
        fell = up < 0.3                                          # 胴が約 70° 以上傾いた
        low = data.xpos[self._pelvis, 2] < 0.25
        return fell | low | jp.isnan(data.qpos).any()

    def _contact_found(self, data, sensor_ids):
        return jp.array([data.sensordata[self._mj_model.sensor_adr[s]] > 0 for s in sensor_ids])

    def _get_obs(self, data, info):
        info["rng"], *keys = jax.random.split(info["rng"], 5)
        nz = self._config.noise_config

        def noisy(x, scale, key):
            return x + (2 * jax.random.uniform(key, shape=x.shape) - 1) * nz.level * scale

        gyro = mjx_env.get_sensor_data(self.mj_model, data, "gyro_pelvis")
        gravity = data.site_xmat[self._mj_model.site("imu_in_pelvis").id].T @ jp.array([0, 0, -1])
        i = self._ref_index(info)
        q_ref = self._ref[i, 7:7 + N_ACT]
        q_next = self._ref[jp.minimum(i + 5, self._ref_len - 1), 7:7 + N_ACT]
        t_since = jp.maximum(info["step"] - info["go"], -50) * self.dt
        phase = jp.clip(t_since / (self._ref_len * self.dt), -0.5, 1.2)
        q = data.qpos[7:7 + N_ACT]
        state = jp.hstack([
            noisy(gyro, nz.scales.gyro, keys[0]),
            noisy(gravity, nz.scales.gravity, keys[1]),
            noisy(q, nz.scales.joint_pos, keys[2]) - q_ref,
            noisy(data.qvel[6:6 + N_ACT], nz.scales.joint_vel, keys[3]) * 0.1,
            q_ref, q_next - q_ref,
            info["last_act"],
            jp.array([phase, (info["step"] >= info["go"]).astype(jp.float32)]),
        ])
        privileged = jp.hstack([state, data.qpos[:7], data.qvel[:6], data.qpos[7:], data.qvel[6:],
                                self._ref_h[i], self._ref_tp[i], data.actuator_force])
        return {"state": state, "privileged_state": privileged}

    def _get_reward(self, data, action, info, done):
        c = self._config.reward_config
        i = self._ref_index(info)
        q_ref = self._ref[i]
        dq = data.qpos[7:7 + N_ACT] - q_ref[7:7 + N_ACT]
        h = data.xpos[self._pelvis, 2]
        R = data.xmat[self._torso].reshape(3, 3)
        tp = jp.arctan2(R[0, 2], R[2, 2])
        feet = data.site_xpos[self._feet_site]
        slip = jp.sum(jp.square(feet[:, :2] - self._foot_pos[:, :2]))
        contact = self._contact_found(data, self._feet_floor_found_sensor)
        # 机に当たったか：手と胴の当たり判定（カプセル）の両端が、机の板の範囲（前の縁より奥、板の下面より上）に入ったか
        pos = data.geom_xpos[self._desk_geoms]                    # (3, 3)
        axis = data.geom_xmat[self._desk_geoms].reshape(-1, 3, 3)[:, :, 2]
        half = self._desk_half[:, None]
        ends = jp.concatenate([pos + axis * half, pos - axis * half])
        rad = jp.concatenate([self._desk_rad, self._desk_rad])
        desk_hit = jp.any((ends[:, 0] + rad > STAND_CLEAR) & (ends[:, 2] + rad > DESK_TOP - 0.03)
                          & (jp.abs(ends[:, 1]) < 0.6))
        return {
            "track_q": jp.exp(-jp.sum(jp.square(dq)) / c.sigma_q),
            "track_height": jp.exp(-jp.square(h - self._ref_h[i]) / c.sigma_h),
            "track_torso": jp.exp(-jp.square(tp - self._ref_tp[i]) / c.sigma_torso),
            "feet_slip": jp.minimum(slip, 0.05) / 0.05,
            "feet_air": jp.sum(~contact).astype(jp.float32) / 2.0,
            "desk_contact": desk_hit.astype(jp.float32),
            "action_rate": jp.sum(jp.square(action - info["last_act"])),
            "torques": jp.sum(jp.square(data.actuator_force)),
            "termination": done.astype(jp.float32),
            "alive": jp.array(1.0),
        }


def domain_randomize(model: mjx.Model, rng: jax.Array, env: StandUp):
    """床の摩擦・関節の摩擦・リンクの質量・椅子の座面の高さと前後の位置を少し変える。"""
    seat = env.mj_model.geom("seat").id

    @jax.vmap
    def rand(rng):
        rng, k = jax.random.split(rng)
        friction = jax.random.uniform(k, minval=0.5, maxval=1.0)
        pair_friction = model.pair_friction.at[0:2, 0:2].set(friction)
        rng, k = jax.random.split(rng)
        dof_frictionloss = model.dof_frictionloss.at[6:].set(
            model.dof_frictionloss[6:] * jax.random.uniform(k, shape=(29,), minval=0.5, maxval=2.0))
        rng, k = jax.random.split(rng)
        body_mass = model.body_mass * jax.random.uniform(k, shape=(model.nbody,), minval=0.92, maxval=1.08)
        rng, k1, k2 = jax.random.split(rng, 3)
        geom_pos = model.geom_pos.at[seat, 2].add(jax.random.uniform(k1, minval=-0.015, maxval=0.0))
        geom_pos = geom_pos.at[seat, 0].add(jax.random.uniform(k2, minval=-0.03, maxval=0.03))
        return pair_friction, dof_frictionloss, body_mass, geom_pos

    pair_friction, frictionloss, body_mass, geom_pos = rand(rng)
    in_axes = jax.tree_util.tree_map(lambda x: None, model)
    in_axes = in_axes.tree_replace({"pair_friction": 0, "dof_frictionloss": 0, "body_mass": 0, "geom_pos": 0})
    model = model.tree_replace({"pair_friction": pair_friction, "dof_frictionloss": frictionloss,
                                "body_mass": body_mass, "geom_pos": geom_pos})
    return model, in_axes
