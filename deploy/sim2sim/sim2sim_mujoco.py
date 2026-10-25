#!/usr/bin/env python3
"""
Sim2Sim Deployment Pipeline for Unitree Go2: IsaacLab DreamWaQ to MuJoCo.

Features:
- Unified DreamWaQ ONNX model execution (CENet + Actor fused)
- Exact 45-dim current observation + 225-dim history observation (term-major)
- Strict joint topology mapping between IsaacLab and MuJoCo Menagerie
- Low-level PD torque control with motor limits (Kp=25.0, Kd=0.5, clip=±23.7 Nm)
- Supports flat ground and stairs obstacle scenes
- Interactive passive viewer on DISPLAY=:0 with real-time HUD telemetry
"""

import argparse
import os
import sys
import time
from pathlib import Path
import numpy as np
import mujoco
import onnxruntime as ort

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_DIR = str(REPO_ROOT / "logs" / "rsl_rl" / "unitree" / "unitree_go2_dreamwaq_rough" / "2026-10-04_20-02_trot_dreamwaq_smooth_continuous" / "exported")
DEFAULT_ASSETS_DIR = "/home/rum/himloco_assets/mujoco_menagerie/unitree_go2"


def quat_apply_inverse(quat, vec):
    """Rotate a 3D vector by the conjugate (inverse) of a quaternion [w, x, y, z]."""
    w = quat[0]
    xyz = quat[1:]
    t = np.cross(xyz, vec) * 2.0
    return vec - w * t + np.cross(xyz, t)


def quat_to_euler_rpy(quat):
    """Convert quaternion [w, x, y, z] to Roll-Pitch-Yaw (in degrees)."""
    w, x, y, z = quat
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    if np.abs(sinp) >= 1:
        pitch = np.copysign(np.pi / 2, sinp)
    else:
        pitch = np.arcsin(sinp)

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return np.degrees([roll, pitch, yaw])


class DreamWaQMuJoCoSim2Sim:
    """Manages the full Sim2Sim inference and simulation loop for Go2 DreamWaQ in MuJoCo."""

    def __init__(
        self,
        scene_path,
        model_dir,
        flat_model_dir=None,
        stairs_model_dir=None,
        kp=25.0,
        kd=0.5,
        torque_limit=23.7,
        sim_dt=0.002,
        decimation=10,
        filter_alpha=0.7,
    ):
        self.kp = kp
        self.kd = kd
        self.torque_limit = torque_limit
        self.sim_dt = sim_dt
        self.decimation = decimation
        self.policy_dt = self.sim_dt * self.decimation  # 0.02s = 50Hz
        self.filter_alpha = filter_alpha

        # 1. Load MuJoCo model
        if not os.path.exists(scene_path):
            raise FileNotFoundError(f"MuJoCo scene XML not found: {scene_path}")
        self.model = mujoco.MjModel.from_xml_path(scene_path)
        self.data = mujoco.MjData(self.model)

        self.scene_path = scene_path
        init_x = -3.5 if "stair" in scene_path.lower() else 0.0
        self.default_init_pos = (init_x, 0.0, 0.35)
        self.model.opt.timestep = self.sim_dt

        # Zero out Menagerie's passive joint damping & friction loss to align with IsaacLab PhysX
        self.model.dof_damping[6:] = 0.0
        self.model.dof_frictionloss[6:] = 0.0

        # Physical geometry & contact alignment with IsaacLab:
        # 1. IsaacLab sets `replace_cylinders_with_capsules = True`. Replace calf cylinder geoms with capsules
        #    to eliminate sharp 90-degree cylinder base rims snagging on stair step edges (Edge Snagging).
        for i in range(self.model.ngeom):
            if self.model.geom_type[i] == mujoco.mjtGeom.mjGEOM_CYLINDER:
                b_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, self.model.geom_bodyid[i])
                if b_name and "calf" in b_name:
                    self.model.geom_type[i] = mujoco.mjtGeom.mjGEOM_CAPSULE

        # 2. Align feet contact dimension to 3D point Coulomb contact (eliminate Menagerie's condim=6 rolling/torsional friction latch)
        for foot_name in ["FL", "FR", "RL", "RR"]:
            gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, foot_name)
            if gid != -1:
                self.model.geom_condim[gid] = 3
                self.model.geom_friction[gid][1] = 0.0
                self.model.geom_friction[gid][2] = 0.0

        # 2. Joint ordering & bidirectional mappings
        # DreamWaQ policy joint order: [FL(3), FR(3), RL(3), RR(3)]
        self.isaac_joint_names = [
            "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
            "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
            "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
            "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint"
        ]
        self.default_joint_pos_isaac = np.array([
            0.1, 0.8, -1.5,
            -0.1, 0.8, -1.5,
            0.1, 1.0, -1.5,
            -0.1, 1.0, -1.5
        ], dtype=np.float32)

        # MuJoCo joints (1 to 12)
        self.mujoco_joint_names = [
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
            for i in range(1, 13)
        ]

        # Permutations
        self.mujoco_to_isaac = np.array([self.mujoco_joint_names.index(name) for name in self.isaac_joint_names])
        self.isaac_to_mujoco = np.array([self.isaac_joint_names.index(name) for name in self.mujoco_joint_names])
        self.default_joint_pos_mujoco = self.default_joint_pos_isaac[self.isaac_to_mujoco]

        # 3. Observation scales & configuration
        self.obs_scales = {
            "ang_vel": 0.20,
            "gravity": 1.0,
            "cmd": 1.0,
            "joint_pos": 1.0,
            "joint_vel": 0.05,
            "last_action": 1.0,
        }
        self.action_scale = 0.25

        # 4. History configuration (term-major, oldest-to-newest, 5 steps)
        self.term_dims = [3, 3, 3, 12, 12, 12]  # ang_vel, gravity, cmd, jpos, jvel, last_action
        self.history_len = 5
        self.term_history = [np.zeros((self.history_len, d), dtype=np.float32) for d in self.term_dims]
        self.last_action = np.zeros(12, dtype=np.float32)
        self.current_target_pos_mujoco = self.default_joint_pos_mujoco.copy()

        # 5. Load unified ONNX models (supporting Multi-Policy Hot-Swap)
        self.policies = {}
        so = ort.SessionOptions()
        so.intra_op_num_threads = 2

        def _load_sess(m_dir):
            if not m_dir:
                return None, None
            for cand in ["policy.onnx", "dreamwaq_full_policy.onnx"]:
                p = os.path.join(m_dir, cand)
                if os.path.exists(p):
                    return ort.InferenceSession(p, sess_options=so, providers=["CPUExecutionProvider"]), p
            return None, None

        if flat_model_dir and stairs_model_dir:
            sess_flat, p_flat = _load_sess(flat_model_dir)
            sess_stairs, p_stairs = _load_sess(stairs_model_dir)
            if sess_flat and sess_stairs:
                self.policies[1] = {"name": "4598 Flat-Hover (Symmetric Trot)", "sess": sess_flat, "path": p_flat}
                self.policies[2] = {"name": "6642 Stairs-Climb (Obstacle Mode)", "sess": sess_stairs, "path": p_stairs}
                print(f"[INFO] 🚀 Dual-Policy Hot-Swap enabled:\n  [1] Flat: {p_flat}\n  [2] Stairs: {p_stairs}")

        if not self.policies:
            sess, p = _load_sess(model_dir)
            if sess is None:
                raise FileNotFoundError(f"ONNX model not found in {model_dir}")
            self.policies[1] = {"name": "Default Policy", "sess": sess, "path": p}

        self.active_policy_id = 1
        self.ort_sess = self.policies[self.active_policy_id]["sess"]

        # Foot geom IDs for contact detection
        self.foot_geom_ids = {}
        for foot_name in ["FL", "FR", "RL", "RR"]:
            gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, foot_name)
            if gid != -1:
                self.foot_geom_ids[foot_name] = gid

        self.reset()

    def switch_policy(self, policy_id):
        """无缝热切换策略模型，保持关节滤波与历史状态平滑连续。"""
        if policy_id in self.policies and policy_id != self.active_policy_id:
            self.active_policy_id = policy_id
            self.ort_sess = self.policies[policy_id]["sess"]
            name = self.policies[policy_id]["name"]
            print(f"\n[HOT-SWAP] 🚀 已无缝热切换至策略 [{policy_id}]: {name}")
            return True
        return False

    def cycle_policy(self):
        """轮换切换已载入的策略模型。"""
        policy_ids = sorted(list(self.policies.keys()))
        if len(policy_ids) <= 1:
            return False
        curr_idx = policy_ids.index(self.active_policy_id)
        next_id = policy_ids[(curr_idx + 1) % len(policy_ids)]
        return self.switch_policy(next_id)

    def get_active_policy_name(self):
        """获取当前激活策略的友好名称。"""
        return self.policies.get(self.active_policy_id, {}).get("name", "Unknown")

    def reset(self, init_pos=None):
        """Reset robot pose and history buffers."""
        if init_pos is None:
            init_pos = self.default_init_pos
        mujoco.mj_resetData(self.model, self.data)

        self.data.qpos[0:3] = init_pos
        self.data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]  # identity orientation
        self.data.qpos[7:19] = self.default_joint_pos_mujoco
        self.data.qvel[:] = 0.0

        self.last_action.fill(0.0)
        self.current_target_pos_mujoco = self.default_joint_pos_mujoco.copy()

        # Settle robot gently onto the terrain
        for _ in range(150):
            q_curr = self.data.qpos[7:19]
            dq_curr = self.data.qvel[6:18]
            tau = self.kp * (self.default_joint_pos_mujoco - q_curr) - self.kd * dq_curr
            self.data.ctrl[:] = np.clip(tau, -self.torque_limit, self.torque_limit)
            mujoco.mj_step(self.model, self.data)

        # Initialize history buffers with initial state
        init_terms = self._get_current_terms(np.zeros(3, dtype=np.float32))
        for i, val in enumerate(init_terms):
            self.term_history[i][:] = val

    def _get_current_terms(self, cmd):
        """Construct the 6 observation terms in IsaacLab format."""
        q_mj = self.data.qpos[7:19]
        dq_mj = self.data.qvel[6:18]
        q_isaac = q_mj[self.mujoco_to_isaac]
        dq_isaac = dq_mj[self.mujoco_to_isaac]

        quat = self.data.qpos[3:7]  # [w, x, y, z]
        # In MuJoCo freejoint, qvel[3:6] is ALREADY expressed in the local body frame
        ang_vel_b = self.data.qvel[3:6].copy()
        proj_gravity_b = quat_apply_inverse(quat, np.array([0.0, 0.0, -1.0], dtype=np.float32))

        term_ang_vel = np.clip(ang_vel_b * self.obs_scales["ang_vel"], -100.0, 100.0).astype(np.float32)
        term_gravity = np.clip(proj_gravity_b * self.obs_scales["gravity"], -100.0, 100.0).astype(np.float32)
        term_cmd = np.clip(cmd * self.obs_scales["cmd"], -100.0, 100.0).astype(np.float32)
        term_jpos = np.clip((q_isaac - self.default_joint_pos_isaac) * self.obs_scales["joint_pos"], -100.0, 100.0).astype(np.float32)
        term_jvel = np.clip(dq_isaac * self.obs_scales["joint_vel"], -100.0, 100.0).astype(np.float32)
        term_action = np.clip(self.last_action * self.obs_scales["last_action"], -100.0, 100.0).astype(np.float32)

        return [term_ang_vel, term_gravity, term_cmd, term_jpos, term_jvel, term_action]

    def step(self, cmd):
        """Execute one policy step (50Hz) and corresponding physics substeps."""
        current_terms = self._get_current_terms(cmd)

        # Update term history (shift oldest out, put newest at index -1)
        for i, val in enumerate(current_terms):
            self.term_history[i][:-1] = self.term_history[i][1:]
            self.term_history[i][-1] = val

        current_obs = np.concatenate(current_terms).astype(np.float32).reshape(1, 45)
        history_obs = np.concatenate([th.flatten() for th in self.term_history]).astype(np.float32).reshape(1, 225)

        # Forward pass through ONNX model
        ort_inputs = {
            "current_obs": current_obs,
            "history_obs": history_obs,
        }
        action = self.ort_sess.run(None, ort_inputs)[0].flatten()
        action = np.clip(action, -100.0, 100.0)
        self.last_action = action.copy()

        # Action to joint targets (IsaacLab convention: target = nominal + action_scale * action)
        target_pos_isaac = self.default_joint_pos_isaac + self.action_scale * action
        target_pos_mujoco = target_pos_isaac[self.isaac_to_mujoco]
        if self.filter_alpha < 1.0:
            self.current_target_pos_mujoco = (
                self.filter_alpha * target_pos_mujoco
                + (1.0 - self.filter_alpha) * self.current_target_pos_mujoco
            )
            target_pos_mujoco = self.current_target_pos_mujoco.copy()
        else:
            self.current_target_pos_mujoco = target_pos_mujoco.copy()

        # Substep physics
        peak_torque = 0.0
        for _ in range(self.decimation):
            q_curr = self.data.qpos[7:19]
            dq_curr = self.data.qvel[6:18]
            tau = self.kp * (target_pos_mujoco - q_curr) - self.kd * dq_curr
            tau = np.clip(tau, -self.torque_limit, self.torque_limit)
            peak_torque = max(peak_torque, float(np.max(np.abs(tau))))
            self.data.ctrl[:] = tau
            mujoco.mj_step(self.model, self.data)

        # Detect foot contacts
        contacts = {}
        for foot_name, gid in self.foot_geom_ids.items():
            in_contact = False
            for c_idx in range(self.data.ncon):
                con = self.data.contact[c_idx]
                if con.geom1 == gid or con.geom2 == gid:
                    in_contact = True
                    break
            contacts[foot_name] = in_contact

        # Info telemetry
        pos = self.data.qpos[0:3].copy()
        quat = self.data.qpos[3:7].copy()
        vel_w = self.data.qvel[0:3].copy()
        rpy = quat_to_euler_rpy(quat)

        info = {
            "pos": pos,
            "quat": quat,
            "rpy": rpy,
            "vel": vel_w,
            "contacts": contacts,
            "peak_torque": peak_torque,
            "target_pos_isaac": target_pos_isaac,
            "actual_pos_isaac": self.data.qpos[7:19][self.mujoco_to_isaac],
        }
        return info


def compute_adaptive_heading_cmd(env, target_yaw_rad=0.0, kp=2.0, kd=0.12, track_k=0.5, scene="flat"):
    """自适应自抗扰航向与轨道吸附闭环控制器 (Adaptive Heading & Track Lock Controller)。"""
    current_rpy = quat_to_euler_rpy(env.data.qpos[3:7])
    yaw_rad = np.deg2rad(current_rpy[2])
    pos_y = env.data.qpos[1]

    # Cross-track correction (only active on flat straight)
    track_angle = -np.clip(track_k * pos_y, -0.06, 0.06) if "flat" in scene else 0.0
    effective_target = target_yaw_rad + track_angle

    yaw_err = (effective_target - yaw_rad + np.pi) % (2 * np.pi) - np.pi
    gyro_wz = env.data.qvel[5]

    if abs(yaw_err) < np.deg2rad(0.2):
        yaw_err = 0.0

    cmd_wz = kp * yaw_err - kd * gyro_wz
    return float(np.clip(cmd_wz, -0.6, 0.6))


def run_sim2sim(args):
    """Entry point for MuJoCo Sim2Sim execution."""
    scene_map = {
        "flat": "scene.xml",
        "stairs": "scene_stairs_5steps.xml",
        "stairs_10cm": "scene_stairs_10cm.xml",
        "stairs_5steps": "scene_stairs_5steps.xml",
        "stairs_5steps_5cm": "scene_stairs_5steps_5cm.xml",
        "stairs_5steps_15cm": "scene_stairs_5steps_15cm.xml",
    }
    scene_file = scene_map.get(args.scene, "scene.xml")
    scene_path = os.path.join(args.assets_dir, scene_file)

    print("\n" + "="*65)
    print("      Unitree Go2 DreamWaQ Sim2Sim Pipeline (MuJoCo)")
    print("="*65)
    print(f"  Model Dir   : {args.model_dir}")
    print(f"  Scene       : {args.scene} ({scene_path})")
    print(f"  Command     : vx={args.vx:.2f} m/s, vy={args.vy:.2f} m/s, wz={args.wz:.2f} rad/s")
    print(f"  PD Gains    : Kp={args.kp:.1f}, Kd={args.kd:.2f}")
    print(f"  Substeps    : decimation={args.decimation} (sim_dt={args.sim_dt}s -> policy_dt={args.sim_dt*args.decimation}s)")
    print(f"  Heading Mode: {args.heading_mode}")
    print(f"  Display     : {os.environ.get('DISPLAY', 'Not set')}")
    print("="*65 + "\n")

    env = DreamWaQMuJoCoSim2Sim(
        scene_path=scene_path,
        model_dir=args.model_dir,
        flat_model_dir=args.flat_model_dir,
        stairs_model_dir=args.stairs_model_dir,
        kp=args.kp,
        kd=args.kd,
        sim_dt=args.sim_dt,
        decimation=args.decimation,
        filter_alpha=args.filter_alpha,
    )
    if hasattr(args, "init_policy") and args.init_policy in env.policies:
        env.switch_policy(args.init_policy)

    cmd = np.array([args.vx, args.vy, args.wz], dtype=np.float32)
    target_yaw_rad = 0.0
    num_steps = int(args.duration / env.policy_dt) if args.duration > 0 else int(1e9)

    if args.headless:
        start_time = time.time()
        for step_idx in range(num_steps):
            if args.heading_mode and abs(args.wz) < 0.05:
                cmd[2] = compute_adaptive_heading_cmd(
                    env,
                    target_yaw_rad=target_yaw_rad,
                    kp=args.heading_kp,
                    kd=args.heading_kd,
                    track_k=args.track_kp,
                    scene=args.scene,
                )

            info = env.step(cmd)
            sim_time = (step_idx + 1) * env.policy_dt

            if step_idx % 25 == 0 or step_idx == num_steps - 1:  # Print every 0.5s
                c_str = " ".join([f"{k}:{'●' if v else '○'}" for k, v in info["contacts"].items()])
                print(
                    f"[{sim_time:5.2f}s] Pos: x={info['pos'][0]:5.2f}, y={info['pos'][1]:5.2f}, z={info['pos'][2]:5.2f}m | "
                    f"Vel: vx={info['vel'][0]:5.2f}, vy={info['vel'][1]:5.2f}m/s | "
                    f"Yaw: {info['rpy'][2]:+5.1f}° | Wz: {env.data.qvel[5]:+5.2f}rad/s | Feet: [{c_str}] | Tau: {info['peak_torque']:4.1f}Nm"
                )
        total_wall = time.time() - start_time
        print(f"\n[Done] Executed {num_steps} policy steps in {total_wall:.2f}s (Speedup: {args.duration / total_wall:.1f}x real-time)")
    else:
        import mujoco.viewer
        import threading
        def activate_window_task():
            for _ in range(8):
                time.sleep(1.5)
                try:
                    from Xlib import X, display, protocol
                    d = display.Display(':0')
                    root = d.screen().root
                    _NET_CLIENT_LIST = d.intern_atom('_NET_CLIENT_LIST')
                    _NET_ACTIVE_WINDOW = d.intern_atom('_NET_ACTIVE_WINDOW')
                    _NET_WM_NAME = d.intern_atom('_NET_WM_NAME')
                    WM_NAME = d.intern_atom('WM_NAME')
                    client_list = root.get_full_property(_NET_CLIENT_LIST, X.AnyPropertyType)
                    if client_list:
                        for wid in client_list.value:
                            w = d.create_resource_object('window', wid)
                            name = ''
                            for atom in [_NET_WM_NAME, WM_NAME]:
                                prop = w.get_full_property(atom, 0)
                                if prop and prop.value:
                                    name = prop.value.decode('utf-8', errors='ignore') if isinstance(prop.value, bytes) else str(prop.value)
                                    break
                            if 'mujoco' in name.lower():
                                event = protocol.event.ClientMessage(
                                    window=w,
                                    client_type=_NET_ACTIVE_WINDOW,
                                    data=(32, [1, X.CurrentTime, 0, 0, 0])
                                )
                                root.send_event(event, event_mask=X.SubstructureRedirectMask | X.SubstructureNotifyMask)
                                w.configure(stack_mode=X.Above)
                                d.sync()
                                return
                except Exception:
                    pass
        threading.Thread(target=activate_window_task, daemon=True).start()

        # Interactive Teleoperation State
        target_cmd = np.array([args.vx, args.vy, args.wz], dtype=np.float32)
        filtered_cmd = target_cmd.copy()
        reset_requested = [False]
        viewer_ref = [None]
        base_body_id = env.model.body("base").id

        def teleop_key_callback(key):
            # GLFW Keys: C=67, V=86 - Camera Switch works in all modes
            if key in (67, ord('c'), ord('C'), 86, ord('v'), ord('V')):
                if viewer_ref[0] is not None:
                    v = viewer_ref[0]
                    if v.cam.type == mujoco.mjtCamera.mjCAMERA_TRACKING:
                        v.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
                        v.cam.lookat[:] = env.data.qpos[0:3]
                        print(f"\r[CAMERA] 切换为自由漫游视角 (Free Camera) - 鼠标可自由平移/360°旋转/缩放    ", end="", flush=True)
                    else:
                        v.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
                        v.cam.trackbodyid = base_body_id
                        print(f"\r[CAMERA] 切换为跟随视角 (Tracking Camera) - 随狗移动，鼠标可360°环绕    ", end="", flush=True)
                return

            # GLFW Keys: 1=49, 2=50, Tab=258, T=84 - Multi-Policy Hot-Swap works in all modes
            if key in (49, ord('1')):
                env.switch_policy(1)
                return
            elif key in (50, ord('2')):
                env.switch_policy(2)
                return
            elif key in (258, 84, ord('t'), ord('T')):
                env.cycle_policy()
                return

            if not args.teleop:
                return
            # GLFW Keys: W=87, S=83, A=65, D=68, Q=81, E=69, Space=32, X=88, R=82
            # Arrow Keys: UP=265, DOWN=264, LEFT=263, RIGHT=262
            if key in (87, ord('w'), ord('W'), 265):
                target_cmd[0] = float(np.clip(target_cmd[0] + 0.15, -0.6, 1.2))
                print(f"\r[TELEOP] W / ↑ (前进) -> Vx: {target_cmd[0]:+.2f} m/s, Vy: {target_cmd[1]:+.2f} m/s, Wz: {target_cmd[2]:+.2f} rad/s    ", end="", flush=True)
            elif key in (83, ord('s'), ord('S'), 264):
                target_cmd[0] = float(np.clip(target_cmd[0] - 0.15, -0.6, 1.2))
                print(f"\r[TELEOP] S / ↓ (后退) -> Vx: {target_cmd[0]:+.2f} m/s, Vy: {target_cmd[1]:+.2f} m/s, Wz: {target_cmd[2]:+.2f} rad/s    ", end="", flush=True)
            elif key in (65, ord('a'), ord('A')):
                target_cmd[1] = float(np.clip(target_cmd[1] + 0.10, -0.5, 0.5))
                print(f"\r[TELEOP] A (左横移) -> Vx: {target_cmd[0]:+.2f} m/s, Vy: {target_cmd[1]:+.2f} m/s, Wz: {target_cmd[2]:+.2f} rad/s    ", end="", flush=True)
            elif key in (68, ord('d'), ord('D')):
                target_cmd[1] = float(np.clip(target_cmd[1] - 0.10, -0.5, 0.5))
                print(f"\r[TELEOP] D (右横移) -> Vx: {target_cmd[0]:+.2f} m/s, Vy: {target_cmd[1]:+.2f} m/s, Wz: {target_cmd[2]:+.2f} rad/s    ", end="", flush=True)
            elif key in (81, ord('q'), ord('Q'), 263):
                target_cmd[2] = float(np.clip(target_cmd[2] + 0.20, -1.0, 1.0))
                print(f"\r[TELEOP] Q / ← (左自转) -> Vx: {target_cmd[0]:+.2f} m/s, Vy: {target_cmd[1]:+.2f} m/s, Wz: {target_cmd[2]:+.2f} rad/s    ", end="", flush=True)
            elif key in (69, ord('e'), ord('E'), 262):
                target_cmd[2] = float(np.clip(target_cmd[2] - 0.20, -1.0, 1.0))
                print(f"\r[TELEOP] E / → (右自转) -> Vx: {target_cmd[0]:+.2f} m/s, Vy: {target_cmd[1]:+.2f} m/s, Wz: {target_cmd[2]:+.2f} rad/s    ", end="", flush=True)
            elif key in (32, 88, ord('x'), ord('X')):
                target_cmd[:] = [0.0, 0.0, 0.0]
                print(f"\r[TELEOP] SPACE / X (急停刹车) -> 速度归零!                                             ", end="", flush=True)
            elif key in (82, ord('r'), ord('R')):
                reset_requested[0] = True
                target_cmd[:] = [0.0, 0.0, 0.0]
                print(f"\r[TELEOP] R (复位) -> 机器狗回到原点!                                                 ", end="", flush=True)

        if args.teleop or len(env.policies) > 1:
            print("\n" + "="*65)
            print("  🎮 Go2 交互控制与多策略热插拔 (Dual-Policy Hot-Swap Active)")
            print("="*65)
            if len(env.policies) > 1:
                print("  [ 1 ] 切换至策略 1: 4598 平地悬浮 (极稳直腿, 对角 Trot)")
                print("  [ 2 ] 切换至策略 2: 6642 台阶攻坚 (越障模式, 自适应抬腿)")
                print("  [Tab/T] 轮流切换策略 (无缝平滑切换，无力矩冲击)")
            if args.teleop:
                print("  [ W ] 前进加速 (+0.15 m/s)      [ S ] 后退减速 (-0.15 m/s)")
                print("  [ A ] 向左横移 (+0.10 m/s)      [ D ] 向右横移 (-0.10 m/s)")
                print("  [ Q ] 原地左自转 (+0.20 rad/s)   [ E ] 原地右自转 (-0.20 rad/s)")
                print("  [Space / X] 急停刹车 (速度归零)   [ R ] 复位机器狗回原点")
            print("  [ C / V ] 切换自由视角 / 自动跟随 (鼠标左键360°旋转, 右键/滚轮缩放)")
            print("  * 核心机制：已锁死 MuJoCo 默认快捷键 (Space 不会暂停物理引擎)")
            print("="*65 + "\n")

        print("[Viewer] Launching MuJoCo passive viewer window on DISPLAY=:0...")
        with mujoco.viewer.launch_passive(env.model, env.data, key_callback=teleop_key_callback) as viewer:
            viewer_ref[0] = viewer
            # Set native tracking camera: follows robot in 3D while allowing 360° mouse rotation and zoom
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            viewer.cam.trackbodyid = base_body_id
            viewer.cam.distance = 2.4
            viewer.cam.elevation = -15
            viewer.cam.azimuth = 135

            step_idx = 0
            fall_cooldown = 0
            while viewer.is_running() and step_idx < num_steps:
                step_start = time.time()
                if fall_cooldown > 0:
                    fall_cooldown -= 1

                if args.teleop:
                    if reset_requested[0]:
                        env.reset()
                        target_cmd[:] = [0.0, 0.0, 0.0]
                        filtered_cmd[:] = [0.0, 0.0, 0.0]
                        fall_cooldown = 30
                        if viewer.cam.type == mujoco.mjtCamera.mjCAMERA_FREE:
                            viewer.cam.lookat[:] = [1.0, 0.0, 0.4]
                        reset_requested[0] = False

                    # Smooth command interpolation
                    filtered_cmd = 0.88 * filtered_cmd + 0.12 * target_cmd
                    cmd = filtered_cmd.copy()

                    if args.heading_mode:
                        if abs(target_cmd[2]) < 0.05:
                            cmd[2] = compute_adaptive_heading_cmd(
                                env,
                                target_yaw_rad=target_yaw_rad,
                                kp=args.heading_kp,
                                kd=args.heading_kd,
                                track_k=0.0,
                                scene=args.scene,
                            )
                        else:
                            curr_rpy = quat_to_euler_rpy(env.data.qpos[3:7])
                            target_yaw_rad = np.deg2rad(curr_rpy[2])
                else:
                    if args.heading_mode and abs(args.wz) < 0.05:
                        cmd[2] = compute_adaptive_heading_cmd(
                            env,
                            target_yaw_rad=target_yaw_rad,
                            kp=args.heading_kp,
                            kd=args.heading_kd,
                            track_k=args.track_kp,
                            scene=args.scene,
                        )

                info = env.step(cmd)

                # Auto reset
                if not args.teleop and (("stair" in args.scene and info["pos"][0] > 50.0) or ("flat" in args.scene and info["pos"][0] > 25.0) or (step_idx > 50 and info["pos"][2] < 0.20)):
                    print("[INFO] Reached end of run, looping back to start...")
                    env.reset()
                    if viewer.cam.type == mujoco.mjtCamera.mjCAMERA_FREE:
                        viewer.cam.lookat[:] = [1.0, 0.0, 0.4]
                elif args.teleop and fall_cooldown == 0 and (step_idx > 50 and info["pos"][2] < 0.15):
                    print(f"\n[INFO] Robot fell over at X={info['pos'][0]:.2f}m, Z={info['pos'][2]:.2f}m, resetting...")
                    env.reset()
                    fall_cooldown = 40  # 40 steps = 0.8s 稳定站立冷却
                    target_cmd[:] = [0.0, 0.0, 0.0]
                    filtered_cmd[:] = [0.0, 0.0, 0.0]
                    if viewer.cam.type == mujoco.mjtCamera.mjCAMERA_FREE:
                        viewer.cam.lookat[:] = [1.0, 0.0, 0.4]

                # Teleop / Hot-Swap on-screen HUD
                if (args.teleop or len(env.policies) > 1) and step_idx % 10 == 0:
                    try:
                        hud_l = (
                            f"Go2 Status | Mode: [{env.active_policy_id}]\n"
                            f"Policy: {env.get_active_policy_name()}\n"
                            f"Vx : {cmd[0]:+5.2f} m/s\n"
                            f"Vy : {cmd[1]:+5.2f} m/s\n"
                            f"Wz : {cmd[2]:+5.2f} rad/s\n"
                            f"Pos: X={info['pos'][0]:.1f}m Z={info['pos'][2]:.2f}m\n"
                            f"Yaw: {info['rpy'][2]:+5.1f} deg"
                        )
                        hud_r = (
                            "Hot-Swap & Controls:\n"
                            "[1] Flat-4598 | [2] Stairs-6642\n"
                            "[Tab/T] Cycle Active Policy\n"
                            "[W/S] Vx | [A/D] Vy | [Q/E] Wz\n"
                            "[Space/X] Brake | [R] Reset\n"
                            "[C/V] Camera | Mouse: Rot/Zoom"
                        )
                        viewer.set_texts([
                            (mujoco.mjtFontScale.mjFONTSCALE_100, mujoco.mjtGridPos.mjGRID_TOPLEFT, hud_l, hud_r)
                        ])
                    except Exception:
                        pass

                # Camera is natively handled by MuJoCo (mjCAMERA_TRACKING / mjCAMERA_FREE)
                # Mouse orbit, pan, and zoom are completely unlocked without fighting
                viewer.sync()

                sim_time = (step_idx + 1) * env.policy_dt
                if step_idx % 25 == 0 and not args.teleop:
                    c_str = " ".join([f"{k}:{'●' if v else '○'}" for k, v in info["contacts"].items()])
                    print(
                        f"[{sim_time:5.2f}s] Pos: ({info['pos'][0]:5.2f}, {info['pos'][1]:5.2f}, {info['pos'][2]:5.2f})m | "
                        f"Vx: {info['vel'][0]:5.2f}m/s | Yaw: {info['rpy'][2]:+5.1f}° | Feet: [{c_str}] | Tau: {info['peak_torque']:4.1f}Nm"
                    )

                step_idx += 1

                # High-precision frame synchronization
                elapsed = time.time() - step_start
                if elapsed < env.policy_dt:
                    time.sleep(env.policy_dt - elapsed)

        print("[Viewer] Window closed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Go2 DreamWaQ MuJoCo Sim2Sim Runner")
    parser.add_argument("--model_dir", type=str, default=DEFAULT_MODEL_DIR, help="Path to exported checkpoint directory")
    parser.add_argument("--flat_model_dir", type=str, default=None, help="Path to flat terrain policy model dir (e.g. 4598)")
    parser.add_argument("--stairs_model_dir", type=str, default=None, help="Path to stairs obstacle policy model dir (e.g. 6642)")
    parser.add_argument("--assets_dir", type=str, default=DEFAULT_ASSETS_DIR, help="Path to Menagerie Go2 directory")
    parser.add_argument("--scene", type=str, choices=["flat", "stairs", "stairs_10cm", "stairs_5steps", "stairs_5steps_5cm", "stairs_5steps_15cm"], default="flat", help="Scene type")
    parser.add_argument("--vx", type=float, default=0.6, help="Forward velocity command (m/s)")
    parser.add_argument("--vy", type=float, default=0.0, help="Lateral velocity command (m/s)")
    parser.add_argument("--wz", type=float, default=0.0, help="Yaw angular velocity command (rad/s)")
    parser.add_argument("--duration", type=float, default=15.0, help="Simulation duration (seconds, 0 for infinite)")
    parser.add_argument("--kp", type=float, default=28.0, help="Joint stiffness Kp (default 28.0)")
    parser.add_argument("--kd", type=float, default=0.75, help="Joint damping Kd (default 0.75, eliminates joint flutter)")
    parser.add_argument("--sim_dt", type=float, default=0.002, help="Physics simulation timestep")
    parser.add_argument("--decimation", type=int, default=10, help="Physics substeps per policy step")
    parser.add_argument("--heading_mode", action="store_true", default=True, help="Enable automatic adaptive heading and track lock controller")
    parser.add_argument("--heading_kp", type=float, default=2.0, help="Adaptive heading P-gain (default 2.0)")
    parser.add_argument("--heading_kd", type=float, default=0.12, help="Adaptive heading D-gain (default 0.12, gyro damping)")
    parser.add_argument("--track_kp", type=float, default=0.5, help="Cross-track virtual guide gain (default 0.5)")
    parser.add_argument("--filter_alpha", type=float, default=0.6, help="Action EMA filter factor (default 0.6 for buttery smoothness)")
    parser.add_argument("--init_policy", type=int, default=1, choices=[1, 2], help="Initial active policy (1: flat, 2: stairs)")
    parser.add_argument("--teleop", action="store_true", help="Enable interactive QWEASD keyboard teleoperation and lock MuJoCo defaults")
    parser.add_argument("--headless", action="store_true", help="Run without graphical viewer")

    args = parser.parse_args()
    run_sim2sim(args)

