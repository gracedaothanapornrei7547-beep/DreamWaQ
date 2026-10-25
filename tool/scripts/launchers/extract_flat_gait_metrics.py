#!/usr/bin/env python3
"""
Quantitative Gait & Body Dynamics Extraction Tool for Unitree Go2 in MuJoCo Sim2Sim.
Evaluates foot trajectories, swing/stance kinematics, and body vibrations over a sprint.
"""

import os
import sys
from pathlib import Path
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from deploy.sim2sim.sim2sim_mujoco import DreamWaQMuJoCoSim2Sim, compute_adaptive_heading_cmd, quat_apply_inverse, quat_to_euler_rpy

DEFAULT_ASSETS_DIR = "/home/rum/himloco_assets/mujoco_menagerie/unitree_go2"


def evaluate_model(model_dir, speed=1.0, duration=8.0, scene="flat", assets_dir=DEFAULT_ASSETS_DIR):
    scene_path = os.path.join(assets_dir, "scene.xml")
    env = DreamWaQMuJoCoSim2Sim(
        scene_path=scene_path,
        model_dir=model_dir,
        kp=28.0,
        kd=0.75,
        filter_alpha=0.6,
    )

    cmd = np.array([speed, 0.0, 0.0], dtype=np.float32)
    num_steps = int(duration / env.policy_dt)

    # Telemetry storage
    pos_history = []
    rpy_history = []
    vel_history = []
    torque_diffs = []
    foot_pos_w = {k: [] for k in ["FL", "FR", "RL", "RR"]}
    foot_pos_b = {k: [] for k in ["FL", "FR", "RL", "RR"]}
    contacts_history = {k: [] for k in ["FL", "FR", "RL", "RR"]}

    last_tau = None

    for step in range(num_steps):
        cmd[2] = compute_adaptive_heading_cmd(env, target_yaw_rad=0.0, kp=2.0, kd=0.12, track_k=0.5, scene=scene)
        info = env.step(cmd)

        tau = env.data.ctrl.copy()
        if last_tau is not None:
            torque_diffs.append(np.abs(tau - last_tau))
        last_tau = tau

        # Only evaluate after initial settling (first 1.0s / 50 steps)
        if step >= 50:
            pos_history.append(info["pos"].copy())
            rpy_history.append(info["rpy"].copy())
            vel_history.append(info["vel"].copy())

            base_pos = env.data.qpos[0:3].copy()
            base_quat = env.data.qpos[3:7].copy()

            for fn in ["FL", "FR", "RL", "RR"]:
                gid = env.foot_geom_ids[fn]
                geom_pos_w = env.data.geom_xpos[gid].copy()
                geom_pos_b = quat_apply_inverse(base_quat, geom_pos_w - base_pos)

                foot_pos_w[fn].append(geom_pos_w)
                foot_pos_b[fn].append(geom_pos_b)
                contacts_history[fn].append(info["contacts"][fn])

    pos_arr = np.array(pos_history)
    rpy_arr = np.array(rpy_history)
    vel_arr = np.array(vel_history)
    tau_diff_arr = np.array(torque_diffs)

    # Body Metrics
    height_ptp = np.ptp(pos_arr[:, 2]) * 1000.0  # mm
    height_std = np.std(pos_arr[:, 2]) * 1000.0  # mm
    height_mean = np.mean(pos_arr[:, 2])

    roll_ptp = np.ptp(rpy_arr[:, 0])  # deg
    roll_std = np.std(rpy_arr[:, 0])  # deg
    pitch_ptp = np.ptp(rpy_arr[:, 1])  # deg
    pitch_std = np.std(rpy_arr[:, 1])  # deg
    yaw_ptp = np.ptp(rpy_arr[:, 2])  # deg
    yaw_max_drift = np.max(np.abs(rpy_arr[:, 2]))  # deg

    lateral_drift_max = np.max(np.abs(pos_arr[:, 1])) * 1000.0  # mm
    forward_speed_mean = np.mean(vel_arr[:, 0])

    mean_delta_tau = float(np.mean(tau_diff_arr)) if len(tau_diff_arr) > 0 else 0.0

    # Foot Kinematics
    rl_swing_y = []
    rr_swing_y = []
    rl_swing_z = []
    rr_swing_z = []

    for i in range(len(contacts_history["RL"])):
        if not contacts_history["RL"][i]:
            rl_swing_y.append(foot_pos_b["RL"][i][1])
            rl_swing_z.append(foot_pos_w["RL"][i][2])
        if not contacts_history["RR"][i]:
            rr_swing_y.append(foot_pos_b["RR"][i][1])
            rr_swing_z.append(foot_pos_w["RR"][i][2])

    rl_y_peak = float(np.max(np.abs(rl_swing_y))) * 1000.0 if rl_swing_y else 0.0
    rr_y_peak = float(np.max(np.abs(rr_swing_y))) * 1000.0 if rr_swing_y else 0.0
    rear_swing_asymm = abs(rl_y_peak - rr_y_peak)

    # Foot clearance (peak height in swing)
    rl_clearance = float(np.max(rl_swing_z) - np.min(rl_swing_z)) * 1000.0 if rl_swing_z else 0.0
    rr_clearance = float(np.max(rr_swing_z) - np.min(rr_swing_z)) * 1000.0 if rr_swing_z else 0.0

    # Stance width (distance between left and right feet in base Y)
    rear_width = float(np.mean([abs(foot_pos_b["RL"][i][1] - foot_pos_b["RR"][i][1]) for i in range(len(pos_arr))])) * 1000.0

    return {
        "model_dir": model_dir,
        "forward_speed_mean": forward_speed_mean,
        "height_mean": height_mean,
        "height_ptp_mm": height_ptp,
        "height_std_mm": height_std,
        "roll_ptp_deg": roll_ptp,
        "roll_std_deg": roll_std,
        "pitch_ptp_deg": pitch_ptp,
        "pitch_std_deg": pitch_std,
        "yaw_ptp_deg": yaw_ptp,
        "yaw_max_drift_deg": yaw_max_drift,
        "lateral_drift_max_mm": lateral_drift_max,
        "rear_swing_asymm_mm": rear_swing_asymm,
        "rl_y_peak_mm": rl_y_peak,
        "rr_y_peak_mm": rr_y_peak,
        "rear_clearance_mm": (rl_clearance + rr_clearance) / 2.0,
        "rear_width_mm": rear_width,
        "mean_delta_tau_nm": mean_delta_tau,
    }


if __name__ == "__main__":
    if len(sys.argv) > 1:
        md = sys.argv[1]
    else:
        md = "/home/rum/dreamwaq_lab/logs/rsl_rl/unitree/unitree_go2_dreamwaq_flat/2026-10-11_19-35_waxberry_flat_straight_symmetry_finetune_300iter/exported"
    res = evaluate_model(md)
    for k, v in res.items():
        if isinstance(v, float):
            print(f"{k:25s}: {v:8.3f}")
        else:
            print(f"{k:25s}: {v}")
