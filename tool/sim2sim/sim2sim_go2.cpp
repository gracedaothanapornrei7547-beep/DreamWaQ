#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <chrono>
#include <cstring>
#include <algorithm>
#include <iomanip>
#include <mujoco/mujoco.h>
#include <onnxruntime_cxx_api.h>

// ----------------------------------------------------------------------------
// Quaternion math helpers
// ----------------------------------------------------------------------------
struct Vec3 {
    double x, y, z;
};

// Rotate vector by inverse of quaternion [w, x, y, z]
Vec3 quat_apply_inverse(const mjtNum quat[4], const mjtNum vec[3]) {
    double w = quat[0], qx = quat[1], qy = quat[2], qz = quat[3];
    // Cross product xyz x vec
    double tx = 2.0 * (qy * vec[2] - qz * vec[1]);
    double ty = 2.0 * (qz * vec[0] - qx * vec[2]);
    double tz = 2.0 * (qx * vec[1] - qy * vec[0]);

    double vtx = ty * qz - tz * qy;
    double vty = tz * qx - tx * qz;
    double vtz = tx * qy - ty * qx;

    return {
        vec[0] - w * tx + vtx,
        vec[1] - w * ty + vty,
        vec[2] - w * tz + vtz
    };
}

void quat_to_rpy_deg(const mjtNum quat[4], double& roll, double& pitch, double& yaw) {
    double w = quat[0], x = quat[1], y = quat[2], z = quat[3];
    double sinr_cosp = 2.0 * (w * x + y * z);
    double cosr_cosp = 1.0 - 2.0 * (x * x + y * y);
    roll = std::atan2(sinr_cosp, cosr_cosp) * 180.0 / M_PI;

    double sinp = 2.0 * (w * y - z * x);
    if (std::abs(sinp) >= 1.0)
        pitch = std::copysign(90.0, sinp);
    else
        pitch = std::asin(sinp) * 180.0 / M_PI;

    double siny_cosp = 2.0 * (w * z + x * y);
    double cosy_cosp = 1.0 - 2.0 * (y * y + z * z);
    yaw = std::atan2(siny_cosp, cosy_cosp) * 180.0 / M_PI;
}

// ----------------------------------------------------------------------------
// Main C++ MuJoCo Sim2Sim Engine
// ----------------------------------------------------------------------------
int main(int argc, char** argv) {
    std::string model_path = "/home/rum/isaac/dreamwaq_lab/logs/DreamQaW_P2/1_straight/exported/policy.onnx";
    std::string scene_path = "/home/rum/himloco_assets/mujoco_menagerie/unitree_go2/scene.xml";
    double duration = 10.0;
    double cmd_vx = 0.8;
    double cmd_vy = 0.0;
    double cmd_wz = 0.0;
    bool heading_lock = true;
    double heading_kp = 0.8;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--model" && i + 1 < argc) model_path = argv[++i];
        else if (arg == "--scene" && i + 1 < argc) scene_path = argv[++i];
        else if (arg == "--duration" && i + 1 < argc) duration = std::stod(argv[++i]);
        else if (arg == "--vx" && i + 1 < argc) cmd_vx = std::stod(argv[++i]);
        else if (arg == "--no_heading_lock") heading_lock = false;
    }

    std::cout << "=================================================================\n";
    std::cout << "      Unitree Go2 DreamWaQ Native C++ Sim2Sim Pipeline\n";
    std::cout << "=================================================================\n";
    std::cout << "  MuJoCo Version : " << mj_versionString() << "\n";
    std::cout << "  ONNX Model     : " << model_path << "\n";
    std::cout << "  Scene XML      : " << scene_path << "\n";
    std::cout << "  Target Command : vx=" << cmd_vx << " m/s, vy=" << cmd_vy << " m/s, heading_lock=" << (heading_lock ? "ON" : "OFF") << "\n";
    std::cout << "  Duration       : " << duration << " seconds\n";
    std::cout << "=================================================================\n" << std::endl;

    // 1. Load MuJoCo Model
    char error[1000] = "Could not load XML model";
    mjModel* m = mj_loadXML(scene_path.c_str(), nullptr, error, 1000);
    if (!m) {
        std::cerr << "[ERROR] mj_loadXML failed: " << error << std::endl;
        return 1;
    }
    mjData* d = mj_makeData(m);

    // Disable passive damping on joints 1..12
    for (int i = 6; i < m->nv; ++i) {
        m->dof_damping[i] = 0.0;
    }

    // Default joint angles in IsaacLab order: FL(3), FR(3), RL(3), RR(3)
    const float default_jpos_isaac[12] = {
        0.1f, 0.8f, -1.5f,
       -0.1f, 0.8f, -1.5f,
        0.1f, 1.0f, -1.5f,
       -0.1f, 1.0f, -1.5f
    };

    // Joint permutation map (MuJoCo Menagerie order is FL, FR, RL, RR by leg)
    // Menagerie joint order:
    // 0..2: FL_hip, FL_thigh, FL_calf
    // 3..5: FR_hip, FR_thigh, FR_calf
    // 6..8: RL_hip, RL_thigh, RL_calf
    // 9..11: RR_hip, RR_thigh, RR_calf
    // In IsaacLab DreamWaQ: same limb-grouped order!
    // So permutation is 1:1 identity for Go2 Menagerie!
    
    // Set initial robot state
    mj_resetData(m, d);
    d->qpos[0] = 0.0; d->qpos[1] = 0.0; d->qpos[2] = 0.35; // pos
    d->qpos[3] = 1.0; d->qpos[4] = 0.0; d->qpos[5] = 0.0; d->qpos[6] = 0.0; // identity quat
    for (int i = 0; i < 12; ++i) {
        d->qpos[7 + i] = default_jpos_isaac[i];
    }
    mj_forward(m, d);

    // 2. Initialize ONNXRuntime C++ Session
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "dreamwaq_sim2sim_cpp");
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(2);
    session_options.SetGraphOptimizationLevel(ORT_ENABLE_EXTENDED);

    Ort::Session session(env, model_path.c_str(), session_options);
    Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

    const char* input_names[] = {"current_obs", "history_obs"};
    const char* output_names[] = {"actions"};

    const int64_t current_obs_shape[] = {1, 45};
    const int64_t history_obs_shape[] = {1, 225};

    // Observation buffers
    std::vector<float> current_obs(45, 0.0f);
    std::vector<float> history_obs(225, 0.0f);
    
    // Term history ring buffer: 5 steps of 6 terms
    // term 0: ang_vel (3)
    // term 1: gravity (3)
    // term 2: cmd (3)
    // term 3: jpos (12)
    // term 4: jvel (12)
    // term 5: last_action (12)
    const int term_dims[6] = {3, 3, 3, 12, 12, 12};
    std::vector<std::vector<float>> term_history(6);
    for (int t = 0; t < 6; ++t) {
        term_history[t].resize(5 * term_dims[t], 0.0f);
    }

    std::vector<float> last_action(12, 0.0f);
    std::vector<float> current_action(12, 0.0f);
    std::vector<float> target_pos_filtered(12, 0.0f);
    for (int i = 0; i < 12; ++i) target_pos_filtered[i] = default_jpos_isaac[i];

    const double sim_dt = 0.002;
    const int decimation = 10;
    const double policy_dt = sim_dt * decimation; // 0.02s (50Hz)
    const double Kp = 25.0;
    const double Kd = 0.50;
    const double torque_limit = 23.7;
    const double filter_alpha = 0.70;
    const double action_scale = 0.25;

    int total_policy_steps = static_cast<int>(duration / policy_dt);
    auto start_wall_time = std::chrono::high_resolution_clock::now();

    std::cout << "[INFO] Entering 500Hz MuJoCo physics & 50Hz C++ policy loop..." << std::endl;

    for (int step = 0; step < total_policy_steps; ++step) {
        double current_time = step * policy_dt;

        // --- 1. Compute Base IMU & Orientation ---
        mjtNum quat[4] = {d->qpos[3], d->qpos[4], d->qpos[5], d->qpos[6]};
        mjtNum ang_vel_w[3] = {d->qvel[3], d->qvel[4], d->qvel[5]};
        Vec3 ang_vel_b = quat_apply_inverse(quat, ang_vel_w);
        
        mjtNum gravity_w[3] = {0.0, 0.0, -1.0};
        Vec3 proj_gravity_b = quat_apply_inverse(quat, gravity_w);

        double roll, pitch, yaw;
        quat_to_rpy_deg(quat, roll, pitch, yaw);

        // Heading lock controller
        double eff_wz = cmd_wz;
        if (heading_lock) {
            double heading_rad = yaw * M_PI / 180.0;
            // Target heading = 0.0 rad
            double heading_error = -heading_rad;
            while (heading_error > M_PI) heading_error -= 2.0 * M_PI;
            while (heading_error < -M_PI) heading_error += 2.0 * M_PI;
            eff_wz = std::clamp(heading_kp * heading_error, -1.0, 1.0);
        }

        // --- 2. Build 6 current observation terms ---
        std::vector<float> cur_terms[6];
        cur_terms[0] = {static_cast<float>(ang_vel_b.x * 0.20), static_cast<float>(ang_vel_b.y * 0.20), static_cast<float>(ang_vel_b.z * 0.20)};
        cur_terms[1] = {static_cast<float>(proj_gravity_b.x), static_cast<float>(proj_gravity_b.y), static_cast<float>(proj_gravity_b.z)};
        cur_terms[2] = {static_cast<float>(cmd_vx), static_cast<float>(cmd_vy), static_cast<float>(eff_wz)};
        
        cur_terms[3].resize(12);
        cur_terms[4].resize(12);
        for (int i = 0; i < 12; ++i) {
            float jpos = static_cast<float>(d->qpos[7 + i]) - default_jpos_isaac[i];
            float jvel = static_cast<float>(d->qvel[6 + i]) * 0.05f;
            cur_terms[3][i] = std::clamp(jpos, -100.0f, 100.0f);
            cur_terms[4][i] = std::clamp(jvel, -100.0f, 100.0f);
        }
        cur_terms[5] = last_action;

        // Assemble current_obs (45 dim)
        int obs_idx = 0;
        for (int t = 0; t < 6; ++t) {
            for (float val : cur_terms[t]) {
                current_obs[obs_idx++] = std::clamp(val, -100.0f, 100.0f);
            }
        }

        // Update term_history (shift oldest out, push newest to back)
        for (int t = 0; t < 6; ++t) {
            int dim = term_dims[t];
            // Shift left by dim
            std::copy(term_history[t].begin() + dim, term_history[t].end(), term_history[t].begin());
            // Put current at last dim
            std::copy(cur_terms[t].begin(), cur_terms[t].end(), term_history[t].end() - dim);
        }

        // Flatten term_history into history_obs (225 dim, term-major ordering)
        int hist_idx = 0;
        for (int t = 0; t < 6; ++t) {
            for (float val : term_history[t]) {
                history_obs[hist_idx++] = val;
            }
        }

        // --- 3. ONNX Inference ---
        Ort::Value input_tensors[] = {
            Ort::Value::CreateTensor<float>(mem_info, current_obs.data(), current_obs.size(), current_obs_shape, 2),
            Ort::Value::CreateTensor<float>(mem_info, history_obs.data(), history_obs.size(), history_obs_shape, 2)
        };

        auto output_tensors = session.Run(
            Ort::RunOptions{nullptr},
            input_names, input_tensors, 2,
            output_names, 1
        );

        float* raw_action = output_tensors[0].GetTensorMutableData<float>();
        for (int i = 0; i < 12; ++i) {
            current_action[i] = std::clamp(raw_action[i], -100.0f, 100.0f);
            last_action[i] = current_action[i];
        }

        // --- 4. Compute Joint Targets with EMA Smoothing ---
        float target_pos[12];
        for (int i = 0; i < 12; ++i) {
            float raw_target = default_jpos_isaac[i] + static_cast<float>(action_scale) * current_action[i];
            target_pos_filtered[i] = static_cast<float>(filter_alpha) * raw_target + static_cast<float>(1.0 - filter_alpha) * target_pos_filtered[i];
            target_pos[i] = target_pos_filtered[i];
        }

        // --- 5. Step MuJoCo Physics (500Hz decimation) ---
        double peak_tau = 0.0;
        for (int sub = 0; sub < decimation; ++sub) {
            for (int i = 0; i < 12; ++i) {
                double q = d->qpos[7 + i];
                double dq = d->qvel[6 + i];
                double tau = Kp * (target_pos[i] - q) - Kd * dq;
                tau = std::clamp(tau, -torque_limit, torque_limit);
                peak_tau = std::max(peak_tau, std::abs(tau));
                d->ctrl[i] = tau;
            }
            mj_step(m, d);
        }

        // --- 6. Telemetry Output every 0.5s ---
        if (step % 25 == 0 || step == total_policy_steps - 1) {
            mjtNum x = d->qpos[0], y = d->qpos[1], z = d->qpos[2];
            mjtNum vx = d->qvel[0], vy = d->qvel[1];
            std::cout << std::fixed << std::setprecision(2)
                      << "[" << std::setw(5) << current_time << "s] "
                      << "Pos: x=" << std::setw(5) << x << "m, y=" << std::setw(5) << y << "m, z=" << std::setw(5) << z << "m | "
                      << "Vel: vx=" << std::setw(5) << vx << "m/s, vy=" << std::setw(5) << vy << "m/s | "
                      << "RPY: [" << std::setw(5) << roll << "°, " << std::setw(5) << pitch << "°, " << std::setw(5) << yaw << "°] | "
                      << "MaxTau: " << std::setw(4) << peak_tau << "Nm" << std::endl;
        }
    }

    auto end_wall_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_wall_time - start_wall_time;

    std::cout << "\n=================================================================\n";
    std::cout << "           C++ Sim2Sim (MuJoCo) 验证圆满完成报告\n";
    std::cout << "=================================================================\n";
    std::cout << "  物理总模拟时长 : " << duration << " 秒\n";
    std::cout << "  C++ 推理与仿真耗时: " << elapsed.count() << " 秒\n";
    std::cout << "  计算加速比     : " << (duration / elapsed.count()) << " 倍实时 (超极速)\n";
    std::cout << "  最终前向位移 X  : " << d->qpos[0] << " 米\n";
    std::cout << "  最终横向偏倚 Y  : " << d->qpos[1] << " 米 (仅 " << (d->qpos[1] * 100.0) << " 厘米)\n";
    std::cout << "  最终机身高度 Z  : " << d->qpos[2] << " 米 (标称 0.29~0.30m)\n";
    std::cout << "=================================================================\n" << std::endl;

    mj_deleteData(d);
    mj_deleteModel(m);
    return 0;
}

