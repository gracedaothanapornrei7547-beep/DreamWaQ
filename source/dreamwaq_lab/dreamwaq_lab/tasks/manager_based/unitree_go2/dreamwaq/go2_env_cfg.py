# Copyright (c) 2024 LunarQuadruped Project.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""宇树 Go2 四足机器人 DreamWaQ 核心基础环境配置 (Base Environment Configuration)。

【架构与设计哲学】
1. 本文件定义了 Unitree Go2 运行 DreamWaQ 强化学习算法的通用骨架配置：
   - 交互场景定义 (InteractiveSceneCfg): 地形导入、物理材料、高程扫描射线、接触力传感器及光照。
   - 速度指令生成器 (CommandsCfg): 包含课程自适应范围与驻车零速指令机制。
   - 动作空间 (ActionsCfg): 采用 12 关节标称位置偏移控制（PD 闭环底层跟踪）。
   - 观测系统 (ObservationsCfg): 严格遵循 DreamWaQ 的非对称信息架构（Asymmetric Actor-Critic + CENet/VAE）。
   - 随机化事件 (EventCfg): 域随机化（Domain Randomization）覆盖摩擦、负载、质心、电机强度与脉冲扰动。
   - 基础奖励项 (RewardsCfg): 全套标准化运动奖励、能耗抑制与姿态阻尼项。
   - 终止条件 (TerminationsCfg): 摔倒、触底碰撞、极限倾覆与超时截断。
   - 课程机制 (CurriculumCfg): 地形等级递进与速度指令范围扩展的双重课程。
"""

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from dreamwaq_lab.terrains import GO2_EXTENDED_ROUGH_TERRAINS_CFG
from dreamwaq_lab.robots.unitree_go2 import UNITREE_GO2_CFG

from .. import mdp


# ==============================================================================
# 全局常量定义 (Global Constants)
# ==============================================================================
# 【足端刚体名称列表】：用于接触力传感器采样、触地时间统计与足端规整项
FOOT_BODY_NAMES = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]

# 【机身目标标称对地高度 (单位: 米)】：
# 【物理依据】：Go2 机械腿自然站立全伸展高度约 0.38m，折叠常态高度推荐 0.31m ~ 0.35m。
# 【调参指南】：
#   - 0.35m：腿部更加直立挺拔，下肢能耗更低，但跨越超高台阶时俯仰下蹲裕度略小；
#   - 0.31m~0.32m：低重心稳健姿态，爬坡与抗冲击能力极强，但关节扭矩略有上升。
BASE_HEIGHT_TARGET = 0.35

# 【电机扭矩放大系数随机化范围】：(0.9 ~ 1.1)
# 【物理依据】：模拟实机电机驱动板在不同温升与母线电压下的执行力衰减（±10% 动力波动）。
GO2_MOTOR_STRENGTH_RANGE = (0.9, 1.1)

# 【显式 12 关节拓扑顺序定义】：
# 【工程避坑要点】：绝不依赖正则表达式或乱序字典遍历！固定顺序确保训练、推理、ONNX导出与实机 C++ 部署的绝对一致。
GO2_ACTION_JOINT_NAMES = [
    "FL_hip_joint",
    "FL_thigh_joint",
    "FL_calf_joint",
    "FR_hip_joint",
    "FR_thigh_joint",
    "FR_calf_joint",
    "RL_hip_joint",
    "RL_thigh_joint",
    "RL_calf_joint",
    "RR_hip_joint",
    "RR_thigh_joint",
    "RR_calf_joint",
]


# ==============================================================================
# 1. 物理仿真交互场景配置 (Interactive Scene Configuration)
# ==============================================================================
@configclass
class UnitreeGo2DreamWaQSceneCfg(InteractiveSceneCfg):
    """Go2 DreamWaQ 复杂地形交互场景配置。

    【核心理念】：
    - 实机部署时机器狗为“盲狗”（Blind Locomotion，无深度相机），因此 Actor 不包含高程图输入；
    - 在仿真中保留高度扫描射线（height_scanner）作为特权信息（Privileged Information）供给 Critic 网络，
      引导 Critic 精确评估地形势能并计算足端高程清除奖励。
    """

    # --- 地形导入器配置 ---
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=GO2_EXTENDED_ROUGH_TERRAINS_CFG,
        max_init_terrain_level=1,  # 初始重生最高地形等级（从低难度起步）
        collision_group=-1,        # 碰撞过滤分组
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",  # 摩擦系数合成模式：相乘
            restitution_combine_mode="average", # 恢复系数合成模式：取平均
            static_friction=1.0,               # 标称静摩擦系数
            dynamic_friction=1.0,              # 标称动摩擦系数
            restitution=0.0,                   # 完全非弹性碰撞（无回弹）
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )

    # --- 机器人刚体模型配置 ---
    # 基于宇树官方 Go2 USD 资产定义，使用 ENV_REGEX_NS 占位符实例化多并行环境
    robot: ArticulationCfg = UNITREE_GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # --- 机身特权高度扫描仪 (Height Scanner / RayCaster) ---
    # 【机制说明】：从机身 base 坐标系正上方 20.0m 向下发射射线，计算地面高度网格
    # 【射线网格】：长 1.6m x 宽 1.0m，分辨率 0.1m（共 17 x 11 = 187 根射线），沿机身 Yaw 对齐
    # 【作用】：仅供给特权 Critic 评估状态价值，以及局部足端清除度奖励函数
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )

    # --- 足端接触力传感器 (Contact Sensor) ---
    # 记录触地法向力、触地标志及腾空时间（track_air_time=True），保留 3 帧历史记录
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        track_air_time=True,
    )

    # --- 场景天空光照 ---
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


# ==============================================================================
# 2. 运动指令生成器配置 (Command Specifications)
# ==============================================================================
@configclass
class CommandsCfg:
    """DreamWaQ 速度追踪指令配置。"""

    # 【线速度与角速度指令采样器】：
    # 【机制说明】：
    # 1. 周期性（每隔 10.0s）在速度范围内随机抽取一组 [vx, vy, yaw_rate] 指令。
    # 2. rel_standing_envs: 强制分配 5% 的环境输入全零指令 [0, 0, 0]，训练静止站立防抖能力。
    # 3. ranges vs limit_ranges: 初始采样范围较窄（降低冷启动难度），随着课程推进自适应扩展至 limit_ranges。
    base_velocity = mdp.UniformLevelVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),  # 指令重采样周期 (秒)
        rel_standing_envs=0.05,             # 静态站立环境比例 (5%)
        debug_vis=False,                    # 禁用指令箭头 USD，防止无头集群启动依赖网络延迟
        # 初始速度采样范围 (Stage 0 阶段)：
        ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.5, 0.5),           # 前进/后退: -0.5 ~ 0.5 m/s
            lin_vel_y=(-0.2, 0.2),           # 侧移: -0.2 ~ 0.2 m/s
            ang_vel_z=(-0.3, 0.3),           # 转向: -0.3 ~ 0.3 rad/s (~17°/s)
        ),
        # 课程扩展极限范围 (完全体能力上限)：
        limit_ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.5, 1.5),           # 极速前冲: -1.5 ~ 1.5 m/s
            lin_vel_y=(-0.6, 0.6),           # 极速侧行: -0.6 ~ 0.6 m/s
            ang_vel_z=(-0.8, 0.8),           # 极速旋转: -0.8 ~ 0.8 rad/s (~45°/s)
        ),
    )


# ==============================================================================
# 3. 动作空间配置 (Action Specifications)
# ==============================================================================
@configclass
class ActionsCfg:
    """Go2 关节位置控制输出规范。"""

    # 【关节位置残差动作 (PD Joint Position Delta)】：
    # 【机制与公式】：
    #   q_target = q_nominal + scale * action
    # 【工程调参机理】：
    #   - scale = 0.25：策略输出归一化在 [-1, 1] 区间，乘以 0.25 对应每个关节最大角度偏置为 ±0.25 rad (~14.3°)。
    #   - 增大幅度（如 0.5）：动作更激进，跨越高台阶更容易，但极易引发电机高频震荡与超调；
    #   - 减小幅度（如 0.15）：步态极其平稳，但台阶通过性不足。0.25 是工业界四足狗的黄金标杆。
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=GO2_ACTION_JOINT_NAMES,
        scale=0.25,
        use_default_offset=True,
        preserve_order=True,
        clip={".*": (-100.0, 100.0)},
    )


# ==============================================================================
# 4. 观测空间配置 (Observation Specifications - DreamWaQ 核心)
# ==============================================================================
@configclass
class ObservationsCfg:
    """DreamWaQ 专属非对称观测组定义。

    【核心结构】：
    1. ProprioceptionCfg (带噪本体感知): 包含 IMU、关节角度、速度、指令与上一帧动作，注入现实级噪声，供给部署 Actor。
    2. DreamWaQPolicyHistoryCfg (5 步历史滑动窗): 连续 5 帧本体感知时序，送入 CENet (VAE) 推理隐式动力学表征 z。
    3. DreamWaQReconstructionTargetCfg (干净本体感知下一帧): 作为 VAE 重构自监督 Loss 的真实监督信号。
    4. DreamWaQCriticCfg (特权全局感知): 供给特权 Critic 训练，包含真实机身线速度、扭矩、加速度、足端冲击力与局部高程网格。
    """

    @configclass
    class ProprioceptionCfg(ObsGroup):
        """部署 Actor 所能获取的带噪传感器读数（Noisy Proprioception）。"""

        # 机身角速度 (IMU 陀螺仪): 缩放 0.2，添加 ±0.2 rad/s 均匀噪声
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2, clip=(-100, 100), noise=Unoise(n_min=-0.2, n_max=0.2))

        # 机身投影重力矢量 [gx, gy, gz]: 描述机身倾角姿态，添加 ±0.05 噪声
        projected_gravity = ObsTerm(func=mdp.projected_gravity, clip=(-100, 100), noise=Unoise(n_min=-0.05, n_max=0.05))

        # 当前速度目标指令 [vx_cmd, vy_cmd, yaw_rate_cmd]: 无噪声
        velocity_commands = ObsTerm(func=mdp.generated_commands, clip=(-100, 100), params={"command_name": "base_velocity"})

        # 12 关节相对标称位置角度误差: 编码器精度误差注入 ±0.01 rad (~0.57°)
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            clip=(-100, 100),
            noise=Unoise(n_min=-0.01, n_max=0.01),
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=GO2_ACTION_JOINT_NAMES,
                    preserve_order=True,
                )
            },
        )

        # 12 关节角速度: 缩放 0.05，差分测速噪声注入 ±1.5 rad/s
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            scale=0.05,
            clip=(-100, 100),
            noise=Unoise(n_min=-1.5, n_max=1.5),
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=GO2_ACTION_JOINT_NAMES,
                    preserve_order=True,
                )
            },
        )

        # 上一时刻动作输出 a_{t-1}: 提供控制自回归记忆
        last_action = ObsTerm(func=mdp.last_action, clip=(-100, 100))

        def __post_init__(self):
            self.enable_corruption = True  # 启用噪声注入
            self.concatenate_terms = True  # 拼合成单一平坦特征向量

    @configclass
    class CleanProprioceptionCfg(ObsGroup):
        """无噪声干净本体感知（用于特权 Critic 与 VAE 重构目标）。"""

        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2, clip=(-100, 100))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, clip=(-100, 100))
        velocity_commands = ObsTerm(func=mdp.generated_commands, clip=(-100, 100), params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            clip=(-100, 100),
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=GO2_ACTION_JOINT_NAMES,
                    preserve_order=True,
                )
            },
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            scale=0.05,
            clip=(-100, 100),
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=GO2_ACTION_JOINT_NAMES,
                    preserve_order=True,
                )
            },
        )
        last_actions = ObsTerm(func=mdp.last_action, clip=(-100, 100))

        def __post_init__(self):
            self.enable_corruption = False  # 禁用噪声
            self.concatenate_terms = True

    @configclass
    class DreamWaQPolicyCfg(ProprioceptionCfg):
        """Actor 当前帧单步感知输入。"""

    @configclass
    class DreamWaQPolicyHistoryCfg(ProprioceptionCfg):
        """CENet (VAE) 所需的 5 帧滑动历史感知缓存 (维度: 5 * Dim_Proprioception)。"""

        def __post_init__(self):
            super().__post_init__()
            self.history_length = 5
            self.flatten_history_dim = True

    @configclass
    class DreamWaQReconstructionTargetCfg(CleanProprioceptionCfg):
        """CENet 下一时刻状态自监督重构目标。"""

    @configclass
    class DreamWaQCriticCfg(CleanProprioceptionCfg):
        """特权 Critic 输入：无噪声本体感知 + 真实线速度 + 动力学加速度/扭矩 + 接触力 + 地形高程。"""

        # 真实机身世界/机体系 3D 线速度 (实机中通常无法精准直测)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, clip=(-100, 100))

        # 12 关节瞬时角加速度 (缩放 1e-4)
        joint_acc = ObsTerm(
            func=mdp.joint_acc,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=GO2_ACTION_JOINT_NAMES,
                    preserve_order=True,
                )
            },
            clip=(-100, 100),
            scale=1.0e-4,
        )

        # 12 关节驱动力矩 (缩放 0.01)
        joint_torque = ObsTerm(
            func=mdp.joint_effort,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=GO2_ACTION_JOINT_NAMES,
                    preserve_order=True,
                )
            },
            clip=(-100, 100),
            scale=0.01,
        )

        # 四足足端接触力范数 (缩放 1e-3)
        contact_force = ObsTerm(
            func=mdp.foot_contact_force_norm,
            params={
                "sensor_cfg": SceneEntityCfg(
                    "contact_forces",
                    body_names=FOOT_BODY_NAMES,
                    preserve_order=True,
                )
            },
            clip=(-100, 100),
            scale=1.0e-3,
        )

        # 局部 187 点高程扫描网格 (截断在 ±1.0m 范围内)
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            clip=(-1.0, 1.0),
        )

    # 注册各观测组实例
    dreamwaq_policy: DreamWaQPolicyCfg | None = DreamWaQPolicyCfg()
    dreamwaq_policy_history: DreamWaQPolicyHistoryCfg | None = DreamWaQPolicyHistoryCfg()
    dreamwaq_reconstruction_target: DreamWaQReconstructionTargetCfg | None = DreamWaQReconstructionTargetCfg()
    dreamwaq_critic: DreamWaQCriticCfg | None = DreamWaQCriticCfg()


# ==============================================================================
# 5. 域随机化事件配置 (Domain Randomization Events)
# ==============================================================================
@configclass
class EventCfg:
    """物理动力学随机化与环境重置事件。"""

    # 【电机驱动强度系数随机化】：每次重置时在 (0.9, 1.1) 之间缩放扭矩
    motor_strength = EventTerm(
        func=mdp.randomize_motor_strength,
        mode="reset",
        params={"strength_range": GO2_MOTOR_STRENGTH_RANGE},
    )

    # 【刚体接触材质摩擦力随机化】：
    # 【模式】：启动时（startup）随机分布到 64 个桶中，覆盖 0.4 ~ 1.25 的动静摩擦
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.4, 1.25),
            "dynamic_friction_range": (0.4, 1.25),
            "restitution_range": (0.0, 0.5),
            "num_buckets": 64,
        },
    )

    # 【机身附加质量随机化】：(-0.5kg ~ +2.0kg)
    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "mass_distribution_params": (-0.5, 2.0),
            "operation": "add",
        },
    )

    # 【腿部各连杆质量比例微调】：(0.95 ~ 1.05)，重新计算转动惯量
    scale_other_body_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="^(?!.*base).*"),
            "mass_distribution_params": (0.95, 1.05),
            "operation": "scale",
            "recompute_inertia": True,
        },
    )

    # 【质心微小偏移 (CoM Jitter)】：打破对称平衡，训练重心漂移鲁棒性
    base_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "com_range": {"x": (-0.01, 0.01), "y": (-0.005, 0.005), "z": (-0.003, 0.003)},
        },
    )

    # 【机身恒定外力/力矩】：基类默认置零
    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "force_range": (0.0, 0.0),
            "torque_range": (0.0, 0.0),
        },
    )

    # 【重生初始位姿采样】：X/Y 在 ±0.5m 内抖动，Yaw 偏航全向随机 (-π ~ π)
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
    )

    # 【重生关节角度缩放】：(0.95 ~ 1.05) 初始站立角度扰动
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.95, 1.05),
            "velocity_range": (0.0, 0.0),
        },
    )

    # 【关节执行器 PD 增益缩放】：Kp、Kd 在 ±10% 范围内均匀缩放
    robot_joint_stiffness_and_damping = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (0.9, 1.1),
            "damping_distribution_params": (0.9, 1.1),
            "operation": "scale",
            "distribution": "uniform",
        },
    )

    # 【电机零点标定偏置 (Zero Offset)】：模拟电机安装装配或编码器机械零点微小漂移 (±0.035 rad, 约 ±2°)
    motor_zero_offset = EventTerm(
        func=mdp.randomize_action_joint_pos_offset,
        mode="reset",
        params={
            "action_term_name": "joint_pos",
            "offset_range": (-0.035, 0.035),
        },
    )

    # 【间歇性外部速度推力脉冲 (Periodic Pushes)】：每 10~15 秒施加一次 X 轴 ±0.2m/s, Y 轴 ±0.1m/s 的瞬时冲击
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params={"velocity_range": {"x": (-0.2, 0.2), "y": (-0.1, 0.1)}},
    )


# ==============================================================================
# 6. 强化学习奖励函数族 (Base Reward Terms)
# ==============================================================================
@configclass
class RewardsCfg:
    """Go2 DreamWaQ 全套标准奖励函数项定义。"""

    # --- 速度指令追踪项 ---
    # 【线速度高斯核追踪】：exp(-||v_xy - cmd_xy||^2 / std^2)
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=1.5,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    # 【角速度高斯核追踪】：exp(-(w_z - cmd_wz)^2 / std^2)
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )

    # --- 姿态与振荡抑制项 ---
    # 【机身 Z 轴线速度二次方惩罚】：抑制颠簸上下蹦跳
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)

    # 【机身 Roll/Pitch 角速度二次方惩罚】：抑制左右颠簸与俯仰剧烈晃动（基类标称 -0.05，在训练配置中微调至 -0.20）
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)

    # 【机身水平投影重力倾角惩罚】：约束重力矢量与机身 Z 轴平行（基类标称 -0.2，训练配置微调至 -1.0）
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-0.2)
    roll_orientation_l2 = RewTerm(func=mdp.roll_orientation_l2, weight=0.0)
    terrain_adaptive_pitch_orientation_l2 = RewTerm(
        func=mdp.terrain_adaptive_pitch_orientation_l2,
        weight=0.0,
        params={"asset_cfg": SceneEntityCfg("robot"), "stairs_pitch_threshold": 0.08},
    )

    # --- 防侧漂与姿态规整项 (Anti-Drift & Symmetrical Posture) ---
    # 【正交侧向速度二次方惩罚】：惩罚垂直于前进指令方向的横向漂移，消除“螃蟹式滑步”
    lin_vel_xy_orthogonal_l2 = RewTerm(
        func=mdp.lin_vel_xy_orthogonal_l2,
        weight=0.0,
        params={"command_name": "base_velocity", "command_threshold": 0.1},
    )

    # 【髋关节标称位姿偏离惩罚 (WaxberryS 标准)】：当直行时惩罚大腿髋关节偏离 0 度的八字脚外撇，根除偏航力矩不对称
    hip_pos_deviation_l1 = RewTerm(
        func=mdp.hip_pos_deviation_l1,
        weight=0.0,
        params={
            "command_name": "base_velocity",
            "command_threshold": 0.1,
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_joint"]),
        },
    )

    # 【摆动相专属髋关节外展画弧惩罚 (带全向与自转动态门控)】
    swing_hip_abduction_l2 = RewTerm(
        func=mdp.swing_hip_abduction_l2,
        weight=0.0,
        params={
            "command_name": "base_velocity",
            "command_threshold": 0.08,
            "yaw_threshold": 0.15,
            "contact_threshold": 1.0,
            "contact_sensor_cfg": SceneEntityCfg("contact_forces", body_names=["FL_foot", "FR_foot", "RL_foot", "RR_foot"]),
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_joint"]),
        },
    )

    # 【后肢对称外展约束 (消除单侧独甩)】
    rear_hip_swing_symmetry_l2 = RewTerm(
        func=mdp.rear_hip_swing_symmetry_l2,
        weight=0.0,
        params={
            "command_name": "base_velocity",
            "command_threshold": 0.08,
            "yaw_threshold": 0.15,
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_joint"]),
        },
    )

    # 【平地绝对偏航角误差惩罚】：直行时惩罚偏航角偏离 0 度的二次方误差，促使策略主动反向纠偏打轮
    flat_heading_error_l2 = RewTerm(
        func=mdp.flat_heading_error_l2,
        weight=0.0,
        params={"command_name": "base_velocity", "yaw_threshold": 0.15, "target_heading_rad": 0.0},
    )

    # 【双向等变镜像对称约束 (无截断，杜绝内扣套利)】
    bilateral_hip_symmetry_equivariance_l2 = RewTerm(
        func=mdp.bilateral_hip_symmetry_equivariance_l2,
        weight=0.0,
        params={
            "command_name": "base_velocity",
            "command_threshold": 0.08,
            "yaw_threshold": 0.15,
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_joint"]),
        },
    )

    # 【前腿髋关节双向中立零位对中锁】
    front_hip_neutral_pos_l2 = RewTerm(
        func=mdp.front_hip_neutral_pos_l2,
        weight=0.0,
        params={
            "command_name": "base_velocity",
            "command_threshold": 0.08,
            "yaw_threshold": 0.15,
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_joint"]),
        },
    )

    # 【四肢机械功率方差惩罚 (对标 agent.md 功耗均衡)】
    power_distribution_variance_l2 = RewTerm(
        func=mdp.power_distribution_variance_l2,
        weight=0.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    # 【直行偏航角速度刚性阻尼惩罚 (Stage 9 锁死直线)】
    straight_ang_vel_z = RewTerm(
        func=mdp.straight_ang_vel_z_l2,
        weight=0.0,
        params={"command_name": "base_velocity", "yaw_threshold": 0.10},
    )

    # 【连续接触速度阻尼 (Stage 9 柔化撞阶，消灭碰阶回缩假死与卡顿)】
    soft_contact_velocity_damping = RewTerm(
        func=mdp.soft_contact_velocity_damping_l2,
        weight=0.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*thigh", ".*calf"]),
            "asset_cfg": SceneEntityCfg("robot"),
            "contact_force_threshold": 2.0,
        },
    )

    # 【后腿髋关节双向中立零位锁 (Stage 9 消灭后驱单侧偏推)】
    rear_hip_neutral_pos = RewTerm(
        func=mdp.rear_hip_neutral_pos_l2,
        weight=0.0,
        params={
            "command_name": "base_velocity",
            "command_threshold": 0.08,
            "yaw_threshold": 0.15,
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_joint"]),
        },
    )

    # --- 足端软着陆阻尼与接触力缓冲 (Soft Landing Damping & Contact Force Cushioning) ---
    feet_contact_velocity_z = RewTerm(
        func=mdp.feet_contact_velocity_z_l2,
        weight=0.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*foot"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*foot"),
            "contact_threshold": 1.0,
        },
    )

    feet_contact_force_excess = RewTerm(
        func=mdp.feet_contact_force_excess_l2,
        weight=0.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*foot"),
            "max_contact_force": 85.0,
        },
    )

    action_smoothness_2nd = RewTerm(
        func=mdp.action_smoothness_2nd_custom,
        weight=0.0,
    )


    # --- 高度锁定与静止驻立 ---
    # 【机身相对局部地形高度惩罚】：保证机身始终悬浮于标称高度 BASE_HEIGHT_TARGET 之上
    base_height_relative_l2 = RewTerm(
        func=mdp.base_height_relative_l2,
        weight=-0.5,
        params={
            "target_height": BASE_HEIGHT_TARGET,
            "sensor_cfg": SceneEntityCfg("height_scanner"),
            "footprint_size": (0.45, 0.3),
            "reduction": "mean",
        },
    )

    # 【零速指令时关节复位惩罚】：静止时不许小碎步踩踏，必须规整站立
    stand_still_joint_deviation_l1 = RewTerm(
        func=mdp.stand_still_joint_deviation_l1,
        weight=-0.2,
        params={"command_name": "base_velocity", "command_threshold": 0.06},
    )

    # --- 动力学能耗与平滑度项 ---
    dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-1.0e-5)      # 关节力矩惩罚
    joint_power_l1 = RewTerm(func=mdp.joint_power_l1, weight=-2.0e-5)        # 机械功耗惩罚 (|tau * v|)
    dof_vel_l2 = RewTerm(func=mdp.joint_vel_l2, weight=0.0)                  # 关节角速度惩罚 (默认关闭)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)              # 关节角加速度惩罚 (防高频抖动)
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-1.0)          # 关节软限位超限惩罚

    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)           # 一阶动作指令变化率惩罚
    action_smoothness_l2 = RewTerm(func=mdp.action_smoothness_2nd_custom, weight=-0.01)  # 二阶动作加速度平滑惩罚

    # --- 足端步态与地形清除度备选项 (按需启用) ---
    feet_regulation = RewTerm(
        func=mdp.feet_regulation,
        weight=0.0,
        params={
            "base_height_target": BASE_HEIGHT_TARGET,
            "asset_cfg": SceneEntityCfg("robot", body_names=FOOT_BODY_NAMES),
            "sensor_cfg": SceneEntityCfg("height_scanner"),
            "footprint_size": (0.4, 0.3),
        },
    )
    feet_air_time = RewTerm(
        func=mdp.feet_air_time,
        weight=0.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*foot"),
            "command_name": "base_velocity",
            "threshold": 0.5,
        },
    )
    feet_air_time_limit = RewTerm(
        func=mdp.feet_air_time_limit,
        weight=0.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FOOT_BODY_NAMES),
            "max_air_time": 0.2,
        },
    )
    foot_clearance_reward = RewTerm(
        func=mdp.foot_clearance_reward,
        weight=0.0,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FOOT_BODY_NAMES),
            "asset_cfg": SceneEntityCfg("robot", body_names=FOOT_BODY_NAMES),
            "height_sensor_cfg": SceneEntityCfg("height_scanner"),
            "target_clearance": 0.05,
            "contact_threshold": 1.0,
            "top_k": 8,
        },
    )
    terrain_relative_foot_clearance_l1 = RewTerm(
        func=mdp.terrain_relative_foot_clearance_l1,
        weight=-0.0,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
            "height_sensor_cfg": SceneEntityCfg("height_scanner"),
            "target_clearance": 0.06,
            "contact_threshold": 5.0,
            "top_k": 8,
            "speed_scale": 2.0,
            "normalize": True,
        },
    )
    paper_foot_clearance_l2 = RewTerm(
        func=mdp.paper_foot_clearance_l2_new,
        weight=-0.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=FOOT_BODY_NAMES),
            "target_height": 0.05,
            "nominal_base_height": BASE_HEIGHT_TARGET,
        },
    )
    foot_swing_height_l2 = RewTerm(
        func=mdp.foot_swing_height_l2,
        weight=0.0,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FOOT_BODY_NAMES),
            "asset_cfg": SceneEntityCfg("robot", body_names=FOOT_BODY_NAMES),
            "target_height": 0.055,
            "contact_threshold": 1.0,
        },
    )
    # 【Trot 对角肢体对称性惩罚】：消除左右大腿/小腿不对称造成的跛行与侧漂
    joint_diagonal_symmetry_l2 = RewTerm(
        func=mdp.joint_diagonal_symmetry_l2,
        weight=0.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    # 【四足滞空/支撑时间方差惩罚】：惩罚四腿滞空时间方差，杜绝单腿常悬偷分
    feet_air_time_variance = RewTerm(
        func=mdp.feet_air_time_variance_penalty,
        weight=0.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot")},
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=0.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FOOT_BODY_NAMES),
            "asset_cfg": SceneEntityCfg("robot", body_names=FOOT_BODY_NAMES),
        },
    )
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=0.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*thigh"]),
            "threshold": 5.0,
        },
    )


# ==============================================================================
# 7. 回合终止条件配置 (Termination Terms)
# ==============================================================================
@configclass
class TerminationsCfg:
    """回合截断与摔倒判定配置。"""

    # 【单回合最长时间截断】：达到 20.0s (1000 policy steps) 正常截断并重置
    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    # 【机身触底碰撞终止】：base 刚体与地面接触力超过 1.0N 即判负终止
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"), "threshold": 1.0},
    )

    # 【机身极限倾覆姿态终止】：
    # 【工程安全阈值】：
    #   - 横滚角 (Roll) 超过 1.0 rad (~57.3°)
    #   - 俯仰角 (Pitch) 超过 1.2 rad (~68.8°)
    #   超出即判定彻底翻车，立即终止并重置，防止梯度崩塌。
    bad_orientation = DoneTerm(
        func=mdp.bad_roll_pitch_orientation,
        params={"limit_roll": 1.0, "limit_pitch": 1.2},
    )

    # 【机身最低对地高度终止 (高度安全锁)】：
    # 【物理机理】：机身高度低于 0.22m 时直接终止。彻底粉碎策略“趴在地上蹭地前进”的平凡欺骗解。
    low_base_height = DoneTerm(
        func=mdp.low_base_height,
        params={"minimum_height": 0.22},
    )


# ==============================================================================
# 8. 课程学习自适应配置 (Curriculum Specifications)
# ==============================================================================
@configclass
class CurriculumCfg:
    """DreamWaQ 地形难度与速度范围双重课程学习。"""

    # 【地形等级晋级课程 (Terrain Curriculum)】：
    # 【晋升判定】：当机器人速度跟踪进度超过 90% (0.9) 且指令速度大于 0.15m/s 时，允许晋级到更高难度地形网格。
    terrain_levels = CurrTerm(
        func=mdp.terrain_levels_vel_new,
        params={
            "command_name": "base_velocity",
            "min_command_speed": 0.15,
            "min_command_progress_for_terrain": 0.9,
        },
    )

    # 【指令速度范围自适应扩充课程 (Command Range Expansion)】：
    # 【扩展判定】：当跟踪奖励超过 0.8 时，逐步将采样范围扩充 step_size (0.01)，直到达到极限上限 limit_ranges。
    command_ranges = CurrTerm(
        func=mdp.modify_command_ranges_on_tracking_reward,
        params={
            "term_name": "base_velocity",
            "step_size": 0.01,
            "reward_threshold": 0.8,
            "min_command_speed": 0.15,
            "sample_window": 1024,
            "lin_vel_reward_term_name": "track_lin_vel_xy_exp",
            "ang_vel_reward_term_name": "track_ang_vel_z_exp",
            "include_ang_vel": True,
        },
    )


# ==============================================================================
# 9. 顶级环境基类组装 (Base Environment Configuration Class)
# ==============================================================================
@configclass
class UnitreeGo2DreamWaQBaseEnvCfg(ManagerBasedRLEnvCfg):
    """Go2 DreamWaQ 顶级环境基类。

    【时间离散化与频率】：
    - 物理引擎仿真步长 (sim.dt): 0.005s (200Hz 物理求解器)
    - 策略抽取抽取率 (decimation): 4
    - 强化学习控制步长: 4 * 0.005s = 0.02s (50Hz 控制环，与实机 CAN 总线主流刷新率严格匹配)
    """

    scene: UnitreeGo2DreamWaQSceneCfg = UnitreeGo2DreamWaQSceneCfg(
        num_envs=4096,
        env_spacing=8.0,
    )
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 20.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15

        # 传感器采样周期对齐
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = self.decimation * self.sim.dt  # 50Hz 更新
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt                    # 200Hz 触地监测

        # 课程生成器联动
        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True
        else:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = False


__all__ = [
    "BASE_HEIGHT_TARGET",
    "FOOT_BODY_NAMES",
    "GO2_ACTION_JOINT_NAMES",
    "GO2_MOTOR_STRENGTH_RANGE",
    "ActionsCfg",
    "CommandsCfg",
    "CurriculumCfg",
    "EventCfg",
    "ObservationsCfg",
    "RewardsCfg",
    "TerminationsCfg",
    "UnitreeGo2DreamWaQBaseEnvCfg",
    "UnitreeGo2DreamWaQSceneCfg",
]
