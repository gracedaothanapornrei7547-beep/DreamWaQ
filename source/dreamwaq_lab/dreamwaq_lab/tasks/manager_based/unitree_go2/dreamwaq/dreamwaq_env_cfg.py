# Copyright (c) 2024 LunarQuadruped Project.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""宇树 Go2 四足机器人 DreamWaQ 复杂地形（Rough-Terrain）任务环境配置。

【配置架构与定位】
1. 继承自 UnitreeGo2DreamWaQBaseEnvCfg，专职负责多地形训练集（Curriculum Sub-terrains）、
   域随机化（Domain Randomization / Events）以及运动性能奖励函数权重（Reward Weights）的特化调优。
2. DreamWaQ 核心思想：通过 VAE / CENet 在无外部视觉时隐式重构环境与外界扰动，因此训练环境需提供
   高挑战性的地形与物理扰动，迫使策略建立对地形高程差和外力扰动的内在动力学鲁棒表征。
"""

from isaaclab.terrains import MeshPlaneTerrainCfg
from isaaclab.utils import configclass

from dreamwaq_lab.terrains import MeshStraightAscendingStairsTerrainCfg, apply_play_terrain

from .go2_env_cfg import BASE_HEIGHT_TARGET, UnitreeGo2DreamWaQBaseEnvCfg
from .. import mdp


@configclass
class UnitreeGo2DreamWaQRoughEnvCfg(UnitreeGo2DreamWaQBaseEnvCfg):
    """宇树 Go2 复杂地形训练配置（针对台阶、斜坡、非结构化崎岖路面的全地形强化学习）。"""

    def __post_init__(self):
        super().__post_init__()

        # ==============================================================================
        # 1. 多地形分布与难度范围配置 (Terrain Distribution & Geometric Ranges)
        # ==============================================================================
        # 【机制说明】：
        # - 所有地形由 Isaac Lab TerrainGenerator 按 proportion 比例混合生成网格矩阵（Mesh）。
        # - 课程机制（Curriculum）会在各地形的最小与最大几何参数之间按等级（Level 0~9）进行线性插值。
        # ------------------------------------------------------------------------------

        # --- 地形类型比例 (Sum 应当等于 1.0) ---
        # 【正向金字塔台阶 (上楼梯)】：强化抬腿高度与前向冲量
        # 【调参指南】：若爬楼梯能力偏弱可增至 0.35；过高会导致机器狗在平地上过度踏步
        self.scene.terrain.terrain_generator.sub_terrains["pyramid_stairs"].proportion = 0.25

        # 【倒置金字塔台阶 (下楼梯)】：强化后倾重力支撑与膝关节防挫屈能力
        # 【调参指南】：下楼梯容易发生俯冲倒地（Pitch倾覆），调参时可与上楼梯维持 1:1 对等比例
        self.scene.terrain.terrain_generator.sub_terrains["pyramid_stairs_inv"].proportion = 0.25

        # 【离散方块障碍 (Discrete Boxes)】：模拟碎石、路面凸起，考验离散步态适应
        # 【调参指南】：保持 0.10 即可；比例过高易导致机器狗学出停滞防滑的小碎步
        self.scene.terrain.terrain_generator.sub_terrains["boxes"].proportion = 0.10

        # 【随机连续粗糙路面 (Random Rough)】：连续微小起伏，压制高频足端冲击
        # 【调参指南】：提供高频触地阻尼感知，推荐 0.15~0.25
        self.scene.terrain.terrain_generator.sub_terrains["random_rough"].proportion = 0.20

        # 【金字塔上斜坡 (Slope Up)】：测试持续爬坡力矩与机身俯仰重心自适应
        # 【调参指南】：0.10 足够覆盖坡面倾斜感知
        self.scene.terrain.terrain_generator.sub_terrains["hf_pyramid_slope"].proportion = 0.10

        # 【倒置金字塔下斜坡 (Slope Down)】：测试坡面下行制动稳定性
        # 【调参指南】：与上坡保持 1:1，维持纵向受力对称性
        self.scene.terrain.terrain_generator.sub_terrains["hf_pyramid_slope_inv"].proportion = 0.10

        # --- 地形几何难度范围 (Level 0 对应下限，Level 9 对应上限) ---
        # 【金字塔台阶高度】：(最小 0.03m ~ 最大 0.25m)
        # 【物理机理】：Go2 小腿有效臂长约 0.21m，台阶超过 0.18m 即进入极限攀爬工况。
        # 【调参经验】：若初期训练收敛缓慢，可将上限保守设为 0.18m；若要冲击极限台阶再放宽至 0.25m。
        self.scene.terrain.terrain_generator.sub_terrains["pyramid_stairs"].step_height_range = (0.03, 0.25)
        self.scene.terrain.terrain_generator.sub_terrains["pyramid_stairs_inv"].step_height_range = (0.03, 0.25)

        # 【离散方块障碍高度】：(最小 0.02m ~ 最大 0.06m)
        # 【调参经验】：大于 0.08m 易直接卡住足端球体，引发大腿撞击惩罚，维持在 2~6cm 为宜。
        self.scene.terrain.terrain_generator.sub_terrains["boxes"].grid_height_range = (0.02, 0.06)

        # 【随机粗糙地形噪声范围与离散步长】：
        # 【物理机理】：noise_range 决定凹凸波幅，noise_step 决定离散台阶颗粒度。
        # 【调参经验】：保持 2~6cm 扰动即可迫使网络自适应调整四足虚拟阻抗。
        self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_range = (0.02, 0.06)
        self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_step = 0.01

        # 【斜坡坡度范围】：(0.0 ~ 0.35 rad，即 0° ~ 20.0°)
        # 【调参经验】：Go2 橡胶脚垫摩擦极限下最大爬坡约 25°，0.35 rad（约20°）兼顾安全性与鲁棒性。
        self.scene.terrain.terrain_generator.sub_terrains["hf_pyramid_slope"].slope_range = (0.0, 0.35)
        self.scene.terrain.terrain_generator.sub_terrains["hf_pyramid_slope_inv"].slope_range = (0.0, 0.35)


        # ==============================================================================
        # 2. 域随机化事件配置 (Domain Randomization / Events)
        # ==============================================================================
        # 【机制说明】：
        # - 在仿真中主动注入传感器噪声、动力学参数扰动与外力冲击，以消除 Sim-to-Real 的现实差距（Reality Gap）。
        # ------------------------------------------------------------------------------

        # 【地面动/静摩擦因数随机化】：(0.4 ~ 1.25)
        # 【物理机理】：涵盖从光滑湿瓷砖（0.4）到高抓地力橡胶跑道（1.25）的真实摩擦力范围。
        # 【调参避坑】：下限不宜低于 0.2（会导致策略过早放弃前进转为趴地防滑）；上限不宜超过 1.5。
        self.events.physics_material.params["static_friction_range"] = (0.4, 1.25)
        self.events.physics_material.params["dynamic_friction_range"] = (0.4, 1.25)

        # 【机身附加质量随机化】：(-0.5kg ~ +2.0kg)
        # 【物理机理】：Go2 整机空载约 15kg，此范围模拟电池压降重量差、外部背负负载（计算模块/雷达等）。
        # 【调参避坑】：若要背负机械臂等重物，可调大至 (+1.0, +5.0kg) 并配合调大电机驱动增益。
        self.events.add_base_mass.params["mass_distribution_params"] = (-0.5, 2.0)

        # 【机身质心位置偏移随机化 (CoM Offset)】：X轴 ±5cm, Y轴 ±2cm, Z轴 ±2cm
        # 【物理机理】：打破物理完美对称性，强迫策略学会抵抗重心偏移带来的非对称倾覆力矩。
        # 【调参经验】：X 轴（前后）容忍度较大；Y 轴（左右）若给得过大（如 >0.05m）易导致机器狗单侧瘫痪。
        self.events.base_com.params["com_range"] = {
            "x": (-0.05, 0.05),
            "y": (-0.02, 0.02),
            "z": (-0.02, 0.02),
        }

        # 【重置时持续外力与外力矩】：常态重置时保持 0，突发扰动由 push_robot 事件负责。
        self.events.base_external_force_torque.params["force_range"] = (0.0, 0.0)
        self.events.base_external_force_torque.params["torque_range"] = (0.0, 0.0)

        # 【初始机身位姿与速度重置】：
        # 【物理机理】：位置轻微抖动（X/Y ±0.5m）防止记忆固定台阶位置；全向偏航角（Yaw ±π）迫使网络学会全方位调头。
        self.events.reset_base.params = {
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }

        # 【初始关节角度缩放扰动】：(0.8 ~ 1.2)
        # 【物理机理】：每次重生时不以绝对标称角度站立，增强非标准初始姿态下的自救起身能力。
        self.events.reset_robot_joints.params["position_range"] = (0.8, 1.2)
        self.events.reset_robot_joints.params["velocity_range"] = (0.0, 0.0)

        # 【电机驱动刚度 (Kp) 与阻尼 (Kd) 缩放随机化】：(0.9 ~ 1.1)
        # 【物理机理】：模拟电机温升、母线电压波动与各关节执行器响应特性的个体差异（±10% 不确定性）。
        self.events.robot_joint_stiffness_and_damping.params["stiffness_distribution_params"] = (0.9, 1.1)
        self.events.robot_joint_stiffness_and_damping.params["damping_distribution_params"] = (0.9, 1.1)

        # 【运行中随机速度脉冲冲击 (Push Robot)】：X轴 ±0.3m/s, Y轴 ±0.1m/s
        # 【物理机理】：模拟行进中受到侧向踢击或踩空突发冲量，激发快速侧跨步（Cross-step recovery）平衡反射。
        self.events.push_robot.params["velocity_range"] = {"x": (-0.3, 0.3), "y": (-0.1, 0.1)}


        # ==============================================================================
        # 2.5 运动指令采样配置 (对齐 WaxberryS 标杆，全向全速对称采样)
        # ==============================================================================
        # 【物理机理】：全向速度指令 (-0.8~1.0, -0.8~0.8, -1.0~1.0) 迫使机器狗以相同概率承受左/右转、
        # 左/右侧行压力，任何单侧发力或代偿偏向都会在反向指令下遭受巨大惩罚，天然从物理机制层面逼出对称 Trot！
        self.curriculum.command_ranges = None
        self.commands.base_velocity.ranges.lin_vel_x = (-0.8, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.8, 0.8)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        self.commands.base_velocity.rel_standing_envs = 0.05

        # ==============================================================================
        # 3. 强化学习奖励函数权重配置 (对标 WaxberryS 黄金标准)
        # ==============================================================================
        # --- 核心任务目标：指令速度跟踪 (对标 WaxberryS: 1.0 / 0.5) ---
        self.rewards.track_lin_vel_xy_exp.weight = 1.0
        self.rewards.track_ang_vel_z_exp.weight = 0.5

        # --- 姿态与振荡抑制 (抗晃动核心防线) ---
        self.rewards.lin_vel_z_l2.weight = -2.0
        self.rewards.ang_vel_xy_l2.weight = -0.05
        self.rewards.flat_orientation_l2.weight = -0.20

        # --- 执行器平滑性与硬件寿命保护 ---
        self.rewards.dof_acc_l2.weight = -2.5e-7
        self.rewards.joint_power_l1.weight = -2.0e-5

        # --- 机身相对高度 (对标 0.32m 黄金站高) ---
        self.rewards.base_height_relative_l2.weight = -1.0
        self.rewards.base_height_relative_l2.params["target_height"] = 0.32

        # --- 足端清除度与地形交互 (对齐 WaxberryS: 降回 -0.01，根治样本一暴躁砸腿) ---
        self.rewards.terrain_relative_foot_clearance_l1.weight = -0.01
        self.rewards.terrain_relative_foot_clearance_l1.params["target_clearance"] = 0.06
        self.rewards.paper_foot_clearance_l2 = None

        # --- 动作平滑度 (Sim-to-Real 移植基石) ---
        self.rewards.action_rate_l2.weight = -0.01
        self.rewards.action_smoothness_l2.weight = -0.01

        # --- 关节软限位超限惩罚 ---
        self.rewards.dof_pos_limits.weight = -1.0

        # --- 静止指令时关节标称位姿锁定惩罚 ---
        self.rewards.stand_still_joint_deviation_l1.weight = -0.20

        # --- 非预期接触惩罚 (撞击防护) ---
        self.rewards.undesired_contacts.weight = -0.20
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = [
            ".*thigh",
            ".*calf",
        ]

        # --- 核心恢复：WaxberryS 髋关节姿态锁 (hip_pos = -0.1，根除八字脚外展与偏航不对称) ---
        self.rewards.hip_pos_deviation_l1.weight = -0.10

        # --- 彻底拆除带有数学漏洞的手工惩罚项与多余项 ---
        self.rewards.joint_diagonal_symmetry_l2.weight = 0.0
        self.rewards.feet_air_time_variance.weight = 0.0
        self.rewards.feet_regulation = None
        self.rewards.lin_vel_xy_orthogonal_l2.weight = 0.0
        self.rewards.dof_torques_l2 = None          # 已由 joint_power 覆盖
        self.rewards.foot_clearance_reward = None   # 使用 terrain_relative_foot_clearance 代替
        self.rewards.feet_air_time = None           # 避免与台阶攀爬的动态步频相冲突
        self.rewards.dof_vel_l2 = None              # 避免过度压制正常摆动腿角速度
        self.rewards.feet_slide = None              # 避免在爬高台阶摩擦抓地时产生误惩罚
        self.rewards.foot_swing_height_l2 = None    # 避免双重足端高程目标混淆
        self.rewards.feet_air_time_limit = None     # 避免限制大步幅跨越台阶的腾空时间


# ==============================================================================
# 4. 推演与验证模式配置 (Evaluation / Play Configuration)
# ==============================================================================
@configclass
class UnitreeGo2DreamWaQRoughEnvCfg_PLAY(UnitreeGo2DreamWaQRoughEnvCfg):
    """Play / 评估专用配置类。

    【核心特性】：
    1. 地形固定：采用标准 4 地形切片（平地、台阶、斜坡、离散坑洼），固定随机种子，保证基准评测绝对确定可复现。
    2. 传感器纯净：关闭所有观测噪声与随机干扰（enable_corruption = False），纯净验证策略部署上限。
    3. 环境与相机：默认单机/百机评测，关闭随机外力与扰动推力。
    """

    terrain_profile: str = "fixed_four"
    terrain_difficulty: float = 0.6
    terrain_seed: int | None = 1

    def __post_init__(self):
        super().__post_init__()

        # 评测环境数与间距
        self.scene.num_envs = 100
        self.scene.env_spacing = 8.0
        self.scene.height_scanner.debug_vis = True
        self.viewer.origin_type = "world"

        # 加载固定的评测地形网格
        apply_play_terrain(
            self,
            profile_name=self.terrain_profile,
            difficulty=self.terrain_difficulty,
            seed=self.terrain_seed,
        )

        # 关闭课程自适应，固定指令测试范围
        self.curriculum.command_ranges = None
        self.commands.base_velocity.rel_standing_envs = 0.02
        self.commands.base_velocity.ranges.lin_vel_x = (0.6, 1.5)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.25, 0.25)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.55, 0.55)

        # 禁用观测噪声与信号失真，评测确定性决策表现
        self.observations.dreamwaq_policy.enable_corruption = False
        self.observations.dreamwaq_policy_history.enable_corruption = False
        self.observations.dreamwaq_reconstruction_target.enable_corruption = False
        self.observations.dreamwaq_critic.enable_corruption = False

        # 禁用持续外力推搡
        self.events.base_external_force_torque = None


__all__ = [
    "UnitreeGo2DreamWaQRoughEnvCfg",
    "UnitreeGo2DreamWaQRoughEnvCfg_PLAY",
    "UnitreeGo2DreamWaQFlatEnvCfg",
    "UnitreeGo2DreamWaQFlatEnvCfg_PLAY",
    "UnitreeGo2DreamWaQStairsEnvCfg",
    "UnitreeGo2DreamWaQStairsEnvCfg_PLAY",
    "UnitreeGo2DreamWaQMixedEnvCfg",
    "UnitreeGo2DreamWaQMixedEnvCfg_PLAY",
]


# ==============================================================================
# 5. 纯平地冷启动步态定型配置 (Flat Ground Training Configuration)
# ==============================================================================
@configclass
class UnitreeGo2DreamWaQFlatEnvCfg(UnitreeGo2DreamWaQRoughEnvCfg):
    """宇树 Go2 纯平地冷启动步态定型与 CENet 速度感知校准配置。

    【纯粹学术路线定位与设计思想】：
    1. 地形彻底清零为 plane 纯平地，排除任何台阶、坑洼对速度回归和步态学习的非线性几何扰动；
    2. 拆除所有生硬物理死锁（禁用正交侧漂锁、髋关节锁、滞空方差锁，恢复原版 flat_orientation=-0.2 与 ang_vel_xy=-0.05）；
    3. 允许动力学自然舒展小跑，重心自由弹性起伏，彻底消除过度刚性约束诱发的原地踏步与单侧代偿歪斜；
    4. 仅保留极柔和对角肢体对称软正则（joint_diagonal_symmetry_l2=-0.02），防止左右肢体长短腿畸变退化；
    5. 依靠 CENet 速度回归损失（velocity_loss_weight=3.0）提供纯净准确的速度感知，从源头消灭幽灵速度漂移。
    """

    def __post_init__(self):
        super().__post_init__()

        # --- 1. 彻底切换为纯平地，关闭地形网格生成器与地形课程 ---
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.curriculum.terrain_levels = None
        self.curriculum.command_ranges = None

        # --- 2. 平地全向平移 (XY移动且Yaw固定) + 原地自转混合双轨指令域 ---
        self.commands.base_velocity.heading_command = True
        self.commands.base_velocity.heading_control_stiffness = 1.0
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)
        self.commands.base_velocity.rel_heading_envs = 0.50          # 50% 环境强制固定航向 0.0 (给xy走yaw不变)；50% 环境自由采样自转角速度
        self.commands.base_velocity.ranges.lin_vel_x = (-0.6, 1.0)   # 覆盖倒退与高速冲刺
        self.commands.base_velocity.ranges.lin_vel_y = (-0.5, 0.5)   # 覆盖全向横向滑行 (螃蟹步)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)   # 覆盖高速原地自转 (快速旋转)
        self.commands.base_velocity.rel_standing_envs = 0.02
        self.events.reset_base.params["pose_range"]["yaw"] = (0.0, 0.0)

        # --- 3. 极致悬浮平顺滑行姿态 (压制颠簸、起伏与俯仰) ---
        self.rewards.track_lin_vel_xy_exp.weight = 1.5
        self.rewards.track_ang_vel_z_exp.weight = 1.0
        self.rewards.lin_vel_z_l2.weight = -3.0
        self.rewards.ang_vel_xy_l2.weight = -0.10
        self.rewards.flat_orientation_l2.weight = -0.40
        self.rewards.base_height_relative_l2.weight = -2.0
        self.rewards.base_height_relative_l2.params["target_height"] = 0.30
        self.rewards.stand_still_joint_deviation_l1.weight = -0.20
        self.rewards.terrain_relative_foot_clearance_l1.weight = -0.01

        # --- 4. 训练端 Yaw 自抗扰与动态门控 (全向平移时航向死锁，自转时门控清零放开自由度) ---
        self.rewards.flat_heading_error_l2.weight = -0.50
        self.rewards.lin_vel_xy_orthogonal_l2.weight = -0.15

        # --- 5. 动态门控外展与对称 (直行时纯矢状面迈步，横移与自转时自适应释放) ---
        self.rewards.hip_pos_deviation_l1.weight = -0.10
        self.rewards.swing_hip_abduction_l2.weight = -0.50
        self.rewards.rear_hip_swing_symmetry_l2.weight = -0.20
        self.rewards.joint_diagonal_symmetry_l2.weight = -0.05
        self.rewards.feet_air_time_variance.weight = 0.0

        # --- 6. 动作平滑与硬件保护 (原版标准) ---
        self.rewards.action_rate_l2.weight = -0.01
        self.rewards.action_smoothness_l2.weight = -0.01
        self.rewards.dof_acc_l2.weight = -2.5e-7
        self.rewards.joint_power_l1.weight = -2.0e-5




@configclass
class UnitreeGo2DreamWaQFlatEnvCfg_PLAY(UnitreeGo2DreamWaQFlatEnvCfg):
    """纯平地评估与可视化配置。"""

    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 1
        self.scene.env_spacing = 8.0
        self.scene.height_scanner.debug_vis = False
        self.viewer.origin_type = "world"

        self.commands.base_velocity.ranges.lin_vel_x = (0.8, 0.8)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.rel_standing_envs = 0.0

        self.observations.dreamwaq_policy.enable_corruption = False
        self.observations.dreamwaq_policy_history.enable_corruption = False
        self.observations.dreamwaq_reconstruction_target.enable_corruption = False
        self.observations.dreamwaq_critic.enable_corruption = False
        self.events.base_external_force_torque = None


# ==============================================================================
# 6. 第二阶段：单向直线阶梯长廊自适应课程攻坚专项微调 (Straight Ascending Stairs Corridor)
# ==============================================================================
@configclass
class UnitreeGo2DreamWaQStairsEnvCfg(UnitreeGo2DreamWaQRoughEnvCfg):
    """宇树 Go2 两阶段微调第二步：单向直线阶梯长廊自适应课程攻坚专项微调。

    【核心特性与动力学设计】：
    1. 彻底淘汰环形金字塔，定制 12m×4m 纯单向正向爬升阶梯长廊（起跑平地 2.5m + 23 级正向阶梯 7m + 登顶平地 2.5m）；
    2. 几何上物理杜绝下楼梯与下坠失重边缘，彻底粉碎“摔跤翻滚骗数据”的投机回路；
    3. 动态自适应课程学习 (Curriculum)：从 3cm 微阶平滑起步，随存活率与速度跟踪动态攀升至 15cm 极限阶高；
    4. 严格法向对齐 (Yaw in [-0.05, 0.05]) + 纯前向指令硬锁 (vx in [0.45, 0.75], vy=0, wz=0)；
    5. 正交侧漂强镇压 (-0.15) + 髋关节 0° 锁 (-0.15) + 对角对称软正则 (-0.02) + 抬腿净空 (0.09m, -0.03)。
    """

    def __post_init__(self):
        super().__post_init__()

        # --- 1. 定制单向单道直线阶梯长廊 (12.0m x 4.0m) ---
        self.scene.terrain.terrain_generator.size = (12.0, 4.0)
        self.scene.terrain.terrain_generator.border_width = 5.0
        self.scene.terrain.terrain_generator.curriculum = True
        self.scene.terrain.max_init_terrain_level = 0  # Stage 2 起步：从 Level 0 (3cm 微阶) 展开，随自适应课程平滑晋级至 15cm
        self.scene.terrain.terrain_generator.sub_terrains = {
            "straight_stairs": MeshStraightAscendingStairsTerrainCfg(
                proportion=1.0,
                step_height_range=(0.03, 0.15),
                step_width=0.3,
                platform_start=2.5,
                platform_end=2.5,
            )
        }

        # --- 2. 刷新重生严格正对台阶法向 (Orthogonal Spawn Lock) ---
        # 初始误差严格限制在 ±2.8° 内，出生在平地起跑区
        self.events.reset_base.params["pose_range"]["yaw"] = (-0.05, 0.05)
        self.events.reset_base.params["pose_range"]["x"] = (-0.1, 0.1)
        self.events.reset_base.params["pose_range"]["y"] = (-0.1, 0.1)

        # --- 3. WaxberryS 正向爬阶指令：正向攻坚长廊，航向锁紧 ---
        self.commands.base_velocity.heading_command = True
        self.commands.base_velocity.heading_control_stiffness = 1.0
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)
        self.commands.base_velocity.rel_heading_envs = 1.0
        self.commands.base_velocity.ranges.lin_vel_x = (0.45, 0.75)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.rel_standing_envs = 0.0

        # --- 4. 彻底解耦死锁：纯物理翻阶距离自适应课程 (Waxberry 正统规范) ---
        self.curriculum.terrain_levels.func = mdp.terrain_levels_vel
        self.curriculum.terrain_levels.params = {}
        self.curriculum.command_ranges = None

        # --- 5. 动力学奖励微调 (拆除姿态锁红线 W4，强化跨阶净空) ---
        self.rewards.track_lin_vel_xy_exp.weight = 1.0
        self.rewards.track_ang_vel_z_exp.weight = 0.50

        # 爬楼梯必然有向上正向 vz 与自然俯仰，放宽垂直速度阻尼并彻底清零水平姿态锁 (红线 W4)
        self.rewards.lin_vel_z_l2.weight = -0.10
        self.rewards.ang_vel_xy_l2.weight = -0.05
        self.rewards.flat_orientation_l2.weight = 0.0
        self.rewards.base_height_relative_l2.weight = -0.20

        # 核心防漂移与对称性继承 (阶段一成果注入加码)
        self.rewards.lin_vel_xy_orthogonal_l2.weight = -0.15
        self.rewards.hip_pos_deviation_l1.weight = -0.05
        self.rewards.joint_diagonal_symmetry_l2.weight = -0.02

        # 摆动相专属髋关节外展画弧惩罚与四肢全对称镜像约束 (带全向与自转动态门控)
        self.rewards.swing_hip_abduction_l2.weight = -0.50
        self.rewards.rear_hip_swing_symmetry_l2.weight = -0.15

        # 核心防撞阶：足端清除度激励 (目标净空提升至 0.16m 覆盖 15cm 阶高，权重加强至 -0.15)
        self.rewards.terrain_relative_foot_clearance_l1.weight = -0.15
        self.rewards.terrain_relative_foot_clearance_l1.params["target_clearance"] = 0.16

        # 软限位与非预期碰撞防护 (小腿杆磕阶碰撞适度约束 -0.20，引导抬腿同时防畏缩)
        self.rewards.dof_pos_limits.weight = -1.0
        self.rewards.undesired_contacts.weight = -0.20
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = [
            ".*thigh",
            ".*calf",
        ]

        # --- 6. 台阶截断优化 (彻底移除射线误判的 low_base_height) ---
        self.terminations.low_base_height = None
        self.terminations.base_contact.params["threshold"] = 25.0


@configclass
class UnitreeGo2DreamWaQStairsEnvCfg_PLAY(UnitreeGo2DreamWaQStairsEnvCfg):
    """单向直线阶梯长廊评估与可视化配置。"""

    terrain_difficulty: float = 1.0  # 1.0 对应 15cm 极限阶高验收

    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 4
        self.scene.env_spacing = 6.0
        self.scene.height_scanner.debug_vis = True
        self.viewer.origin_type = "world"

        # PLAY 评估直接固定在指定难度 (例如 1.0 为 15cm 阶高)
        self.scene.terrain.terrain_generator.size = (12.0, 4.0)
        self.scene.terrain.terrain_generator.num_rows = 1
        self.scene.terrain.terrain_generator.num_cols = 4
        self.scene.terrain.terrain_generator.curriculum = True
        self.scene.terrain.terrain_generator.difficulty_range = (self.terrain_difficulty, self.terrain_difficulty)
        self.scene.terrain.terrain_generator.sub_terrains = {
            "straight_stairs": MeshStraightAscendingStairsTerrainCfg(
                proportion=1.0,
                step_height_range=(0.03, 0.15),
                step_width=0.3,
                platform_start=2.5,
                platform_end=2.5,
            )
        }

        self.curriculum.command_ranges = None
        self.curriculum.terrain_levels = None
        self.commands.base_velocity.heading_command = True
        self.commands.base_velocity.heading_control_stiffness = 1.0
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)
        self.commands.base_velocity.rel_heading_envs = 1.0
        self.commands.base_velocity.rel_standing_envs = 0.0
        self.commands.base_velocity.ranges.lin_vel_x = (0.6, 0.6)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)

        self.observations.dreamwaq_policy.enable_corruption = False
        self.observations.dreamwaq_policy_history.enable_corruption = False
        self.observations.dreamwaq_reconstruction_target.enable_corruption = False
        self.observations.dreamwaq_critic.enable_corruption = False
        self.events.base_external_force_torque = None


# ==============================================================================
# 7. 第三阶段：平地+楼梯混合长廊多任务融合微调 (Stage 3 Mixed Corridor Fusion Fine-Tuning)
# ==============================================================================
@configclass
class UnitreeGo2DreamWaQMixedEnvCfg(UnitreeGo2DreamWaQStairsEnvCfg):
    """宇树 Go2 两阶段断层弥合微调：40% 平地长廊 + 60% 单向阶梯长廊联合多任务攻坚。

    【核心特性与动力学设计】：
    1. 混合长廊地形：40% 纯平地长廊 (MeshPlaneTerrainCfg) + 60% 单向阶梯长廊 (MeshStraightAscendingStairsTerrainCfg)；
    2. CENet 隐空间去混淆：迫使变分自编码器 16 维潜变量 z 同时经历平地光滑流形与台阶接触阻抗，消灭潜变量混淆与平地高频打颤；
    3. 俯仰自适应单边净空 (Pitch-Adaptive Clearance)：
       - 平地 (|pitch| < 3°): 净空门限自动回缩至 0.04m，步态轻盈紧凑，根治高抬腿猛跺地诱发的地面反力冲击；
       - 台阶 (pitch > 6°): 自适应放大至 0.16m，强力引导深屈膝高提膝翻越 15cm 大台阶；
    4. 姿态精准解耦 (roll_orientation_l2 = -0.25)：锁死机身左右横滚侧倾，彻底释放 Pitch 自由度顺应 15°~25° 楼梯坡度；
    5. 平衡垂直速度阻尼 (lin_vel_z_l2 = -0.30)：平地抑制起伏颠簸，爬楼梯保障向上攀爬动能；
    6. 继承对角 Trot 对称软正则 (-0.02) 与摆动相髋关节外展画弧惩罚 (-0.50)，步态优美端正。
    """

    def __post_init__(self):
        super().__post_init__()

        # --- 1. 定制混合长廊地形网格 (40% 平地 + 60% 阶梯) ---
        self.scene.terrain.terrain_generator.size = (12.0, 4.0)
        self.scene.terrain.terrain_generator.border_width = 5.0
        self.scene.terrain.terrain_generator.curriculum = True
        self.scene.terrain.max_init_terrain_level = 0  # 随自适应课程平滑晋级
        self.scene.terrain.terrain_generator.sub_terrains = {
            "flat_corridor": MeshPlaneTerrainCfg(proportion=0.40),
            "straight_stairs": MeshStraightAscendingStairsTerrainCfg(
                proportion=0.60,
                step_height_range=(0.03, 0.15),
                step_width=0.3,
                platform_start=2.5,
                platform_end=2.5,
            ),
        }

        # --- 2. 刷新出生正对长廊法向 ---
        self.events.reset_base.params["pose_range"]["yaw"] = (-0.05, 0.05)
        self.events.reset_base.params["pose_range"]["x"] = (-0.1, 0.1)
        self.events.reset_base.params["pose_range"]["y"] = (-0.1, 0.1)

        # --- 3. 全域指令域覆盖：平地巡航与爬阶冲顶一体化 ---
        self.commands.base_velocity.heading_command = True
        self.commands.base_velocity.heading_control_stiffness = 1.0
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)
        self.commands.base_velocity.rel_heading_envs = 1.0
        self.commands.base_velocity.ranges.lin_vel_x = (0.45, 0.85)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.rel_standing_envs = 0.0

        # --- 4. 动力学奖励重构 (平地悬浮滑行 + 楼梯霸气翻越) ---
        self.rewards.track_lin_vel_xy_exp.weight = 1.2
        self.rewards.track_ang_vel_z_exp.weight = 0.50

        # 姿态精准解耦：横滚刚性锁定，俯仰彻底释放
        self.rewards.flat_orientation_l2.weight = 0.0
        self.rewards.roll_orientation_l2.weight = -0.25

        # 均衡阻尼与恒高
        self.rewards.lin_vel_z_l2.weight = -0.30
        self.rewards.ang_vel_xy_l2.weight = -0.05
        self.rewards.base_height_relative_l2.weight = -0.25
        self.rewards.base_height_relative_l2.params["target_height"] = 0.30

        # 核心防侧漂与对称性 (继承黄金成果)
        self.rewards.lin_vel_xy_orthogonal_l2.weight = -0.15
        self.rewards.hip_pos_deviation_l1.weight = -0.05
        self.rewards.joint_diagonal_symmetry_l2.weight = -0.02
        self.rewards.swing_hip_abduction_l2.weight = -0.50
        self.rewards.rear_hip_swing_symmetry_l2.weight = -0.15

        # 核心：俯仰自适应单边足端净空
        self.rewards.terrain_relative_foot_clearance_l1.func = mdp.pitch_adaptive_foot_clearance_l1
        self.rewards.terrain_relative_foot_clearance_l1.weight = -0.15
        self.rewards.terrain_relative_foot_clearance_l1.params["flat_target_clearance"] = 0.04
        self.rewards.terrain_relative_foot_clearance_l1.params["stairs_target_clearance"] = 0.16

        # 硬件碰撞与截断保护
        self.rewards.dof_pos_limits.weight = -1.0
        self.rewards.undesired_contacts.weight = -0.20
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = [
            ".*thigh",
            ".*calf",
        ]

        self.terminations.low_base_height = None
        self.terminations.base_contact.params["threshold"] = 25.0


@configclass
class UnitreeGo2DreamWaQMixedEnvCfg_PLAY(UnitreeGo2DreamWaQMixedEnvCfg):
    """平地与台阶混合长廊评估与可视化配置。"""

    terrain_difficulty: float = 1.0

    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 4
        self.scene.env_spacing = 6.0
        self.scene.height_scanner.debug_vis = True
        self.viewer.origin_type = "world"

        self.scene.terrain.terrain_generator.size = (12.0, 4.0)
        self.scene.terrain.terrain_generator.num_rows = 1
        self.scene.terrain.terrain_generator.num_cols = 4
        self.scene.terrain.terrain_generator.curriculum = True
        self.scene.terrain.terrain_generator.difficulty_range = (self.terrain_difficulty, self.terrain_difficulty)
        self.scene.terrain.terrain_generator.sub_terrains = {
            "flat_corridor": MeshPlaneTerrainCfg(proportion=0.40),
            "straight_stairs": MeshStraightAscendingStairsTerrainCfg(
                proportion=0.60,
                step_height_range=(0.03, 0.15),
                step_width=0.3,
                platform_start=2.5,
                platform_end=2.5,
            ),
        }

        self.curriculum.command_ranges = None
        self.curriculum.terrain_levels = None
        self.commands.base_velocity.heading_command = True
        self.commands.base_velocity.heading_control_stiffness = 1.0
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)
        self.commands.base_velocity.rel_heading_envs = 1.0
        self.commands.base_velocity.rel_standing_envs = 0.0
        self.commands.base_velocity.ranges.lin_vel_x = (0.6, 0.6)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)

        self.observations.dreamwaq_policy.enable_corruption = False
        self.observations.dreamwaq_policy_history.enable_corruption = False
        self.observations.dreamwaq_reconstruction_target.enable_corruption = False
        self.observations.dreamwaq_critic.enable_corruption = False
        self.events.base_external_force_torque = None


@configclass
class UnitreeGo2DreamWaQSoftLandingEnvCfg(UnitreeGo2DreamWaQMixedEnvCfg):
    """Stage 4: 注入足端软着陆阻尼与接触力缓冲微调配置。

    目标：在保持 100% 翻越 5 级台阶的基础上，将平地机身垂直抖动彻底压进 < 10.0mm 黄金线以内。

    核心调优项：
    1. 注入足端垂直撞击软着陆阻尼：feet_contact_velocity_z (weight = -0.06)
    2. 注入足端超额接地力缓冲：feet_contact_force_excess (weight = -0.0001, max_force = 85.0)
    3. 注入动作二阶差分平滑：action_smoothness_2nd (weight = -0.005)
    4. 强化机身垂直速度阻尼：lin_vel_z_l2 从 -0.30 提升至 -0.75 (平地强力压平颠簸)
    5. 加固横滚姿态刚性：roll_orientation_l2 从 -0.25 加固至 -0.40 (压制侧倾摆幅)
    6. 收紧平地目标净空：flat_target_clearance 从 0.04m 收紧至 0.025m (台阶依然维持 0.16m 暴力提膝)
    """

    def __post_init__(self):
        super().__post_init__()

        # --- 核心强化项 1: 足端软着陆阻尼 (消除平地跺地冲击与力矩尖峰) ---
        self.rewards.feet_contact_velocity_z.weight = -0.06
        self.rewards.feet_contact_force_excess.weight = -0.0001
        self.rewards.action_smoothness_2nd.weight = -0.005

        # --- 核心强化项 2: 机身垂直速度阻尼强化 (-0.30 -> -0.75，压平起伏) ---
        self.rewards.lin_vel_z_l2.weight = -0.75

        # --- 核心强化项 3: 横滚姿态刚性加固 (-0.25 -> -0.40) ---
        self.rewards.roll_orientation_l2.weight = -0.40

        # --- 核心强化项 4: 平地目标净空收紧 (0.04m -> 0.025m，台阶依然维持 0.16m) ---
        self.rewards.terrain_relative_foot_clearance_l1.params["flat_target_clearance"] = 0.025
        self.rewards.terrain_relative_foot_clearance_l1.params["stairs_target_clearance"] = 0.16


@configclass
class UnitreeGo2DreamWaQSoftLandingEnvCfg_PLAY(UnitreeGo2DreamWaQSoftLandingEnvCfg):
    """平地与台阶软着陆评估与可视化配置。"""

    terrain_difficulty: float = 1.0

    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 4
        self.scene.env_spacing = 6.0
        self.scene.height_scanner.debug_vis = True
        self.viewer.origin_type = "world"

        self.scene.terrain.terrain_generator.size = (12.0, 4.0)
        self.scene.terrain.terrain_generator.num_rows = 1
        self.scene.terrain.terrain_generator.num_cols = 4
        self.scene.terrain.terrain_generator.curriculum = True
        self.scene.terrain.terrain_generator.difficulty_range = (self.terrain_difficulty, self.terrain_difficulty)
        self.scene.terrain.terrain_generator.sub_terrains = {
            "flat_corridor": MeshPlaneTerrainCfg(proportion=0.40),
            "straight_stairs": MeshStraightAscendingStairsTerrainCfg(
                proportion=0.60,
                step_height_range=(0.03, 0.15),
                step_width=0.3,
                platform_start=2.5,
                platform_end=2.5,
            ),
        }

        self.curriculum.command_ranges = None
        self.curriculum.terrain_levels = None
        self.commands.base_velocity.heading_command = True
        self.commands.base_velocity.heading_control_stiffness = 1.0
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)
        self.commands.base_velocity.rel_heading_envs = 1.0
        self.commands.base_velocity.rel_standing_envs = 0.0
        self.commands.base_velocity.ranges.lin_vel_x = (0.6, 0.6)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)

        self.observations.dreamwaq_policy.enable_corruption = False
        self.observations.dreamwaq_policy_history.enable_corruption = False
        self.observations.dreamwaq_reconstruction_target.enable_corruption = False
        self.observations.dreamwaq_critic.enable_corruption = False
        self.events.base_external_force_torque = None


@configclass
class UnitreeGo2DreamWaQDecoupledEnvCfg(UnitreeGo2DreamWaQSoftLandingEnvCfg):
    """Stage 5: 解耦微调配置 (Decoupled Flat Pitch Lock & Dual-Bounded Foot Clearance).

    目标：彻底根除平地机身俯仰跷跷板摇晃（Pitch Oscillation）与足端高抬腿跺地冲击（Stamping Impact），
    实现平地如磁悬浮般平顺巡航，同时遇阶自适应释放俯仰与提膝，保持 100% 翻越 5 级台阶。

    核心解耦与加固项：
    1. 注入解耦平地俯仰刚性锁：terrain_adaptive_pitch_orientation_l2 (weight = -0.40)
       - 平地巡航时严格锁死 pitch 晃动 (压回 <= 1.5°，消除机头点头颠簸)；
       - 台阶大仰角时 gate 平滑衰减至 0，彻底释放爬阶自由度，严格遵循红线 W4。
    2. 注入双向限幅足端净空惩罚：dual_bounded_pitch_adaptive_foot_clearance_l1 (weight = -0.20)
       - 平地设置严苛上限 flat_max_clearance = 0.028m，超额抬腿处以 3.0x 重罚，死死按压在 10~25mm 悬浮小跑；
       - 台阶上超额惩罚彻底清零，允许并激励深屈膝提膝至 0.16m，无损翻越 15cm 台阶。
    3. 垂向蹦跳严厉阻尼加倍：lin_vel_z_l2 从 -0.75 加码至 -1.50 (深度压平机身起伏)。
    4. 机身角速度与高度刚性加固：
       - ang_vel_xy_l2 从 -0.05 加固至 -0.10 (抑制高速晃荡角速度)；
       - base_height_relative_l2 从 -0.25 恢复至 -1.00 (target = 0.30m，锁死恒高)。
    5. 足端触地软着陆与动作平滑深度强化：
       - feet_contact_velocity_z 从 -0.06 提升至 -0.08；
       - action_smoothness_2nd 从 -0.005 提升至 -0.008。
    """

    def __post_init__(self):
        super().__post_init__()

        # --- 核心解耦项 1: 平地俯仰刚性锁 (-0.40，平地消除跷跷板摇晃，遇阶自动脱锁) ---
        self.rewards.terrain_adaptive_pitch_orientation_l2.weight = -0.40
        self.rewards.roll_orientation_l2.weight = -0.40

        # --- 核心解耦项 2: 双向限幅足端净空 (平地超额严惩，台阶高提膝无损) ---
        self.rewards.terrain_relative_foot_clearance_l1.func = mdp.dual_bounded_pitch_adaptive_foot_clearance_l1
        self.rewards.terrain_relative_foot_clearance_l1.weight = -0.20
        self.rewards.terrain_relative_foot_clearance_l1.params["flat_target_clearance"] = 0.022
        self.rewards.terrain_relative_foot_clearance_l1.params["flat_max_clearance"] = 0.028
        self.rewards.terrain_relative_foot_clearance_l1.params["stairs_target_clearance"] = 0.16
        self.rewards.terrain_relative_foot_clearance_l1.params["excess_penalty_scale"] = 3.0

        # --- 核心强化项 3: 垂向阻尼与恒高刚性锁 (彻底消灭颠簸沉浮) ---
        self.rewards.lin_vel_z_l2.weight = -1.50
        self.rewards.ang_vel_xy_l2.weight = -0.10
        self.rewards.base_height_relative_l2.weight = -1.00
        self.rewards.base_height_relative_l2.params["target_height"] = 0.30

        # --- 核心强化项 4: 触地缓冲与动作二阶平滑 ---
        self.rewards.feet_contact_velocity_z.weight = -0.08
        self.rewards.action_smoothness_2nd.weight = -0.008


@configclass
class UnitreeGo2DreamWaQDecoupledEnvCfg_PLAY(UnitreeGo2DreamWaQDecoupledEnvCfg):
    """平地与台阶解耦微调评估与可视化配置。"""

    terrain_difficulty: float = 1.0

    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 4
        self.scene.env_spacing = 6.0
        self.scene.height_scanner.debug_vis = True
        self.viewer.origin_type = "world"

        self.scene.terrain.terrain_generator.size = (12.0, 4.0)
        self.scene.terrain.terrain_generator.num_rows = 1
        self.scene.terrain.terrain_generator.num_cols = 4
        self.scene.terrain.terrain_generator.curriculum = True
        self.scene.terrain.terrain_generator.difficulty_range = (self.terrain_difficulty, self.terrain_difficulty)
        self.scene.terrain.terrain_generator.sub_terrains = {
            "flat_corridor": MeshPlaneTerrainCfg(proportion=0.40),
            "straight_stairs": MeshStraightAscendingStairsTerrainCfg(
                proportion=0.60,
                step_height_range=(0.03, 0.15),
                step_width=0.3,
                platform_start=2.5,
                platform_end=2.5,
            ),
        }

        self.curriculum.command_ranges = None
        self.curriculum.terrain_levels = None
        self.commands.base_velocity.heading_command = True
        self.commands.base_velocity.heading_control_stiffness = 1.0
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)
        self.commands.base_velocity.rel_heading_envs = 1.0
        self.commands.base_velocity.rel_standing_envs = 0.0
        self.commands.base_velocity.ranges.lin_vel_x = (0.6, 0.6)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)

        self.observations.dreamwaq_policy.enable_corruption = False
        self.observations.dreamwaq_policy_history.enable_corruption = False
        self.observations.dreamwaq_reconstruction_target.enable_corruption = False
        self.observations.dreamwaq_critic.enable_corruption = False
        self.events.base_external_force_torque = None


@configclass
class UnitreeGo2DreamWaQSurgeryEnvCfg(UnitreeGo2DreamWaQDecoupledEnvCfg):
    """Stage 6: 全域等变镜像对称与前肢双向对中硬手术配置 (Unified Symmetry Surgery & Equivariance Lock).

    目标：彻底矫正 Stage 5 模型在台阶训练中遗留的左前腿内偏（FL Inward Adduction Bias -15.0° / 6.3cm），
    构建同时具备【4598 平地绝对镜像对称滑行】与【6493 强悍 100% 越障】的终极大一统模型。

    核心手术项：
    1. 注入全域等变镜像对称硬锁：bilateral_hip_symmetry_equivariance_l2 (weight = -0.50)
       - 彻底剔除 min=0.0 单边截断后门，无论是外甩还是内扣，只要左右不对称一律严惩！
    2. 注入前腿髋关节双向中立零位锁：front_hip_neutral_pos_l2 (weight = -0.80)
       - 强制将左前髋 FL 偏离角从 -9.3°（内偏 15°）强行拉直回 0° ~ 5.7° 黄金支撑角度；
    3. 激活四肢功率均衡方差惩罚：power_distribution_variance_l2 (weight = -1.0e-5)
       - 遵循 agent.md 与 DreamWaQ.md 规范，瓦解单肢内收借力与非对称负荷偷懒；
    4. 保留 Stage 5 全部成果：
       - 自适应平地俯仰刚性锁 (-0.40) + 双向限幅足端净空 (平地 <=28mm 严惩，台阶 0.16m 释放)；
       - 足端软着陆阻尼与动作二阶平滑；
       - 40% 平地长廊 + 60% 直线阶梯自适应混合地形。
    """

    def __post_init__(self):
        super().__post_init__()

        # --- 核心手术项 1: 双向全域等变镜像对称锁 (-0.50，左右必须镜像一致) ---
        self.rewards.bilateral_hip_symmetry_equivariance_l2.weight = -0.50

        # --- 核心手术项 2: 前腿髋关节中立零位硬拉直 (-0.80，将左前腿拉直回对中状态) ---
        self.rewards.front_hip_neutral_pos_l2.weight = -0.80

        # --- 核心手术项 3: 功耗对称方差均衡 (-1.0e-4，强制四腿负荷绝对均摊，消灭单侧主导) ---
        self.rewards.power_distribution_variance_l2.weight = -1.0e-4

        # --- 核心手术项 4: 几何铁笼严惩总髋位姿偏离 (-0.50，强制四腿髋角绝对对中) ---
        self.rewards.hip_pos_deviation_l1.weight = -0.50


@configclass
class UnitreeGo2DreamWaQSurgeryEnvCfg_PLAY(UnitreeGo2DreamWaQSurgeryEnvCfg):
    """Stage 6 评估与可视化配置。"""

    terrain_difficulty: float = 1.0

    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 4
        self.scene.env_spacing = 6.0
        self.scene.height_scanner.debug_vis = True
        self.viewer.origin_type = "world"

        self.scene.terrain.terrain_generator.size = (12.0, 4.0)
        self.scene.terrain.terrain_generator.num_rows = 1
        self.scene.terrain.terrain_generator.num_cols = 4
        self.scene.terrain.terrain_generator.curriculum = True
        self.scene.terrain.terrain_generator.difficulty_range = (self.terrain_difficulty, self.terrain_difficulty)
        self.scene.terrain.terrain_generator.sub_terrains = {
            "flat_corridor": MeshPlaneTerrainCfg(proportion=0.40),
            "straight_stairs": MeshStraightAscendingStairsTerrainCfg(
                proportion=0.60,
                step_height_range=(0.03, 0.15),
                step_width=0.3,
                platform_start=2.5,
                platform_end=2.5,
            ),
        }

        self.curriculum.command_ranges = None
        self.curriculum.terrain_levels = None
        self.commands.base_velocity.heading_command = True
        self.commands.base_velocity.heading_control_stiffness = 1.0
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)
        self.commands.base_velocity.rel_heading_envs = 1.0
        self.commands.base_velocity.rel_standing_envs = 0.0
        self.commands.base_velocity.ranges.lin_vel_x = (0.6, 0.6)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)

        self.observations.dreamwaq_policy.enable_corruption = False
        self.observations.dreamwaq_policy_history.enable_corruption = False
        self.observations.dreamwaq_reconstruction_target.enable_corruption = False
        self.observations.dreamwaq_critic.enable_corruption = False
        self.events.base_external_force_torque = None


@configclass
class UnitreeGo2DreamWaQExtremeStairs15cmEnvCfg(UnitreeGo2DreamWaQSurgeryEnvCfg):
    """Stage 8: 15cm 极限高台阶专项攻坚与自适应提膝跨障配置 (Extreme 15cm Stairs Knee-Lift Fine-tuning).

    目标：彻底克服 15cm 极限阶高（0.75m 攀升）起步切入撞阶摔倒与小腿剐蹭痛点，
    实现大开大合的高抬腿悬空跨越，兼具平地磁悬浮滑行与 15cm~18cm 极限台阶自适应翻越。

    核心攻坚项：
    1. 阶梯课程锁定在 10cm ~ 16cm 极限高度区间：
       - straight_stairs: step_height_range=(0.10, 0.16)
       - 强迫所有智能体专门在极限高阶中进行步态与动力学探索；
    2. 自适应越障足端净空全面拔高：
       - stairs_target_clearance 从 0.16m 调整为 0.15m (悬空整整 15cm)；
       - 平地依然维持 flat_target_clearance = 0.022m, flat_max_clearance = 0.028m，严惩平地跺地；
    3. 小腿/大腿磕碰重罚加码：
       - undesired_contacts: weight 从 -0.20 加码至 -0.40 (严惩胫骨与台阶磕碰，逼迫提膝)；
    4. 继承 Stage 7 的所有等变损失与动力学铁笼：
       - 左右等变镜像硬锁、髋关节中立零位锁、四腿功率均衡方差锁全量生效！
    """

    def __post_init__(self):
        super().__post_init__()

        # --- 1. 极限阶高长廊设定 (10cm ~ 16cm) ---
        self.scene.terrain.terrain_generator.sub_terrains["straight_stairs"] = MeshStraightAscendingStairsTerrainCfg(
            proportion=0.60,
            step_height_range=(0.10, 0.16),
            step_width=0.30,
            platform_start=2.5,
            platform_end=2.5,
        )

        # --- 2. 越障足端提膝拔高与平地磁悬浮严格解耦 ---
        self.rewards.terrain_relative_foot_clearance_l1.func = mdp.dual_bounded_pitch_adaptive_foot_clearance_l1
        self.rewards.terrain_relative_foot_clearance_l1.weight = -0.25
        self.rewards.terrain_relative_foot_clearance_l1.params["flat_target_clearance"] = 0.022
        self.rewards.terrain_relative_foot_clearance_l1.params["flat_max_clearance"] = 0.028
        self.rewards.terrain_relative_foot_clearance_l1.params["stairs_target_clearance"] = 0.15
        self.rewards.terrain_relative_foot_clearance_l1.params["excess_penalty_scale"] = 3.0

        # --- 3. 严惩磕阶与小腿刮蹭 (加码 2 倍) ---
        self.rewards.undesired_contacts.weight = -0.40


@configclass
class UnitreeGo2DreamWaQExtremeStairs15cmEnvCfg_PLAY(UnitreeGo2DreamWaQExtremeStairs15cmEnvCfg):
    """Stage 8 评估与可视化配置。"""

    terrain_difficulty: float = 1.0

    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 4
        self.scene.env_spacing = 6.0
        self.scene.height_scanner.debug_vis = True
        self.viewer.origin_type = "world"

        self.scene.terrain.terrain_generator.size = (12.0, 4.0)
        self.scene.terrain.terrain_generator.num_rows = 1
        self.scene.terrain.terrain_generator.num_cols = 4
        self.scene.terrain.terrain_generator.curriculum = True
        self.scene.terrain.terrain_generator.difficulty_range = (self.terrain_difficulty, self.terrain_difficulty)
        self.scene.terrain.terrain_generator.sub_terrains = {
            "flat_corridor": MeshPlaneTerrainCfg(proportion=0.40),
            "straight_stairs": MeshStraightAscendingStairsTerrainCfg(
                proportion=0.60,
                step_height_range=(0.10, 0.16),
                step_width=0.30,
                platform_start=2.5,
                platform_end=2.5,
            ),
        }

        self.curriculum.command_ranges = None
        self.curriculum.terrain_levels = None
        self.commands.base_velocity.heading_command = True
        self.commands.base_velocity.heading_control_stiffness = 1.0
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)
        self.commands.base_velocity.rel_heading_envs = 1.0
        self.commands.base_velocity.rel_standing_envs = 0.0
        self.commands.base_velocity.ranges.lin_vel_x = (0.6, 0.6)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)

        self.observations.dreamwaq_policy.enable_corruption = False
        self.observations.dreamwaq_policy_history.enable_corruption = False
        self.observations.dreamwaq_reconstruction_target.enable_corruption = False
        self.observations.dreamwaq_critic.enable_corruption = False
        self.events.base_external_force_torque = None


@configclass
class UnitreeGo2DreamWaQSmoothStairsStage9EnvCfg(UnitreeGo2DreamWaQExtremeStairs15cmEnvCfg):
    """Stage 9: 终极平滑与激光级直行大一统手术配置 (Smooth Stairs & Straight Drift Erasure Fine-tuning).

    核心改进与物理机理：
    1. 彻底消灭台阶卡壳/卡顿（消灭恐慌性避险回缩假死 Retraction Reflex）：
       - undesired_contacts: weight 从 -0.40 剧降至 -0.02 (解除对肢体擦阶的粗暴二值封杀)；
       - soft_contact_velocity_damping: weight = -0.05 (连续接触速度阻尼，允许低速顺滑贴附阶梯滑行，仅重惩高速粗暴硬撞)；
    2. 彻底消灭直行向左偏摆问题（锁死激光级直线）：
       - straight_ang_vel_z: weight = -0.25 (在直行指令下严惩机身任何偏航角速度 |w_z|^2)；
       - flat_heading_error_l2: weight = -0.40 (平地绝对航向死锁，强行纠正偏航)；
       - rear_hip_neutral_pos: weight = -0.50 (后腿髋关节双向中立零位锁，消灭 RL/RR 差异偏推)；
    3. 全量继承 Stage 7/8 黄金动力学铁笼：
       - front_hip_neutral_pos_l2: weight = -0.80 (前腿髋关节中立零位锁)；
       - bilateral_hip_symmetry_equivariance_l2: weight = -0.50 (左右镜像等变硬锁)；
       - power_distribution_variance_l2: weight = -1.0e-4 (四肢功耗方差均衡)；
       - hip_pos_deviation_l1: weight = -0.50 (总髋位姿偏离惩罚)；
       - dual_bounded_pitch_adaptive_foot_clearance_l1: weight = -0.25 (15cm 极限高台阶提膝 + 平地 22mm 磁悬浮滑行)。
    """

    def __post_init__(self):
        super().__post_init__()

        # --- 1. 消灭台阶卡顿：碰撞惩罚柔化为连续速度阻尼 ---
        self.rewards.undesired_contacts.weight = -0.02
        self.rewards.soft_contact_velocity_damping.weight = -0.05

        # --- 2. 锁死激光级直线：偏航刚性阻尼 + 绝对航向死锁 + 后腿髋零位锁 ---
        self.rewards.straight_ang_vel_z.weight = -0.25
        self.rewards.flat_heading_error_l2.weight = -0.40
        self.rewards.rear_hip_neutral_pos.weight = -0.50

        # --- 3. 强化镜像等变与功耗平衡铁笼 ---
        self.rewards.front_hip_neutral_pos_l2.weight = -0.80
        self.rewards.bilateral_hip_symmetry_equivariance_l2.weight = -0.50
        self.rewards.power_distribution_variance_l2.weight = -1.0e-4
        self.rewards.hip_pos_deviation_l1.weight = -0.50


@configclass
class UnitreeGo2DreamWaQSmoothStairsStage9EnvCfg_PLAY(UnitreeGo2DreamWaQSmoothStairsStage9EnvCfg):
    """Stage 9 评估与可视化配置。"""

    terrain_difficulty: float = 1.0

    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 4
        self.scene.env_spacing = 6.0
        self.scene.height_scanner.debug_vis = True
        self.viewer.origin_type = "world"

        self.scene.terrain.terrain_generator.size = (12.0, 4.0)
        self.scene.terrain.terrain_generator.num_rows = 1
        self.scene.terrain.terrain_generator.num_cols = 4
        self.scene.terrain.terrain_generator.curriculum = True
        self.scene.terrain.terrain_generator.difficulty_range = (self.terrain_difficulty, self.terrain_difficulty)
        self.scene.terrain.terrain_generator.sub_terrains = {
            "flat_corridor": MeshPlaneTerrainCfg(proportion=0.40),
            "straight_stairs": MeshStraightAscendingStairsTerrainCfg(
                proportion=0.60,
                step_height_range=(0.10, 0.16),
                step_width=0.30,
                platform_start=2.5,
                platform_end=2.5,
            ),
        }

        self.curriculum.command_ranges = None
        self.curriculum.terrain_levels = None
        self.commands.base_velocity.heading_command = True
        self.commands.base_velocity.heading_control_stiffness = 1.0
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)
        self.commands.base_velocity.rel_heading_envs = 1.0
        self.commands.base_velocity.rel_standing_envs = 0.0
        self.commands.base_velocity.ranges.lin_vel_x = (0.6, 0.6)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)

        self.observations.dreamwaq_policy.enable_corruption = False
        self.observations.dreamwaq_policy_history.enable_corruption = False
        self.observations.dreamwaq_reconstruction_target.enable_corruption = False
        self.observations.dreamwaq_critic.enable_corruption = False
        self.events.base_external_force_torque = None


@configclass
class UnitreeGo2DreamWaQOptimalStage10EnvCfg(UnitreeGo2DreamWaQSmoothStairsStage9EnvCfg):
    """Stage 10: 0 网络修改 · 纯物理与等变 Loss 级最优解 (Optimal Zero-Architecture-Change Equivariant & Smooth Alignment).

    核心改进与物理机理：
    1. 彻底消灭上楼梯卡顿/卡壳（消灭避险回缩假死 Retraction Reflex）：
       - undesired_contacts: weight = 0.0 (彻底清零小腿/大腿碰阶惩罚，允许肢体自然贴合阶梯边缘)；
       - soft_contact_velocity_damping: weight = 0.0 (彻底清零接触速度阻尼，杜绝碰阶降速与停滞)；
       - feet_contact_velocity_z: weight = 0.0 (彻底清零足端触地瞬间的垂直速度阻尼，释放落脚动量，消除蹬阶迟疑)；
    2. 彻底消灭直行向左偏摆（锁死绝对激光直线）：
       - 配合 CENet 物理镜像等变自监督损失 (消除幽灵侧漂速度 v_hat_y)；
       - straight_ang_vel_z: weight = -0.30 (平地直行偏航角速度刚性锁强化)；
       - flat_heading_error_l2: weight = -0.40 (平地绝对航向刚性死锁)；
       - rear_hip_neutral_pos: weight = -0.50 (后腿双髋对称中立锁)；
    3. 全量继承 Stage 7/8/9 黄金动力学铁笼：
       - front_hip_neutral_pos_l2: weight = -0.80 (前腿双髋中立锁)；
       - bilateral_hip_symmetry_equivariance_l2: weight = -0.50 (左右镜像等变硬锁)；
       - power_distribution_variance_l2: weight = -1.0e-4 (四腿功耗方差均衡)；
       - hip_pos_deviation_l1: weight = -0.50 (总髋位姿偏离惩罚)；
       - terrain_relative_foot_clearance_l1: 维持 15cm 极限高台阶提膝 + 平地 22mm 磁悬浮滑行。
    """

    def __post_init__(self):
        super().__post_init__()

        # --- 1. 彻底消灭台阶卡顿与假死：小腿碰阶与触地阻尼全部清零 ---
        self.rewards.undesired_contacts.weight = 0.0
        self.rewards.soft_contact_velocity_damping.weight = 0.0
        self.rewards.feet_contact_velocity_z.weight = 0.0

        # --- 2. 锁死激光级直线：强化偏航刚性阻尼 + 绝对航向死锁 + 后腿髋中立锁 ---
        self.rewards.straight_ang_vel_z.weight = -0.30
        self.rewards.flat_heading_error_l2.weight = -0.40
        self.rewards.rear_hip_neutral_pos.weight = -0.50

        # --- 3. 强化镜像等变与功耗平衡铁笼 ---
        self.rewards.front_hip_neutral_pos_l2.weight = -0.80
        self.rewards.bilateral_hip_symmetry_equivariance_l2.weight = -0.50
        self.rewards.power_distribution_variance_l2.weight = -1.0e-4
        self.rewards.hip_pos_deviation_l1.weight = -0.50


@configclass
class UnitreeGo2DreamWaQOptimalStage10EnvCfg_PLAY(UnitreeGo2DreamWaQOptimalStage10EnvCfg):
    """Stage 10 评估与可视化配置。"""

    terrain_difficulty: float = 1.0

    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 4
        self.scene.env_spacing = 6.0
        self.scene.height_scanner.debug_vis = True
        self.viewer.origin_type = "world"

        self.scene.terrain.terrain_generator.size = (12.0, 4.0)
        self.scene.terrain.terrain_generator.num_rows = 1
        self.scene.terrain.terrain_generator.num_cols = 4
        self.scene.terrain.terrain_generator.curriculum = True
        self.scene.terrain.terrain_generator.difficulty_range = (self.terrain_difficulty, self.terrain_difficulty)
        self.scene.terrain.terrain_generator.sub_terrains = {
            "flat_corridor": MeshPlaneTerrainCfg(proportion=0.40),
            "straight_stairs": MeshStraightAscendingStairsTerrainCfg(
                proportion=0.60,
                step_height_range=(0.10, 0.16),
                step_width=0.30,
                platform_start=2.5,
                platform_end=2.5,
            ),
        }

        self.curriculum.command_ranges = None
        self.curriculum.terrain_levels = None
        self.commands.base_velocity.heading_command = True
        self.commands.base_velocity.heading_control_stiffness = 1.0
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)
        self.commands.base_velocity.rel_heading_envs = 1.0
        self.commands.base_velocity.rel_standing_envs = 0.0
        self.commands.base_velocity.ranges.lin_vel_x = (0.6, 0.6)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)

        self.observations.dreamwaq_policy.enable_corruption = False
        self.observations.dreamwaq_policy_history.enable_corruption = False
        self.observations.dreamwaq_reconstruction_target.enable_corruption = False
        self.observations.dreamwaq_critic.enable_corruption = False

@configclass
class UnitreeGo2DreamWaQStage11EnvCfg(UnitreeGo2DreamWaQOptimalStage10EnvCfg):
    """Stage 11: 黄金基准手术配置 (Gold Baseline: 0.3s Lookahead Knee-Lift + Flexible Compliance Decoupling).

    核心手术与物理机理（路线 B：柔性自适应松绑流）：
    1. 彻底消灭台阶切入卡顿与悬停（0.3s 前瞻感知 + 落脚保护门控 Landing Gate）：
       - dual_bounded_pitch_adaptive_foot_clearance_l1 引入前向 0.3s (0.05~0.40m) 高差射线前瞻，撞阶前 0.3s 提前将前腿目标净空拔高至 16cm；
       - 同时免除平地超额惩罚，允许前腿大弧度高提膝，一脚跨上第 1 阶踏面；
       - clearance 权重微调至 -0.10，配合落地保护门控消除落脚惩罚死锁；
       - 彻底保持 undesired_contacts=0, soft_contact_velocity_damping=0，消除立面畏缩；
    2. 彻底消灭直行向右摆动（刚性约束全面松绑 80%，归还 Trot 顺应性）：
       - 打破前四项高达 -2.30 的髋关节焊死铁笼，全部降至 -0.05 极弱引导；
       - 归还左右前腿与后腿在对角支撑相中的微动吸震自由度，消除刚体反弹与过矫正扭力；
       - straight_ang_vel_z 与 flat_heading_error_l2 柔化至 -0.10，保留长程航向引导的同时消除过阻尼航向震荡；
       - power_distribution_variance_l2 柔化至 -1.0e-5，允许四腿按需动态发力。
    """

    def __post_init__(self):
        super().__post_init__()

        # --- 1. 锁死走廊直行主轴，杜绝重生坐标系扭曲 ---
        self.events.reset_base.params["pose_range"]["yaw"] = (0.0, 0.0)

        # --- 2. 根治台阶落脚卡顿：0.3s 前瞻自适应净空 + 落地平滑门控 ---
        self.rewards.terrain_relative_foot_clearance_l1.weight = -0.10
        self.rewards.undesired_contacts.weight = 0.0
        self.rewards.soft_contact_velocity_damping.weight = 0.0
        self.rewards.feet_contact_velocity_z.weight = 0.0

        # --- 3. 路线 B 柔性自适应松绑流：全面削减 80% 刚性锁，归还 Trot 顺应性，终结向右偏摆 ---
        self.rewards.straight_ang_vel_z.weight = -0.10
        self.rewards.flat_heading_error_l2.weight = -0.10
        self.rewards.rear_hip_neutral_pos.weight = -0.05
        self.rewards.front_hip_neutral_pos_l2.weight = -0.05
        self.rewards.bilateral_hip_symmetry_equivariance_l2.weight = -0.05
        self.rewards.power_distribution_variance_l2.weight = -1.0e-5
        self.rewards.hip_pos_deviation_l1.weight = -0.05


@configclass
class UnitreeGo2DreamWaQStage11EnvCfg_PLAY(UnitreeGo2DreamWaQStage11EnvCfg):
    """Stage 11 评估与可视化配置。"""

    terrain_difficulty: float = 1.0

    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 4
        self.scene.env_spacing = 6.0
        self.scene.height_scanner.debug_vis = True
        self.viewer.origin_type = "world"

        self.scene.terrain.terrain_generator.size = (12.0, 4.0)
        self.scene.terrain.terrain_generator.num_rows = 1
        self.scene.terrain.terrain_generator.num_cols = 4
        self.scene.terrain.terrain_generator.curriculum = True
        self.scene.terrain.terrain_generator.difficulty_range = (self.terrain_difficulty, self.terrain_difficulty)
        self.scene.terrain.terrain_generator.sub_terrains = {
            "flat_corridor": MeshPlaneTerrainCfg(proportion=0.40),
            "straight_stairs": MeshStraightAscendingStairsTerrainCfg(
                proportion=0.60,
                step_height_range=(0.10, 0.16),
                step_width=0.30,
                platform_start=2.5,
                platform_end=2.5,
            ),
        }

        self.curriculum.command_ranges = None
        self.curriculum.terrain_levels = None
        self.commands.base_velocity.heading_command = True
        self.commands.base_velocity.heading_control_stiffness = 1.0
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)
        self.commands.base_velocity.rel_heading_envs = 1.0
        self.commands.base_velocity.rel_standing_envs = 0.0
        self.commands.base_velocity.ranges.lin_vel_x = (0.6, 0.6)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)

        self.observations.dreamwaq_policy.enable_corruption = False
        self.observations.dreamwaq_policy_history.enable_corruption = False
        self.observations.dreamwaq_reconstruction_target.enable_corruption = False
        self.observations.dreamwaq_critic.enable_corruption = False
        self.events.base_external_force_torque = None


# ==============================================================================
# Stage 13: 纯平地洗髓定型配置 (Pure Flat Equivariant Symmetry Purify - 200 iterations)
# ==============================================================================
@configclass
class UnitreeGo2DreamWaQStage13FlatPurifyEnvCfg(UnitreeGo2DreamWaQFlatEnvCfg):
    """Stage 13: 纯平地洗髓定型手术 (Pure Flat Equivariant Symmetry Purify).

    核心定位与设计思想（方案 B）：
    1. 彻底切断任何楼梯地形（100% 纯平面 plane），0 台阶几何扰动；
    2. 建立纯直行激光走廊指令空间 (vx in [0.6, 0.8], vy=0, wz=0, heading=0)；
    3. 全域施加【对角对称 + 双向全域等变镜像对称 + 四肢功耗均等】强力洗髓力场：
       - bilateral_hip_symmetry_equivariance_l2 = -0.50 强制左右髋关节偏差严格镜像对称
       - joint_diagonal_symmetry_l2 = -0.10 强制对角肢体运动学绝对对称
       - power_distribution_variance_l2 = -1.0e-4 强制四腿机械功率绝对均摊，根除单侧推力偏大
       - straight_ang_vel_z = -0.35 与 flat_heading_error_l2 = -0.50 锁死激光级航向
       - lin_vel_xy_orthogonal_l2 = -0.30 严惩任何侧滑横移
    4. 200 轮纯平地快训，彻底洗净前 7000 轮在楼梯上残留的单侧偏展与代偿神经突触，把零位彻底调平！
    """

    def __post_init__(self):
        super().__post_init__()

        # --- 1. 彻底切换为纯平面，排除任何台阶网格扰动 ---
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.curriculum.terrain_levels = None
        self.curriculum.command_ranges = None

        # --- 2. 纯直行激光走廊指令空间：锁死主轴，杜绝重生偏航撕扯 ---
        self.events.reset_base.params["pose_range"]["yaw"] = (0.0, 0.0)
        self.commands.base_velocity.heading_command = True
        self.commands.base_velocity.heading_control_stiffness = 1.0
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)
        self.commands.base_velocity.rel_heading_envs = 1.0
        self.commands.base_velocity.rel_standing_envs = 0.0
        self.commands.base_velocity.ranges.lin_vel_x = (0.6, 0.8)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)

        # --- 3. 对角对称 + 镜像等变洗髓力场 (Equivariant Symmetry Wash) ---
        self.rewards.bilateral_hip_symmetry_equivariance_l2.weight = -0.50
        self.rewards.joint_diagonal_symmetry_l2.weight = -0.10
        self.rewards.power_distribution_variance_l2.weight = -1.0e-4
        self.rewards.front_hip_neutral_pos_l2.weight = -0.20
        self.rewards.rear_hip_neutral_pos.weight = -0.20
        self.rewards.hip_pos_deviation_l1.weight = -0.10

        # --- 4. 激光直行航向与侧滑严惩 ---
        self.rewards.straight_ang_vel_z.weight = -0.35
        self.rewards.flat_heading_error_l2.weight = -0.50
        self.rewards.lin_vel_xy_orthogonal_l2.weight = -0.30

        # --- 5. 平地姿态悬浮与动作平滑 ---
        self.rewards.track_lin_vel_xy_exp.weight = 1.5
        self.rewards.track_ang_vel_z_exp.weight = 1.0
        self.rewards.lin_vel_z_l2.weight = -2.5
        self.rewards.ang_vel_xy_l2.weight = -0.10
        self.rewards.flat_orientation_l2.weight = -0.30
        self.rewards.base_height_relative_l2.weight = -1.5
        self.rewards.base_height_relative_l2.params["target_height"] = 0.30
        self.rewards.terrain_relative_foot_clearance_l1.weight = -0.01

        # 禁用碰撞阻尼与脚接触突变
        self.rewards.undesired_contacts.weight = 0.0
        self.rewards.soft_contact_velocity_damping.weight = 0.0
        self.rewards.feet_contact_velocity_z.weight = 0.0


@configclass
class UnitreeGo2DreamWaQStage13FlatPurifyEnvCfg_PLAY(UnitreeGo2DreamWaQStage13FlatPurifyEnvCfg):
    """Stage 13 纯平地洗髓评估与可视化配置。"""

    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 4
        self.scene.env_spacing = 5.0
        self.viewer.origin_type = "world"

        self.commands.base_velocity.ranges.lin_vel_x = (0.6, 0.6)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)

        self.observations.dreamwaq_policy.enable_corruption = False
        self.observations.dreamwaq_policy_history.enable_corruption = False
        self.observations.dreamwaq_reconstruction_target.enable_corruption = False
        self.observations.dreamwaq_critic.enable_corruption = False
        self.events.base_external_force_torque = None


# ==============================================================================
# Stage 14: 纯平地洗直 + 15cm 台阶跨越大一统融合 (Unified Flat-Stair Synergy)
# ==============================================================================
@configclass
class UnitreeGo2DreamWaQStage14UnifiedEnvCfg(UnitreeGo2DreamWaQOptimalStage10EnvCfg):
    """Stage 14: 大一统黄金融合手术配置 (Unified Flat-Stair Synergy Baseline).

    核心定位与设计思想：
    1. 以 Stage 13 纯平地洗直检查点 model_7189.pt 为基石，继承其高度对称的零位突触；
    2. 返回走廊地形（40% 平地走廊 + 60% 10~16cm 连续台阶）；
    3. 激活 0.3s 前瞻感知门控（alpha_ahead -> 1.0, clearance=-0.12），撞阶前 0.3s 强制拔高前膝；
    4. 施加黄金固化等变力场（bilateral_hip=-0.20, power_var=-5e-5, straight_wz=-0.25, flat_heading=-0.35），
       将洗平的激光级直行步态与 15cm 台阶高提膝翻越完美融合，终结偏摆与卡顿！
    """

    def __post_init__(self):
        super().__post_init__()

        # --- 1. 锁死走廊直行主轴，杜绝重生坐标系扭曲 ---
        self.events.reset_base.params["pose_range"]["yaw"] = (0.0, 0.0)

        # --- 2. 0.3s 前瞻感知自适应净空 + 落地平滑门控 ---
        self.rewards.terrain_relative_foot_clearance_l1.weight = -0.12
        self.rewards.undesired_contacts.weight = 0.0
        self.rewards.soft_contact_velocity_damping.weight = 0.0
        self.rewards.feet_contact_velocity_z.weight = 0.0

        # --- 3. 固化 Stage 13 洗直对称性：等变镜像 + 功耗均等 + 航向阻尼 ---
        self.rewards.bilateral_hip_symmetry_equivariance_l2.weight = -0.20
        self.rewards.joint_diagonal_symmetry_l2.weight = -0.05
        self.rewards.power_distribution_variance_l2.weight = -5.0e-5
        self.rewards.straight_ang_vel_z.weight = -0.25
        self.rewards.flat_heading_error_l2.weight = -0.35
        self.rewards.front_hip_neutral_pos_l2.weight = -0.10
        self.rewards.rear_hip_neutral_pos.weight = -0.10
        self.rewards.hip_pos_deviation_l1.weight = -0.08


@configclass
class UnitreeGo2DreamWaQStage14UnifiedEnvCfg_PLAY(UnitreeGo2DreamWaQStage14UnifiedEnvCfg):
    """Stage 14 评估与可视化配置。"""

    terrain_difficulty: float = 1.0

    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 4
        self.scene.env_spacing = 6.0
        self.scene.height_scanner.debug_vis = True
        self.viewer.origin_type = "world"

        self.scene.terrain.terrain_generator.size = (12.0, 4.0)
        self.scene.terrain.terrain_generator.num_rows = 1
        self.scene.terrain.terrain_generator.num_cols = 4
        self.scene.terrain.terrain_generator.curriculum = True
        self.scene.terrain.terrain_generator.difficulty_range = (self.terrain_difficulty, self.terrain_difficulty)
        self.scene.terrain.terrain_generator.sub_terrains = {
            "flat_corridor": MeshPlaneTerrainCfg(proportion=0.40),
            "straight_stairs": MeshStraightAscendingStairsTerrainCfg(
                proportion=0.60,
                step_height_range=(0.10, 0.16),
                step_width=0.30,
                platform_start=2.5,
                platform_end=2.5,
            ),
        }

        self.curriculum.command_ranges = None
        self.curriculum.terrain_levels = None
        self.commands.base_velocity.heading_command = True
        self.commands.base_velocity.heading_control_stiffness = 1.0
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)
        self.commands.base_velocity.rel_heading_envs = 1.0
        self.commands.base_velocity.rel_standing_envs = 0.0
        self.commands.base_velocity.ranges.lin_vel_x = (0.6, 0.6)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)

        self.observations.dreamwaq_policy.enable_corruption = False
        self.observations.dreamwaq_policy_history.enable_corruption = False
        self.observations.dreamwaq_reconstruction_target.enable_corruption = False
        self.observations.dreamwaq_critic.enable_corruption = False
        self.events.base_external_force_torque = None










