# Copyright (c) 2024 LunarQuadruped Project.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""RSL-RL config for the independent Unitree Go2 DreamWaQ task."""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class UnitreeGo2DreamWaQActorCriticCfg(RslRlPpoActorCriticCfg):
    """DreamWaQ actor-critic config for Unitree Go2."""

    class_name: str = "DreamWaQActorCritic"
    history_length: int = 5
    proprio_obs_dim: int = -1
    estimated_velocity_dim: int = 3
    context_latent_dim: int = 16
    cenet_encoder_hidden_dims: list[int] = [128, 64]
    cenet_decoder_hidden_dims: list[int] = [128, 128]
    beta_vae_kl_weight: float = 0.01                      # beta参数值
    use_cenet: bool = True
    history_is_term_major: bool = True
    history_term_dims: list[int] | None = None
    detach_cenet_actor_input: bool = True
    reconstruct_next_observation: bool = True
    policy_group_name: str = "dreamwaq_policy"
    history_group_name: str = "dreamwaq_policy_history"
    reconstruction_target_group_name: str = "dreamwaq_reconstruction_target"
    critic_group_name: str = "dreamwaq_critic"


@configclass
class UnitreeGo2DreamWaQAlgorithmCfg(RslRlPpoAlgorithmCfg):
    """PPO config with DreamWaQ CENet loss controls."""

    class_name: str = "DreamWaQPPO"
    cenet_loss_weight: float = 1.0
    velocity_loss_weight: float = 1.0
    reconstruction_loss_weight: float = 1.0
    kl_loss_weight: float = 1.0
    kl_warmup_iterations: int = 2000           # KL散度热身迭代次数
    vae_learning_rate: float = 1.0e-3
    use_adaboot: bool = False
    adaboot_reward_window: int = 2048
    adaboot_min_episodes: int = 32
    adaboot_min_probability: float = 0.0
    adaboot_max_probability: float = 1.0
    adaboot_eps: float = 1.0e-6


@configclass
class UnitreeGo2DreamWaQRoughRunnerCfg(RslRlOnPolicyRunnerCfg):
    """Go2 DreamWaQ runner config with an independent task path and log directory."""

    seed = 1
    num_steps_per_env = 24
    max_iterations = 20000
    save_interval = 50
    experiment_name = "unitree/unitree_go2_dreamwaq_rough"
    obs_groups = {
        "policy": ["dreamwaq_policy", "dreamwaq_policy_history"],
        "critic": ["dreamwaq_critic"],
    }

    policy = UnitreeGo2DreamWaQActorCriticCfg(
        init_noise_std=1.0,
        noise_std_type="scalar",
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )

    algorithm = UnitreeGo2DreamWaQAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        cenet_loss_weight=1.0,
        velocity_loss_weight=3.0,          # 注入速度头损失权重补丁 3.0，使显式速度估计梯度充沛，根除符号倒挂
        reconstruction_loss_weight=1.0,
        kl_loss_weight=1.0,
        vae_learning_rate=1.0e-3,
    )


@configclass
class UnitreeGo2DreamWaQFlatRunnerCfg(RslRlOnPolicyRunnerCfg):
    """Go2 DreamWaQ 纯平地冷启动步态定型与 CENet 速度感知校准 Runner 配置。"""

    seed = 1
    num_steps_per_env = 24
    max_iterations = 300
    save_interval = 25
    experiment_name = "unitree/unitree_go2_dreamwaq_flat"
    obs_groups = {
        "policy": ["dreamwaq_policy", "dreamwaq_policy_history"],
        "critic": ["dreamwaq_critic"],
    }

    policy = UnitreeGo2DreamWaQActorCriticCfg(
        init_noise_std=1.0,
        noise_std_type="scalar",
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        beta_vae_kl_weight=0.01,
    )

    algorithm = UnitreeGo2DreamWaQAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=2.0e-4,              # 微调降温学习率 2.0e-4，平稳吸收残余力矩
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        cenet_loss_weight=1.0,
        velocity_loss_weight=3.0,          # 注入速度头损失权重补丁 3.0，彻底根除符号倒挂
        reconstruction_loss_weight=1.0,
        kl_loss_weight=1.0,
        kl_warmup_iterations=300,          # 300 轮微调退火
        vae_learning_rate=2.0e-4,
    )


@configclass
class UnitreeGo2DreamWaQStairsRunnerCfg(RslRlOnPolicyRunnerCfg):
    """Go2 DreamWaQ 两阶段微调第二步：台阶/楼梯攻坚专项微调 Runner 配置。"""

    seed = 1
    num_steps_per_env = 24
    max_iterations = 1000
    save_interval = 25
    experiment_name = "unitree/unitree_go2_dreamwaq_stairs_straight_corridor"
    obs_groups = {
        "policy": ["dreamwaq_policy", "dreamwaq_policy_history"],
        "critic": ["dreamwaq_critic"],
    }

    policy = UnitreeGo2DreamWaQActorCriticCfg(
        init_noise_std=1.0,
        noise_std_type="scalar",
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        beta_vae_kl_weight=0.01,
    )

    algorithm = UnitreeGo2DreamWaQAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3.0e-4,              # 阶梯攻坚微调降温学习率 3.0e-4，平滑吸收台阶高频反力，保护 Trot 底模
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        cenet_loss_weight=1.0,
        velocity_loss_weight=3.0,          # 注入速度头损失权重补丁 3.0，彻底根除符号倒挂
        reconstruction_loss_weight=1.0,
        kl_loss_weight=1.0,
        kl_warmup_iterations=500,          # 阶梯微调 KL 预热
        vae_learning_rate=3.0e-4,
    )


@configclass
class UnitreeGo2DreamWaQMixedRunnerCfg(RslRlOnPolicyRunnerCfg):
    """Go2 DreamWaQ 第三阶段：平地+楼梯混合长廊多任务融合微调 Runner 配置。"""

    seed = 1
    num_steps_per_env = 24
    max_iterations = 500
    save_interval = 25
    experiment_name = "unitree/unitree_go2_dreamwaq_mixed_stage3"
    obs_groups = {
        "policy": ["dreamwaq_policy", "dreamwaq_policy_history"],
        "critic": ["dreamwaq_critic"],
    }

    policy = UnitreeGo2DreamWaQActorCriticCfg(
        init_noise_std=0.8,
        noise_std_type="scalar",
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        beta_vae_kl_weight=0.01,
    )

    algorithm = UnitreeGo2DreamWaQAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.008,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=2.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        cenet_loss_weight=1.0,
        velocity_loss_weight=3.0,
        reconstruction_loss_weight=1.0,
        kl_loss_weight=1.0,
        kl_warmup_iterations=300,
        vae_learning_rate=2.0e-4,
    )


@configclass
class UnitreeGo2DreamWaQSoftLandingRunnerCfg(RslRlOnPolicyRunnerCfg):
    """Go2 DreamWaQ 第四阶段：足端软着陆阻尼微调 Runner 配置 (压进 <10mm 黄金线)。"""

    seed = 1
    num_steps_per_env = 24
    max_iterations = 300
    save_interval = 25
    experiment_name = "unitree/unitree_go2_dreamwaq_softlanding_stage4"
    obs_groups = {
        "policy": ["dreamwaq_policy", "dreamwaq_policy_history"],
        "critic": ["dreamwaq_critic"],
    }

    policy = UnitreeGo2DreamWaQActorCriticCfg(
        init_noise_std=0.6,
        noise_std_type="scalar",
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        beta_vae_kl_weight=0.01,
    )

    algorithm = UnitreeGo2DreamWaQAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-4,              # 精准微调降温学习率 1.0e-4
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        cenet_loss_weight=1.0,
        velocity_loss_weight=3.0,
        reconstruction_loss_weight=1.0,
        kl_loss_weight=1.0,
        kl_warmup_iterations=300,
        vae_learning_rate=1.0e-4,
    )


@configclass
class UnitreeGo2DreamWaQDecoupledRunnerCfg(UnitreeGo2DreamWaQSoftLandingRunnerCfg):
    """Go2 DreamWaQ 第五阶段：解耦平地俯仰刚性锁与双向限幅足端净空 Runner 配置。"""

    experiment_name = "unitree/unitree_go2_dreamwaq_decoupled_stage5"
    max_iterations = 300
    save_interval = 25


@configclass
class UnitreeGo2DreamWaQSurgeryRunnerCfg(UnitreeGo2DreamWaQDecoupledRunnerCfg):
    """Go2 DreamWaQ 第六阶段：全域等变镜像对称与前肢双向对中硬手术 Runner 配置。"""

    experiment_name = "unitree/unitree_go2_dreamwaq_surgery_stage6"
    max_iterations = 150
    save_interval = 25


@configclass
class UnitreeGo2DreamWaQEquivariantStage7RunnerCfg(UnitreeGo2DreamWaQSurgeryRunnerCfg):
    """Go2 DreamWaQ 第七阶段：CENet 左右镜像等变 + 对角功耗方差硬约束大一统终极对齐 Runner。"""

    experiment_name = "unitree/unitree_go2_dreamwaq_equivariant_stage7"
    max_iterations = 200
    save_interval = 25


@configclass
class UnitreeGo2DreamWaQExtremeStairs15cmRunnerCfg(UnitreeGo2DreamWaQEquivariantStage7RunnerCfg):
    """Go2 DreamWaQ 第八阶段：15cm 极限高台阶自适应提膝攻坚 Runner。"""

    experiment_name = "unitree/unitree_go2_dreamwaq_extreme15cm_stage8"
    max_iterations = 150
    save_interval = 25


@configclass
class UnitreeGo2DreamWaQSmoothStairsStage9RunnerCfg(UnitreeGo2DreamWaQExtremeStairs15cmRunnerCfg):
    """Go2 DreamWaQ 第九阶段：终极平滑与激光级直行大一统手术 Runner 配置。"""

    experiment_name = "unitree/unitree_go2_dreamwaq_smooth_stage9"
    max_iterations = 150
    save_interval = 25


@configclass
class UnitreeGo2DreamWaQOptimalStage10RunnerCfg(UnitreeGo2DreamWaQSmoothStairsStage9RunnerCfg):
    """Go2 DreamWaQ 第十阶段：0 网络修改 · 纯物理与等变 Loss 级最优解 Runner 配置。"""

    experiment_name = "unitree/unitree_go2_dreamwaq_optimal_stage10"
    max_iterations = 150
    save_interval = 25


@configclass
class UnitreeGo2DreamWaQStage11RunnerCfg(UnitreeGo2DreamWaQOptimalStage10RunnerCfg):
    """Go2 DreamWaQ 第十一阶段：黄金基准手术加训 Runner 配置 (500 轮)。"""

    experiment_name = "unitree/unitree_go2_dreamwaq_gold_stage11"
    max_iterations = 500
    save_interval = 25


@configclass
class UnitreeGo2DreamWaQStage13FlatPurifyRunnerCfg(UnitreeGo2DreamWaQFlatRunnerCfg):
    """Go2 DreamWaQ 第十三阶段：纯平地镜像等变洗髓 Runner 配置 (200 轮)。"""

    experiment_name = "unitree/unitree_go2_dreamwaq_flat_purify_stage13"
    max_iterations = 200
    save_interval = 25


@configclass
class UnitreeGo2DreamWaQStage14UnifiedRunnerCfg(UnitreeGo2DreamWaQOptimalStage10RunnerCfg):
    """Go2 DreamWaQ 第十四阶段：大一统黄金融合加训 Runner 配置 (300 轮)。"""

    experiment_name = "unitree/unitree_go2_dreamwaq_unified_stage14"
    max_iterations = 300
    save_interval = 25


__all__ = [
    "UnitreeGo2DreamWaQActorCriticCfg",
    "UnitreeGo2DreamWaQAlgorithmCfg",
    "UnitreeGo2DreamWaQRoughRunnerCfg",
    "UnitreeGo2DreamWaQFlatRunnerCfg",
    "UnitreeGo2DreamWaQStairsRunnerCfg",
    "UnitreeGo2DreamWaQMixedRunnerCfg",
    "UnitreeGo2DreamWaQSoftLandingRunnerCfg",
    "UnitreeGo2DreamWaQDecoupledRunnerCfg",
    "UnitreeGo2DreamWaQSurgeryRunnerCfg",
    "UnitreeGo2DreamWaQEquivariantStage7RunnerCfg",
    "UnitreeGo2DreamWaQExtremeStairs15cmRunnerCfg",
    "UnitreeGo2DreamWaQSmoothStairsStage9RunnerCfg",
    "UnitreeGo2DreamWaQOptimalStage10RunnerCfg",
    "UnitreeGo2DreamWaQStage11RunnerCfg",
    "UnitreeGo2DreamWaQStage13FlatPurifyRunnerCfg",
    "UnitreeGo2DreamWaQStage14UnifiedRunnerCfg",
]




