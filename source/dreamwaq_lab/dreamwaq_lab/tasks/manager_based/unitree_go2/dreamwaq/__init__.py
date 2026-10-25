"""Unitree Go2 DreamWaQ task registrations."""

import gymnasium as gym

from . import agents

TASK_ID = "Unitree-Go2-DreamWaQ-Rough"
PLAY_TASK_ID = "Unitree-Go2-DreamWaQ-Rough-Play"
FLAT_TASK_ID = "Unitree-Go2-DreamWaQ-Flat"
FLAT_PLAY_TASK_ID = "Unitree-Go2-DreamWaQ-Flat-Play"
STAIRS_TASK_ID = "Unitree-Go2-DreamWaQ-Stairs"
STAIRS_PLAY_TASK_ID = "Unitree-Go2-DreamWaQ-Stairs-Play"
MIXED_TASK_ID = "Unitree-Go2-DreamWaQ-Mixed"
MIXED_PLAY_TASK_ID = "Unitree-Go2-DreamWaQ-Mixed-Play"
SOFTLANDING_TASK_ID = "Unitree-Go2-DreamWaQ-SoftLanding"
SOFTLANDING_PLAY_TASK_ID = "Unitree-Go2-DreamWaQ-SoftLanding-Play"
DECOUPLED_TASK_ID = "Unitree-Go2-DreamWaQ-Decoupled"
DECOUPLED_PLAY_TASK_ID = "Unitree-Go2-DreamWaQ-Decoupled-Play"
SURGERY_TASK_ID = "Unitree-Go2-DreamWaQ-Surgery"
SURGERY_PLAY_TASK_ID = "Unitree-Go2-DreamWaQ-Surgery-Play"
EQUIVARIANT_TASK_ID = "Unitree-Go2-DreamWaQ-Equivariant"
EQUIVARIANT_PLAY_TASK_ID = "Unitree-Go2-DreamWaQ-Equivariant-Play"
EXTREME15CM_TASK_ID = "Unitree-Go2-DreamWaQ-Extreme15cm"
EXTREME15CM_PLAY_TASK_ID = "Unitree-Go2-DreamWaQ-Extreme15cm-Play"
SMOOTH_STAGE9_TASK_ID = "Unitree-Go2-DreamWaQ-Smooth-Stage9"
SMOOTH_STAGE9_PLAY_TASK_ID = "Unitree-Go2-DreamWaQ-Smooth-Stage9-Play"
OPTIMAL_STAGE10_TASK_ID = "Unitree-Go2-DreamWaQ-Optimal-Stage10"
OPTIMAL_STAGE10_PLAY_TASK_ID = "Unitree-Go2-DreamWaQ-Optimal-Stage10-Play"
GOLD_STAGE11_TASK_ID = "Unitree-Go2-DreamWaQ-Gold-Stage11"
GOLD_STAGE11_PLAY_TASK_ID = "Unitree-Go2-DreamWaQ-Gold-Stage11-Play"
STAGE13_FLAT_PURIFY_TASK_ID = "Unitree-Go2-DreamWaQ-FlatPurify-Stage13"
STAGE13_FLAT_PURIFY_PLAY_TASK_ID = "Unitree-Go2-DreamWaQ-FlatPurify-Stage13-Play"
STAGE14_UNIFIED_TASK_ID = "Unitree-Go2-DreamWaQ-Unified-Stage14"
STAGE14_UNIFIED_PLAY_TASK_ID = "Unitree-Go2-DreamWaQ-Unified-Stage14-Play"



gym.register(
    id=TASK_ID,
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.dreamwaq_env_cfg:UnitreeGo2DreamWaQRoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_dreamwaq_cfg:UnitreeGo2DreamWaQRoughRunnerCfg",
    },
)

gym.register(
    id=PLAY_TASK_ID,
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.dreamwaq_env_cfg:UnitreeGo2DreamWaQRoughEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_dreamwaq_cfg:UnitreeGo2DreamWaQRoughRunnerCfg",
    },
)

gym.register(
    id=FLAT_TASK_ID,
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.dreamwaq_env_cfg:UnitreeGo2DreamWaQFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_dreamwaq_cfg:UnitreeGo2DreamWaQFlatRunnerCfg",
    },
)

gym.register(
    id=FLAT_PLAY_TASK_ID,
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.dreamwaq_env_cfg:UnitreeGo2DreamWaQFlatEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_dreamwaq_cfg:UnitreeGo2DreamWaQFlatRunnerCfg",
    },
)

gym.register(
    id=STAIRS_TASK_ID,
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.dreamwaq_env_cfg:UnitreeGo2DreamWaQStairsEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_dreamwaq_cfg:UnitreeGo2DreamWaQStairsRunnerCfg",
    },
)

gym.register(
    id=STAIRS_PLAY_TASK_ID,
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.dreamwaq_env_cfg:UnitreeGo2DreamWaQStairsEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_dreamwaq_cfg:UnitreeGo2DreamWaQStairsRunnerCfg",
    },
)

gym.register(
    id=MIXED_TASK_ID,
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.dreamwaq_env_cfg:UnitreeGo2DreamWaQMixedEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_dreamwaq_cfg:UnitreeGo2DreamWaQMixedRunnerCfg",
    },
)

gym.register(
    id=MIXED_PLAY_TASK_ID,
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.dreamwaq_env_cfg:UnitreeGo2DreamWaQMixedEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_dreamwaq_cfg:UnitreeGo2DreamWaQMixedRunnerCfg",
    },
)

gym.register(
    id=SOFTLANDING_TASK_ID,
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.dreamwaq_env_cfg:UnitreeGo2DreamWaQSoftLandingEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_dreamwaq_cfg:UnitreeGo2DreamWaQSoftLandingRunnerCfg",
    },
)

gym.register(
    id=SOFTLANDING_PLAY_TASK_ID,
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.dreamwaq_env_cfg:UnitreeGo2DreamWaQSoftLandingEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_dreamwaq_cfg:UnitreeGo2DreamWaQSoftLandingRunnerCfg",
    },
)

gym.register(
    id=DECOUPLED_TASK_ID,
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.dreamwaq_env_cfg:UnitreeGo2DreamWaQDecoupledEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_dreamwaq_cfg:UnitreeGo2DreamWaQDecoupledRunnerCfg",
    },
)

gym.register(
    id=DECOUPLED_PLAY_TASK_ID,
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.dreamwaq_env_cfg:UnitreeGo2DreamWaQDecoupledEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_dreamwaq_cfg:UnitreeGo2DreamWaQDecoupledRunnerCfg",
    },
)

gym.register(
    id=SURGERY_TASK_ID,
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.dreamwaq_env_cfg:UnitreeGo2DreamWaQSurgeryEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_dreamwaq_cfg:UnitreeGo2DreamWaQSurgeryRunnerCfg",
    },
)

gym.register(
    id=SURGERY_PLAY_TASK_ID,
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.dreamwaq_env_cfg:UnitreeGo2DreamWaQSurgeryEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_dreamwaq_cfg:UnitreeGo2DreamWaQSurgeryRunnerCfg",
    },
)

gym.register(
    id=EQUIVARIANT_TASK_ID,
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.dreamwaq_env_cfg:UnitreeGo2DreamWaQSurgeryEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_dreamwaq_cfg:UnitreeGo2DreamWaQEquivariantStage7RunnerCfg",
    },
)

gym.register(
    id=EQUIVARIANT_PLAY_TASK_ID,
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.dreamwaq_env_cfg:UnitreeGo2DreamWaQSurgeryEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_dreamwaq_cfg:UnitreeGo2DreamWaQEquivariantStage7RunnerCfg",
    },
)

gym.register(
    id=EXTREME15CM_TASK_ID,
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.dreamwaq_env_cfg:UnitreeGo2DreamWaQExtremeStairs15cmEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_dreamwaq_cfg:UnitreeGo2DreamWaQExtremeStairs15cmRunnerCfg",
    },
)

gym.register(
    id=EXTREME15CM_PLAY_TASK_ID,
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.dreamwaq_env_cfg:UnitreeGo2DreamWaQExtremeStairs15cmEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_dreamwaq_cfg:UnitreeGo2DreamWaQExtremeStairs15cmRunnerCfg",
    },
)

gym.register(
    id=SMOOTH_STAGE9_TASK_ID,
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.dreamwaq_env_cfg:UnitreeGo2DreamWaQSmoothStairsStage9EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_dreamwaq_cfg:UnitreeGo2DreamWaQSmoothStairsStage9RunnerCfg",
    },
)

gym.register(
    id=SMOOTH_STAGE9_PLAY_TASK_ID,
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.dreamwaq_env_cfg:UnitreeGo2DreamWaQSmoothStairsStage9EnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_dreamwaq_cfg:UnitreeGo2DreamWaQSmoothStairsStage9RunnerCfg",
    },
)

gym.register(
    id=OPTIMAL_STAGE10_TASK_ID,
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.dreamwaq_env_cfg:UnitreeGo2DreamWaQOptimalStage10EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_dreamwaq_cfg:UnitreeGo2DreamWaQOptimalStage10RunnerCfg",
    },
)

gym.register(
    id=OPTIMAL_STAGE10_PLAY_TASK_ID,
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.dreamwaq_env_cfg:UnitreeGo2DreamWaQOptimalStage10EnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_dreamwaq_cfg:UnitreeGo2DreamWaQOptimalStage10RunnerCfg",
    },
)

gym.register(
    id=GOLD_STAGE11_TASK_ID,
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.dreamwaq_env_cfg:UnitreeGo2DreamWaQStage11EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_dreamwaq_cfg:UnitreeGo2DreamWaQStage11RunnerCfg",
    },
)

gym.register(
    id=GOLD_STAGE11_PLAY_TASK_ID,
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.dreamwaq_env_cfg:UnitreeGo2DreamWaQStage11EnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_dreamwaq_cfg:UnitreeGo2DreamWaQStage11RunnerCfg",
    },
)

gym.register(
    id=STAGE13_FLAT_PURIFY_TASK_ID,
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.dreamwaq_env_cfg:UnitreeGo2DreamWaQStage13FlatPurifyEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_dreamwaq_cfg:UnitreeGo2DreamWaQStage13FlatPurifyRunnerCfg",
    },
)

gym.register(
    id=STAGE13_FLAT_PURIFY_PLAY_TASK_ID,
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.dreamwaq_env_cfg:UnitreeGo2DreamWaQStage13FlatPurifyEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_dreamwaq_cfg:UnitreeGo2DreamWaQStage13FlatPurifyRunnerCfg",
    },
)

gym.register(
    id=STAGE14_UNIFIED_TASK_ID,
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.dreamwaq_env_cfg:UnitreeGo2DreamWaQStage14UnifiedEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_dreamwaq_cfg:UnitreeGo2DreamWaQStage14UnifiedRunnerCfg",
    },
)

gym.register(
    id=STAGE14_UNIFIED_PLAY_TASK_ID,
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.dreamwaq_env_cfg:UnitreeGo2DreamWaQStage14UnifiedEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_dreamwaq_cfg:UnitreeGo2DreamWaQStage14UnifiedRunnerCfg",
    },
)







