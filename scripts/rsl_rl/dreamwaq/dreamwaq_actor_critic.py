"""DreamWaQ actor-critic for RSL-RL.
This module implements the stage-3 policy network only. PPO integration with an
extra CENet loss is intentionally left to the stage-4 algorithm.
这份实现是 DreamWaQ 专用 ActorCritic，不影响普通 PPO、teacher 或 student。
它负责三件事：
1. 从 observation groups 里取当前 proprio、history、critic obs。
2. 用 CENet(history) 得到 estimated velocity 和 latent。
3. 把 estimated velocity + latent + current proprio 拼给 actor。
"""

from __future__ import annotations

from typing import Any, NoReturn

import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import TensorDict
from torch.distributions import Normal

try:
    from dreamwaq_modules import CENetForwardOutput, DreamWaQCENet
except ImportError:  # pragma: no cover - useful when imported as scripts.rsl_rl.*
    from scripts.rsl_rl.dreamwaq_modules import CENetForwardOutput, DreamWaQCENet

from rsl_rl.networks import EmpiricalNormalization, MLP


class DreamWaQActorCritic(nn.Module):
    """Single-stage asymmetric DreamWaQ actor-critic.
    Actor path:
        dreamwaq_policy + CENet(dreamwaq_policy_history) -> actor MLP
    Critic path:
        dreamwaq_critic -> critic MLP
    The actor never reads ``dreamwaq_critic`` or any other privileged group.
    """

    is_recurrent: bool = False

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        num_actions: int,
        actor_obs_normalization: bool = False,
        critic_obs_normalization: bool = False,
        actor_hidden_dims: tuple[int] | list[int] = (512, 256, 128),
        critic_hidden_dims: tuple[int] | list[int] = (512, 256, 128),
        activation: str = "elu",
        init_noise_std: float = 1.0,
        noise_std_type: str = "scalar",
        state_dependent_std: bool = False,
        history_length: int = 5,
        proprio_obs_dim: int = -1,
        estimated_velocity_dim: int = 3,
        context_latent_dim: int = 16,
        cenet_encoder_hidden_dims: tuple[int] | list[int] = (128, 64),
        cenet_decoder_hidden_dims: tuple[int] | list[int] = (128, 128),
        beta_vae_kl_weight: float = 1.0,
        use_cenet: bool = True,
        history_is_term_major: bool = True,
        history_term_dims: tuple[int, ...] | list[int] | None = None,
        detach_cenet_actor_input: bool = True,
        reconstruct_next_observation: bool = True,
        policy_group_name: str = "dreamwaq_policy",
        history_group_name: str = "dreamwaq_policy_history",
        reconstruction_target_group_name: str = "dreamwaq_reconstruction_target",
        critic_group_name: str = "dreamwaq_critic",
        **kwargs: dict[str, Any],
    ) -> None:
        if kwargs:
            print("DreamWaQActorCritic.__init__ ignored unexpected arguments: " + str(list(kwargs.keys())))
        super().__init__()

        self.obs_groups = obs_groups
        self.policy_group_name = policy_group_name
        self.history_group_name = history_group_name
        self.reconstruction_target_group_name = reconstruction_target_group_name
        self.critic_group_name = critic_group_name
        self.cached_velocity_group_name = "dreamwaq_cached_velocity"
        self.cached_latent_group_name = "dreamwaq_cached_latent"
        self.history_length = int(history_length)
        self.estimated_velocity_dim = int(estimated_velocity_dim)
        self.context_latent_dim = int(context_latent_dim)
        self.beta_vae_kl_weight = float(beta_vae_kl_weight)
        self.use_cenet = bool(use_cenet)
        self.history_is_term_major = bool(history_is_term_major)
        self.detach_cenet_actor_input = bool(detach_cenet_actor_input)
        self.reconstruct_next_observation = bool(reconstruct_next_observation)
        self.state_dependent_std = state_dependent_std
        self.use_adaboot = False
        # Probability of feeding the actor CENet's estimated velocity. When
        # AdaBoot is disabled this stays at 1.0, matching the plain DreamWaQ path.
        self.adaboot_probability = 1.0

        # 这里直接从 IsaacLab/RSL-RL 传进来的第一批 obs 推断维度，避免手填维度和实际观测不一致。
        policy_obs = self._require_vector_group(obs, self.policy_group_name)
        history_obs = self._require_vector_group(obs, self.history_group_name)
        critic_obs = self._require_vector_group(obs, self.critic_group_name)

        inferred_proprio_dim = int(policy_obs.shape[-1])
        if proprio_obs_dim > 0 and int(proprio_obs_dim) != inferred_proprio_dim:
            raise ValueError(
                f"Configured proprio_obs_dim={proprio_obs_dim}, but group '{self.policy_group_name}' "
                f"has dim {inferred_proprio_dim}."
            )
        self.proprio_obs_dim = inferred_proprio_dim
        self.history_obs_dim = int(history_obs.shape[-1])

        # DreamWaQ history 应该是 H 个 proprio 帧拼起来，因此 history_dim 必须能整除 proprio_dim。
        if self.history_obs_dim % self.proprio_obs_dim != 0:
            raise ValueError(
                f"History dim {self.history_obs_dim} is not a multiple of proprio dim {self.proprio_obs_dim}."
            )
        inferred_history_length = self.history_obs_dim // self.proprio_obs_dim
        if self.history_length > 0 and inferred_history_length != self.history_length:
            raise ValueError(
                f"Configured history_length={self.history_length}, but history group implies "
                f"{inferred_history_length}."
            )
        self.history_length = inferred_history_length
        self.history_term_dims = self._resolve_history_term_dims(history_term_dims, num_actions)

        if self.estimated_velocity_dim <= 0:
            raise ValueError(f"estimated_velocity_dim must be positive, got {self.estimated_velocity_dim}.")
        if self.context_latent_dim <= 0:
            raise ValueError(f"context_latent_dim must be positive, got {self.context_latent_dim}.")

        self.critic_obs_dim = self._count_group_dim(obs, obs_groups["critic"])
        self.reconstruction_target_dim = self._infer_reconstruction_target_dim(obs)
        self._check_velocity_target_layout(critic_obs)

        # CENet 只吃 history，不吃当前 privileged critic obs。critic obs 只用于 value 和速度监督目标。
        self.cenet = DreamWaQCENet(
            history_dim=self.history_obs_dim,
            reconstruction_dim=self.reconstruction_target_dim,
            estimated_velocity_dim=self.estimated_velocity_dim,
            latent_dim=self.context_latent_dim,
            encoder_hidden_dims=list(cenet_encoder_hidden_dims),
            decoder_hidden_dims=list(cenet_decoder_hidden_dims),
            activation=activation,
        )

        # Actor 的真实输入不是 RSL-RL 的 raw policy group，而是：
        # estimated base velocity + CENet latent + 当前 proprioception。
        actor_input_dim = self.estimated_velocity_dim + self.context_latent_dim + self.proprio_obs_dim
        self.actor_input_dim = actor_input_dim
        self.num_actions = int(num_actions)
        if self.state_dependent_std:
            self.actor = MLP(actor_input_dim, [2, num_actions], list(actor_hidden_dims), activation)
        else:
            self.actor = MLP(actor_input_dim, num_actions, list(actor_hidden_dims), activation)
        self.critic = MLP(self.critic_obs_dim, 1, list(critic_hidden_dims), activation)

        self.actor_obs_normalization = actor_obs_normalization
        self.critic_obs_normalization = critic_obs_normalization
        # normalizer 作用在 actor/critic 实际吃进去的张量上，而不是原始 observation group 上。
        self.actor_obs_normalizer = (
            EmpiricalNormalization(actor_input_dim) if actor_obs_normalization else nn.Identity()
        )
        self.critic_obs_normalizer = (
            EmpiricalNormalization(self.critic_obs_dim) if critic_obs_normalization else nn.Identity()
        )

        self.noise_std_type = noise_std_type
        if self.state_dependent_std:
            torch.nn.init.zeros_(self.actor[-2].weight[num_actions:])
            if self.noise_std_type == "scalar":
                torch.nn.init.constant_(self.actor[-2].bias[num_actions:], init_noise_std)
            elif self.noise_std_type == "log":
                torch.nn.init.constant_(
                    self.actor[-2].bias[num_actions:], torch.log(torch.tensor(init_noise_std + 1.0e-7))
                )
            else:
                raise ValueError(f"Unknown noise_std_type={self.noise_std_type!r}.")
        else:
            if self.noise_std_type == "scalar":
                self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
            elif self.noise_std_type == "log":
                self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
            else:
                raise ValueError(f"Unknown noise_std_type={self.noise_std_type!r}.")

        self.distribution: Normal | None = None
        self.last_cenet_outputs: CENetForwardOutput | None = None
        Normal.set_default_validate_args(False)

        # 构建 45 维单帧与 225 维时序历史的严格双向镜像置换与符号掩码 (用于 CENet 左右镜像等变损失)
        single_frame_perm = [
            0, 1, 2,        # ang_vel: roll, pitch, yaw
            3, 4, 5,        # gravity: gx, gy, gz
            6, 7, 8,        # cmd: vx, vy, wz
            12, 13, 14,     # FL_jpos <- FR (hip, thigh, calf)
            9, 10, 11,      # FR_jpos <- FL
            18, 19, 20,     # RL_jpos <- RR
            15, 16, 17,     # RR_jpos <- RL
            24, 25, 26,     # FL_jvel <- FR
            21, 22, 23,     # FR_jvel <- FL
            30, 31, 32,     # RL_jvel <- RR
            27, 28, 29,     # RR_jvel <- RL
            36, 37, 38,     # FL_act <- FR
            33, 34, 35,     # FR_act <- FL
            42, 43, 44,     # RL_act <- RR
            39, 40, 41,     # RR_act <- RL
        ]
        single_frame_sign = [
            -1.0, 1.0, -1.0,  # ang_vel: roll(-), pitch(+), yaw(-)
            1.0, -1.0, 1.0,   # gravity: gx(+), gy(-), gz(+)
            1.0, -1.0, -1.0,  # cmd: vx(+), vy(-), wz(-)
            -1.0, 1.0, 1.0,   # FL_jpos: hip(-), thigh(+), calf(+)
            -1.0, 1.0, 1.0,   # FR_jpos: hip(-), thigh(+), calf(+)
            -1.0, 1.0, 1.0,   # RL_jpos
            -1.0, 1.0, 1.0,   # RR_jpos
            -1.0, 1.0, 1.0,   # FL_jvel
            -1.0, 1.0, 1.0,   # FR_jvel
            -1.0, 1.0, 1.0,   # RL_jvel
            -1.0, 1.0, 1.0,   # RR_jvel
            -1.0, 1.0, 1.0,   # FL_act
            -1.0, 1.0, 1.0,   # FR_act
            -1.0, 1.0, 1.0,   # RL_act
            -1.0, 1.0, 1.0,   # RR_act
        ]
        full_perm = [p + 45 * t for t in range(self.history_length) for p in single_frame_perm]
        full_sign = single_frame_sign * self.history_length
        self.register_buffer("history_mirror_perm", torch.tensor(full_perm, dtype=torch.long), persistent=False)
        self.register_buffer("history_mirror_sign", torch.tensor(full_sign, dtype=torch.float32), persistent=False)
        self.register_buffer("proprio_mirror_perm", torch.tensor(single_frame_perm, dtype=torch.long), persistent=False)
        self.register_buffer("proprio_mirror_sign", torch.tensor(single_frame_sign, dtype=torch.float32), persistent=False)

        print("DreamWaQ CENet:", self.cenet)
        print("[DreamWaQ CENet Debug]")
        print(f"history_len = {self.history_length}")
        print(f"proprio_obs_dim = {self.proprio_obs_dim}")
        print(f"history_obs_dim = {self.history_obs_dim}")
        print(f"cenet_input_dim = {self.history_obs_dim}")
        print(f"latent_dim = {self.context_latent_dim}")
        print(f"velocity_hat_dim = {self.estimated_velocity_dim}")
        print(f"actor_obs_dim = {self.proprio_obs_dim}")
        print(f"actor_input_dim = {actor_input_dim}")
        print(f"critic_obs_dim = {self.critic_obs_dim}")
        print(f"action_dim = {num_actions}")
        print(f"reconstruction_target_dim = {self.reconstruction_target_dim}")
        print(f"history_is_term_major = {self.history_is_term_major}")
        print(f"use_cenet = {self.use_cenet}")
        print(f"DreamWaQ actor MLP: {self.actor}")
        print(f"DreamWaQ critic MLP: {self.critic}")

    def reset(self, dones: torch.Tensor | None = None) -> None:
        pass

    def forward(self) -> NoReturn:
        raise NotImplementedError

    @property
    def action_mean(self) -> torch.Tensor:
        return self.distribution.mean

    @property
    def action_std(self) -> torch.Tensor:
        return self.distribution.stddev

    @property
    def entropy(self) -> torch.Tensor:
        return self.distribution.entropy().sum(dim=-1)

    @staticmethod
    def _count_group_dim(obs: TensorDict, groups: list[str]) -> int:
        obs_dim = 0
        for obs_group in groups:
            tensor = DreamWaQActorCritic._require_vector_group(obs, obs_group)
            obs_dim += int(tensor.shape[-1])
        return obs_dim

    @staticmethod
    def _require_vector_group(obs: TensorDict, group_name: str) -> torch.Tensor:
        if group_name not in obs.keys():
            raise KeyError(f"Missing observation group '{group_name}'. Available groups: {list(obs.keys())}.")
        tensor = obs[group_name]
        if len(tensor.shape) != 2:
            raise ValueError(f"Observation group '{group_name}' must be [batch, dim], got {tuple(tensor.shape)}.")
        return tensor

    def _infer_reconstruction_target_dim(self, obs: TensorDict) -> int:
        if self.reconstruction_target_group_name not in obs.keys():
            return self.proprio_obs_dim
        target = self._require_vector_group(obs, self.reconstruction_target_group_name)
        return int(target.shape[-1])

    def _check_velocity_target_layout(self, critic_obs: torch.Tensor) -> None:
        min_dim = self.proprio_obs_dim + self.estimated_velocity_dim
        if critic_obs.shape[-1] < min_dim:
            raise ValueError(
                f"Critic group '{self.critic_group_name}' must contain clean proprio plus base_lin_vel. "
                f"Need at least {min_dim} dims, got {critic_obs.shape[-1]}."
            )

    def _resolve_history_term_dims(
        self,
        history_term_dims: tuple[int, ...] | list[int] | None,
        num_actions: int,
    ) -> tuple[int, ...]:
        if history_term_dims is not None:
            term_dims = tuple(int(dim) for dim in history_term_dims)
            if sum(term_dims) != self.proprio_obs_dim:
                raise ValueError(
                    f"history_term_dims sum to {sum(term_dims)}, but proprio_obs_dim is {self.proprio_obs_dim}."
                )
            return term_dims

        standard_dim = 9 + 3 * int(num_actions)
        if self.proprio_obs_dim == standard_dim:
            action_dim = int(num_actions)
            return (3, 3, 3, action_dim, action_dim, action_dim)

        inferred_joint_dim = (self.proprio_obs_dim - 9) // 3
        if self.proprio_obs_dim >= 9 and 9 + 3 * inferred_joint_dim == self.proprio_obs_dim:
            return (3, 3, 3, inferred_joint_dim, inferred_joint_dim, inferred_joint_dim)

        raise ValueError(
            "Could not infer DreamWaQ history term dimensions from "
            f"proprio_obs_dim={self.proprio_obs_dim}. Set policy.history_term_dims explicitly."
        )

    def _term_major_history_to_frame_major(self, history_obs: torch.Tensor) -> torch.Tensor:
        if not self.history_is_term_major:
            return history_obs

        # IsaacLab 的 ObsGroup history 可能是“每个 term 自己拼 H 帧”：
        # [ang_vel_t-H:t, gravity_t-H:t, ..., actions_t-H:t]。
        # CENet 更容易按帧理解，所以这里重排成：
        # [frame_0_all_terms, frame_1_all_terms, ..., frame_H-1_all_terms]。
        term_chunks: list[torch.Tensor] = []
        cursor = 0
        for term_dim in self.history_term_dims:
            chunk_size = self.history_length * term_dim
            term_chunk = history_obs[..., cursor : cursor + chunk_size]
            term_chunks.append(term_chunk.reshape(*history_obs.shape[:-1], self.history_length, term_dim))
            cursor += chunk_size
        if cursor != history_obs.shape[-1]:
            raise ValueError(f"History cursor ended at {cursor}, expected {history_obs.shape[-1]}.")

        frame_major = torch.cat(term_chunks, dim=-1)
        return frame_major.reshape(*history_obs.shape[:-1], self.history_obs_dim)

    def _concat_groups(self, obs: TensorDict, group_key: str) -> torch.Tensor:
        return torch.cat([self._require_vector_group(obs, group) for group in self.obs_groups[group_key]], dim=-1)

    def get_current_proprio_obs(self, obs: TensorDict) -> torch.Tensor:
        """Return the actor's current proprioceptive frame only."""

        return self._require_vector_group(obs, self.policy_group_name)

    def get_history_obs(self, obs: TensorDict) -> torch.Tensor:
        """Return the flattened proprioceptive history used by CENet."""

        history_obs = self._require_vector_group(obs, self.history_group_name)
        return self._term_major_history_to_frame_major(history_obs)

    def get_critic_obs(self, obs: TensorDict) -> torch.Tensor:
        """Return privileged critic observations."""

        return self._concat_groups(obs, "critic")

    def compute_cenet_outputs(self, obs: TensorDict, sample_latent: bool | None = None) -> CENetForwardOutput:
        """Run CENet on proprioceptive history."""

        if sample_latent is None:
            sample_latent = self.training
        if not self.use_cenet:
            history = self.get_history_obs(obs)
            batch_shape = history.shape[:-1]
            zeros_velocity = history.new_zeros(*batch_shape, self.estimated_velocity_dim)
            zeros_latent = history.new_zeros(*batch_shape, self.context_latent_dim)
            zeros_reconstruction = history.new_zeros(*batch_shape, self.reconstruction_target_dim)
            outputs = CENetForwardOutput(
                code=torch.cat((zeros_velocity, zeros_latent), dim=-1),
                estimated_velocity=zeros_velocity,
                latent=zeros_latent,
                velocity_mu=zeros_velocity,
                velocity_logvar=zeros_velocity,
                latent_mu=zeros_latent,
                latent_logvar=zeros_latent,
                reconstruction=zeros_reconstruction,
                encoder_features=history.new_zeros(*batch_shape, 0),
            )
            self.last_cenet_outputs = outputs
            return outputs
        outputs = self.cenet(self.get_history_obs(obs), sample_latent=sample_latent)
        self.last_cenet_outputs = outputs
        return outputs

    def compute_actor_input(
        self, obs: TensorDict, sample_latent: bool | None = None
    ) -> tuple[torch.Tensor, CENetForwardOutput]:
        """Build actor input with deterministic velocity + latent mean + current proprio.

        ``sample_latent`` is kept for API compatibility, but the actor control path
        always uses CENet means so PPO does not see VAE sampling noise.
        """

        current_proprio = self.get_current_proprio_obs(obs)
        # PPO update 阶段可先显式刷新 CENet 表征，再把 detached 的 v/z 写入 batch。
        # 这和参考项目的 no_grad refresh latent 方式一致；没有缓存时仍实时计算 CENet。
        if self.cached_velocity_group_name in obs.keys() and self.cached_latent_group_name in obs.keys():
            actor_velocity = obs[self.cached_velocity_group_name]
            actor_latent = obs[self.cached_latent_group_name]
            if actor_velocity.shape[-1] != self.estimated_velocity_dim:
                raise ValueError(
                    f"Cached velocity dim must be {self.estimated_velocity_dim}, got {actor_velocity.shape[-1]}."
                )
            if actor_latent.shape[-1] != self.context_latent_dim:
                raise ValueError(f"Cached latent dim must be {self.context_latent_dim}, got {actor_latent.shape[-1]}.")
            cenet_outputs = self._build_cached_cenet_outputs(obs, actor_velocity, actor_latent)
        else:
            # Actor 控制路径始终使用确定性 CENet 输出，避免 PPO 输入被 VAE 采样噪声扰动。
            cenet_outputs = self.compute_cenet_outputs(obs, sample_latent=False)
            actor_velocity = cenet_outputs.estimated_velocity
            actor_latent = cenet_outputs.latent_mu
        # AdaBoot 训练技巧：早期可按概率把真实 base_lin_vel 喂给 actor，后期逐步依赖 CENet 估计。
        if self.training and self.use_adaboot:
            velocity_target = self.get_velocity_target(obs)
            if self.adaboot_probability <= 0.0:
                actor_velocity = velocity_target
            elif self.adaboot_probability < 1.0:
                boot_mask = torch.rand(
                    actor_velocity.shape[:-1] + (1,),
                    device=actor_velocity.device,
                    dtype=actor_velocity.dtype,
                ) < self.adaboot_probability
                actor_velocity = torch.where(boot_mask, actor_velocity, velocity_target)
        if self.training and self.detach_cenet_actor_input:
            # PPO actor loss 不直接更新 CENet，CENet 由辅助 loss 的 optimizer 单独更新。
            actor_velocity = actor_velocity.detach()
            actor_latent = actor_latent.detach()
        actor_input = torch.cat(
            (actor_velocity, actor_latent, current_proprio),
            dim=-1,
        )
        return actor_input, cenet_outputs

    def _build_cached_cenet_outputs(
        self, obs: TensorDict, actor_velocity: torch.Tensor, actor_latent: torch.Tensor
    ) -> CENetForwardOutput:
        """Build a lightweight CENet output object around refreshed batch features."""

        history = self.get_history_obs(obs)
        batch_shape = history.shape[:-1]
        zeros_reconstruction = history.new_zeros(*batch_shape, self.reconstruction_target_dim)
        zeros_logvar_velocity = actor_velocity.new_zeros(*actor_velocity.shape)
        zeros_logvar_latent = actor_latent.new_zeros(*actor_latent.shape)
        return CENetForwardOutput(
            code=torch.cat((actor_velocity, actor_latent), dim=-1),
            estimated_velocity=actor_velocity,
            latent=actor_latent,
            velocity_mu=actor_velocity,
            velocity_logvar=zeros_logvar_velocity,
            latent_mu=actor_latent,
            latent_logvar=zeros_logvar_latent,
            reconstruction=zeros_reconstruction,
            encoder_features=history.new_zeros(*batch_shape, 0),
        )

    def refresh_actor_cenet_cache(self, obs: TensorDict) -> None:
        """Refresh deterministic v/z for a training mini-batch without linking PPO gradients to CENet."""

        if not self.use_cenet:
            return
        with torch.no_grad():
            outputs = self.compute_cenet_outputs(obs, sample_latent=False)
            obs[self.cached_velocity_group_name] = outputs.estimated_velocity.detach()
            obs[self.cached_latent_group_name] = outputs.latent_mu.detach()

    def get_actor_obs(self, obs: TensorDict) -> torch.Tensor:
        """Return the encoded actor input expected by RSL-RL."""

        actor_input, _ = self.compute_actor_input(obs)
        return actor_input

    def _update_distribution(self, actor_input: torch.Tensor) -> None:
        if self.state_dependent_std:
            mean_and_std = self.actor(actor_input)
            if self.noise_std_type == "scalar":
                mean, std = torch.unbind(mean_and_std, dim=-2)
            elif self.noise_std_type == "log":
                mean, log_std = torch.unbind(mean_and_std, dim=-2)
                std = torch.exp(log_std)
            else:
                raise ValueError(f"Unknown noise_std_type={self.noise_std_type!r}.")
        else:
            mean = self.actor(actor_input)
            if self.noise_std_type == "scalar":
                std = self.std.expand_as(mean)
            elif self.noise_std_type == "log":
                std = torch.exp(self.log_std).expand_as(mean)
            else:
                raise ValueError(f"Unknown noise_std_type={self.noise_std_type!r}.")
        self.distribution = Normal(mean, std)

    def act(self, obs: TensorDict, **kwargs: dict[str, Any]) -> torch.Tensor:
        actor_input = self.actor_obs_normalizer(self.get_actor_obs(obs))
        self._update_distribution(actor_input)
        return self.distribution.sample()

    def act_inference(self, obs: TensorDict) -> torch.Tensor:
        actor_input, _ = self.compute_actor_input(obs, sample_latent=False)
        actor_input = self.actor_obs_normalizer(actor_input)
        if self.state_dependent_std:
            return self.actor(actor_input)[..., 0, :]
        return self.actor(actor_input)

    def evaluate(self, obs: TensorDict, **kwargs: dict[str, Any]) -> torch.Tensor:
        critic_obs = self.critic_obs_normalizer(self.get_critic_obs(obs))
        return self.critic(critic_obs)

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(actions).sum(dim=-1)

    def update_normalization(self, obs: TensorDict) -> None:
        if self.actor_obs_normalization:
            actor_input, _ = self.compute_actor_input(obs, sample_latent=False)
            self.actor_obs_normalizer.update(actor_input)
        if self.critic_obs_normalization:
            self.critic_obs_normalizer.update(self.get_critic_obs(obs))

    def set_adaboot_enabled(self, enabled: bool) -> None:
        """Enable AdaBoot velocity bootstrapping during training."""

        self.use_adaboot = bool(enabled)
        if not self.use_adaboot:
            self.adaboot_probability = 1.0

    def set_adaboot_probability(self, probability: float) -> None:
        """Set the probability of using estimated velocity in AdaBoot."""

        self.adaboot_probability = float(min(max(probability, 0.0), 1.0))

    def get_velocity_target(self, obs: TensorDict) -> torch.Tensor:
        """Extract base linear velocity from dreamwaq_critic.

        The DreamWaQ critic layout is clean proprio first, followed immediately
        by base linear velocity.
        """

        # 约定 dreamwaq_critic = clean proprio + base_lin_vel + 其它 privileged 信息。
        critic_obs = self._require_vector_group(obs, self.critic_group_name)
        start = self.proprio_obs_dim
        stop = start + self.estimated_velocity_dim
        return critic_obs[..., start:stop]

    def get_reconstruction_target(self, obs: TensorDict, next_obs: TensorDict | None = None) -> torch.Tensor:
        """Return the decoder target.
        The default DreamWaQ training target is the next clean proprioceptive
        observation stored by ``DreamWaQPPO``. Current-frame reconstruction is
        still available by setting ``reconstruct_next_observation=False``.
        """

        source = (
            next_obs
            if (
                self.reconstruct_next_observation
                and next_obs is not None
                and self.reconstruction_target_group_name in next_obs.keys()
            )
            else obs
        )
        # reconstruction target 使用 clean proprioception，避免把 actor observation noise 当成重构目标。
        if self.reconstruction_target_group_name not in source.keys():
            raise KeyError(
                f"Missing reconstruction target group '{self.reconstruction_target_group_name}'. "
                "Enable the DreamWaQ reconstruction observation group before training CENet."
            )
        target = self._require_vector_group(source, self.reconstruction_target_group_name)
        if target.shape[-1] != self.reconstruction_target_dim:
            raise ValueError(
                f"Expected reconstruction target dim {self.reconstruction_target_dim}, got {target.shape[-1]}."
            )
        return target

    def get_cenet_loss(
        self,
        obs: TensorDict,
        next_obs: TensorDict | None = None,
        live_mask: torch.Tensor | None = None,
        velocity_loss_weight: float = 1.0,
        reconstruction_loss_weight: float = 1.0,
        kl_loss_weight: float = 1.0,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute DreamWaQ CENet auxiliary loss.

        ``next_obs`` is supplied by ``DreamWaQPPO`` during training. If omitted,
        the configured reconstruction target in ``obs`` is used as a fallback.
        """

        if not self.use_cenet:
            zero = self.get_current_proprio_obs(obs).new_zeros(())
            return zero, {
                "cenet_loss": 0.0,
                "cenet_velocity_loss": 0.0,
                "cenet_reconstruction_loss": 0.0,
                "cenet_kl_loss": 0.0,
                "cenet_beta_vae_kl_weight": self.beta_vae_kl_weight,
            }

        outputs = self.cenet(self.get_history_obs(obs), sample_latent=True)
        # velocity target 来自 clean critic obs；reconstruction target 来自 clean reconstruction group。
        velocity_target = self.get_velocity_target(obs)
        reconstruction_target = self.get_reconstruction_target(obs, next_obs=next_obs)

        if live_mask is not None:
            valid = live_mask.to(device=outputs.estimated_velocity.device).reshape(-1) > 0.5
            velocity_pred = outputs.estimated_velocity.reshape(valid.numel(), -1)
            velocity_target = velocity_target.reshape(valid.numel(), -1)
            reconstruction_pred = outputs.reconstruction.reshape(valid.numel(), -1)
            reconstruction_target = reconstruction_target.reshape(valid.numel(), -1)
            kl_per_sample = -0.5 * torch.sum(
                1.0 + outputs.latent_logvar - outputs.latent_mu.pow(2) - outputs.latent_logvar.exp(),
                dim=-1,
            ).reshape(valid.numel())
            if valid.any():
                # 参考项目做法：done 样本不参与 next observation 重建/速度/KL 的均值。
                velocity_loss = F.mse_loss(velocity_pred[valid], velocity_target[valid])
                reconstruction_loss = F.mse_loss(reconstruction_pred[valid], reconstruction_target[valid])
                kl_loss = kl_per_sample[valid].mean()
            else:
                zero = outputs.estimated_velocity.sum() * 0.0
                velocity_loss = zero
                reconstruction_loss = zero
                kl_loss = zero
        else:
            velocity_loss = F.mse_loss(outputs.estimated_velocity, velocity_target)
            reconstruction_loss = F.mse_loss(outputs.reconstruction, reconstruction_target)
            kl_loss = DreamWaQCENet.kl_divergence(outputs.latent_mu, outputs.latent_logvar)

        # --- 核心改进: 物理镜像等变自监督损失 (Physical Bilateral Mirror Equivariance Loss) ---
        # 1. 对时序历史本体感知施加严格左右镜像翻转 flip(h_t)
        history_obs = self.get_history_obs(obs)
        history_flipped = history_obs[..., self.history_mirror_perm] * self.history_mirror_sign
        outputs_flipped = self.cenet(history_flipped, sample_latent=False)

        # 2. 显式线速度等变性约束: vx 保持, vy 符号严格反转 (-vy), vz 保持
        # 物理本质: 机器人直行时左右输入近似对称，强约束 \hat{v}_y = -\hat{v}_y = 0，从源头彻底斩断幽灵侧漂偏置
        vel_pred = outputs.estimated_velocity
        vel_expected_flip = torch.stack([vel_pred[..., 0], -vel_pred[..., 1], vel_pred[..., 2]], dim=-1)

        # 3. 干净本体感知重构等变性约束: 翻转输入的重构 与 原始重构的镜像翻转 严格等价
        # Decoder(flip(h_t)) ≡ flip(Decoder(h_t))
        recon_expected_flip = outputs.reconstruction[..., self.proprio_mirror_perm] * self.proprio_mirror_sign

        if live_mask is not None and valid.any():
            vel_equiv_loss = F.mse_loss(
                outputs_flipped.estimated_velocity.reshape(valid.numel(), -1)[valid],
                vel_expected_flip.reshape(valid.numel(), -1)[valid],
            )
            recon_equiv_loss = F.mse_loss(
                outputs_flipped.reconstruction.reshape(valid.numel(), -1)[valid],
                recon_expected_flip.reshape(valid.numel(), -1)[valid],
            )
        else:
            vel_equiv_loss = F.mse_loss(outputs_flipped.estimated_velocity, vel_expected_flip)
            recon_equiv_loss = F.mse_loss(outputs_flipped.reconstruction, recon_expected_flip)

        # 严格加权：速度等变权重 2.0 (重惩侧偏)，重构等变权重 1.0
        equiv_loss = 2.0 * vel_equiv_loss + 1.0 * recon_equiv_loss

        equiv_loss_weight = 1.0
        total_loss = (
            float(velocity_loss_weight) * velocity_loss
            + float(reconstruction_loss_weight) * reconstruction_loss
            + float(kl_loss_weight) * self.beta_vae_kl_weight * kl_loss
            + float(equiv_loss_weight) * equiv_loss
        )
        metrics = {
            "cenet_loss": total_loss.detach().cpu().item(),
            "cenet_velocity_loss": velocity_loss.detach().cpu().item(),
            "cenet_reconstruction_loss": reconstruction_loss.detach().cpu().item(),
            "cenet_kl_loss": kl_loss.detach().cpu().item(),
            "cenet_vel_equiv_loss": vel_equiv_loss.detach().cpu().item(),
            "cenet_recon_equiv_loss": recon_equiv_loss.detach().cpu().item(),
            "cenet_beta_vae_kl_weight": self.beta_vae_kl_weight,
        }
        return total_loss, metrics

    def get_dreamwaq_dimensions(self) -> dict[str, int]:
        """Return dimensions useful for startup sanity checks."""

        return {
            "proprio_dim": self.proprio_obs_dim,
            "history_dim": self.history_obs_dim,
            "history_length": self.history_length,
            "estimated_velocity_dim": self.estimated_velocity_dim,
            "latent_dim": self.context_latent_dim,
            "actor_input_dim": self.actor_input_dim,
            "critic_obs_dim": self.critic_obs_dim,
            "reconstruction_target_dim": self.reconstruction_target_dim,
        }

    def load_state_dict(self, state_dict: dict, strict: bool = False) -> bool:
        super().load_state_dict(state_dict, strict=strict)
        return True


def register_dreamwaq_actor_critic() -> None:
    """Expose DreamWaQActorCritic to RSL-RL's eval-based runner lookup."""

    import rsl_rl.modules as rsl_modules
    import rsl_rl.runners.on_policy_runner as on_policy_runner_module

    rsl_modules.DreamWaQActorCritic = DreamWaQActorCritic
    on_policy_runner_module.DreamWaQActorCritic = DreamWaQActorCritic


register_dreamwaq_modules = register_dreamwaq_actor_critic


__all__ = ["DreamWaQActorCritic", "register_dreamwaq_actor_critic", "register_dreamwaq_modules"]
