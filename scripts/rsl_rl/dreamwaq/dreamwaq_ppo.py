"""DreamWaQ PPO extension for RSL-RL."""

from __future__ import annotations

from collections import deque
from typing import Any, Generator

import torch
import torch.nn as nn
import torch.optim as optim
from tensordict import TensorDict

from rsl_rl.algorithms import PPO


class DreamWaQPPO(PPO):
    """PPO with the DreamWaQ CENet auxiliary loss.

    The PPO loss is kept aligned with the installed RSL-RL implementation.

        PPO and CENet are optimized in two separate steps, matching the
        reference DreamWaQ implementation:

        actor/critic/std <- L_ppo
        CENet            <- L_velocity + L_reconstruction + beta * L_KL

        A next-observation buffer is stored so the default decoder target can be
        the next clean proprioceptive observation.
    AdaBoot updates the probability of feeding the actor estimated body velocity
    from episodic reward variation.
    """

    def __init__(
        self,
        policy,
        *args: Any,
        cenet_loss_weight: float = 1.0,
        velocity_loss_weight: float = 1.0,
        reconstruction_loss_weight: float = 1.0,
        kl_loss_weight: float = 1.0,
        kl_warmup_iterations: int = 0,
        vae_learning_rate: float = 1.0e-3,
        use_adaboot: bool = False,
        adaboot_reward_window: int = 2048,
        adaboot_min_episodes: int = 32,
        adaboot_min_probability: float = 0.0,
        adaboot_max_probability: float = 1.0,
        adaboot_eps: float = 1.0e-6,
        **kwargs: Any,
    ) -> None:
        super().__init__(policy, *args, **kwargs)
        if self.policy.is_recurrent:
            raise ValueError("DreamWaQPPO currently supports feed-forward DreamWaQActorCritic only.")
        if self.rnd is not None:
            raise ValueError("DreamWaQPPO does not support RND in the current baseline.")
        if self.symmetry is not None:
            raise ValueError("DreamWaQPPO does not support RSL-RL symmetry augmentation in the current baseline.")
        if not hasattr(self.policy, "get_cenet_loss"):
            raise TypeError("DreamWaQPPO requires a policy with get_cenet_loss().")

        # PPO 参数和 CENet 参数分开优化，保持和参考 DreamWaQ 的训练方式一致：
        # actor/critic/std 走 PPO loss，CENet 走 velocity/reconstruction/KL 辅助 loss。
        self.rl_parameters = [
            parameter for name, parameter in self.policy.named_parameters() if not name.startswith("cenet.")
        ]
        self.cenet_parameters = list(self.policy.cenet.parameters())
        self.optimizer = optim.Adam(self.rl_parameters, lr=self.learning_rate)
        self.cenet_optimizer = optim.Adam(self.cenet_parameters, lr=float(vae_learning_rate))
        self.vae_learning_rate = float(vae_learning_rate)

        self.cenet_loss_weight = float(cenet_loss_weight)
        self.velocity_loss_weight = float(velocity_loss_weight)
        self.reconstruction_loss_weight = float(reconstruction_loss_weight)
        self.kl_loss_weight = float(kl_loss_weight)
        self.kl_warmup_iterations = max(0, int(kl_warmup_iterations))
        self.update_counter = 0

        self.use_adaboot = bool(use_adaboot)
        self.adaboot_reward_window = max(1, int(adaboot_reward_window))
        self.adaboot_min_episodes = max(1, int(adaboot_min_episodes))
        self.adaboot_min_probability = float(min(max(adaboot_min_probability, 0.0), 1.0))
        self.adaboot_max_probability = float(min(max(adaboot_max_probability, 0.0), 1.0))
        if self.adaboot_min_probability > self.adaboot_max_probability:
            raise ValueError(
                "adaboot_min_probability must be less than or equal to adaboot_max_probability, got "
                f"{self.adaboot_min_probability} > {self.adaboot_max_probability}."
            )
        self.adaboot_eps = max(float(adaboot_eps), 1.0e-12)
        self.adaboot_probability = self.adaboot_min_probability if self.use_adaboot else 1.0
        self.adaboot_episode_reward_sums = deque(maxlen=self.adaboot_reward_window)
        self.current_episode_rewards: torch.Tensor | None = None
        self.last_adaboot_cv = 0.0
        self.last_adaboot_reward_mean = 0.0
        self.last_adaboot_reward_std = 0.0

        self.next_observations: TensorDict | None = None

        # AdaBoot 是可选训练技巧；默认关闭时 actor 始终使用 CENet 的速度估计。
        if hasattr(self.policy, "set_adaboot_enabled"):
            self.policy.set_adaboot_enabled(self.use_adaboot)
        if hasattr(self.policy, "set_adaboot_probability"):
            self.policy.set_adaboot_probability(self.adaboot_probability)

    def _get_effective_kl_loss_weight(self) -> float:
        if self.kl_warmup_iterations <= 0:
            return self.kl_loss_weight
        progress = min(float(self.update_counter + 1) / float(self.kl_warmup_iterations), 1.0)
        return self.kl_loss_weight * progress

    def _set_adaboot_probability(self, probability: float) -> None:
        self.adaboot_probability = float(
            min(max(probability, self.adaboot_min_probability), self.adaboot_max_probability)
        )
        if hasattr(self.policy, "set_adaboot_probability"):
            self.policy.set_adaboot_probability(self.adaboot_probability)

    def _update_adaboot_probability(self) -> None:
        if not self.use_adaboot:
            self.adaboot_probability = 1.0
            if hasattr(self.policy, "set_adaboot_probability"):
                self.policy.set_adaboot_probability(self.adaboot_probability)
            return

        episode_count = len(self.adaboot_episode_reward_sums)
        if episode_count < self.adaboot_min_episodes:
            self.last_adaboot_cv = 0.0
            self.last_adaboot_reward_mean = 0.0
            self.last_adaboot_reward_std = 0.0
            self._set_adaboot_probability(self.adaboot_min_probability)
            return

        rewards = torch.as_tensor(
            list(self.adaboot_episode_reward_sums),
            dtype=torch.float32,
            device=self.device,
        )
        reward_mean = rewards.mean()
        reward_std = rewards.std(unbiased=False)
        cv = reward_std / torch.clamp(reward_mean.abs(), min=self.adaboot_eps)
        probability = 1.0 - torch.tanh(cv)

        self.last_adaboot_cv = cv.detach().cpu().item()
        self.last_adaboot_reward_mean = reward_mean.detach().cpu().item()
        self.last_adaboot_reward_std = reward_std.detach().cpu().item()
        self._set_adaboot_probability(probability.detach().cpu().item())

    def init_storage(
        self,
        training_type: str,
        num_envs: int,
        num_transitions_per_env: int,
        obs: TensorDict,
        actions_shape: tuple[int] | list[int],
        ) -> None:
        super().init_storage(training_type, num_envs, num_transitions_per_env, obs, actions_shape)
        self.current_episode_rewards = torch.zeros(num_envs, device=self.device)
        # RSL-RL 原始 storage 只存当前 obs。DreamWaQ decoder 默认重构 next clean obs，
        # 所以这里额外保存 step 后返回的 next observation。
        self.next_observations = TensorDict(
            {
                key: torch.zeros(num_transitions_per_env, *value.shape, device=self.device)
                for key, value in obs.items()
            },
            batch_size=[num_transitions_per_env, num_envs],
            device=self.device,
        )

    def process_env_step(
        self, obs: TensorDict, rewards: torch.Tensor, dones: torch.Tensor, extras: dict[str, torch.Tensor]
    ) -> None:
        if self.storage is None or self.next_observations is None:
            raise RuntimeError("DreamWaQPPO storage is not initialized.")
        if self.current_episode_rewards is not None:
            rewards_flat = rewards.detach().view(-1)
            dones_flat = dones.detach().view(-1).bool()
            if rewards_flat.numel() != self.current_episode_rewards.numel():
                raise ValueError(
                    "Reward tensor does not match initialized DreamWaQ environment count: "
                    f"{rewards_flat.numel()} vs {self.current_episode_rewards.numel()}."
                )
            if dones_flat.numel() != self.current_episode_rewards.numel():
                raise ValueError(
                    "Done tensor does not match initialized DreamWaQ environment count: "
                    f"{dones_flat.numel()} vs {self.current_episode_rewards.numel()}."
                )
            self.current_episode_rewards += rewards_flat
            if dones_flat.any():
                done_rewards = self.current_episode_rewards[dones_flat].detach().cpu().tolist()
                self.adaboot_episode_reward_sums.extend(float(reward) for reward in done_rewards)
                self.current_episode_rewards[dones_flat] = 0.0
        super().process_env_step(obs, rewards, dones, extras)
        last_step = self.storage.step - 1
        if last_step < 0:
            raise RuntimeError("Rollout storage step was not advanced after process_env_step().")
        # 此处的 obs 是 env.step(action) 之后的新 obs，正好对应上一条 transition 的 next_obs。
        self.next_observations[last_step].copy_(obs)

    def _mini_batch_generator_with_next_obs(self) -> Generator:
        if self.storage is None or self.next_observations is None:
            raise RuntimeError("DreamWaQPPO storage is not initialized.")
        if self.storage.training_type != "rl":
            raise ValueError("DreamWaQPPO only supports reinforcement learning storage.")

        batch_size = self.storage.num_envs * self.storage.num_transitions_per_env
        mini_batch_size = batch_size // self.num_mini_batches
        indices = torch.randperm(self.num_mini_batches * mini_batch_size, requires_grad=False, device=self.device)

        observations = self.storage.observations.flatten(0, 1)
        next_observations = self.next_observations.flatten(0, 1)
        actions = self.storage.actions.flatten(0, 1)
        dones = self.storage.dones.flatten(0, 1)
        values = self.storage.values.flatten(0, 1)
        returns = self.storage.returns.flatten(0, 1)
        old_actions_log_prob = self.storage.actions_log_prob.flatten(0, 1)
        advantages = self.storage.advantages.flatten(0, 1)
        old_mu = self.storage.mu.flatten(0, 1)
        old_sigma = self.storage.sigma.flatten(0, 1)

        for _ in range(self.num_learning_epochs):
            for mini_batch_index in range(self.num_mini_batches):
                start = mini_batch_index * mini_batch_size
                stop = (mini_batch_index + 1) * mini_batch_size
                batch_idx = indices[start:stop]
                yield (
                    observations[batch_idx],
                    next_observations[batch_idx],
                    1.0 - dones[batch_idx].float(),
                    actions[batch_idx],
                    values[batch_idx],
                    advantages[batch_idx],
                    returns[batch_idx],
                    old_actions_log_prob[batch_idx],
                    old_mu[batch_idx],
                    old_sigma[batch_idx],
                )

    def update(self) -> dict[str, float]:
        mean_value_loss = 0.0
        mean_surrogate_loss = 0.0
        mean_entropy = 0.0
        mean_cenet_loss = 0.0
        mean_cenet_velocity_loss = 0.0
        mean_cenet_reconstruction_loss = 0.0
        mean_cenet_kl_loss = 0.0
        effective_kl_loss_weight = self._get_effective_kl_loss_weight()

        for (
            obs_batch,
            next_obs_batch,
            live_batch,
            actions_batch,
            target_values_batch,
            advantages_batch,
            returns_batch,
            old_actions_log_prob_batch,
            old_mu_batch,
            old_sigma_batch,
        ) in self._mini_batch_generator_with_next_obs():
            original_batch_size = obs_batch.batch_size[0]

            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    advantages_batch = (advantages_batch - advantages_batch.mean()) / (advantages_batch.std() + 1.0e-8)

            if self.cenet_loss_weight != 0.0 and getattr(self.policy, "use_cenet", True):
                # 参考项目式顺序：先用辅助 loss 更新 CENet，再 no_grad 刷新 v/z 给 actor 使用。
                cenet_loss, cenet_metrics = self.policy.get_cenet_loss(
                    obs_batch,
                    next_obs=next_obs_batch,
                    live_mask=live_batch,
                    velocity_loss_weight=self.velocity_loss_weight,
                    reconstruction_loss_weight=self.reconstruction_loss_weight,
                    kl_loss_weight=effective_kl_loss_weight,
                )
                self.cenet_optimizer.zero_grad(set_to_none=True)
                (self.cenet_loss_weight * cenet_loss).backward()

                if self.is_multi_gpu:
                    self.reduce_parameters()

                nn.utils.clip_grad_norm_(self.cenet_parameters, self.max_grad_norm)
                self.cenet_optimizer.step()
                self.cenet_optimizer.zero_grad(set_to_none=True)
                if hasattr(self.policy, "refresh_actor_cenet_cache"):
                    self.policy.refresh_actor_cenet_cache(obs_batch)
            else:
                cenet_metrics = {
                    "cenet_loss": 0.0,
                    "cenet_velocity_loss": 0.0,
                    "cenet_reconstruction_loss": 0.0,
                    "cenet_kl_loss": 0.0,
                }

            self.policy.act(obs_batch)
            actions_log_prob_batch = self.policy.get_actions_log_prob(actions_batch)
            value_batch = self.policy.evaluate(obs_batch)
            mu_batch = self.policy.action_mean[:original_batch_size]
            sigma_batch = self.policy.action_std[:original_batch_size]
            entropy_batch = self.policy.entropy[:original_batch_size]

            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
                        + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch))
                        / (2.0 * torch.square(sigma_batch))
                        - 0.5,
                        axis=-1,
                    )
                    kl_mean = torch.mean(kl)

                    if self.is_multi_gpu:
                        torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
                        kl_mean /= self.gpu_world_size

                    if self.gpu_global_rank == 0:
                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1.0e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1.0e-2, self.learning_rate * 1.5)

                    if self.is_multi_gpu:
                        lr_tensor = torch.tensor(self.learning_rate, device=self.device)
                        torch.distributed.broadcast(lr_tensor, src=0)
                        self.learning_rate = lr_tensor.item()

                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                    -self.clip_param, self.clip_param
                )
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            loss = (
                surrogate_loss
                + self.value_loss_coef * value_loss
                - self.entropy_coef * entropy_batch.mean()
            )

            # 第一步：只更新 PPO 相关参数，不让 PPO loss 直接改 CENet。
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()

            if self.is_multi_gpu:
                self.reduce_parameters()

            nn.utils.clip_grad_norm_(self.rl_parameters, self.max_grad_norm)
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy_batch.mean().item()
            mean_cenet_loss += cenet_metrics["cenet_loss"]
            mean_cenet_velocity_loss += cenet_metrics["cenet_velocity_loss"]
            mean_cenet_reconstruction_loss += cenet_metrics["cenet_reconstruction_loss"]
            mean_cenet_kl_loss += cenet_metrics["cenet_kl_loss"]

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates
        mean_cenet_loss /= num_updates
        mean_cenet_velocity_loss /= num_updates
        mean_cenet_reconstruction_loss /= num_updates
        mean_cenet_kl_loss /= num_updates

        self.update_counter += 1
        self._update_adaboot_probability()
        self.storage.clear()

        return {
            "value_function": mean_value_loss,
            "surrogate": mean_surrogate_loss,
            "entropy": mean_entropy,
            "cenet": mean_cenet_loss,
            "cenet_velocity": mean_cenet_velocity_loss,
            "cenet_reconstruction": mean_cenet_reconstruction_loss,
            "cenet_kl": mean_cenet_kl_loss,
            "cenet_kl_weight": effective_kl_loss_weight,
            "cenet_loss_weight": self.cenet_loss_weight,
            "cenet_beta_vae_kl_weight": float(getattr(self.policy, "beta_vae_kl_weight", 1.0)),
            "adaboot_probability": self.adaboot_probability,
            "adaboot_cv": self.last_adaboot_cv,
            "adaboot_reward_mean": self.last_adaboot_reward_mean,
            "adaboot_reward_std": self.last_adaboot_reward_std,
            "adaboot_episode_count": float(len(self.adaboot_episode_reward_sums)),
        }


def register_dreamwaq_ppo() -> None:
    """Expose DreamWaQPPO to RSL-RL's eval-based runner lookup."""

    import rsl_rl.algorithms as rsl_algorithms
    import rsl_rl.runners.on_policy_runner as on_policy_runner_module

    rsl_algorithms.DreamWaQPPO = DreamWaQPPO
    on_policy_runner_module.DreamWaQPPO = DreamWaQPPO


__all__ = ["DreamWaQPPO", "register_dreamwaq_ppo"]
