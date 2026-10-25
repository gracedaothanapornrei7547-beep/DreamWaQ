from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.envs.mdp import UniformVelocityCommandCfg
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


@configclass
class UniformLevelVelocityCommandCfg(UniformVelocityCommandCfg):
    limit_ranges: UniformVelocityCommandCfg.Ranges = MISSING
    


class WalkTheseWaysCommand(CommandTerm):
    """Command generator for Walk These Ways / Multiplicity of Behavior.

    The command layout follows the reference implementation:
    ``[vx, vy, yaw, body_height, frequency, phase, offset, bound, duration,
    footswing_height, pitch, roll, stance_width, stance_length, aux]``.
    It also maintains the gait phase buffers consumed by WTW observations and
    rewards.
    """

    cfg: "WalkTheseWaysCommandCfg"

    def __init__(self, cfg: "WalkTheseWaysCommandCfg", env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        self.robot: Articulation = env.scene[cfg.asset_name]
        self._gait_names = ("pronk", "trot", "pace", "bound")
        self.command_buf = torch.zeros(self.num_envs, 15, device=self.device)
        self.gait_indices = torch.zeros(self.num_envs, device=self.device)
        self.foot_indices = torch.zeros(self.num_envs, 4, device=self.device)
        self.clock_inputs = torch.zeros(self.num_envs, 4, device=self.device)
        self.desired_contact_states = torch.zeros(self.num_envs, 4, device=self.device)
        self.desired_footswing_height = torch.zeros(self.num_envs, device=self.device)
        self.gait_category = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.is_standing_env = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        self.metrics["error_vel_xy"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_vel_yaw"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["curriculum_tracking_lin_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["curriculum_tracking_ang_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["curriculum_tracking_contacts_force"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["curriculum_tracking_contacts_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["curriculum_area"] = torch.zeros(self.num_envs, device=self.device)
        self._reward_metric_names = (
            "tracking_lin_vel",
            "tracking_ang_vel",
            "tracking_contacts_force",
            "tracking_contacts_vel",
            "body_height",
            "orientation",
            "raibert",
            "feet_clearance",
            "feet_slip",
            "action_smoothness_1",
            "action_smoothness_2",
            "lin_vel_z",
            "ang_vel_xy",
            "torques",
            "dof_vel",
            "dof_acc",
            "dof_pos_limits",
            "collision",
            "pos_sum",
            "neg_sum",
            "penalty_gate",
            "total",
        )
        for metric_name in self._reward_metric_names:
            self.metrics[f"reward/{metric_name}"] = torch.zeros(self.num_envs, device=self.device)

        self._curriculum_reward_sums = torch.zeros(self.num_envs, 4, device=self.device)
        self._curriculum_reward_steps = torch.zeros(self.num_envs, device=self.device)
        self._curriculum_command_bins = torch.full((self.num_envs,), -1, dtype=torch.long, device=self.device)
        self._curriculum_command_categories = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._skip_next_curriculum_accumulation = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._init_curriculum_distribution()

    @property
    def command(self) -> torch.Tensor:
        """WTW command tensor with shape ``(num_envs, 15)``."""

        return self.command_buf

    def __str__(self) -> str:
        return (
            "WalkTheseWaysCommand:\n"
            f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
            f"\tResampling time range: {self.cfg.resampling_time_range}\n"
            f"\tStanding probability: {self.cfg.rel_standing_envs}\n"
            f"\tFixed gait: {self.cfg.fixed_gait}"
        )

    def _update_metrics(self) -> None:
        max_command_time = self.cfg.resampling_time_range[1]
        max_command_step = max_command_time / self._env.step_dt
        self.metrics["error_vel_xy"] += (
            torch.norm(self.command_buf[:, :2] - self.robot.data.root_lin_vel_b[:, :2], dim=-1) / max_command_step
        )
        self.metrics["error_vel_yaw"] += (
            torch.abs(self.command_buf[:, 2] - self.robot.data.root_ang_vel_b[:, 2]) / max_command_step
        )
        self._accumulate_curriculum_rewards()
        self._update_curriculum_metrics()

    def update_reward_metrics(self, values: dict[str, torch.Tensor]) -> None:
        """Accumulate diagnostic-only WTW reward terms for command-manager logging."""

        max_command_time = self.cfg.resampling_time_range[1]
        max_command_step = max_command_time / self._env.step_dt
        for metric_name, metric_value in values.items():
            metric_key = f"reward/{metric_name}"
            if metric_key in self.metrics:
                self.metrics[metric_key] += metric_value.detach() / max_command_step

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        """Update the WTW command curriculum before resetting command windows."""

        if env_ids is None:
            env_ids_tensor = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        else:
            env_ids_tensor = self._as_tensor_ids(env_ids)
        if self.cfg.curriculum.enabled and env_ids_tensor.numel() > 0:
            self._accumulate_curriculum_rewards(env_ids_tensor)
            self._update_curriculum(env_ids_tensor)
            self._skip_next_curriculum_accumulation[env_ids_tensor] = True
        return super().reset(env_ids_tensor)

    def _resample_command(self, env_ids: Sequence[int]) -> None:
        env_ids = self._as_tensor_ids(env_ids)
        if env_ids.numel() == 0:
            return

        self._update_curriculum(env_ids)
        if self.cfg.curriculum.enabled:
            self._sample_curriculum_commands(env_ids)
        else:
            self._sample_uniform_commands(env_ids)

        n = env_ids.numel()
        self.is_standing_env[env_ids] = torch.rand(n, device=self.device) <= self.cfg.rel_standing_envs
        self.command_buf[env_ids, :2] *= (
            torch.norm(self.command_buf[env_ids, :2], dim=1) > self.cfg.zero_velocity_threshold
        ).unsqueeze(1)
        self.command_buf[self.is_standing_env, :3] = 0.0
        self._reset_curriculum_accumulators(env_ids)

    def _update_command(self) -> None:
        self.command_buf[self.is_standing_env, :3] = 0.0
        self._step_contact_targets()

    def _sample_uniform_commands(self, env_ids: torch.Tensor) -> None:
        ranges = self.cfg.ranges
        n = env_ids.numel()
        self.command_buf[env_ids, 0] = self._uniform(n, ranges.lin_vel_x)
        self.command_buf[env_ids, 1] = self._uniform(n, ranges.lin_vel_y)
        self.command_buf[env_ids, 2] = self._uniform(n, ranges.ang_vel_yaw)
        self.command_buf[env_ids, 3] = self._uniform(n, ranges.body_height)
        self.command_buf[env_ids, 4] = self._uniform(n, ranges.gait_frequency)
        self.command_buf[env_ids, 5] = self._uniform(n, ranges.gait_phase)
        self.command_buf[env_ids, 6] = self._uniform(n, ranges.gait_offset)
        self.command_buf[env_ids, 7] = self._uniform(n, ranges.gait_bound)
        self.command_buf[env_ids, 8] = self._uniform(n, ranges.gait_duration)
        self.command_buf[env_ids, 9] = self._uniform(n, ranges.footswing_height)
        self.command_buf[env_ids, 10] = self._uniform(n, ranges.body_pitch)
        self.command_buf[env_ids, 11] = self._uniform(n, ranges.body_roll)
        self.command_buf[env_ids, 12] = self._uniform(n, ranges.stance_width)
        self.command_buf[env_ids, 13] = self._uniform(n, ranges.stance_length)
        self.command_buf[env_ids, 14] = self._uniform(n, ranges.aux_reward_coef)
        self._sample_gait_categories(env_ids)

    def _sample_gait_categories(self, env_ids: torch.Tensor) -> None:
        n = env_ids.numel()
        fixed_gait = self.cfg.fixed_gait
        if fixed_gait is not None:
            category = self._gait_name_to_id(fixed_gait)
            categories = torch.full((n,), category, dtype=torch.long, device=self.device)
        else:
            categories = torch.randint(0, 4, (n,), device=self.device)
        self.gait_category[env_ids] = categories
        self._apply_gait_category_transforms(env_ids, categories)

    def _apply_gait_category_transforms(self, env_ids: torch.Tensor, categories: torch.Tensor) -> None:
        self.gait_category[env_ids] = categories

        for category_id, category_name in enumerate(self._gait_names):
            category_env_ids = env_ids[categories == category_id]
            if category_env_ids.numel() == 0:
                continue
            if category_name == "pronk":
                self.command_buf[category_env_ids, 5] = (self.command_buf[category_env_ids, 5] / 2.0 - 0.25) % 1.0
                self.command_buf[category_env_ids, 6] = (self.command_buf[category_env_ids, 6] / 2.0 - 0.25) % 1.0
                self.command_buf[category_env_ids, 7] = (self.command_buf[category_env_ids, 7] / 2.0 - 0.25) % 1.0
            elif category_name == "trot":
                self.command_buf[category_env_ids, 5] = self.command_buf[category_env_ids, 5] / 2.0 + 0.25
                self.command_buf[category_env_ids, 6] = 0.0
                self.command_buf[category_env_ids, 7] = 0.0
            elif category_name == "pace":
                self.command_buf[category_env_ids, 5] = 0.0
                self.command_buf[category_env_ids, 6] = self.command_buf[category_env_ids, 6] / 2.0 + 0.25
                self.command_buf[category_env_ids, 7] = 0.0
            elif category_name == "bound":
                self.command_buf[category_env_ids, 5] = 0.0
                self.command_buf[category_env_ids, 6] = 0.0
                self.command_buf[category_env_ids, 7] = self.command_buf[category_env_ids, 7] / 2.0 + 0.25

        if self.cfg.binary_phases:
            self.command_buf[env_ids, 5] = (torch.round(2.0 * self.command_buf[env_ids, 5]) / 2.0) % 1.0
            self.command_buf[env_ids, 6] = (torch.round(2.0 * self.command_buf[env_ids, 6]) / 2.0) % 1.0
            self.command_buf[env_ids, 7] = (torch.round(2.0 * self.command_buf[env_ids, 7]) / 2.0) % 1.0

    def _init_curriculum_distribution(self) -> None:
        cfg = self.cfg.curriculum
        self._curriculum_grid = None
        self._curriculum_bin_sizes = None
        self._curriculum_weights = None
        if not cfg.enabled:
            return

        low, high = self._ranges_to_tensors(cfg.limit_ranges)
        num_bins = torch.tensor(cfg.num_bins, dtype=torch.long, device=self.device)
        if num_bins.numel() != 15:
            raise ValueError(f"WTW curriculum num_bins must have 15 entries, got {tuple(cfg.num_bins)}.")
        if torch.any(num_bins <= 0):
            raise ValueError(f"WTW curriculum num_bins must all be positive, got {tuple(cfg.num_bins)}.")

        self._curriculum_low = low
        self._curriculum_high = high
        self._curriculum_bin_sizes = (high - low) / num_bins.to(dtype=torch.float32)
        axes = [
            torch.linspace(
                low[i] + 0.5 * self._curriculum_bin_sizes[i],
                high[i] - 0.5 * self._curriculum_bin_sizes[i],
                int(num_bins[i].item()),
                device=self.device,
            )
            for i in range(15)
        ]
        mesh = torch.meshgrid(*axes, indexing="ij")
        self._curriculum_grid = torch.stack([axis.reshape(-1) for axis in mesh], dim=1)
        self._curriculum_weights = torch.zeros(len(self._gait_names), self._curriculum_grid.shape[0], device=self.device)
        self._curriculum_local_range = torch.tensor(cfg.local_range, dtype=torch.float32, device=self.device)
        if self._curriculum_local_range.numel() != 15:
            raise ValueError(f"WTW curriculum local_range must have 15 entries, got {tuple(cfg.local_range)}.")
        self._set_curriculum_weights_to_ranges(self.cfg.ranges, value=1.0)

    def _sample_curriculum_commands(self, env_ids: torch.Tensor) -> None:
        if self._curriculum_weights is None or self._curriculum_grid is None:
            self._sample_uniform_commands(env_ids)
            return

        n = env_ids.numel()
        if self.cfg.fixed_gait is not None:
            categories = torch.full(
                (n,), self._gait_name_to_id(self.cfg.fixed_gait), dtype=torch.long, device=self.device
            )
        else:
            categories = torch.randint(0, len(self._gait_names), (n,), device=self.device)
        self._curriculum_command_categories[env_ids] = categories

        for category_id in range(len(self._gait_names)):
            category_env_ids = env_ids[categories == category_id]
            if category_env_ids.numel() == 0:
                continue
            weights = self._curriculum_weights[category_id]
            if torch.sum(weights) <= 0.0:
                weights = torch.ones_like(weights)
            bin_ids = torch.multinomial(weights, category_env_ids.numel(), replacement=True)
            centers = self._curriculum_grid[bin_ids]
            low = centers - 0.5 * self._curriculum_bin_sizes
            high = centers + 0.5 * self._curriculum_bin_sizes
            commands = low + torch.rand_like(centers) * (high - low)
            commands = torch.clamp(commands, min=self._curriculum_low, max=self._curriculum_high)
            self.command_buf[category_env_ids, :] = commands
            self._curriculum_command_bins[category_env_ids] = bin_ids

        self._apply_gait_category_transforms(env_ids, categories)

    def _update_curriculum(self, env_ids: torch.Tensor) -> None:
        if not self.cfg.curriculum.enabled or self._curriculum_weights is None:
            return

        valid = (
            (self.command_counter[env_ids] > 0)
            & (self._curriculum_command_bins[env_ids] >= 0)
            & (self._curriculum_reward_steps[env_ids] > 0)
        )
        if not torch.any(valid):
            return

        valid_env_ids = env_ids[valid]
        avg_rewards = self._curriculum_reward_sums[valid_env_ids] / self._curriculum_reward_steps[valid_env_ids].unsqueeze(1)
        thresholds = torch.tensor(
            (
                self.cfg.curriculum.tracking_lin_vel_threshold,
                self.cfg.curriculum.tracking_ang_vel_threshold,
                self.cfg.curriculum.tracking_contacts_force_threshold,
                self.cfg.curriculum.tracking_contacts_vel_threshold,
            ),
            device=self.device,
            dtype=avg_rewards.dtype,
        )
        successes = torch.all(avg_rewards > thresholds, dim=1)
        if not torch.any(successes):
            return

        update_rate = float(self.cfg.curriculum.update_rate)
        success_env_ids = valid_env_ids[successes]
        success_bins = self._curriculum_command_bins[success_env_ids]
        success_categories = self._curriculum_command_categories[success_env_ids]
        for category_id in torch.unique(success_categories):
            category_bins = torch.unique(success_bins[success_categories == category_id])
            category_weights = self._curriculum_weights[int(category_id.item())]
            for bin_id in category_bins:
                local_mask = self._get_curriculum_local_mask(int(bin_id.item()))
                category_weights[local_mask] = torch.clamp(category_weights[local_mask] + update_rate, 0.0, 1.0)

    def _accumulate_curriculum_rewards(self, env_ids: torch.Tensor | None = None) -> None:
        if not self.cfg.curriculum.enabled:
            return
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        if env_ids.numel() == 0:
            return

        skip = self._skip_next_curriculum_accumulation[env_ids]
        active_env_ids = env_ids[~skip]
        if torch.any(skip):
            self._skip_next_curriculum_accumulation[env_ids[skip]] = False
        if active_env_ids.numel() == 0:
            return

        reward_names = (
            "_wtw_curriculum_tracking_lin_vel",
            "_wtw_curriculum_tracking_ang_vel",
            "_wtw_curriculum_tracking_contacts_force",
            "_wtw_curriculum_tracking_contacts_vel",
        )
        rewards = []
        for reward_name in reward_names:
            reward = getattr(self._env, reward_name, None)
            if reward is None:
                return
            rewards.append(reward[active_env_ids])
        self._curriculum_reward_sums[active_env_ids] += torch.stack(rewards, dim=1)
        self._curriculum_reward_steps[active_env_ids] += 1.0

    def _reset_curriculum_accumulators(self, env_ids: torch.Tensor) -> None:
        self._curriculum_reward_sums[env_ids] = 0.0
        self._curriculum_reward_steps[env_ids] = 0.0

    def _update_curriculum_metrics(self) -> None:
        if not self.cfg.curriculum.enabled:
            return
        steps = self._curriculum_reward_steps.clamp(min=1.0).unsqueeze(1)
        avg_rewards = self._curriculum_reward_sums / steps
        self.metrics["curriculum_tracking_lin_vel"][:] = avg_rewards[:, 0]
        self.metrics["curriculum_tracking_ang_vel"][:] = avg_rewards[:, 1]
        self.metrics["curriculum_tracking_contacts_force"][:] = avg_rewards[:, 2]
        self.metrics["curriculum_tracking_contacts_vel"][:] = avg_rewards[:, 3]
        if self._curriculum_weights is not None:
            area = torch.mean((self._curriculum_weights > 0.0).to(dtype=torch.float32))
            self.metrics["curriculum_area"][:] = area

    def _get_curriculum_local_mask(self, bin_id: int) -> torch.Tensor:
        center = self._curriculum_grid[bin_id]
        return torch.all(
            (self._curriculum_grid >= center - self._curriculum_local_range)
            & (self._curriculum_grid <= center + self._curriculum_local_range),
            dim=1,
        )

    def _set_curriculum_weights_to_ranges(self, ranges: "WalkTheseWaysCommandCfg.Ranges", value: float) -> None:
        low, high = self._ranges_to_tensors(ranges)
        mask = torch.all((self._curriculum_grid >= low) & (self._curriculum_grid <= high), dim=1)
        if not torch.any(mask):
            raise ValueError("WTW curriculum initial command ranges produce an empty command domain.")
        self._curriculum_weights[:, mask] = float(value)

    def _ranges_to_tensors(self, ranges: "WalkTheseWaysCommandCfg.Ranges") -> tuple[torch.Tensor, torch.Tensor]:
        low_high = (
            ranges.lin_vel_x,
            ranges.lin_vel_y,
            ranges.ang_vel_yaw,
            ranges.body_height,
            ranges.gait_frequency,
            ranges.gait_phase,
            ranges.gait_offset,
            ranges.gait_bound,
            ranges.gait_duration,
            ranges.footswing_height,
            ranges.body_pitch,
            ranges.body_roll,
            ranges.stance_width,
            ranges.stance_length,
            ranges.aux_reward_coef,
        )
        low = torch.tensor([value[0] for value in low_high], dtype=torch.float32, device=self.device)
        high = torch.tensor([value[1] for value in low_high], dtype=torch.float32, device=self.device)
        if torch.any(high < low):
            raise ValueError(f"WTW command ranges must have high >= low, got {ranges}.")
        return low, high

    def _step_contact_targets(self) -> None:
        frequencies = self.command_buf[:, 4]
        phases = self.command_buf[:, 5]
        offsets = self.command_buf[:, 6]
        bounds = self.command_buf[:, 7]
        durations = self.command_buf[:, 8].clamp(0.05, 0.95)

        self.gait_indices = torch.remainder(self.gait_indices + self._env.step_dt * frequencies, 1.0)

        if self.cfg.pacing_offset:
            raw_foot_indices = (
                self.gait_indices + phases + offsets + bounds,
                self.gait_indices + bounds,
                self.gait_indices + offsets,
                self.gait_indices + phases,
            )
        else:
            raw_foot_indices = (
                self.gait_indices + phases + offsets + bounds,
                self.gait_indices + offsets,
                self.gait_indices + bounds,
                self.gait_indices + phases,
            )

        self.foot_indices = torch.remainder(torch.stack(raw_foot_indices, dim=1), 1.0)
        warped_foot_indices = []
        for raw_indices in raw_foot_indices:
            remainder = torch.remainder(raw_indices, 1.0)
            stance_phase = remainder * (0.5 / durations)
            swing_phase = 0.5 + (remainder - durations) * (0.5 / (1.0 - durations))
            warped_foot_indices.append(torch.where(remainder < durations, stance_phase, swing_phase))

        warped = torch.stack(warped_foot_indices, dim=1)
        self.clock_inputs = torch.sin(2.0 * math.pi * warped)
        self.desired_contact_states = self._smooth_contact_probability(warped)
        self.desired_footswing_height = self.command_buf[:, 9]

    def _smooth_contact_probability(self, warped_foot_indices: torch.Tensor) -> torch.Tensor:
        sigma = float(self.cfg.kappa_gait_probs)
        sqrt_two = math.sqrt(2.0)

        def normal_cdf(x: torch.Tensor) -> torch.Tensor:
            return 0.5 * (1.0 + torch.erf(x / (sigma * sqrt_two)))

        phase = torch.remainder(warped_foot_indices, 1.0)
        return normal_cdf(phase) * (1.0 - normal_cdf(phase - 0.5)) + normal_cdf(phase - 1.0) * (
            1.0 - normal_cdf(phase - 1.5)
        )

    def _uniform(self, count: int, value_range: tuple[float, float]) -> torch.Tensor:
        return torch.empty(count, device=self.device).uniform_(float(value_range[0]), float(value_range[1]))

    def _as_tensor_ids(self, env_ids: Sequence[int]) -> torch.Tensor:
        if isinstance(env_ids, slice):
            return torch.arange(self.num_envs, device=self.device, dtype=torch.long)[env_ids]
        if isinstance(env_ids, torch.Tensor):
            return env_ids.to(device=self.device, dtype=torch.long)
        return torch.as_tensor(env_ids, device=self.device, dtype=torch.long)

    @staticmethod
    def _gait_name_to_id(gait_name: str) -> int:
        gait_ids = {"pronk": 0, "pronking": 0, "trot": 1, "trotting": 1, "pace": 2, "pacing": 2, "bound": 3, "bounding": 3}
        try:
            return gait_ids[gait_name.lower()]
        except KeyError as exc:
            raise ValueError(f"Unknown WTW gait {gait_name!r}. Expected one of {sorted(gait_ids)}.") from exc


@configclass
class WalkTheseWaysCommandCfg(CommandTermCfg):
    """Configuration for :class:`WalkTheseWaysCommand`."""

    class_type: type = WalkTheseWaysCommand

    asset_name: str = MISSING
    rel_standing_envs: float = 0.0
    #zero_velocity_threshold: float = 0.2
    zero_velocity_threshold: float = 0.1
    binary_phases: bool = True
    pacing_offset: bool = True
    fixed_gait: str | None = None
    kappa_gait_probs: float = 0.07

    @configclass
    class Ranges:
        lin_vel_x: tuple[float, float] = (-1.0, 1.0)
        lin_vel_y: tuple[float, float] = (-0.6, 0.6)
        ang_vel_yaw: tuple[float, float] = (-1.0, 1.0)
        body_height: tuple[float, float] = (-0.25, 0.15)
        gait_frequency: tuple[float, float] = (2.0, 4.0)
        gait_phase: tuple[float, float] = (0.0, 1.0)
        gait_offset: tuple[float, float] = (0.0, 1.0)
        gait_bound: tuple[float, float] = (0.0, 1.0)
        gait_duration: tuple[float, float] = (0.5, 0.5)
        footswing_height: tuple[float, float] = (0.03, 0.35)
        body_pitch: tuple[float, float] = (-0.4, 0.4)
        body_roll: tuple[float, float] = (0.0, 0.0)
        stance_width: tuple[float, float] = (0.10, 0.45)
        stance_length: tuple[float, float] = (0.35, 0.45)
        aux_reward_coef: tuple[float, float] = (0.0, 0.01)

    @configclass
    class Curriculum:
        enabled: bool = False
        update_rate: float = 0.2
        tracking_lin_vel_threshold: float = 0.8
        tracking_ang_vel_threshold: float = 0.7
        tracking_contacts_force_threshold: float = 0.90
        tracking_contacts_vel_threshold: float = 0.90
        # Local expansion radius in the 15-D command space. This follows the
        # official WTW values, with velocity/yaw/body-height in metric units and
        # gait phases in normalized cycles.
        local_range: tuple[float, ...] = (
            0.55,
            0.55,
            0.55,
            0.55,
            0.35,
            0.25,
            0.25,
            0.25,
            0.25,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
        )
        # Official WTW uses vx/yaw limits of [-5, 5]. The cfg that enables the
        # curriculum can override these to a LunarQuadruped-safe envelope.
        limit_ranges: "WalkTheseWaysCommandCfg.Ranges" = MISSING
        num_bins: tuple[int, ...] = (
            21,
            1,
            21,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
        )

    ranges: Ranges = Ranges()
    curriculum: Curriculum = Curriculum(limit_ranges=Ranges())
