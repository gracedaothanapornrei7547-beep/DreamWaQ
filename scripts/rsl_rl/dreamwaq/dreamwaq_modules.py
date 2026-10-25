"""Neural building blocks for DreamWaQ policies.
本文件对应参考项目里的 ``vae.py``：这里没有单独叫 VAE，而是把
DreamWaQ 的 CENet/VAE 结构命名为 ``DreamWaQCENet``。它只负责把
proprioception history 编码成“显式速度估计 + 隐变量 latent”，并重构
clean proprioception target；actor/critic 和 PPO 训练逻辑在其它文件里。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
import torch.nn as nn

def build_activation(name: str) -> nn.Module:
    """Return the activation module used by the local DreamWaQ networks."""

    activation_map: dict[str, type[nn.Module]] = {
        "elu": nn.ELU,
        "relu": nn.ReLU,
        "selu": nn.SELU,
        "lrelu": nn.LeakyReLU,
        "leakyrelu": nn.LeakyReLU,
        "tanh": nn.Tanh,
        "sigmoid": nn.Sigmoid,
    }
    return activation_map.get(name.lower(), nn.ELU)()


def build_mlp(input_dim: int, hidden_dims: Sequence[int], output_dim: int, activation: str) -> nn.Sequential:
    """Build a simple MLP with a linear output layer."""

    layers: list[nn.Module] = []
    in_dim = input_dim
    for hidden_dim in hidden_dims:
        layers.append(nn.Linear(in_dim, int(hidden_dim)))
        layers.append(build_activation(activation))
        in_dim = int(hidden_dim)
    layers.append(nn.Linear(in_dim, output_dim))
    return nn.Sequential(*layers)


def build_backbone(input_dim: int, hidden_dims: Sequence[int], activation: str) -> tuple[nn.Module, int]:
    """Build an encoder backbone and return its output dimension."""

    hidden_dims = [int(dim) for dim in hidden_dims]
    if not hidden_dims:
        return nn.Identity(), input_dim
    layers: list[nn.Module] = []
    in_dim = input_dim
    for hidden_dim in hidden_dims:
        layers.append(nn.Linear(in_dim, hidden_dim))
        layers.append(build_activation(activation))
        in_dim = hidden_dim
    return nn.Sequential(*layers), hidden_dims[-1]


@dataclass
class CENetForwardOutput:
    """Forward tensors produced by DreamWaQ CENet.

    code = estimated_velocity + latent，是 decoder 使用的上下文信息。
    actor 控制路径应使用 deterministic estimated_velocity + latent_mu。
    reconstruction 用于辅助 loss，不直接输入 actor。
    """

    code: torch.Tensor
    estimated_velocity: torch.Tensor
    latent: torch.Tensor
    velocity_mu: torch.Tensor
    velocity_logvar: torch.Tensor
    latent_mu: torch.Tensor
    latent_logvar: torch.Tensor
    reconstruction: torch.Tensor
    encoder_features: torch.Tensor


class DreamWaQCENet(nn.Module):
    """Context encoder network from proprioceptive history.
    The network follows the DreamWaQ CENet/VAE structure:
    proprio history -> shared encoder -> deterministic velocity estimate
                                |-----> latent mu/logvar -> sampled latent
    deterministic velocity + sampled latent -> decoder
    """

    def __init__(
        self,
        history_dim: int,                     # history_dim = history_length * proprio_obs_dim
        reconstruction_dim: int,              # reconstruction_dim = clean proprio target dim
        estimated_velocity_dim: int = 3,      # estimated_velocity_dim = 3
        latent_dim: int = 16,                 # latent_dim = 16
        encoder_hidden_dims: Sequence[int] = (128, 64),
        decoder_hidden_dims: Sequence[int] = (128, 128),
        activation: str = "elu",
        min_logvar: float = -5.0,
        max_logvar: float = 5.0,
    ) -> None:
        super().__init__()
        if history_dim <= 0:
            raise ValueError(f"history_dim must be positive, got {history_dim}.")
        if reconstruction_dim <= 0:
            raise ValueError(f"reconstruction_dim must be positive, got {reconstruction_dim}.")
        if estimated_velocity_dim <= 0:
            raise ValueError(f"estimated_velocity_dim must be positive, got {estimated_velocity_dim}.")
        if latent_dim <= 0:
            raise ValueError(f"latent_dim must be positive, got {latent_dim}.")

        self.history_dim = int(history_dim)
        self.reconstruction_dim = int(reconstruction_dim)
        self.estimated_velocity_dim = int(estimated_velocity_dim)
        self.latent_dim = int(latent_dim)
        self.min_logvar = float(min_logvar)
        self.max_logvar = float(max_logvar)

        self.activation_name = activation
        # 共享 encoder：只看历史本体感知，不看 critic privileged information。
        self.encoder, encoder_feature_dim = build_backbone(history_dim, encoder_hidden_dims, activation)
        # 显式速度头：学习确定性估计 base linear velocity，部署时 actor 只能用这个估计值。
        self.velocity_mu_head = nn.Linear(encoder_feature_dim, estimated_velocity_dim)
        # 保留旧字段以兼容已有 checkpoint；当前 DreamWaQ 路径不再采样 velocity。
        self.velocity_logvar_head = nn.Sequential(
            nn.Linear(encoder_feature_dim, estimated_velocity_dim),
            nn.Hardtanh(min_val=self.min_logvar, max_val=self.max_logvar),
        )
        # 隐变量头：学习地形/扰动/动力学上下文等不容易直接观测到的信息。
        self.latent_mu_head = nn.Linear(encoder_feature_dim, latent_dim)
        self.latent_logvar_head = nn.Sequential(
            nn.Linear(encoder_feature_dim, latent_dim),
            nn.Hardtanh(min_val=self.min_logvar, max_val=self.max_logvar),
        )
        # decoder 只用于 CENet 辅助训练，目标通常是 clean proprioception。
        self.decoder = build_mlp(
            estimated_velocity_dim + latent_dim,
            decoder_hidden_dims,
            reconstruction_dim,
            activation,
        )

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor, sample: bool) -> torch.Tensor:
        """Sample z with the VAE reparameterization trick, or return mu."""

        if not sample:
            return mu
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    @staticmethod
    def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Mean KL(q(z|x) || N(0, I)) over the batch."""

        kl_per_sample = -0.5 * torch.sum(1.0 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
        return kl_per_sample.mean()

    def forward(self, proprio_history: torch.Tensor, sample_latent: bool = True) -> CENetForwardOutput:
        if proprio_history.shape[-1] != self.history_dim:
            raise ValueError(f"Expected proprio_history dim {self.history_dim}, got {proprio_history.shape[-1]}.")

        # history -> encoder features -> deterministic velocity + latent Gaussian parameters.
        features = self.encoder(proprio_history)
        velocity_mu = self.velocity_mu_head(features)
        # velocity 是监督回归量，不走 VAE 采样；该 logvar 仅作为输出占位以兼容旧接口。
        velocity_logvar = torch.zeros_like(velocity_mu)
        latent_mu = self.latent_mu_head(features)
        latent_logvar = self.latent_logvar_head(features)

        # v_hat 是确定性速度估计；z 在 VAE 训练路径中采样，推理/actor 控制路径使用 latent_mu。
        estimated_velocity = velocity_mu
        latent = self.reparameterize(latent_mu, latent_logvar, sample_latent)
        code = torch.cat((estimated_velocity, latent), dim=-1)
        # reconstruction 参与辅助 loss，帮助 latent 保留历史中与运动状态相关的信息。
        reconstruction = self.decoder(code)
        return CENetForwardOutput(
            code=code,
            estimated_velocity=estimated_velocity,
            latent=latent,
            velocity_mu=velocity_mu,
            velocity_logvar=velocity_logvar,
            latent_mu=latent_mu,
            latent_logvar=latent_logvar,
            reconstruction=reconstruction,
            encoder_features=features,
        )
