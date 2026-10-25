"""DreamWaQ 专用导出工具。

IsaacLab/RSL-RL 默认 exporter 会直接导出 ``policy.actor``，这对 DreamWaQ
来说只得到 base policy，输入维度是：

    estimated_velocity(3) + context_latent(16) + current_proprio(45) = 64

真机部署需要的是完整链路：

    current_proprio(45) + proprio_history(225) -> CENet -> actor -> action(12)

本文件只用于训练机/PC 离线导出；Jetson 真机运行时仍然走 C++ 推理后端。
"""

from __future__ import annotations

import copy
import hashlib
import os
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from tensordict import TensorDict


class DreamWaQFullPolicyExporter(nn.Module):
    """可导出的完整 DreamWaQ 推理 wrapper。

    输入保持部署侧最容易构造的两个张量：
    - ``current_obs``: 当前 45 维本体感知观测。
    - ``history_obs``: IsaacLab ``dreamwaq_policy_history`` 同布局的 225 维历史观测。

    输出为 12 维 action。推理时不采样 latent，直接使用 CENet 的均值头，保证动作确定。
    """

    def __init__(self, policy: nn.Module) -> None:
        super().__init__()

        required_attrs = (
            "cenet",
            "actor",
            "actor_obs_normalizer",
            "proprio_obs_dim",
            "history_obs_dim",
            "history_length",
            "history_term_dims",
            "history_is_term_major",
            "state_dependent_std",
        )
        missing = [name for name in required_attrs if not hasattr(policy, name)]
        if missing:
            raise AttributeError(f"Policy is missing DreamWaQ attributes required for export: {missing}")

        # 深拷贝后只保留推理路径，避免导出时修改训练中的 policy 对象状态。
        self.cenet = copy.deepcopy(policy.cenet)
        self.actor = copy.deepcopy(policy.actor)
        self.normalizer = copy.deepcopy(policy.actor_obs_normalizer)

        self.proprio_obs_dim = int(policy.proprio_obs_dim)
        self.history_obs_dim = int(policy.history_obs_dim)
        self.history_length = int(policy.history_length)
        self.history_term_dims = tuple(int(dim) for dim in policy.history_term_dims)
        self.history_is_term_major = bool(policy.history_is_term_major)
        self.state_dependent_std = bool(policy.state_dependent_std)

    def _history_to_cenet_input(self, history_obs: torch.Tensor) -> torch.Tensor:
        if not self.history_is_term_major:
            return history_obs

        # IsaacLab history group 在当前配置中是 term-major：
        # [ang_vel_H, gravity_H, cmd_H, joint_pos_H, joint_vel_H, action_H]。
        # CENet 训练时实际吃 frame-major，所以这里复现 DreamWaQActorCritic 的重排。
        term_chunks: list[torch.Tensor] = []
        cursor = 0
        for term_dim in self.history_term_dims:
            chunk_size = self.history_length * term_dim
            term_chunk = history_obs[..., cursor : cursor + chunk_size]
            term_chunks.append(term_chunk.reshape(history_obs.shape[0], self.history_length, term_dim))
            cursor += chunk_size

        frame_major = torch.cat(term_chunks, dim=-1)
        return frame_major.reshape(history_obs.shape[0], self.history_obs_dim)

    def forward(self, current_obs: torch.Tensor, history_obs: torch.Tensor) -> torch.Tensor:
        history = self._history_to_cenet_input(history_obs)

        # 推理时 sample_latent=False，因此只需要均值头，不需要 logvar/reparameterize/decoder。
        features = self.cenet.encoder(history)
        estimated_velocity = self.cenet.velocity_mu_head(features)
        latent = self.cenet.latent_mu_head(features)

        actor_input = torch.cat((estimated_velocity, latent, current_obs), dim=-1)
        actor_input = self.normalizer(actor_input)
        actions = self.actor(actor_input)
        if self.state_dependent_std:
            return actions[..., 0, :]
        return actions


def verify_full_policy_matches_python(policy: nn.Module, obs: TensorDict, exporter: nn.Module) -> float:
    """比较完整导出 wrapper 和原 Python DreamWaQ act_inference 的输出误差。"""

    current_obs = obs[policy.policy_group_name]
    history_obs = obs[policy.history_group_name]
    with torch.inference_mode():
        expected = policy.act_inference(obs)
        actual = exporter(current_obs, history_obs)
    return float(torch.max(torch.abs(expected - actual)).detach().cpu())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _deploy_bundle_dir(export_dir: Path) -> Path:
    """Return a bundle directory tagged with its training-run directory name."""

    run_name = export_dir.parent.name.strip()
    if not run_name:
        raise ValueError(f"Cannot infer the training-run name from export directory: {export_dir}")
    return export_dir / f"deploy_bundle_{run_name}"


def _write_deploy_bundle(
    export_dir: Path,
    jit_path: Path,
    onnx_path: Path,
    current_obs: torch.Tensor,
    history_obs: torch.Tensor,
    expected_actions: torch.Tensor,
) -> Path | None:
    """Package models, the resolved contract, hashes, and one golden I/O sample."""

    deploy_cfg_path = export_dir.parent / "params" / "deploy.yaml"
    if not deploy_cfg_path.is_file():
        print(f"[WARN] Deployment contract not found; skipped deploy bundle: {deploy_cfg_path}")
        return None

    with deploy_cfg_path.open("r", encoding="utf-8") as stream:
        deploy_cfg = yaml.safe_load(stream)
    if not isinstance(deploy_cfg, dict):
        raise ValueError(f"Deployment contract is not a YAML mapping: {deploy_cfg_path}")

    bundle_dir = _deploy_bundle_dir(export_dir)
    bundle_dir.mkdir(parents=True, exist_ok=True)
    bundle_jit_path = bundle_dir / "policy.pt"
    bundle_onnx_path = bundle_dir / "policy.onnx"
    shutil.copy2(jit_path, bundle_jit_path)
    shutil.copy2(onnx_path, bundle_onnx_path)

    golden_path = bundle_dir / "golden_io.npz"
    np.savez_compressed(
        golden_path,
        current_obs=current_obs.detach().cpu().numpy(),
        history_obs=history_obs.detach().cpu().numpy(),
        expected_actions=expected_actions.detach().cpu().numpy(),
    )

    model_hashes = {
        "torchscript": _sha256(bundle_jit_path),
        "onnx": _sha256(bundle_onnx_path),
    }
    deploy_cfg.setdefault("model", {})["files"] = {
        "torchscript": bundle_jit_path.name,
        "onnx": bundle_onnx_path.name,
    }
    deploy_cfg["model"]["sha256"] = model_hashes
    deploy_cfg.setdefault("rl_sar", {})["model_name"] = bundle_jit_path.name

    bundle_deploy_path = bundle_dir / "deploy.yaml"
    with bundle_deploy_path.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(deploy_cfg, stream, sort_keys=False, allow_unicode=True)

    manifest_paths = (bundle_jit_path, bundle_onnx_path, golden_path, bundle_deploy_path)
    manifest_path = bundle_dir / "sha256.txt"
    manifest_path.write_text(
        "".join(f"{_sha256(path)}  {path.name}\n" for path in manifest_paths),
        encoding="utf-8",
    )
    print(f"[INFO] Wrote self-contained DreamWaQ deploy bundle to: {bundle_dir}")
    return bundle_dir


def export_dreamwaq_full_policy(
    policy: nn.Module,
    path: str | os.PathLike[str],
    filename_prefix: str = "policy",
    example_obs: TensorDict | None = None,
    opset_version: int = 11,
) -> tuple[Path, Path]:
    """导出完整 DreamWaQ TorchScript 和 ONNX 模型。"""

    export_dir = Path(path)
    export_dir.mkdir(parents=True, exist_ok=True)

    was_training = policy.training
    policy.eval()

    device = next(policy.parameters()).device
    exporter = DreamWaQFullPolicyExporter(policy).to(device=device)
    exporter.eval()

    current_obs = torch.zeros(1, exporter.proprio_obs_dim, device=device)
    history_obs = torch.zeros(1, exporter.history_obs_dim, device=device)
    if example_obs is not None:
        current_obs = example_obs[policy.policy_group_name][:1].to(device=device)
        history_obs = example_obs[policy.history_group_name][:1].to(device=device)
        max_error = verify_full_policy_matches_python(policy, example_obs, exporter)
        print(f"[INFO] DreamWaQ full exporter max abs error vs Python act_inference: {max_error:.6e}")

    jit_path = export_dir / f"{filename_prefix}.pt"
    onnx_path = export_dir / f"{filename_prefix}.onnx"

    with torch.inference_mode():
        traced = torch.jit.trace(exporter, (current_obs, history_obs), check_trace=False)
        traced.save(str(jit_path))

        torch.onnx.export(
            exporter,
            (current_obs, history_obs),
            str(onnx_path),
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=["current_obs", "history_obs"],
            output_names=["actions"],
            dynamic_axes={
                "current_obs": {0: "batch"},
                "history_obs": {0: "batch"},
                "actions": {0: "batch"},
            },
        )
        expected_actions = exporter(current_obs, history_obs)

    _write_deploy_bundle(
        export_dir,
        jit_path,
        onnx_path,
        current_obs,
        history_obs,
        expected_actions,
    )

    # 额外保存语义更明确的别名，避免和旧的 actor-only policy.onnx 混淆。
    if filename_prefix == "policy":
        alias_jit_path = export_dir / "dreamwaq_full_policy.pt"
        alias_onnx_path = export_dir / "dreamwaq_full_policy.onnx"
        shutil.copy2(jit_path, alias_jit_path)
        shutil.copy2(onnx_path, alias_onnx_path)
        print(f"[INFO] Also wrote full DreamWaQ TorchScript alias to: {alias_jit_path}")
        print(f"[INFO] Also wrote full DreamWaQ ONNX alias to: {alias_onnx_path}")

    if was_training:
        policy.train()

    print(f"[INFO] Exported full DreamWaQ TorchScript policy to: {jit_path}")
    print(f"[INFO] Exported full DreamWaQ ONNX policy to: {onnx_path}")
    print(
        "[INFO] Full DreamWaQ model inputs: "
        f"current_obs[1,{exporter.proprio_obs_dim}], history_obs[1,{exporter.history_obs_dim}] -> actions[1,*]"
    )
    return jit_path, onnx_path


__all__ = ["DreamWaQFullPolicyExporter", "export_dreamwaq_full_policy"]
