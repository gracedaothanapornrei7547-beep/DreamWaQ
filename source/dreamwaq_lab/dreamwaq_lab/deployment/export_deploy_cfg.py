"""Export a resolved policy contract for rl_sar and other deployment runtimes.

Unlike ``params/env.yaml``, this file contains the values resolved by Isaac
Lab's action and observation managers.  It is intentionally limited to
deployment-time data and excludes rewards, privileged critic observations, and
training noise.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Sequence

import yaml


_RL_SAR_OBSERVATION_TYPES = {
    "base_ang_vel": "ang_vel",
    "projected_gravity": "gravity_vec",
    "generated_commands": "commands",
    "joint_pos_rel": "dof_pos",
    "joint_vel_rel": "dof_vel",
    "last_action": "actions",
}


def _python_value(value: Any) -> Any:
    """Convert tensors/config objects to safe YAML primitives."""

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "tolist"):
        return _python_value(value.tolist())
    if isinstance(value, dict):
        return {str(key): _python_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_python_value(item) for item in value]
    if hasattr(value, "to_dict"):
        return _python_value(value.to_dict())
    return str(value)


def _flatten_first_batch(value: Any) -> list[float]:
    value = _python_value(value)
    while isinstance(value, list) and value and isinstance(value[0], list):
        value = value[0]
    if not isinstance(value, list):
        return [float(value)]
    result: list[float] = []
    for item in value:
        if isinstance(item, list):
            result.extend(float(x) for x in item)
        else:
            result.append(float(item))
    return result


def _expanded_vector(value: Any, length: int, *, default: float) -> list[float]:
    if value is None:
        return [default] * length
    values = _flatten_first_batch(value)
    if len(values) == 1:
        return values * length
    if len(values) != length:
        raise ValueError(f"Expected scalar or {length} values, got {len(values)}.")
    return values


def _selected_vector(value: Any, joint_ids: Sequence[int], total_joints: int) -> list[float]:
    values = _flatten_first_batch(value)
    # Check the full asset vector first: an action may select every joint while
    # still permuting them into a policy-specific order.
    if len(values) == total_joints:
        return [values[index] for index in joint_ids]
    if len(values) == len(joint_ids):
        return values
    raise ValueError(
        f"Resolved joint vector has {len(values)} values; expected "
        f"{len(joint_ids)} policy joints or {total_joints} asset joints."
    )


def _nominal_default_joint_pos(
    asset: Any,
    joint_ids: Sequence[int],
    total_joints: int,
) -> list[float]:
    """Return the asset's nominal default joint positions in policy order."""

    return _selected_vector(asset.data.default_joint_pos[0], joint_ids, total_joints)


def _action_offset(
    action_term: Any,
    nominal_default_joint_pos: Sequence[float],
    action_dim: int,
) -> list[float]:
    """Return an action offset that is safe to freeze into a deployment bundle.

    With ``use_default_offset=True``, a joint-position action is defined
    relative to the asset's nominal default joint positions. Runtime domain
    randomization may mutate ``action_term._offset`` per environment, so
    exporting that tensor would freeze one sampled motor-zero bias.
    """

    action_cfg = getattr(action_term, "cfg", None)
    is_joint_position_action = action_term.__class__.__name__.endswith("JointPositionAction")
    if is_joint_position_action and bool(getattr(action_cfg, "use_default_offset", False)):
        values = [float(value) for value in nominal_default_joint_pos]
        if len(values) != action_dim:
            raise ValueError(
                f"Expected {action_dim} nominal default joint positions, got {len(values)}."
            )
        return values
    return _expanded_vector(action_term._offset, action_dim, default=0.0)


def _joint_ids(action_term: Any, total_joints: int) -> list[int]:
    ids = action_term._joint_ids
    if isinstance(ids, slice):
        return list(range(total_joints))[ids]
    ids = _python_value(ids)
    return [int(index) for index in ids]


def _action_clip(action_term: Any, action_dim: int) -> tuple[list[float], list[float]]:
    clip = getattr(action_term, "_clip", None)
    if clip is None:
        return ([-math.inf] * action_dim, [math.inf] * action_dim)
    values = _python_value(clip)
    # Isaac Lab expands action clips to [num_envs, action_dim, 2].  Deployment
    # needs the resolved per-action bounds, which are identical across envs.
    while (
        isinstance(values, list)
        and values
        and isinstance(values[0], list)
        and values[0]
        and isinstance(values[0][0], list)
    ):
        values = values[0]
    if not values:
        return ([-math.inf] * action_dim, [math.inf] * action_dim)
    if len(values) == 2 and not isinstance(values[0], list):
        return ([float(values[0])] * action_dim, [float(values[1])] * action_dim)
    if len(values) != action_dim or any(not isinstance(item, list) or len(item) != 2 for item in values):
        raise ValueError(f"Unsupported resolved action clip shape: {values}")
    return (
        [float(item[0]) for item in values],
        [float(item[1]) for item in values],
    )


def _term_clip(value: Any) -> list[float] | None:
    if value is None:
        return None
    clip = _python_value(value)
    if isinstance(clip, list) and len(clip) == 2:
        return [float(clip[0]), float(clip[1])]
    raise ValueError(f"Unsupported observation clip: {clip}")


def _observation_group(env: Any, group_name: str, *, history_is_term_major: bool) -> dict[str, Any]:
    manager = env.observation_manager
    if group_name not in manager.active_terms:
        raise KeyError(f"Observation group '{group_name}' is not active.")

    names = manager.active_terms[group_name]
    cfgs = manager._group_obs_term_cfgs[group_name]
    group_cfg = getattr(env.cfg.observations, group_name)
    group_history = getattr(group_cfg, "history_length", None) or 1

    terms: list[dict[str, Any]] = []
    current_dim = 0
    for term_name, term_cfg in zip(names, cfgs):
        sample = term_cfg.func(env, **term_cfg.params)
        dim = int(math.prod(sample.shape[1:]))
        current_dim += dim
        scale = _expanded_vector(term_cfg.scale, dim, default=1.0)
        func_name = getattr(term_cfg.func, "__name__", str(term_cfg.func).split(":")[-1])
        terms.append(
            {
                "name": term_name,
                "type": func_name,
                "dim": dim,
                "params": _python_value(term_cfg.params),
                "clip": _term_clip(term_cfg.clip),
                "scale": scale,
            }
        )

    return {
        "current_dim": current_dim,
        "model_dim": current_dim * int(group_history),
        "history_length": int(group_history),
        "history_order": "oldest_to_newest",
        "history_layout": "term_major" if history_is_term_major else "frame_major",
        "terms": terms,
    }


def _command_ranges(env: Any) -> dict[str, list[float]]:
    commands = getattr(env.cfg, "commands", None)
    base_velocity = getattr(commands, "base_velocity", None)
    if base_velocity is None:
        return {}
    ranges = getattr(base_velocity, "limit_ranges", None) or getattr(base_velocity, "ranges", None)
    if ranges is None:
        return {}
    result: dict[str, list[float]] = {}
    for name in ("lin_vel_x", "lin_vel_y", "ang_vel_z"):
        value = getattr(ranges, name, None)
        if value is not None:
            result[name] = [float(value[0]), float(value[1])]
    return result


def _runtime_actuator(
    env: Any,
    asset: Any,
    joint_ids: Sequence[int],
    total_joints: int,
) -> dict[str, Any]:
    actuators = getattr(asset, "actuators", None)
    if isinstance(actuators, dict):
        actuator_items = list(actuators.items())
    else:
        actuator_items = [(str(index), actuator) for index, actuator in enumerate(actuators or [])]
    if not actuator_items:
        return {}
    actuator_name, actuator = actuator_items[0]
    # The training reset event randomizes _effort_y1/_effort_y2 in-place.  It
    # preserves their nominal tensors here; exporting an arbitrary environment
    # instance would otherwise bake a random motor-strength sample into deploy.yaml.
    nominal_motor_strength = getattr(env, "_unitree_motor_strength_defaults", {})
    fields = {
        "velocity_x1": "_velocity_x1",
        "velocity_x2": "_velocity_x2",
        "effort_y1": "_effort_y1",
        "effort_y2": "_effort_y2",
        "friction_static": "_friction_static",
        "friction_dynamic": "_friction_dynamic",
        "activation_velocity": "_activation_vel",
    }
    result: dict[str, Any] = {"type": actuator.__class__.__name__}
    for output_name, attr_name in fields.items():
        value = nominal_motor_strength.get(
            (actuator_name, attr_name),
            getattr(actuator, attr_name, None),
        )
        if value is not None:
            result[output_name] = _selected_vector(value, joint_ids, total_joints)
    return result


def _uniform_scale(term: dict[str, Any], expected_type: str) -> float:
    if term["type"] != expected_type:
        raise ValueError(f"Expected observation type '{expected_type}', got '{term['type']}'.")
    scales = term["scale"]
    if not scales or any(abs(value - scales[0]) > 1.0e-9 for value in scales[1:]):
        raise ValueError(f"rl_sar requires a uniform scale for '{expected_type}': {scales}")
    return float(scales[0])


def _require_identity_scale(term: dict[str, Any]) -> None:
    if any(abs(value - 1.0) > 1.0e-9 for value in term["scale"]):
        raise ValueError(
            f"rl_sar does not have a separate scale field for observation "
            f"'{term['type']}': {term['scale']}"
        )


def _rl_sar_compatibility(
    *,
    robot_name: str,
    control: dict[str, Any],
    robot: dict[str, Any],
    action: dict[str, Any],
    default_dof_pos: Sequence[float],
    groups: dict[str, dict[str, Any]],
    actor_groups: Sequence[str],
    command_ranges: dict[str, list[float]],
    actuator: dict[str, Any],
    model_name: str,
) -> dict[str, Any]:
    current_group = groups[actor_groups[0]]
    legacy_observations: list[str] = []
    terms_by_type: dict[str, dict[str, Any]] = {}
    for term in current_group["terms"]:
        canonical_type = term["type"]
        if canonical_type not in _RL_SAR_OBSERVATION_TYPES:
            raise ValueError(
                f"Observation type '{canonical_type}' cannot be represented by the current rl_sar runtime."
            )
        legacy_observations.append(_RL_SAR_OBSERVATION_TYPES[canonical_type])
        terms_by_type[canonical_type] = term

    required_types = {
        "base_ang_vel",
        "projected_gravity",
        "generated_commands",
        "joint_pos_rel",
        "joint_vel_rel",
        "last_action",
    }
    missing_types = sorted(required_types.difference(terms_by_type))
    if missing_types:
        raise ValueError(f"rl_sar compatibility is missing observation types: {missing_types}")
    _require_identity_scale(terms_by_type["projected_gravity"])
    _require_identity_scale(terms_by_type["last_action"])

    history_indices: list[int] = []
    history_priority = "time"
    history_mode = "concat"
    if len(actor_groups) > 1:
        history_group = groups[actor_groups[1]]
        history_length = int(history_group["history_length"])
        history_indices = list(range(history_length - 1, -1, -1))
        history_priority = "term" if history_group["history_layout"] == "term_major" else "time"
        history_mode = "separate"

    term_clips = [term["clip"] for term in current_group["terms"]]
    if all(clip is None for clip in term_clips):
        clip_obs = math.inf
    else:
        if any(clip is None for clip in term_clips):
            raise ValueError("rl_sar requires one shared clip range for all observations.")
        first_clip = term_clips[0]
        if any(clip != first_clip for clip in term_clips[1:]):
            raise ValueError(f"rl_sar cannot represent per-term observation clips: {term_clips}")
        if abs(first_clip[0] + first_clip[1]) > 1.0e-9:
            raise ValueError(f"rl_sar requires a symmetric observation clip: {first_clip}")
        clip_obs = abs(first_clip[1])

    commands_lower = [command_ranges.get(key, [-math.inf, math.inf])[0] for key in ("lin_vel_x", "lin_vel_y", "ang_vel_z")]
    commands_upper = [command_ranges.get(key, [-math.inf, math.inf])[1] for key in ("lin_vel_x", "lin_vel_y", "ang_vel_z")]

    rl_sar = {
        "model_name": model_name,
        "dt": control["sim_dt"],
        "decimation": control["decimation"],
        "num_observations": current_group["current_dim"],
        "observations": legacy_observations,
        "observations_history": history_indices,
        "observations_history_priority": history_priority,
        "observations_history_input_mode": history_mode,
        "clip_obs": clip_obs,
        "clip_actions_lower": action["clip_lower"],
        "clip_actions_upper": action["clip_upper"],
        "rl_kp": control["stiffness"],
        "rl_kd": control["damping"],
        "num_of_dofs": robot["num_dofs"],
        "action_scale": action["scale"],
        "action_offset": action["offset"],
        "wheel_indices": [],
        "lin_vel_scale": 1.0,
        "ang_vel_scale": _uniform_scale(terms_by_type["base_ang_vel"], "base_ang_vel"),
        "dof_pos_scale": _uniform_scale(terms_by_type["joint_pos_rel"], "joint_pos_rel"),
        "dof_vel_scale": _uniform_scale(terms_by_type["joint_vel_rel"], "joint_vel_rel"),
        "commands_scale": terms_by_type["generated_commands"]["scale"],
        "commands_lower": commands_lower,
        "commands_upper": commands_upper,
        "default_dof_pos": [float(value) for value in default_dof_pos],
        "policy_joint_names": robot["policy_joint_names"],
        "backend_joint_names": robot["backend_joint_names"],
        "joint_mapping": robot["joint_mapping"],
    }
    if actuator:
        rl_sar["actuator_model"] = "unitree_torque_speed"
        for key, value in actuator.items():
            if key != "type":
                rl_sar[f"actuator_{key}"] = value
        rl_sar["torque_limits"] = actuator.get("effort_y2", actuator.get("effort_y1", []))
    return {"root_key": f"{robot_name}/bundle", **rl_sar}


def export_deploy_cfg(
    env: Any,
    log_dir: str | Path,
    *,
    actor_groups: Sequence[str],
    model_input_names: Sequence[str] | None = None,
    robot_name: str = "go2",
    model_name: str = "policy.pt",
    history_is_term_major: bool = True,
) -> Path:
    """Write ``params/deploy.yaml`` from a live ManagerBasedRLEnv."""

    if not actor_groups:
        raise ValueError("At least one deployable actor observation group is required.")

    asset = env.scene["robot"]
    all_joint_names = list(asset.data.joint_names)
    total_joints = len(all_joint_names)
    action_items = list(env.action_manager._terms.items())
    if len(action_items) != 1:
        raise ValueError(f"Deployment exporter currently requires one action term, got {len(action_items)}.")
    action_name, action_term = action_items[0]
    joint_ids = _joint_ids(action_term, total_joints)
    policy_joint_names = [all_joint_names[index] for index in joint_ids]

    backend_joint_names = list(getattr(env.cfg.scene.robot, "joint_sdk_names", None) or all_joint_names)
    missing = [name for name in policy_joint_names if name not in backend_joint_names]
    if missing:
        raise ValueError(f"Policy joints are missing from backend joint names: {missing}")
    joint_mapping = [backend_joint_names.index(name) for name in policy_joint_names]

    action_dim = int(action_term.action_dim)
    action_scale = _expanded_vector(action_term._scale, action_dim, default=1.0)
    nominal_default_joint_pos = _nominal_default_joint_pos(asset, joint_ids, total_joints)
    action_offset = _action_offset(action_term, nominal_default_joint_pos, action_dim)
    clip_lower, clip_upper = _action_clip(action_term, action_dim)

    stiffness = _selected_vector(asset.data.default_joint_stiffness[0], joint_ids, total_joints)
    damping = _selected_vector(asset.data.default_joint_damping[0], joint_ids, total_joints)
    control = {
        "sim_dt": float(env.cfg.sim.dt),
        "decimation": int(env.cfg.decimation),
        "policy_dt": float(env.cfg.sim.dt * env.cfg.decimation),
        "stiffness": stiffness,
        "damping": damping,
    }
    robot = {
        "name": robot_name,
        "num_dofs": action_dim,
        "policy_joint_names": policy_joint_names,
        "backend_joint_names": backend_joint_names,
        "joint_mapping": joint_mapping,
    }
    action = {
        "name": action_name,
        "type": action_term.__class__.__name__,
        "scale": action_scale,
        "offset": action_offset,
        "clip_lower": clip_lower,
        "clip_upper": clip_upper,
    }
    groups = {
        group_name: _observation_group(
            env,
            group_name,
            history_is_term_major=history_is_term_major,
        )
        for group_name in actor_groups
    }
    command_ranges = _command_ranges(env)
    actuator = _runtime_actuator(env, asset, joint_ids, total_joints)

    if model_input_names is None:
        model_input_names = ["obs"] if len(actor_groups) == 1 else [f"input_{index}" for index in range(len(actor_groups))]
    if len(model_input_names) != len(actor_groups):
        raise ValueError("model_input_names and actor_groups must have the same length.")
    model_inputs = [
        {
            "name": input_name,
            "observation_group": group_name,
            "dim": groups[group_name]["model_dim"],
        }
        for input_name, group_name in zip(model_input_names, actor_groups)
    ]

    document = {
        "schema_version": 1,
        "robot": robot,
        "control": control,
        "commands": {"base_velocity": {"ranges": command_ranges}} if command_ranges else {},
        "action": action,
        "observations": {"groups": groups},
        "actuator": actuator,
        "model": {
            "files": {"torchscript": model_name, "onnx": "policy.onnx"},
            "inputs": model_inputs,
            "outputs": [{"name": "actions", "dim": action_dim}],
        },
    }
    document["rl_sar"] = _rl_sar_compatibility(
        robot_name=robot_name,
        control=control,
        robot=robot,
        action=action,
        default_dof_pos=nominal_default_joint_pos,
        groups=groups,
        actor_groups=actor_groups,
        command_ranges=command_ranges,
        actuator=actuator,
        model_name=model_name,
    )

    output_path = Path(log_dir) / "params" / "deploy.yaml"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(document, stream, sort_keys=False, allow_unicode=True)
    print(f"[INFO] Exported resolved deployment contract to: {output_path}")
    return output_path


__all__ = ["export_deploy_cfg"]
