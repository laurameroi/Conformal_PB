from dataclasses import dataclass
from typing import Any, Dict, List

import torch

from performance_boosting import PBClosedLoop, PBLoss
from ren import ContractiveREN
from robot import ProportionalController, RobotPlant, StabilizedRobot


@dataclass
class EvaluationConfig:
    n_agents: int = 1
    state_dim: int = 4
    input_dim: int = 2
    dt: float = 0.05
    b_nom: float = 1.0
    m_nom: float = 1.0
    b2_nom: float = 0.2
    b_sim: float = 1.0
    m_sim: float = 1.0
    b2_sim: float = 0.2
    kp: float = 1.0
    alpha_obs: float = 100.0


def _build_loss_terms(config: EvaluationConfig, device: torch.device):
    q_agent = torch.diag(torch.tensor([1.0, 1.0, 1.0, 1.0], device=device))
    q = torch.kron(torch.eye(config.n_agents, device=device), q_agent)

    r_agent = torch.eye(2, device=device) * 0.0002
    r = torch.kron(torch.eye(config.n_agents, device=device), r_agent)

    obs_centers = [torch.tensor([-0.5, -0.5], device=device)]
    obs_sigmas = [torch.tensor([0.2, 0.2], device=device)]

    x_target = torch.zeros(4 * config.n_agents, device=device)
    return x_target, q, r, obs_centers, obs_sigmas


def load_ren_from_checkpoint(checkpoint_path: str, device: torch.device) -> Dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    ren_state = checkpoint["ren_state"] if "ren_state" in checkpoint else checkpoint

    required_keys = ["X", "Y", "B2", "C2", "D21", "D22", "D12"]
    missing = [key for key in required_keys if key not in ren_state]
    if missing:
        raise KeyError(f"Missing REN keys in checkpoint: {missing}")

    dim_internal = ren_state["Y"].shape[0]
    dim_in = ren_state["B2"].shape[1]
    dim_out = ren_state["C2"].shape[0]
    dim_nl = ren_state["D21"].shape[1]

    ren_model = ContractiveREN(
        dim_in=dim_in,
        dim_out=dim_out,
        dim_internal=dim_internal,
        dim_nl=dim_nl,
        initialization_std=0.1,
        internal_state_init=None,
    ).to(device)

    ren_state_filtered = {k: v for k, v in ren_state.items() if k not in ["x", "init_x"]}
    load_result = ren_model.load_state_dict(ren_state_filtered, strict=False)
    ren_model.eval()

    return {
        "checkpoint": checkpoint,
        "ren_model": ren_model,
        "load_result": load_result,
        "dims": {
            "dim_in": dim_in,
            "dim_out": dim_out,
            "dim_internal": dim_internal,
            "dim_nl": dim_nl,
        },
    }


def build_evaluation_system(ren_model: ContractiveREN, config: EvaluationConfig, device: torch.device):
    nominal_plant = RobotPlant(b=config.b_nom, b2=config.b2_nom, m=config.m_nom, n_agents=config.n_agents).to(device)
    sim_plant = RobotPlant(b=config.b_sim, b2=config.b2_sim, m=config.m_sim, n_agents=config.n_agents).to(device)

    controller = ProportionalController(kp=config.kp, n_agents=config.n_agents).to(device)

    f_nom = StabilizedRobot(nominal_plant, controller).to(device)
    f_sim = StabilizedRobot(sim_plant, controller).to(device)

    for param in f_nom.parameters():
        param.requires_grad = False
    for param in f_sim.parameters():
        param.requires_grad = False

    pb_loop = PBClosedLoop(ren_model, f_sim, f_nom).to(device)
    pb_loop.eval()

    return {
        "ren": ren_model,
        "pb_loop": pb_loop,
        "f_sim": f_sim,
        "f_nom": f_nom,
    }


def prepare_evaluation_context(
    checkpoint_path: str = "ren_standard_checkpoint.pt",
    device: str = "cpu",
    config: EvaluationConfig | None = None,
) -> Dict[str, Any]:
    config = config or EvaluationConfig()
    device_obj = torch.device(device)

    x_target, q, r, obs_centers, obs_sigmas = _build_loss_terms(config, device_obj)

    ren_bundle = load_ren_from_checkpoint(checkpoint_path=checkpoint_path, device=device_obj)
    eval_system = build_evaluation_system(ren_bundle["ren_model"], config=config, device=device_obj)

    metric = PBLoss(
        x_target,
        q,
        r,
        alpha_obs=config.alpha_obs,
        obs_centers=obs_centers,
        obs_sigmas=obs_sigmas,
        n_agents=config.n_agents,
    ).to(device_obj)

    return {
        "device": device_obj,
        "config": config,
        "dims": ren_bundle["dims"],
        "load_result": ren_bundle["load_result"],
        "eval_system": eval_system,
        "metric": metric,
        "x_target": x_target,
        "Q": q,
        "R": r,
        "obs_centers": obs_centers,
        "obs_sigmas": obs_sigmas,
    }
