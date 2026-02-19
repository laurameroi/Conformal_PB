import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Mapping

import torch


TrajectoryBatch = Mapping[str, torch.Tensor]


@dataclass
class CalibrationResult:
    scores: torch.Tensor
    threshold: torch.Tensor
    alpha: float
    quantile_level: float
    finite_sample: bool


@dataclass
class PBLossConfig:
    x_target: torch.Tensor
    q: torch.Tensor
    r: torch.Tensor
    alpha_obs: float
    obs_centers: list[torch.Tensor]
    obs_sigmas: list[torch.Tensor]
    n_agents: int = 1


class NonConformityScorer(ABC):
    @abstractmethod
    def score_batch(self, trajectories: TrajectoryBatch) -> torch.Tensor:
        """
        Returns one scalar score per trajectory with shape (batch_size,).
        """


class PBLossNonConformity(NonConformityScorer):
    """
    Native PB loss non-conformity (one score per trajectory).
    """

    def __init__(self, config: PBLossConfig):
        self.config = config
        self.state_per_agent = 4

    def _obstacle_cost(self, traj_x: torch.Tensor) -> torch.Tensor:
        batch_size, horizon, _ = traj_x.shape

        x_reshaped = traj_x.view(batch_size, horizon, self.config.n_agents, self.state_per_agent)
        pos = x_reshaped[..., :2]  # (batch, horizon, n_agents, 2)

        p = pos.unsqueeze(3)  # (batch, horizon, n_agents, 1, 2)
        mu = torch.stack(self.config.obs_centers).to(traj_x.device).view(1, 1, 1, -1, 2)
        sigma = torch.stack(self.config.obs_sigmas).to(traj_x.device).view(1, 1, 1, -1, 2)

        exponent = -0.5 * torch.sum(((p - mu) ** 2) / (sigma ** 2 + 1e-6), dim=-1)
        gaussian_density = torch.exp(exponent)
        return self.config.alpha_obs * gaussian_density.sum(dim=(2, 3))

    def score_batch(self, trajectories: TrajectoryBatch) -> torch.Tensor:
        traj_x = trajectories["x"]
        traj_u = trajectories["u"]

        x_target = self.config.x_target.to(traj_x.device)
        q = self.config.q.to(traj_x.device)
        r = self.config.r.to(traj_x.device)

        error = traj_x - x_target
        cost_target_t = torch.einsum("bhi,ij,bhj->bh", error, q, error)
        cost_u_t = torch.einsum("bhi,ij,bhj->bh", traj_u, r, traj_u)
        cost_obs_t = self._obstacle_cost(traj_x)

        return (cost_target_t + cost_u_t + cost_obs_t).mean(dim=1)


class CallableNonConformity(NonConformityScorer):
    def __init__(self, fn: Callable[[TrajectoryBatch], torch.Tensor]):
        self.fn = fn

    def score_batch(self, trajectories: TrajectoryBatch) -> torch.Tensor:
        scores = self.fn(trajectories)
        if scores.ndim != 1:
            raise ValueError(
                f"Custom scorer must return shape (batch,), got {tuple(scores.shape)}"
            )
        return scores


def build_scorer(
    non_conformity_score: str,
    *,
    pb_loss_config: PBLossConfig | None = None,
    custom_callable: Callable[[TrajectoryBatch], torch.Tensor] | None = None,
) -> NonConformityScorer:
    """
    Factory that maps the notebook selector string to a scorer implementation.
    """
    score_name = non_conformity_score.lower()

    if score_name == "pb_loss":
        if pb_loss_config is None:
            raise ValueError("pb_loss_config is required when non_conformity_score='pb_loss'")
        return PBLossNonConformity(config=pb_loss_config)

    if score_name == "custom":
        if custom_callable is None:
            raise ValueError("custom_callable is required when non_conformity_score='custom'")
        return CallableNonConformity(custom_callable)

    raise ValueError(
        f"Unknown non-conformity score '{non_conformity_score}'. "
        "Supported values: 'pb_loss', 'custom'."
    )


def _conformal_quantile_level(num_samples: int, alpha: float, finite_sample: bool) -> float:
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    if num_samples <= 0:
        raise ValueError("num_samples must be > 0")

    if not finite_sample:
        return 1.0 - alpha

    level = math.ceil((num_samples + 1) * (1.0 - alpha)) / num_samples
    return min(max(level, 0.0), 1.0)


def compute_nonconformity_scores(
    trajectories: TrajectoryBatch,
    scorer: NonConformityScorer,
) -> torch.Tensor:
    scores = scorer.score_batch(trajectories)
    if scores.ndim != 1:
        raise ValueError(f"Non-conformity scores must have shape (batch,), got {tuple(scores.shape)}")
    return scores


def calibrate_nonconformity(
    trajectories: TrajectoryBatch,
    scorer: NonConformityScorer,
    alpha: float,
    finite_sample: bool = True,
) -> CalibrationResult:
    """
    Compute per-trajectory non-conformity scores and conformal threshold.
    """
    scores = compute_nonconformity_scores(trajectories=trajectories, scorer=scorer)
    quantile_level = _conformal_quantile_level(
        num_samples=scores.numel(), alpha=alpha, finite_sample=finite_sample
    )
    threshold = torch.quantile(scores, quantile_level, interpolation="higher")
    return CalibrationResult(
        scores=scores,
        threshold=threshold,
        alpha=alpha,
        quantile_level=quantile_level,
        finite_sample=finite_sample,
    )
