"""
Conformal verification pipeline for comparing standard and CVaR models.
Provides utilities for calibration, testing, and visualization.
Orchestrates the core non-conformity primitives from nonconformity.py.
"""

import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any

import nonconformity
from batch_generation import generate_random_batch


@dataclass
class ModelVerificationResult:
    """Result for a single model."""
    model_name: str
    scores: torch.Tensor
    threshold: torch.Tensor
    fraction_below_threshold: float


@dataclass
class VerificationResult:
    """Result from conformal verification across N models."""
    models: Dict[str, ModelVerificationResult]
    alpha: float
    test_dataset_size: int
    non_conformity_score: str


def build_pb_loss_config(
    x_target: torch.Tensor,
    q: torch.Tensor,
    r: torch.Tensor,
    alpha_obs: float,
    obs_centers: List,
    obs_sigmas: List,
    n_agents: int = 1,
) -> nonconformity.PBLossConfig:
    """Build PB loss config from evaluation context parameters."""
    return nonconformity.PBLossConfig(
        x_target=x_target,
        q=q,
        r=r,
        alpha_obs=alpha_obs,
        obs_centers=obs_centers,
        obs_sigmas=obs_sigmas,
        n_agents=n_agents,
    )


def prepare_nonconformity_scorer(
    non_conformity_score: str,
    evaluation_context: Dict[str, Any],
    n_agents: int,
    delta_distance: Optional[float] = None,
) -> nonconformity.NonConformityScorer:
    """
    Build the selected non-conformity scorer from evaluation context.

    This centralizes score/loss preparation logic so notebooks only pass
    high-level parameters and context.
    """
    pb_loss_config = build_pb_loss_config(
        x_target=evaluation_context["x_target"],
        q=evaluation_context["Q"],
        r=evaluation_context["R"],
        alpha_obs=evaluation_context["config"].alpha_obs,
        obs_centers=evaluation_context["obs_centers"],
        obs_sigmas=evaluation_context["obs_sigmas"],
        n_agents=n_agents,
    )

    proximity_config = None
    if non_conformity_score.lower() == "pb_proximity_fraction":
        if delta_distance is None:
            raise ValueError(
                "delta_distance is required when non_conformity_score='pb_proximity_fraction'"
            )
        proximity_config = nonconformity.ObstacleProximityConfig(
            obs_centers=evaluation_context["obs_centers"],
            delta_distance=delta_distance,
            n_agents=n_agents,
        )

    return nonconformity.build_scorer(
        non_conformity_score,
        pb_loss_config=pb_loss_config,
        proximity_config=proximity_config,
    )


def verify_conformal_multi_model(
    pb_loops: Dict[str, Any],
    evaluation_context: Dict[str, Any],
    non_conformity_score: str,
    m_cert: int,
    test_dataset_size: int,
    horizon: int,
    n_agents: int,
    x0_std: float,
    noise_std: float,
    alpha: float,
    device: str,
    delta_distance: Optional[float] = None,
) -> VerificationResult:
    """
    Run conformal verification on N models with shared calibration/test data.
    
    Args:
        pb_loops: Dict mapping model names to pb_loop instances
                 e.g., {"standard": pb_loop1, "cvar": pb_loop2}
        evaluation_context: Shared context from prepare_evaluation_context
        non_conformity_score: Score name ("pb_loss", "pb_collision", "pb_proximity_fraction")
        m_cert: Calibration set size
        test_dataset_size: Test set size
        horizon: Trajectory horizon
        n_agents: Number of agents
        x0_std: Initial state uncertainty
        noise_std: Process noise
        alpha: Significance level (shifted if conditional)
        device: Device (cpu/cuda)
        delta_distance: Proximity threshold for "pb_proximity_fraction"
    
    Returns:
        VerificationResult with per-model results
    """
    if not pb_loops:
        raise ValueError("pb_loops dict must not be empty")
    
    # Build scorer once (shared across all models)
    scorer = prepare_nonconformity_scorer(
        non_conformity_score,
        evaluation_context=evaluation_context,
        n_agents=n_agents,
        delta_distance=delta_distance,
    )
    
    # Generate shared calibration batch
    calibration_w = generate_random_batch(
        batch_size=m_cert,
        horizon=horizon,
        n_agents=n_agents,
        x0_std=x0_std,
        noise_std=noise_std,
        device=device,
    )
    
    # Generate shared test batch
    test_w = generate_random_batch(
        batch_size=test_dataset_size,
        horizon=horizon,
        n_agents=n_agents,
        x0_std=x0_std,
        noise_std=noise_std,
        device=device,
    )
    
    # Process each model
    model_results = {}
    for model_name, pb_loop in pb_loops.items():
        pb_loop.eval()
        
        # Calibration: run model and calibrate threshold
        traj_cal = {}
        with torch.no_grad():
            traj_cal["x"], traj_cal["u"], traj_cal["w_hat"] = pb_loop.run(calibration_w)
        
        calib_result = nonconformity.calibrate_nonconformity(
            trajectories=traj_cal,
            scorer=scorer,
            alpha=alpha,
            finite_sample=True,
        )
        
        # Test: run model and compute scores
        traj_test = {}
        with torch.no_grad():
            traj_test["x"], traj_test["u"], traj_test["w_hat"] = pb_loop.run(test_w)
        
        test_scores = nonconformity.compute_nonconformity_scores(traj_test, scorer)
        fraction = (test_scores < calib_result.threshold).float().mean().item()
        
        model_results[model_name] = ModelVerificationResult(
            model_name=model_name,
            scores=test_scores,
            threshold=calib_result.threshold,
            fraction_below_threshold=fraction,
        )
    
    return VerificationResult(
        models=model_results,
        alpha=alpha,
        test_dataset_size=test_dataset_size,
        non_conformity_score=non_conformity_score,
    )


def plot_verification_comparison(
    result: VerificationResult,
    figsize: Tuple[int, int] = (12, 5),
) -> None:
    """Plot overlaid histograms of scores for N models."""
    # Color palette for up to 10 models
    colors = ["royalblue", "darkorange", "green", "red", "purple", 
              "brown", "pink", "gray", "olive", "cyan"]
    
    plt.figure(figsize=figsize)
    
    for idx, (model_name, model_result) in enumerate(result.models.items()):
        color = colors[idx % len(colors)]
        scores_np = model_result.scores.detach().cpu().numpy()
        
        # Plot histogram
        sns.histplot(scores_np, bins=40, stat="density", color=color, 
                    alpha=0.45, label=model_name)
        
        # Plot threshold line
        plt.axvline(model_result.threshold.item(), color=color, 
                   linestyle="--", linewidth=2, 
                   label=f"{model_name} threshold")
    
    plt.title(f"{result.non_conformity_score} non-conformity comparison (N={result.test_dataset_size})")
    plt.xlabel(f"{result.non_conformity_score} score")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def print_verification_summary(result: VerificationResult) -> None:
    """Print summary statistics from verification."""
    print(f"\n{result.non_conformity_score.upper()} Non-conformity Verification")
    print("=" * 70)
    
    for model_name, model_result in result.models.items():
        print(f"{model_name} model:")
        print(f"  Threshold: {model_result.threshold.item():.6f}")
        print(f"  Fraction(score < threshold): {model_result.fraction_below_threshold:.4f}")
        print()
    
    print(f"Reference 1-alpha: {1.0 - result.alpha:.4f}")
    print("=" * 70)
