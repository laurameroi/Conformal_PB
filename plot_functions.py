import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import LineCollection


def plot_trajectories(traj_tensor, dt=0.05):
    """
    Basic plotting for N-Agent trajectories.
    """
    # Convert to numpy
    traj_np = traj_tensor.detach().cpu().numpy()  # (Batch, Horizon, State)
    batch_sz, steps, state_dim = traj_np.shape

    # Infer number of agents (4 states per agent: x, y, vx, vy)
    n_agents = state_dim // 4

    # Colors for agents: Blue, Red, Green, Purple...
    cmap = plt.get_cmap('tab20')  # 'tab20' has 20 distinct colors; wraps around if N > 20
    agent_colors = [cmap(i % 20) for i in range(n_agents)]

    # --- Figure 1: 2D Spatial Trajectory ---
    plt.figure(figsize=(10, 8))

    # 1. Plot Target (Origin)
    plt.scatter(0, 0, color='black', marker='*', s=300, label='Target', zorder=10)

    # 2. Loop over each agent
    for agent_idx in range(n_agents):
        # Indices for this agent
        ix = agent_idx * 4  # x position index
        iy = agent_idx * 4 + 1  # y position index

        color = agent_colors[agent_idx % len(agent_colors)]
        label = f"Agent {agent_idx + 1}"

        # Plot Start Points (t=0)
        # FIX: Changed `c=color` to `color=color`
        plt.scatter(traj_np[:, 0, ix], traj_np[:, 0, iy],
                    color=color, s=50, edgecolors='k', zorder=5)

        # Plot End Points (t=final)
        # FIX: Changed `c=color` to `color=color`
        plt.scatter(traj_np[:, -1, ix], traj_np[:, -1, iy],
                    color=color, marker='X', s=50, edgecolors='k', zorder=5)

        # Plot Trajectories (Lines)
        for b in range(batch_sz):
            # Only label the first batch to avoid legend clutter
            lbl = label if b == 0 else ""
            plt.plot(traj_np[b, :, ix], traj_np[b, :, iy],
                     color=color, alpha=0.6, linewidth=2, label=lbl)

    plt.title(f"Multi-Agent Trajectories (N={n_agents})")
    plt.xlabel("Position X [m]")
    plt.ylabel("Position Y [m]")
    plt.axis('equal')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.show()


def plot_pb_trajectories(traj_x, traj_u, traj_w_hat, x_target, obs_centers, obs_radii, obs_radii_safe, dt=0.05):
    """
    Advanced plotting for Multi-Agent PB Control.
    Plots obstacles and overlays statistics for any number of agents.
    """

    # --- 1. Helpers & Conversion ---
    def to_numpy(data):
        if isinstance(data, torch.Tensor):
            return data.detach().cpu().numpy()
        if isinstance(data, list):
            return np.array(data)
        return data

    batch_x = to_numpy(traj_x)  # (Batch, Time, State)
    batch_u = to_numpy(traj_u)  # (Batch, Time, Input)
    batch_w = to_numpy(traj_w_hat)  # (Batch, Time, State/Input)

    # Handle dimensions
    if batch_x.ndim == 4: batch_x = batch_x.squeeze(2)
    if batch_u.ndim == 4: batch_u = batch_u.squeeze(2)
    if batch_w.ndim == 4: batch_w = batch_w.squeeze(2)

    batch_sz, horizon, state_dim = batch_x.shape
    n_agents = state_dim // 4
    time_axis = np.arange(horizon) * dt

    # Colors for agents
    cmap = plt.get_cmap('tab20')
    agent_colors = [cmap(i % 20) for i in range(n_agents)]

    # --- FIGURE 1: 2D Map (XY Plane) ---
    fig1, ax = plt.subplots(figsize=(10, 8), dpi=100)

    # A. Draw Obstacles
    if not isinstance(obs_radii, list): obs_radii = [obs_radii]
    if obs_radii_safe is not None and not isinstance(obs_radii_safe, list): obs_radii_safe = [obs_radii_safe]

    import matplotlib.patches as patches

    for i, center in enumerate(obs_centers):
        c_np = to_numpy(center).flatten()

        # --- Handle Physical Radius ---
        r_val = obs_radii[i] if i < len(obs_radii) else obs_radii[0]

        # If it's a simple float, it's a circle (rx = ry)
        if isinstance(r_val, (float, int)):
            rx = ry = float(r_val)
        # If it's a tensor/array, it could be an ellipse
        else:
            r_np = to_numpy(r_val).flatten()
            rx, ry = r_np[0], r_np[1]

        # Draw the actual physical obstacle (Solid Gray)
        e1 = patches.Ellipse(c_np, 2 * rx, 2 * ry, color='#7f8c8d', alpha=0.7, zorder=0)
        ax.add_patch(e1)

        # --- Handle Safety Margin / Inflation Radius ---
        if obs_radii_safe is not None:
            rs_val = obs_radii_safe[i] if i < len(obs_radii_safe) else obs_radii_safe[0]

            if isinstance(rs_val, (float, int)):
                rs_x = rs_y = float(rs_val)
            else:
                rs_np = to_numpy(rs_val).flatten()
                rs_x, rs_y = rs_np[0], rs_np[1]

            e2 = patches.Ellipse(c_np, 2 * rs_x, 2 * rs_y,
                                 edgecolor='#e74c3c', fill=False, ls='--', lw=2, alpha=0.8, zorder=1)
            ax.add_patch(e2)

    # B. Draw Target (Origin)
    ax.scatter(0, 0, c='gold', marker='*', s=400, edgecolors='k', zorder=20, label='Target')

    # C. Draw Trajectories per Agent
    num_to_plot = min(batch_sz, 50)

    for i in range(n_agents):
        ix, iy = i * 4, i * 4 + 1
        col = agent_colors[i]  # Fixed: safely grab the RGBA tuple

        segments = []
        for b in range(num_to_plot):
            points = np.column_stack([batch_x[b, :, ix], batch_x[b, :, iy]])
            segments.append(points)

        # FIXED: Wrap `col` in a list so Matplotlib knows it's a single color for all segments
        lc = LineCollection(segments, colors=[col], linewidths=1.5, alpha=0.4,
                            label=f'Agent {i + 1} Traj')
        ax.add_collection(lc)

        # FIXED: Wrap `col` in a list for scatter as well
        starts = batch_x[:num_to_plot, 0, ix:iy + 1]
        ax.scatter(starts[:, 0], starts[:, 1], c=[col], s=30, edgecolors='white', zorder=15)

    ax.autoscale()
    ax.set_aspect('equal')
    ax.set_title(f"Multi-Robot Trajectories (N={n_agents})")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.grid(True, linestyle=':', alpha=0.6)

    # Custom Legend
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], color=agent_colors[i], lw=2, label=f'Agent {i + 1}')
                       for i in range(n_agents)]
    legend_elements.append(
        Line2D([0], [0], marker='*', color='w', markerfacecolor='gold', markersize=15, label='Target'))
    ax.legend(handles=legend_elements, loc='upper right')

    plt.tight_layout()
    plt.show()

    # --- FIGURE 2: Statistics Dashboard ---
    fig2, axs = plt.subplots(2, 2, figsize=(14, 10), dpi=100)
    fig2.suptitle("Multi-Agent State Analysis", fontweight='bold')

    def plot_agent_cloud(ax, data_batch, label_prefix, color):
        mean = np.mean(data_batch, axis=0)
        std = np.std(data_batch, axis=0)
        ax.plot(time_axis, mean, color=color, lw=2, label=label_prefix)
        ax.fill_between(time_axis, mean - std, mean + std, color=color, alpha=0.1)

    # 1. Position X
    ax = axs[0, 0]
    for i in range(n_agents):
        idx = i * 4
        plot_agent_cloud(ax, batch_x[:, :, idx], f"Ag{i + 1}", agent_colors[i])
    ax.set_ylabel("Position X [m]")
    ax.set_title("X-Position Tracking")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # 2. Position Y
    ax = axs[0, 1]
    for i in range(n_agents):
        idx = i * 4 + 1
        plot_agent_cloud(ax, batch_x[:, :, idx], f"Ag{i + 1}", agent_colors[i])
    ax.set_ylabel("Position Y [m]")
    ax.set_title("Y-Position Tracking")
    ax.grid(True, alpha=0.3)

    # 3. Control Inputs
    ax = axs[1, 0]
    for i in range(n_agents):
        u_ix, u_iy = i * 2, i * 2 + 1
        u_mag = np.sqrt(batch_u[:, :, u_ix] ** 2 + batch_u[:, :, u_iy] ** 2)
        plot_agent_cloud(ax, u_mag, f"Ag{i + 1} Force", agent_colors[i])
    ax.set_ylabel("Force Magnitude [N]")
    ax.set_title("Control Effort")
    ax.grid(True, alpha=0.3)

    # 4. Estimated Disturbance
    ax = axs[1, 1]

    # FIXED: Dynamically determine the size of w_hat per agent to prevent IndexErrors
    w_dim_per_agent = batch_w.shape[-1] // n_agents

    for i in range(n_agents):
        idx = i * w_dim_per_agent  # Safely points to the X-component of the disturbance
        plot_agent_cloud(ax, batch_w[:, :, idx], f"Ag{i + 1} W_hat_x", agent_colors[i])

    ax.set_ylabel("Est. Disturbance X")
    ax.set_title("Disturbance Estimation (X-Axis)")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.show()

def plot_distance_tube(traj_x_erm, traj_x_q, obs_center, collision_radius):
    """
    Plots the conformal distance tube comparing ERM and Quantile trained controllers.

    Args:
        traj_x_erm: State trajectories from ERM model. Shape (batch_size, horizon, state_dim)
        traj_x_q: State trajectories from Quantile model. Shape (batch_size, horizon, state_dim)
        obs_center: The [x, y] coordinates of the obstacle center.
        collision_radius: The radius of the obstacle (e.g., obs_sigmas[0][0]).
    """
    # 1. Ensure tensors are on CPU and converted to numpy arrays
    if isinstance(traj_x_erm, torch.Tensor):
        traj_x_erm = traj_x_erm.detach().cpu().numpy()
    if isinstance(traj_x_q, torch.Tensor):
        traj_x_q = traj_x_q.detach().cpu().numpy()
    if isinstance(obs_center, torch.Tensor):
        obs_center = obs_center.detach().cpu().numpy()

    # Extract only the X and Y positions (assuming they are the first two state dimensions)
    pos_erm = traj_x_erm[:, :, 0:2]
    pos_q = traj_x_q[:, :, 0:2]

    # 2. Calculate distances to the obstacle for every trajectory at every time step
    dist_erm = np.linalg.norm(pos_erm - obs_center, axis=-1)
    dist_q = np.linalg.norm(pos_q - obs_center, axis=-1)

    time_steps = np.arange(dist_erm.shape[1])

    # 3. Calculate the Median (50th) and Lower Bound (5th percentile)
    # We care about the lower bound because closer = more dangerous!
    erm_median = np.percentile(dist_erm, 50, axis=0)
    erm_lower_bound = np.percentile(dist_erm, 5, axis=0)

    q_median = np.percentile(dist_q, 50, axis=0)
    q_lower_bound = np.percentile(dist_q, 5, axis=0)

    # 4. Create the Plot
    plt.figure(figsize=(10, 6))

    # --- Plot ERM ---
    plt.plot(time_steps, erm_median, color='red', linewidth=2, label='ERM Median Distance')
    plt.fill_between(time_steps, erm_lower_bound, erm_median, color='red', alpha=0.2,
                     label='ERM 5th Percentile (Worst-Case)')

    # --- Plot Quantile ---
    plt.plot(time_steps, q_median, color='blue', linewidth=2, label='Quantile Median Distance')
    plt.fill_between(time_steps, q_lower_bound, q_median, color='blue', alpha=0.2,
                     label='Quantile 5th Percentile (Worst-Case)')

    # --- The Collision Boundary ---
    plt.axhline(y=collision_radius, color='black', linestyle='--', linewidth=2, label='Collision Threshold')
    plt.axhspan(0, collision_radius, color='black', alpha=0.1) # Shade the danger zone!

    # --- Formatting ---
    plt.title('Conformal Safety Bounds: Distance to Obstacle Over Time\n(Evaluating across unseen initial conditions)',
              fontsize=14, fontweight='bold')
    plt.xlabel('Time Step ($t$)', fontsize=12)
    plt.ylabel('Distance to Obstacle', fontsize=12)

    plt.ylim(bottom=0) # Distance cannot be negative
    plt.legend(loc='upper right', fontsize=10)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()

    plt.show()




def plot_nonconformity_histogram(calibration_result, bins=30, density=False, ax=None):
    """
    Plot histogram of non-conformity scores with threshold marker.

    Args:
        calibration_result: object with attributes `scores`, `threshold`, `alpha`.
        bins: number of histogram bins.
        density: whether to normalize histogram.
        ax: optional matplotlib axis.
    """
    return plot_nonconformity_scores(
        scores=calibration_result.scores,
        threshold=calibration_result.threshold,
        alpha=calibration_result.alpha,
        bins=bins,
        density=density,
        ax=ax,
    )


def plot_nonconformity_scores(scores, threshold=None, alpha=None, bins=30, density=False, ax=None):
    """
    Plot histogram of non-conformity scores with optional threshold marker.

    Args:
        scores: non-conformity scores (Tensor or ndarray), shape (N,).
        threshold: optional threshold value (scalar Tensor or float).
        alpha: optional alpha label used in legend.
        bins: number of histogram bins.
        density: whether to normalize histogram.
        ax: optional matplotlib axis.
    """

    if isinstance(scores, torch.Tensor):
        scores_np = scores.detach().cpu().numpy()
    else:
        scores_np = np.asarray(scores)

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5), dpi=100)
    else:
        fig = ax.figure

    ax.hist(scores_np, bins=bins, density=density, alpha=0.75, color="#4C72B0", edgecolor="white")

    if threshold is not None:
        if isinstance(threshold, torch.Tensor):
            threshold_value = float(threshold.detach().cpu().item())
        else:
            threshold_value = float(threshold)

        if alpha is None:
            label = "Calibration threshold"
        else:
            label = f"Conditional threshold, alpha={alpha:.3f}"

        ax.axvline(
            threshold_value,
            color="#C44E52",
            linestyle="--",
            linewidth=2,
            label=label,
        )
    ax.set_title("Non-conformity Score Distribution")
    ax.set_xlabel("Non-conformity score")
    ax.set_ylabel("Density" if density else "Count")
    ax.grid(True, linestyle="--", alpha=0.35)
    if threshold is not None:
        ax.legend()

    return fig, ax