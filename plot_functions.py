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


def plot_pb_trajectories(traj_x, traj_u, traj_w_hat, x_target, obs_centers, obs_sigma, dt=0.05):
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
    if not isinstance(obs_sigma, list): obs_sigma = [obs_sigma]

    for i, center in enumerate(obs_centers):
        c_np = to_numpy(center).flatten()
        s_tensor = obs_sigma[i] if i < len(obs_sigma) else obs_sigma[0]
        s_np = to_numpy(s_tensor).flatten()

        e1 = patches.Ellipse(c_np, 2 * s_np[0], 2 * s_np[1], color='#7f8c8d', alpha=0.5, zorder=0)
        ax.add_patch(e1)
        e2 = patches.Ellipse(c_np, 2 * s_np[0] + 0.2, 2 * s_np[1] + 0.2,
                             color='#7f8c8d', fill=False, ls='--', alpha=0.5)
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