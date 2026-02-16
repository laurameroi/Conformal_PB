import torch
from torch.utils.data import Dataset


class RobotControlDataset(Dataset):
    def __init__(self, num_samples, horizon, n_agents=1, x0_std=0.2, noise_std=0):
        """
        Args:
            num_samples: Number of different initial conditions to train on.
            horizon: Simulation length.
            n_agents: Number of robots in the system.
        """
        self.num_samples = num_samples
        self.horizon = horizon
        self.n_agents = n_agents

        # Calculate total state dimension: 4 states per agent (x, y, vx, vy)
        self.state_dim_per_agent = 4
        self.state_dim = self.state_dim_per_agent * n_agents

        # 1. Define Centers based on N agents
        # If we have 2 agents, we want symmetry: (-1, -1) and (+1, -1)
        if n_agents == 2:
            # Agent 1: x=-1, y=-1
            # Agent 2: x=+1, y=-1
            centers = torch.tensor([-1.0, -1.0, 1.0, -1.0])
        else:
            # Fallback for 1 or 3+ agents: All stack at (-1, -1)
            # (Or you could define more complex logic here)
            pos_center_single = torch.tensor([-1.0, -1.0])
            centers = pos_center_single.repeat(n_agents)

        # Reshape to (1, 1, Total_Pos_Dim) for broadcasting
        pos_center = centers.view(1, 1, n_agents * 2)

        # 2. Add Noise
        # Shape: (num_samples, 1, n_agents * 2)
        pos_noise = torch.randn(num_samples, 1, n_agents * 2) * x0_std

        pos = pos_center + pos_noise

        # --- B. Generate Initial Velocities ---
        # Shape: (num_samples, 1, n_agents * 2) -> (vx, vy) for each agent
        vel = torch.zeros(num_samples, 1, n_agents * 2)

        # --- C. Interleave Position and Velocity ---
        # The standard format for RobotPlant is [x1, y1, vx1, vy1, x2, y2, vx2, vy2...]
        # But simply concatenating all Pos then all Vel results in [x1, y1, x2, y2, vx1, vy1...]
        # We need to assemble them correctly per agent.

        # Reshape to (num_samples, 1, n_agents, 2) to separate agents
        pos_reshaped = pos.view(num_samples, 1, n_agents, 2)
        vel_reshaped = vel.view(num_samples, 1, n_agents, 2)

        # Concatenate on the last dim to get (num_samples, 1, n_agents, 4) -> [x, y, vx, vy]
        state_reshaped = torch.cat((pos_reshaped, vel_reshaped), dim=-1)

        # Flatten back to (num_samples, 1, total_state_dim)
        self.x0_data = state_reshaped.view(num_samples, 1, self.state_dim)

        # --- D. Create w (Process Noise / Trajectory placeholder) ---
        # Small noise for all time steps
        self.w = noise_std * torch.randn(num_samples, horizon, self.state_dim)

        # --- E. Set first time step exactly to initial state ---
        self.w[:, 0, :] = self.x0_data[:, 0, :]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Returns (initial_state, input_trajectory)
        return self.w[idx]