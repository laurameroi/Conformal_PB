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

def generate_random_batch_old(
        batch_size,
        horizon,
        n_agents=1,
        x0_centers=None,
        x0_stds=None,
        x0_probs=None,
        noise_std=0.2,
        device='cpu'
):
    """
    Generates a batch of initial conditions from a custom Gaussian Mixture Model.

    Args:
        x0_centers: List of tensors or lists representing the (x, y) centers.
                 Can be shape (2,) which auto-repeats for all agents, or (n_agents * 2,).
        x0_stds: List of standard deviations corresponding to each center.
        x0_probs: List of probabilities for selecting each center (must sum to 1).
    """
    state_dim_per_agent = 4
    state_dim = state_dim_per_agent * n_agents

    # --- 1. Process Inputs & Set Defaults ---
    if x0_centers is None:
        x0_centers = [[-1.0, -1.0]]  # Default to a single center

    # Standardize centers to shape (num_modes, n_agents * 2)
    processed_centers = []
    for c in x0_centers:
        if not isinstance(c, torch.Tensor):
            c = torch.tensor(c, dtype=torch.float32)
        if c.numel() == 2:  # Auto-repeat (x, y) for all agents
            c = c.repeat(n_agents)
        processed_centers.append(c)
    centers_tensor = torch.stack(processed_centers)

    num_modes = centers_tensor.shape[0]

    # Default stds to 0.2 if not provided
    if x0_stds is None:
        stds_tensor = torch.full((num_modes,), 0.2)
    else:
        stds_tensor = torch.tensor(x0_stds, dtype=torch.float32)

    # Default probabilities to uniform if not provided
    if x0_probs is None:
        probs_tensor = torch.ones(num_modes) / num_modes
    else:
        probs_tensor = torch.tensor(x0_probs, dtype=torch.float32)
        probs_tensor = probs_tensor / probs_tensor.sum()  # Ensure they sum to 1

    # --- 2. Sample Modes ---
    # mode_indices shape: (batch_size,)
    mode_indices = torch.multinomial(probs_tensor, batch_size, replacement=True)

    # --- 3. Gather Chosen Parameters ---
    # chosen_centers shape: (batch_size, n_agents * 2)
    chosen_centers = centers_tensor[mode_indices]

    # chosen_stds shape: (batch_size,) -> Reshaped to (batch_size, 1, 1) for broadcasting
    chosen_stds = stds_tensor[mode_indices].view(batch_size, 1, 1)

    # --- 4. Generate Initial States ---
    pos_center = chosen_centers.unsqueeze(1)  # Shape: (batch_size, 1, n_agents * 2)
    pos_noise = torch.randn(batch_size, 1, n_agents * 2) * chosen_stds
    pos = pos_center + pos_noise

    vel = torch.zeros(batch_size, 1, n_agents * 2)

    # Reshape and concatenate
    pos_reshaped = pos.view(batch_size, 1, n_agents, 2)
    vel_reshaped = vel.view(batch_size, 1, n_agents, 2)
    state_reshaped = torch.cat((pos_reshaped, vel_reshaped), dim=-1)

    x0_data = state_reshaped.view(batch_size, 1, state_dim)

    # --- 5. Generate Horizon Noise and Inject Initial State ---
    w = noise_std * torch.randn(batch_size, horizon, state_dim)
    w[:, 0, :] = x0_data[:, 0, :]

    return w.to(device)

def generate_random_batch(config, custom_batch_size=None):
    """
    Generates a batch of initial conditions from a custom Gaussian Mixture Model.

    Args:
        config: The ExperimentConfig object containing all dataset parameters.
        custom_batch_size: Optional int to override the config.batch_size (e.g., for validation/testing).
    """
    # Use custom size if provided, otherwise default to config.batch_size
    batch_size = custom_batch_size if custom_batch_size is not None else config.batch_size

    # Extract remaining parameters directly from config
    horizon = config.horizon
    n_agents = config.n_agents
    x0_centers = config.x0_centers
    x0_stds = config.x0_stds
    x0_probs = config.x0_probs
    noise_std = config.noise_std
    device = config.device

    state_dim_per_agent = 4
    state_dim = state_dim_per_agent * n_agents

    # --- 1. Process Inputs & Set Defaults ---
    if x0_centers is None:
        x0_centers = [[-1.0, -1.0]]  # Default to a single center

    # Standardize centers to shape (num_modes, n_agents * 2)
    processed_centers = []
    for c in x0_centers:
        if not isinstance(c, torch.Tensor):
            c = torch.tensor(c, dtype=torch.float32)
        if c.numel() == 2:  # Auto-repeat (x, y) for all agents
            c = c.repeat(n_agents)
        processed_centers.append(c)
    centers_tensor = torch.stack(processed_centers)

    num_modes = centers_tensor.shape[0]

    # Default stds to 0.2 if not provided
    if x0_stds is None:
        stds_tensor = torch.full((num_modes,), 0.2)
    else:
        stds_tensor = torch.tensor(x0_stds, dtype=torch.float32)

    # Default probabilities to uniform if not provided
    if x0_probs is None:
        probs_tensor = torch.ones(num_modes) / num_modes
    else:
        probs_tensor = torch.tensor(x0_probs, dtype=torch.float32)
        probs_tensor = probs_tensor / probs_tensor.sum()  # Ensure they sum to 1

    # --- 2. Sample Modes ---
    # mode_indices shape: (batch_size,)
    mode_indices = torch.multinomial(probs_tensor, batch_size, replacement=True)

    # --- 3. Gather Chosen Parameters ---
    # chosen_centers shape: (batch_size, n_agents * 2)
    chosen_centers = centers_tensor[mode_indices]

    # chosen_stds shape: (batch_size,) -> Reshaped to (batch_size, 1, 1) for broadcasting
    chosen_stds = stds_tensor[mode_indices].view(batch_size, 1, 1)

    # --- 4. Generate Initial States ---
    pos_center = chosen_centers.unsqueeze(1)  # Shape: (batch_size, 1, n_agents * 2)
    pos_noise = torch.randn(batch_size, 1, n_agents * 2) * chosen_stds
    pos = pos_center + pos_noise

    vel = torch.zeros(batch_size, 1, n_agents * 2)

    # Reshape and concatenate
    pos_reshaped = pos.view(batch_size, 1, n_agents, 2)
    vel_reshaped = vel.view(batch_size, 1, n_agents, 2)
    state_reshaped = torch.cat((pos_reshaped, vel_reshaped), dim=-1)

    x0_data = state_reshaped.view(batch_size, 1, state_dim)

    # --- 5. Generate Horizon Noise and Inject Initial State ---
    w = noise_std * torch.randn(batch_size, horizon, state_dim)
    w[:, 0, :] = x0_data[:, 0, :]

    return w.to(device)