import torch, time, copy
import torch.nn as nn
import numpy as np

class PBClosedLoop(nn.Module):
    """
    Performance boosting controller, following the paper:
        "Learning to Boost the Performance of Stable Nonlinear Systems".
    Implements a state-feedback controller with stability guarantees.
    NOTE: When used in closed-loop, the controller input is the measured state of the plant
          and the controller output is the input to the plant.
    This controller has a memory for the last input ("self.last_input") and the last output ("self.last_output").
    """

    def __init__(self, ren, f_sim, f_nom):
        """
         Args:
        """
        super().__init__()
        self.f_sim = f_sim
        self.f_nom = f_nom
        self.ren = ren

    def estimate_disturbance(self, x_meas, x_prev, u_prev):
        """
        Estimates the disturbance w_hat.

        Args:
            x_meas: The measured state of the plant at the actual time step (t).
            u_prev: The total control input applied at the previous time step (t-1).
            x_prev: The measured state of the plant at the previous time step (t-1).
        """

        #Predict the nominal next state using our mathematical model.
        x_hat = self.f_nom.predict_nominal_next_state(x_prev, u_prev)

        #Calculate the estimated disturbance (w_hat).
        w_hat = x_meas - x_hat
        return w_hat

    def run(self, w):
        """
        Runs a full closed-loop simulation using the PB Controller.

        Args:
            w: disturbance vector (w0 = x0) (Batch, horizon, State_Dim).

        Returns:
            traj_x: State trajectory (batch_size, horizon, state_dim).
            traj_u: Input trajectory (batch_size, horizon, input_dim).
            traj_w: Estimated disturbance trajectory (batch_size, horizon, state_dim).
        """
        batch_size = w.shape[0]
        horizon = w.shape[1]
        x0 = w[:, :1, :]

        # 1. Initialize the Plant and the REN internal states
        self.f_sim.reset(x0, batch_size)
        self.ren.reset()

        # Update the REN parameters ONCE before the trajectory starts. This computes E_inv, F, Lambda, etc. for the whole run.
        self.ren._update_model_param()

        # Lists to store the history for logging and backpropagation
        x_log = []
        u_log = []
        w_hat_log = []

        # Memory variables to store the previous step's data for the nominal model
        u_prev = None
        x_prev = None

        for t in range(horizon):
            # A. Measurement: Get current state from the plant
            x_meas = self.f_sim.x  # Shape: (Batch, 1, State_Dim)
            x_log.append(x_meas)


            # B. Disturbance Estimation: Compute w_hat
            if t == 0:
                # At t=0, we have no history. We treat the initial state
                # deviation as the first estimated disturbance.
                w_hat = x_meas
            else:
                # We estimate the disturbance (unmodeled dynamics/noise) by
                # comparing reality (x_meas) with the nominal model's prediction.
                w_hat = self.estimate_disturbance(x_meas, x_prev, u_prev) + w[:, t:t+1, :]

            w_hat_log.append(w_hat)

            # C. Controller Step: REN Forward
            u = self.ren.forward(w_hat)
            u_log.append(u)

            # D. Apply control to the real system
            self.f_sim.forward(x_meas, u)

            # E. Update Memory: Prepare for the next time step (t+1)
            x_prev = x_meas
            u_prev = u

        # F. Return stacked trajectories as tensors
        traj_x = torch.cat(x_log, dim=1)  # Final: (Batch, Horizon, State_Dim)
        traj_u = torch.cat(u_log, dim=1)  # Final: (Batch, Horizon, Input_Dim)
        traj_w_hat = torch.cat(w_hat_log, dim=1)  # Final: (Batch, Horizon, State_Dim)

        return traj_x, traj_u, traj_w_hat

    # setters and getters
    def get_parameter_shapes(self):
        return self.ren.get_parameter_shapes()

    def get_named_parameters(self):
        return self.ren.get_named_parameters()

    def get_parameters_as_vector(self):
        # TODO: implement without numpy
        return np.concatenate([p.detach().clone().cpu().numpy().flatten() for p in self.ren.parameters()])

    def set_parameter(self, name, value):
        current_val = getattr(self.ren, name)
        value = torch.nn.Parameter(torch.tensor(value.reshape(current_val.shape)))
        setattr(self.ren, name, value)
        self.ren._update_model_param()  # update dependent params

    def set_parameters(self, param_dict):
        for name, value in param_dict.items():
            self.set_parameter(name, value)

    def set_parameters_as_vector(self, value):
        # flatten vec if not batched
        if value.nelement() == self.num_params:
            value = value.flatten()

        idx = 0
        idx_next = 0
        for name, shape in self.get_parameter_shapes().items():
            if len(shape) == 1:
                dim = shape
            elif len(shape) == 2:
                dim = shape[0] * shape[1]
            else:
                dim = shape[-1] * shape[-2]
            idx_next = idx + dim
            # select index
            if len(value.shape) == 1:
                value_tmp = value[idx:idx_next]
            elif len(value.shape) == 2:
                value_tmp = value[:, idx:idx_next]
            elif value.ndim == 3:
                value_tmp = value[:, :, idx:idx_next]
            else:
                raise AssertionError
            # set
            with torch.no_grad():
                self.set_parameter(name, value_tmp.reshape(shape))
            idx = idx_next
        assert idx_next == value.shape[-1]

    def __call__(self, w):
        """
        """
        return self.run(w)


class PBLoss(nn.Module):
    def __init__(self, x_target, Q, R, alpha_obs, obs_centers, obs_sigmas, n_agents=1):
        """
        Args:
            n_agents (int): Number of robots. Required to parse the state vector.
        """
        super().__init__()
        self.alpha_obs = alpha_obs
        self.n_agents = n_agents

        # Dimensions per agent (Assumed from RobotPlant)
        self.state_per_agent = 4

        # --- Register Buffers ---
        self.register_buffer('x_target', x_target)
        self.register_buffer('Q', Q)
        self.register_buffer('R', R)
        self.register_buffer('mu', torch.stack(obs_centers))
        self.register_buffer('sigma', torch.stack(obs_sigmas))

    def compute_per_sample_cost(self, traj_x, traj_u):
        """
        Calculates the scalar cost Psi(D) for EACH trajectory in the batch.
        Returns shape: (Batch_Size,)
        """
        batch_size, horizon, _ = traj_x.shape

        # --- 1. Target Tracking ---
        # Error shape: (Batch, Horizon, State_Dim)
        error = traj_x - self.x_target
        # einsum -> bh: Result is (Batch, Horizon)
        cost_target_t = torch.einsum('bhi,ij,bhj->bh', error, self.Q, error)

        # --- 2. Control Effort ---
        # einsum -> bh: Result is (Batch, Horizon)
        cost_u_t = torch.einsum('bhi,ij,bhj->bh', traj_u, self.R, traj_u)

        # --- 3. Obstacle Avoidance ---
        x_reshaped = traj_x.view(batch_size, horizon, self.n_agents, self.state_per_agent)
        pos = x_reshaped[..., :2]  # (Batch, Horizon, Agents, 2)

        p = pos.unsqueeze(3)  # (Batch, Horizon, Agents, 1, 2)
        m = self.mu.view(1, 1, 1, -1, 2)
        s = self.sigma.view(1, 1, 1, -1, 2)

        # Calculate Gaussian exponent
        exponent = -0.5 * torch.sum(((p - m) ** 2) / (s ** 2 + 1e-6), dim=-1)
        gaussian_density = torch.exp(exponent)

        # Sum over Agents (dim 2) and Obstacles (dim 3) -> (Batch, Horizon)
        cost_obs_t = self.alpha_obs * gaussian_density.sum(dim=(2, 3))

        # --- 4. Aggregate over Time (Horizon) ---
        # We sum (or mean) over the time dimension (dim=1) to get one scalar per trajectory.
        # Using .mean(dim=1) keeps the scale consistent with your original loss.
        # If you prefer cumulative cost, use .sum(dim=1).

        per_sample_cost = (cost_target_t + cost_u_t + cost_obs_t).mean(dim=1)

        return per_sample_cost  # Shape: (Batch_Size,)

    def forward(self, traj_x, traj_u):
        """
        traj_x: (Batch, Horizon, Total_State_Dim)
        """
        batch_size, horizon, _ = traj_x.shape

        # --- 1. Target Tracking (Works for N agents automatically) ---
        error = traj_x - self.x_target
        loss_target = torch.einsum('bhi,ij,bhj->bh', error, self.Q, error).mean()

        # --- 2. Control Effort (Works for N agents automatically) ---
        loss_u = torch.einsum('bhi,ij,bhj->bh', traj_u, self.R, traj_u).mean()

        # --- 3. Multi-Agent Obstacle Avoidance ---
        # Reshape to separate agents
        # Current Shape: (Batch, Horizon, n_agents * 4)
        # New Shape:     (Batch, Horizon, n_agents, 4)
        x_reshaped = traj_x.view(batch_size, horizon, self.n_agents, self.state_per_agent)

        # Extract Positions: (Batch, Horizon, n_agents, 2)
        # We take indices :2 (x, y) from the last dimension
        pos = x_reshaped[..., :2]

        # Vectorized Gaussian Calculation
        # We need to broadcast across 5 dimensions:
        # P (Position):  [Batch, Horizon, N_AGENTS, 1,     2]
        # M (Obs Center):[1,     1,       1,        N_OBS, 2]

        p = pos.unsqueeze(3)  # Add dim for Obstacles
        m = self.mu.view(1, 1, 1, -1, 2)  # Reshape params
        s = self.sigma.view(1, 1, 1, -1, 2)

        # Distance Squared: (Batch, Horizon, N_Agents, N_Obs)
        # We sum over the coordinate dim (last dim)
        exponent = -0.5 * torch.sum(((p - m) ** 2) / (s ** 2 + 1e-6), dim=-1)

        # Calculate Penalty
        gaussian_density = torch.exp(exponent)

        # Sum penalties for ALL agents and ALL obstacles
        # Sum over Agent dim (2) and Obstacle dim (3)
        total_obstacle_cost = gaussian_density.sum(dim=(2, 3))

        loss_obs = self.alpha_obs * total_obstacle_cost.mean()

        total_loss = loss_target + loss_u + loss_obs

        return total_loss, loss_target.item(), loss_u.item(), loss_obs.item()

# OPTION A: Pinball Loss (Targeting (1-alpha) Quantile)
class PinballLossWrapper(nn.Module):
    def __init__(self, alpha, metric):
        super().__init__()
        self.alpha = alpha
        self.metric = metric
        # Initialize tau (threshold) as a learnable parameter
        self.tau = nn.Parameter(torch.tensor(0.0))

    def forward(self, traj_x, traj_u):
        # Ensure metric returns a VECTOR of performance scores [Batch_Size]: Tensor of scalar metrics Psi(D;K) for a batch of datasets.
        performance_scores = self.metric.compute_per_sample_cost(traj_x, traj_u)

        q = 1.0 - self.alpha
        errors = performance_scores - self.tau

        # Pinball loss formula: E[rho(Psi - tau)]
        loss = torch.mean(
            q * torch.relu(errors) + (1 - q) * torch.relu(-errors)
        )
        return loss, performance_scores

# OPTION B: CVaR Loss (Targeting Worst-Case Tail)
class CVaRLossWrapper(nn.Module):
    def __init__(self, alpha, metric):
        super().__init__()
        self.alpha = alpha
        self.metric = metric
        # Initialize tau roughly where you expect the cost to be
        self.tau = nn.Parameter(torch.tensor(0.0))

    def forward(self, traj_x, traj_u):
        # Ensure metric returns a VECTOR of performance scores [Batch_Size]: Tensor of scalar metrics Psi(D;K) for a batch of datasets.
        performance_scores = self.metric.compute_per_sample_cost(traj_x, traj_u)

        # CVaR formula: tau + (1/alpha) * E[max(0, Psi - tau)]
        excess = torch.relu(performance_scores - self.tau)
        loss = self.tau + (1.0 / self.alpha) * torch.mean(excess)

        return loss, performance_scores