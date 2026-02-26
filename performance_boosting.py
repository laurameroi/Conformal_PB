import torch, time, copy
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

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
    def __init__(self, x_target, Q, R, lambda_obs, obs_centers, obs_sigmas,
                 n_agents=1, reduction='mean', collision_type='rbf', r_safe=1.0):
        super().__init__()
        self.lambda_obs = lambda_obs
        self.n_agents = n_agents
        self.state_per_agent = 4
        self.reduction = reduction

        # --- NEW FLAGS FOR COLLISION AVOIDANCE ---
        self.collision_type = collision_type.lower()
        self.r_safe = r_safe  # The physical radius used for the hinge loss

        # Register Buffers (Pre-computed for optimization)
        self.register_buffer('x_target', x_target)
        self.register_buffer('Q', Q)
        self.register_buffer('R', R)
        self.register_buffer('mu', torch.stack(obs_centers).view(1, 1, 1, -1, 2))
        self.register_buffer('variance', (torch.stack(obs_sigmas).view(1, 1, 1, -1, 2) ** 2) + 1e-6)

    def forward(self, traj_x, traj_u):
        batch_size, horizon, _ = traj_x.shape

        # --- 1. Compute Per-Sample Costs [Shape: Batch_Size] ---
        error = traj_x - self.x_target
        cost_x = torch.einsum('bhi,ij,bhj->bh', error, self.Q, error).mean(dim=1)
        cost_u = torch.einsum('bhi,ij,bhj->bh', traj_u, self.R, traj_u).mean(dim=1)

        pos = traj_x.view(batch_size, horizon, self.n_agents, self.state_per_agent)[..., :2]

        # --- 2. Collision Avoidance Logic ---
        if self.collision_type == 'rbf':
            # ORIGINAL: Gaussian Force Field
            exponent = -0.5 * torch.sum(((pos.unsqueeze(3) - self.mu) ** 2) / self.variance, dim=-1)
            cost_coll = self.lambda_obs * torch.exp(exponent).sum(dim=(2, 3)).mean(dim=1)

        elif self.collision_type == 'hinge':
            # NEW: Physical Penetration Depth
            diff = pos.unsqueeze(3) - self.mu
            distances = torch.norm(diff, p=2, dim=-1)  # Euclidean distance [batch, horizon, n_agents, n_obs]

            # softplus instead of ReLU zeros out anything outside the r_safe radius beta controls the sharpness.
            # beta=10 means the warning zone extends about 0.5 meters outside r_safe.
            #penetration = F.softplus(self.r_safe - distances, beta=10.0)
            penetration = torch.relu(self.r_safe - distances)

            # lambda_obs now acts as the penalty multiplier (e.g., penalty per meter of penetration)
            cost_coll = self.lambda_obs * penetration.sum(dim=(2, 3)).mean(dim=1)

        elif self.collision_type == 'squared_hinge':
            diff = pos.unsqueeze(3) - self.mu
            distances = torch.norm(diff, p=2, dim=-1)

            # ReLU strictly zeros out safe distances
            violation = torch.relu(self.r_safe - distances)

            # Square the physical violation
            squared_penetration = violation ** 2

            cost_coll = self.lambda_obs * squared_penetration.sum(dim=(2, 3)).mean(dim=1)

        else:
            raise ValueError(f"Unknown collision_type: '{self.collision_type}'. Use 'rbf' or 'hinge'.")

        # --- 3. Return based on Reduction ---
        if self.reduction == 'none':
            # Used by Wrappers: Return the raw arrays for the whole batch
            total_cost = cost_x + cost_u + cost_coll
            return total_cost, cost_x, cost_u, cost_coll
        else:
            # Used by ERM: Return the scalar means
            loss_x = cost_x.mean()
            loss_u = cost_u.mean()
            loss_coll = cost_coll.mean()
            total_loss = loss_x + loss_u + loss_coll

            return total_loss, loss_x.item(), loss_u.item(), loss_coll.item()

class PinballLossWrapper(nn.Module):
    def __init__(self, alpha, metric):
        super().__init__()
        self.alpha = alpha
        self.metric = metric

        # Force the base loss to return per-sample vectors instead of means
        self.metric.reduction = 'none'
        self.tau = nn.Parameter(torch.tensor(0.0))

    def forward(self, traj_x, traj_u):
        # Retrieve the individual vectors
        total_costs, cost_x, cost_u, cost_coll = self.metric(traj_x, traj_u)

        q = 1.0 - self.alpha
        errors = total_costs - self.tau
        total_pinball_loss = torch.mean(q * torch.relu(errors) + (1 - q) * torch.relu(-errors))

        # Return total loss, and the expected tracking/collision values for logging
        return total_pinball_loss, cost_x.mean().item(), cost_u.mean().item(), cost_coll.mean().item()


class CVaRLossWrapper(nn.Module):
    def __init__(self, alpha, metric):
        super().__init__()
        self.alpha = alpha
        self.metric = metric

        # Force the base loss to return per-sample vectors instead of means
        self.metric.reduction = 'none'

        # tau is the learnable threshold for the (1-alpha) quantile
        self.tau = nn.Parameter(torch.tensor(0.0))

    def forward(self, traj_x, traj_u):
        # Retrieve the individual cost vectors of shape [Batch_Size]
        total_costs, cost_x, cost_u, cost_coll = self.metric(traj_x, traj_u)

        # Apply CVaR formula to the total weighted cost
        excess = torch.relu(total_costs - self.tau)
        cvar_loss = self.tau + (1.0 / self.alpha) * torch.mean(excess)

        # Return the optimized CVaR loss, plus the standard averages for logging
        return cvar_loss, cost_x.mean().item(), cost_u.mean().item(), cost_coll.mean().item()

class SplitCVaRLossWrapper(nn.Module):
    def __init__(self, alpha, lambda_decoupling, metric):
        super().__init__()
        self.alpha = alpha
        self.lambda_decoupling = lambda_decoupling
        self.metric = metric

        # Force the base loss to return per-sample vectors instead of means
        self.metric.reduction = 'none'
        self.tau = nn.Parameter(torch.tensor(0.0))

    def forward(self, traj_x, traj_u):
        # Retrieve the individual vectors of shape [Batch_Size]
        _, cost_x, cost_u, cost_coll = self.metric(traj_x, traj_u)

        # 1. ERM on tracking performance
        expected_x = cost_x.mean()
        expected_u = cost_u.mean()
        expected_perf = expected_x + expected_u

        # 2. CVaR on collision penalty
        excess_coll = torch.relu(cost_coll - self.tau)
        cvar_coll = self.tau + (1.0 / self.alpha) * torch.mean(excess_coll)

        # 3. Total Loss
        total_loss = expected_perf + (self.lambda_decoupling * cvar_coll)

        return total_loss, expected_x.item(), expected_u.item(), cost_coll.mean().item()

class HardConstraintCVaRLossWrapper(nn.Module):
    def __init__(self, alpha, tau_safe_bar, metric):
        super().__init__()
        self.alpha = alpha
        self.tau_safe_bar = tau_safe_bar
        self.metric = metric

        # Force the base loss to return per-sample vectors instead of means
        self.metric.reduction = 'none'

        # Primal parameter (Minimized)
        self.tau = nn.Parameter(torch.tensor(0.0))

        # Dual parameter (Maximized).
        # We optimize a raw value and apply softplus to guarantee lambda >= 0
        self.pre_lambda = nn.Parameter(torch.tensor(0.0))

    def forward(self, traj_x, traj_u):
        # Retrieve the individual vectors of shape [Batch_Size]
        _, cost_x, cost_u, cost_coll = self.metric(traj_x, traj_u)

        # 1. ERM on tracking performance (Psi_perf)
        expected_x = cost_x.mean()
        expected_u = cost_u.mean()
        expected_perf = expected_x + expected_u

        # 2. CVaR on collision penalty (Psi_safe)
        excess_coll = torch.relu(cost_coll - self.tau)
        cvar_coll = self.tau + (1.0 / self.alpha) * torch.mean(excess_coll)

        # 3. Constraint Violation
        # If cvar_coll > tau_safe_bar, this is positive (penalized).
        # If cvar_coll < tau_safe_bar, this is negative (constraint satisfied).
        constraint_violation = cvar_coll - self.tau_safe_bar

        # 4. Lagrangian Formulation
        lambda_dual = F.softplus(self.pre_lambda)
        lagrangian = expected_perf + (lambda_dual * constraint_violation)

        return lagrangian, expected_perf.item(), cvar_coll.item(), lambda_dual.item(), constraint_violation.item()

class SoftmaxWorstCaseLossWrapper(nn.Module):
    def __init__(self, beta, metric):
        """
        Args:
            beta: Temperature parameter. Higher beta = closer to the true maximum.
            metric: The PBLoss instance.
        """
        super().__init__()
        self.beta = beta
        self.metric = metric

        # Force the base loss to return per-sample vectors instead of means
        self.metric.reduction = 'none'

    def forward(self, traj_x, traj_u):
        # Retrieve the individual cost vectors of shape [Batch_Size]
        total_costs, cost_x, cost_u, cost_coll = self.metric(traj_x, traj_u)

        # Apply the log-sum-exp smooth approximation of the maximum to the total costs
        # Formula: (1/beta) * log(sum(exp(beta * raw_costs)))
        softmax_loss = torch.logsumexp(self.beta * total_costs, dim=0) / self.beta

        # Return the optimized Softmax loss, plus the standard averages for logging
        return softmax_loss, cost_x.mean().item(), cost_u.mean().item(), cost_coll.mean().item()

class BarrierLoss(nn.Module):
    def __init__(self, x_target, obs_centers, obs_radii,
                 margin=0.05, weight_obs=10000.0, weight_u=0.01):
        """
        A "Hard Constraint" Loss function (Code 3 Logic) wrapped in a professional class.

        Args:
            x_target (Tensor): Target state [state_dim].
            obs_centers (list of Tensors): List of [2] tensors (x, y) for obstacles.
            obs_radii (list of floats): List of radii for each obstacle.
            margin (float): Safety distance (buffer zone) around the obstacle.
            weight_obs (float): Penalty weight when hitting the barrier (default: 10000.0).
            weight_u (float): Weight for control effort (default: 0.01).
        """
        super().__init__()

        # --- 1. Configuration ---
        self.margin = margin
        self.alpha_obs = weight_obs
        self.weight_u = weight_u

        # --- 2. Register Buffers (Handles Device Moves Automatically) ---
        # We assume x_target is on CPU initially, register_buffer moves it to GPU if needed
        self.register_buffer('x_target', x_target)

        # --- 3. Setup Obstacles ---
        # Stack centers: (N_obs, 2)
        # We ensure they are floats to match trajectory types
        self.register_buffer('obs_centers', torch.stack(obs_centers).float())

        # Stack radii: (N_obs,)
        # Convert list of floats to a tensor
        if isinstance(obs_radii, list):
            radii_tensor = torch.tensor(obs_radii).float()
        else:
            radii_tensor = obs_radii

        self.register_buffer('obs_radii', radii_tensor)

    def forward(self, traj_x, traj_u):
        """
        traj_x: (Batch, Horizon, state_dim)
        traj_u: (Batch, Horizon, input_dim)
        """
        # --- 1. Target Cost (Euclidean Distance) ---
        # Code 3 Logic: dist = norm(traj - target)
        # Broadcasting: (B, H, S) - (S) -> (B, H, S)
        error = traj_x - self.x_target
        # dim=2 computes norm across the state dimension
        dist_to_target = torch.norm(error, dim=2)
        loss_target = dist_to_target.mean()

        # --- 2. Obstacle Cost (ReLU Barrier) ---
        # Extract (x, y) positions: (Batch, Horizon, 2)
        pos = traj_x[:, :, :2]

        # Vectorized Distance Calculation:
        # pos: (Batch, Horizon, 1, 2)
        # obs: (1, 1, N_obs, 2)
        p = pos.unsqueeze(2)
        c = self.obs_centers.view(1, 1, -1, 2)

        # Compute distance to ALL obstacles at once
        # dists shape: (Batch, Horizon, N_obs)
        dists = torch.norm(p - c, dim=-1)

        # Get radii for broadcasting: (1, 1, N_obs)
        r = self.obs_radii.view(1, 1, -1)

        # Code 3 Logic: violation = ReLU((radius + margin) - dist)
        # If dist < limit, result is positive (Penalty).
        # If dist > limit, result is negative -> ReLU makes it 0.0 (No Penalty).
        violation = torch.relu((r + self.margin) - dists)

        # Square the violation (Quadratic Penalty for smoothness upon impact)
        loss_obs = self.alpha_obs * (violation ** 2).mean()

        # --- 3. Control Cost (Simple Energy) ---
        # Code 3 Logic: mean(u^2)
        loss_u = self.weight_u * (traj_u ** 2).mean()

        total_loss = loss_target + loss_obs + loss_u

        return total_loss, loss_target.item(), loss_u.item(), loss_obs.item()


import torch
import torch.nn as nn
import torch.nn.functional as F


class PBLossnew(nn.Module):
    def __init__(self, x_target, Q, R, lambda_obs, obs_centers, obs_sigmas,
                 n_agents=1, reduction='mean', r_safe=1.0, weight_u=0.01):
        super().__init__()
        self.lambda_obs = lambda_obs
        self.n_agents = n_agents
        self.state_per_agent = 4
        self.reduction = reduction
        self.weight_u = weight_u

        # Handle r_safe: Allow it to be a list or a float
        if isinstance(r_safe, list):
            self.register_buffer('r_safe', torch.tensor(r_safe).float().view(1, 1, 1, -1))
        else:
            self.r_safe = float(r_safe)

        # Register Buffers
        self.register_buffer('x_target', x_target)
        self.register_buffer('Q', Q)
        self.register_buffer('R', R)
        self.register_buffer('mu', torch.stack(obs_centers).view(1, 1, 1, -1, 2))
        self.register_buffer('variance', (torch.stack(obs_sigmas).view(1, 1, 1, -1, 2) ** 2) + 1e-6)

    # --- 1. TRACKING TERM ---
    def compute_tracking_loss(self, error, mode='quadratic'):
        """
        Calculates the cost of being away from target.
        mode 'quadratic': e^T Q e (U-shape bowl)
        mode 'euclidean': ||e||_2  (V-shape cone)
        """
        if mode == 'quadratic':
            return torch.einsum('bhi,ij,bhj->bh', error, self.Q, error).mean(dim=1)
        elif mode == 'euclidean':
            return torch.norm(error, dim=-1).mean(dim=1)
        else:
            raise ValueError("Tracking mode must be 'quadratic' or 'euclidean'")

    # --- 2. COLLISION TERM ---
    def compute_collision_loss(self, pos, mode='hinge'):
        """
        Calculates the penalty for being near obstacles.
        """
        batch_size = pos.shape[0]

        if mode == 'rbf':
            exponent = -0.5 * torch.sum(((pos.unsqueeze(3) - self.mu) ** 2) / self.variance, dim=-1)
            return self.lambda_obs * torch.exp(exponent).sum(dim=(2, 3)).mean(dim=1)

        # For all distance-based methods
        diff = pos.unsqueeze(3) - self.mu
        distances = torch.norm(diff, p=2, dim=-1)  # [B, H, Agents, Obs]
        violation = torch.relu(self.r_safe - distances)

        if mode == 'hinge':
            return self.lambda_obs * violation.sum(dim=(2, 3)).mean(dim=1)

        elif mode == 'softplus':
            soft_viol = F.softplus(self.r_safe - distances, beta=10.0)
            return self.lambda_obs * soft_viol.sum(dim=(2, 3)).mean(dim=1)

        elif mode == 'squared_hinge':
            return self.lambda_obs * (violation ** 2).sum(dim=(2, 3)).mean(dim=1)

        elif mode == 'barrier':
            # Matches your old code: Mean over time AND obstacles
            return self.lambda_obs * (violation ** 2).view(batch_size, -1).mean(dim=1)

        else:
            raise ValueError(f"Unknown collision mode: {mode}")

    # --- 3. ACTUATION TERM ---
    def compute_actuation_loss(self, traj_u, mode='quadratic'):
        """
        Calculates the cost of control effort.
        """
        if mode == 'quadratic':
            return torch.einsum('bhi,ij,bhj->bh', traj_u, self.R, traj_u).mean(dim=1)
        elif mode == 'simple':
            # Scalar weight * u^2
            return self.weight_u * (traj_u ** 2).mean(dim=(1, 2))
        else:
            raise ValueError("Actuation mode must be 'quadratic' or 'simple'")

    # --- 4. MAIN FORWARD (SUMMING THEM UP) ---
    def forward(self, traj_x, traj_u, track_mode='quadratic', coll_mode='hinge', act_mode='quadratic'):
        batch_size, horizon, _ = traj_x.shape
        error = traj_x - self.x_target
        pos = traj_x.view(batch_size, horizon, self.n_agents, self.state_per_agent)[..., :2]

        # Call individual components
        cost_x = self.compute_tracking_loss(error, mode=track_mode)
        cost_u = self.compute_actuation_loss(traj_u, mode=act_mode)
        cost_coll = self.compute_collision_loss(pos, mode=coll_mode)

        # Sum them up
        total_cost = cost_x + cost_u + cost_coll

        # Final reduction for PyTorch
        if self.reduction == 'none':
            return total_cost, cost_x, cost_u, cost_coll
        else:
            loss_x, loss_u, loss_coll = cost_x.mean(), cost_u.mean(), cost_coll.mean()
            return (loss_x + loss_u + loss_coll), loss_x.item(), loss_u.item(), loss_coll.item()