import torch
import torch.nn as nn
import torch.nn.functional as F

class PBLoss(nn.Module):
    """
    Base Performance Boosting Loss function.
    Computes tracking, actuation, and collision avoidance costs.
    """
    def __init__(self, x_target, Q, R, lambda_obs, obs_centers, obs_radii_safe, n_agents=1, track_mode='quadratic', coll_mode = 'hinge'):
        """
        Inputs / Dimensions:
            x_target: Target state.
            Q: Tracking penalty matrix.
            R: Actuation penalty matrix.
            lambda_obs: (float) Weight for the collision avoidance penalty.
            obs_centers: List of center coordinates for obstacles.
            obs_sigmas: List of variances for obstacles.
            n_agents: (int) Number of agents in the system.
            track_mode: (str) Mode for tracking loss.
            coll_mode: (str) Mode for collision loss.
            obs_radii_safe: (float or list) Safe radius distance.
        """
        super().__init__()
        self.lambda_obs = lambda_obs
        self.n_agents = n_agents
        self.state_per_agent = 4
        self.track_mode = track_mode
        self.coll_mode = coll_mode

        # Register Buffers for target and penalties
        self.register_buffer('x_target', x_target)
        self.register_buffer('Q', Q)
        self.register_buffer('R', R)

        # 1. Process Obstacle Centers. Shape: [1, 1, 1, num_obs, 2]
        self.register_buffer('mu', torch.stack(obs_centers).view(1, 1, 1, -1, 2))

        self.lambda_agent = 10.0  # Weight penalty for agents hitting each other
        self.agent_margin = 0.2  # Minimum safe distance between two agents (in meters)

        # 2. Process Obstacle Radii

        if isinstance(obs_radii_safe, list):
            processed_radii = []
            for r in obs_radii_safe:
                # If it's a flat number like [0.3, [0.1, 0.5]]
                if isinstance(r, (int, float)):
                    processed_radii.append([float(r), float(r)])
                # If it's a 1-element list like [[0.3], [0.1, 0.5]] (Your exact case)
                elif isinstance(r, list) and len(r) == 1:
                    processed_radii.append([float(r[0]), float(r[0])])
                # If it's a 2-element list for an ellipse
                elif isinstance(r, list) and len(r) == 2:
                    processed_radii.append([float(r[0]), float(r[1])])
                else:
                    raise ValueError(f"Invalid radius format: {r}")

            radii_t = torch.tensor(processed_radii, dtype=torch.float32)

        else:
            # Fallback if the user just passed a single global float like 0.3
            radii_t = torch.tensor(obs_radii_safe, dtype=torch.float32)
            if radii_t.dim() == 0:
                radii_t = radii_t.view(1, 1).repeat(len(obs_centers), 2)

        # Store safe radii and variance. Shape: [1, 1, 1, num_obs, 2]
        self.register_buffer('obs_radii_safe', radii_t.view(1, 1, 1, -1, 2))

        # Variance is defined as (radius / 2)^2
        variance_t = (radii_t / 2.0) ** 2 + 1e-6
        self.register_buffer('variance', variance_t.view(1, 1, 1, -1, 2))

    # --- 1. TRACKING TERM ---
    def compute_tracking_loss(self, error):
        """
        Calculates the cost of being away from target.
        Inputs:  error [batch_size, horizon, state_dim]
        Outputs: cost  [batch_size]
        """
        if self.track_mode == 'quadratic':
            return torch.einsum('bhi,ij,bhj->bh', error, self.Q, error).mean(dim=1)
        elif self.track_mode == 'weighted_euclidean':
            # 1. Compute e^T Q e for every timestep: [Batch, Horizon]
            quad_form = torch.einsum('bhi,ij,bhj->bh', error, self.Q, error)

            # 2. Take the square root to go from quadratic to linear (Euclidean)
            # We add a tiny epsilon (1e-8) to prevent sqrt(0) gradients from exploding
            return torch.sqrt(quad_form + 1e-8).mean(dim=1)
        else:
            raise ValueError("Tracking mode must be 'quadratic' or 'euclidean'")

    # --- 2. COLLISION TERM ---
    def compute_collision_loss(self, pos):
        # ==========================================
        # A. STATIC OBSTACLE COLLISION AVOIDANCE
        # ==========================================
        diff_obs = pos.unsqueeze(3) - self.mu
        # Compute a normalized distance for ellipsoids: norm((x-cx)/rx, (y-cy)/ry)
        # If scaled_dist <= 1.0, the agent is inside the obstacle's safe boundary.
        scaled_diff_obs = diff_obs / self.obs_radii_safe
        scaled_dist_obs = torch.norm(scaled_diff_obs, p=2, dim=-1)

        avg_radii = self.obs_radii_safe.mean(dim=-1)
        violation_obs = torch.relu(1.0 - scaled_dist_obs) * avg_radii

        if self.coll_mode == 'rbf':
            exponent_obs = -0.5 * torch.sum((diff_obs ** 2) / self.variance, dim=-1)
            cost_obs = self.lambda_obs * torch.exp(exponent_obs).sum(dim=(2, 3)).mean(dim=1)

        elif self.coll_mode == 'shifted_rbf':
            exponent_obs = -0.5 * torch.sum((diff_obs ** 2) / self.variance, dim=-1)
            raw_rbf = torch.exp(exponent_obs)
            rbf_at_margin = torch.exp(torch.tensor(-2.0, device=diff_obs.device))
            shifted_rbf = torch.relu(raw_rbf - rbf_at_margin)
            cost_obs = self.lambda_obs * shifted_rbf.sum(dim=(2, 3)).mean(dim=1)

        elif self.coll_mode == 'hinge':
            cost_obs = self.lambda_obs * violation_obs.sum(dim=(2, 3)).mean(dim=1)

        elif self.coll_mode == 'softplus':
            soft_viol_obs = F.softplus(1.0 - scaled_dist_obs, beta=10.0) * avg_radii
            cost_obs = self.lambda_obs * soft_viol_obs.sum(dim=(2, 3)).mean(dim=1)

        elif self.coll_mode == 'squared_hinge':
            cost_obs = self.lambda_obs * (violation_obs ** 2).sum(dim=(2, 3)).mean(dim=1)

        else:
            raise ValueError(f"Unknown collision mode: {self.coll_mode}")

        # ==========================================
        # B. INTER-AGENT COLLISION AVOIDANCE
        # ==========================================
        cost_agents = 0.0

        # Only compute if there are multiple agents and an agent penalty is defined
        lambda_agent = getattr(self, 'lambda_agent', 0.0)

        if self.n_agents > 1 and lambda_agent > 0.0:
            agent_margin = getattr(self, 'agent_margin', 0.2)  # Safe distance between agents

            # Pairwise differences [B, H, N, 1, 2] - [B, H, 1, N, 2] -> [B, H, N, N, 2]
            diff_agents = pos.unsqueeze(3) - pos.unsqueeze(2)
            dist_agents = torch.norm(diff_agents, p=2, dim=-1)  # Shape: [B, H, N, N]

            # Mask out self-collisions (diagonal) by artificially adding a massive distance
            eye = torch.eye(self.n_agents, device=pos.device).view(1, 1, self.n_agents, self.n_agents)
            dist_agents_masked = dist_agents + (eye * 1e6)

            violation_agents = torch.relu(agent_margin - dist_agents_masked)

            # We divide by 2.0 at the end because pair (i,j) and pair (j,i) are counted twice
            if self.coll_mode == 'rbf':
                var_agents = (agent_margin / 2.0) ** 2
                exponent_agents = -0.5 * (dist_agents_masked ** 2) / var_agents
                rbf_agents = torch.exp(exponent_agents) * (1.0 - eye)
                cost_agents = lambda_agent * rbf_agents.sum(dim=(2, 3)).mean(dim=1) / 2.0

            elif self.coll_mode == 'shifted_rbf':
                var_agents = (agent_margin / 2.0) ** 2
                exponent_agents = -0.5 * (dist_agents_masked ** 2) / var_agents
                raw_rbf_agents = torch.exp(exponent_agents)
                rbf_at_margin_agents = torch.exp(torch.tensor(-2.0, device=pos.device))
                shifted_rbf_agents = torch.relu(raw_rbf_agents - rbf_at_margin_agents) * (1.0 - eye)
                cost_agents = lambda_agent * shifted_rbf_agents.sum(dim=(2, 3)).mean(dim=1) / 2.0

            elif self.coll_mode == 'hinge':
                cost_agents = lambda_agent * violation_agents.sum(dim=(2, 3)).mean(dim=1) / 2.0

            elif self.coll_mode == 'softplus':
                soft_viol_agents = F.softplus(agent_margin - dist_agents_masked, beta=10.0) * (1.0 - eye)
                cost_agents = lambda_agent * soft_viol_agents.sum(dim=(2, 3)).mean(dim=1) / 2.0

            elif self.coll_mode == 'squared_hinge':
                cost_agents = lambda_agent * (violation_agents ** 2).sum(dim=(2, 3)).mean(dim=1) / 2.0

        # Return the combined cost!
        return cost_obs + cost_agents

    # --- 3. ACTUATION TERM ---
    def compute_actuation_loss(self, traj_u):
        """
        Calculates the cost of control effort.
        Inputs:  traj_u [batch_size, horizon, input_dim]
        Outputs: cost   [batch_size]
        """
        return torch.einsum('bhi,ij,bhj->bh', traj_u, self.R, traj_u).mean(dim=1)

    # --- 4. MAIN FORWARD (SUMMING THEM UP) ---
    def forward(self, traj_x, traj_u):
        """
        Inputs:
            traj_x: System state trajectories. Shape: [batch_size, horizon, state_dim]
            traj_u: Control input trajectories. Shape: [batch_size, horizon, input_dim]

        Outputs :
            1. total_cost: [scalar tensor] (Optimized loss)
            2. cost_x: [float] (Itemized mean for logging)
            3. cost_u: [float] (Itemized mean for logging)
            4. cost_coll: [float] (Itemized mean for logging)
        """
        batch_size, horizon, _ = traj_x.shape
        error = traj_x - self.x_target
        pos = traj_x.view(batch_size, horizon, self.n_agents, self.state_per_agent)[..., :2]

        # Call individual components
        cost_x = self.compute_tracking_loss(error)
        cost_u = self.compute_actuation_loss(traj_u)
        cost_coll = self.compute_collision_loss(pos)

        # Sum them up
        total_cost = cost_x + cost_u + cost_coll

        return total_cost, cost_x, cost_u, cost_coll

class ERMWrapper(nn.Module):
    def __init__(self, metric):
        super().__init__()
        self.metric = metric

    def forward(self, traj_x, traj_u):
        # Retrieve the individual vectors of shape [batch_size]
        total_costs, cost_x, cost_u, cost_coll = self.metric(traj_x, traj_u)

        # Calculate the mean (ERM)
        total_loss = total_costs.mean()

        return total_loss, cost_x.mean(), cost_u.mean(), cost_coll.mean()


class CVaRLossWrapper(nn.Module):
    def __init__(self, alpha, metric, tau_init=0.0):
        super().__init__()
        self.alpha = alpha
        self.metric = metric

        # tau is the learnable threshold for the (1-alpha) quantile
        self.tau = nn.Parameter(torch.tensor(tau_init))

    def forward(self, traj_x, traj_u):
        """
        Inputs:
            traj_x: [batch_size, horizon, state_dim]
            traj_u: [batch_size, horizon, input_dim]

        Outputs:
            Tuple of 4 elements:
            1. cvar_loss: [scalar tensor] The CVaR loss used for backpropagation.
            2. expected_x: [float] Mean tracking cost for logging.
            3. expected_u: [float] Mean actuation cost for logging.
            4. expected_coll: [float] Mean collision cost for logging.
        """
        # Retrieve the individual cost vectors of shape [Batch_Size]
        total_costs, cost_x, cost_u, cost_coll = self.metric(traj_x, traj_u)

        # Apply CVaR formula to the total weighted cost
        excess = torch.relu(total_costs - self.tau)
        cvar_loss = self.tau + (1.0 / self.alpha) * torch.mean(excess)

        # Return the optimized CVaR loss, plus the standard averages for logging
        return cvar_loss, cost_x.mean(), cost_u.mean(), cost_coll.mean()

class SplitCVaRLossWrapper(nn.Module):
    def __init__(self, alpha, lambda_decoupling, metric, tau_init=0.0):
        super().__init__()
        self.alpha = alpha
        self.lambda_decoupling = lambda_decoupling
        self.metric = metric
        self.tau = nn.Parameter(torch.tensor(tau_init))

    def forward(self, traj_x, traj_u):
        """
        Inputs:
            traj_x: [batch_size, horizon, state_dim]
            traj_u: [batch_size, horizon, input_dim]

        Outputs:
            Tuple of 4 elements:
            1. total_loss: [scalar tensor] Expected performance + CVaR collision risk.
            2. expected_x: [float] Mean tracking cost for logging.
            3. expected_u: [float] Mean actuation cost for logging.
            4. expected_coll: [float] Mean collision cost for logging.
        """
        # Retrieve the individual vectors of shape [Batch_Size]
        _, cost_x, cost_u, cost_coll = self.metric(traj_x, traj_u)

        # 1. ERM on tracking performance
        expected_perf = cost_x.mean() + cost_u.mean()

        # 2. CVaR on collision penalty
        excess_coll = torch.relu(cost_coll - self.tau)
        cvar_coll = self.tau + (1.0 / self.alpha) * torch.mean(excess_coll)

        # 3. Total Loss
        total_loss = expected_perf + (self.lambda_decoupling * cvar_coll)

        return total_loss, cost_x.mean(), cost_u.mean(), cost_coll.mean()

class LagrangianCVaRLossWrapper(nn.Module):
    def __init__(self, alpha, tau_safe_bar, metric, tau_init=0.0, lambda_init=1.0):
        super().__init__()
        self.alpha = alpha
        self.tau_safe_bar = tau_safe_bar
        self.metric = metric

        # Primal parameter (Minimized)
        self.tau = nn.Parameter(torch.tensor(tau_init))

        # Dual parameter (Maximized).
        # We optimize a raw value and apply softplus to guarantee lambda >= 0
        self.pre_lambda = nn.Parameter(torch.tensor(lambda_init))

    def forward(self, traj_x, traj_u):
        """
        Inputs:
            traj_x: [batch_size, horizon, state_dim]
            traj_u: [batch_size, horizon, input_dim]

        Outputs:
            Tuple of 5 elements:
            1. lagrangian: [scalar tensor] The combined objective to be minimized/maximized.
            2. expected_perf: [float] Sum of tracking + actuation costs.
            3. cvar_coll: [float] Calculated CVaR value of collisions.
            4. lambda_dual: [float] Current value of the dual multiplier.
            5. constraint_violation: [float] Raw margin of safety violation.
        """
        # Retrieve the individual vectors of shape [Batch_Size]
        _, cost_x, cost_u, cost_coll = self.metric(traj_x, traj_u)

        # 1. ERM on tracking performance (Psi_perf)
        expected_perf = cost_x.mean() + cost_u.mean()

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

        return lagrangian, expected_perf, cvar_coll, lambda_dual, constraint_violation

class LagrangianERMLossWrapper(nn.Module):
    def __init__(self, alpha, tau_safe_bar, metric, lambda_init=0.0):
        super().__init__()
        self.alpha = alpha
        self.tau_safe_bar = tau_safe_bar
        self.metric = metric

        # Dual parameter (Maximized).
        # We optimize a raw value and apply softplus to guarantee lambda >= 0
        self.pre_lambda = nn.Parameter(torch.tensor(lambda_init))

    def forward(self, traj_x, traj_u):
        """
        Inputs:
            traj_x: [batch_size, horizon, state_dim]
            traj_u: [batch_size, horizon, input_dim]

        Outputs:
            Tuple of 5 elements:
            1. lagrangian: [scalar tensor] The combined objective to be minimized/maximized.
            2. expected_perf: [float] Sum of tracking + actuation costs.
            3. expected_coll: [float] Calculated expected value of collisions.
            4. lambda_dual: [float] Current value of the dual multiplier.
            5. constraint_violation: [float] Raw margin of safety violation.
        """
        # Retrieve the individual vectors of shape [Batch_Size]
        _, cost_x, cost_u, cost_coll = self.metric(traj_x, traj_u)

        # 1. ERM on tracking performance (Psi_perf)
        expected_perf = cost_x.mean() + cost_u.mean()

        # 2. ERM on collision penalty (Psi_safe)
        expected_coll = cost_coll.mean()

        # 3. Constraint Violation
        # If expected_coll > tau_safe_bar, this is positive (penalized).
        # If expected_coll < tau_safe_bar, this is negative (constraint satisfied).
        constraint_violation = expected_coll - self.tau_safe_bar

        # 4. Lagrangian Formulation
        lambda_dual = F.softplus(self.pre_lambda)
        lagrangian = expected_perf + (lambda_dual * constraint_violation)

        return lagrangian, expected_perf, expected_coll, lambda_dual, constraint_violation

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

    def forward(self, traj_x, traj_u):
        """
        Inputs:
            traj_x: [batch_size, horizon, state_dim]
            traj_u: [batch_size, horizon, input_dim]

        Outputs:
            Tuple of 4 elements:
            1. softmax_loss: [scalar tensor] The smoothed max loss used for optimization.
            2. expected_x: [float] Mean tracking cost for logging.
            3. expected_u: [float] Mean actuation cost for logging.
            4. expected_coll: [float] Mean collision cost for logging.
        """
        # Retrieve the individual cost vectors of shape [Batch_Size]
        total_costs, cost_x, cost_u, cost_coll = self.metric(traj_x, traj_u)

        # Apply the log-sum-exp smooth approximation of the maximum to the total costs
        # Formula: (1/beta) * log(sum(exp(beta * raw_costs)))
        softmax_loss = torch.logsumexp(self.beta * total_costs, dim=0) / self.beta

        # Return the optimized Softmax loss, plus the standard averages for logging
        return softmax_loss, cost_x.mean(), cost_u.mean(), cost_coll.mean()

class PinballLossWrapper(nn.Module):
    def __init__(self, alpha, metric, tau_init=0.0):
        super().__init__()
        self.alpha = alpha
        self.metric = metric
        self.tau = nn.Parameter(torch.tensor(tau_init))

    def forward(self, traj_x, traj_u):
        """
        Inputs:
            traj_x: [batch_size, horizon, state_dim]
            traj_u: [batch_size, horizon, input_dim]

        Outputs:
            Tuple of 4 elements:
            1. total_pinball_loss: [scalar tensor] The loss used for backpropagation.
            2. expected_x: [float] Mean tracking cost for logging.
            3. expected_u: [float] Mean actuation cost for logging.
            4. expected_coll: [float] Mean collision cost for logging.
        """
        # Retrieve the individual vectors
        total_costs, cost_x, cost_u, cost_coll = self.metric(traj_x, traj_u)

        q = 1.0 - self.alpha
        errors = total_costs - self.tau
        total_pinball_loss = torch.mean(q * torch.relu(errors) + (1 - q) * torch.relu(-errors))

        # Return total loss, and the expected tracking/collision values for logging
        return total_pinball_loss, cost_x.mean(), cost_u.mean(), cost_coll.mean()