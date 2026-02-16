import torch
import torch.nn as nn
import torch.nn.functional as F

class RobotPlant(torch.nn.Module):
    def __init__(self, m: float = 1.0, ts: float = 0.05,
                 b: float = 1, b2: float = 0.1, n_agents: int = 1):
        """
        Initializes the robot system model.
        """
        super().__init__()  # Initializes the parent torch.nn.Module class.

        # --- Physics Parameters ---
        self.m = m  # Mass of the robot (kg). Used to calculate acceleration from force (F=ma).
        self.ts = ts  # Sampling time (seconds). The duration of each discrete time step.
        self.b = b  # Linear drag coefficient (b1). Represents viscous friction.
        self.b2 = b2  # Nonlinear drag coefficient (b2). Represents the nonlinear friction component.

        # --- Dimensions for each agent ---
        self.n_agents = n_agents  # Number of robots. Hardcoded to 1 as requested.
        self.state_dim_agent = 4  # State vector size per robot: [pos_x, pos_y, vel_x, vel_y].
        self.input_dim_agent = 2  # Input vector size per robot: [force_x, force_y].
        self.output_dim_agent = 4  # Output vector size. We observe the full state.

        # Total dimensions
        self.state_dim = self.state_dim_agent * self.n_agents
        self.input_dim = self.input_dim_agent * self.n_agents
        self.output_dim = self.output_dim_agent * self.n_agents

        # Initialize placeholders for buffers (so we can update them later)
        # We register them as buffers so they are saved with the model state_dict
        self.register_buffer('A', torch.zeros(self.state_dim, self.state_dim))
        self.register_buffer('B', torch.zeros(self.state_dim, self.input_dim))
        # Initialize an internal buffer to hold the current simulation state.
        # 'None' means simulation hasn't started.
        self.register_buffer('x', None)

        # --- COMPUTE MATRICES INITIAL VALUES ---
        # We call the update function immediately to populate A and B
        self.update_params(m, b, b2)

    def update_params(self, m=None, b=None, b2=None):
        """
        Updates the physics parameters and RE-CALCULATES the system matrices (A, B).
        Call this method inside the training loop for Domain Randomization.
        """
        # 1. Update internal values if provided
        if m is not None: self.m = m
        if b is not None: self.b = b
        if b2 is not None: self.b2 = b2

        # --- Stability Check ---
        # If b2 > b1, the effective drag at low speeds (b1 - b2) becomes negative, causing the system to be unstable at the origin.
        assert 0 <= self.b2 < self.b, f"Stability violation: b2 ({self.b2}) must be < b ({self.b})"

        # 3. Detect Device (CPU or GPU) to ensure new matrices match the model
        device = self.A.device if hasattr(self, 'A') else torch.device('cpu')

        # --- B Matrix (Input -> State map) ---
        Bi = torch.tensor([
            [0, 0],
            [0, 0],
            [1 / self.m, 0],  # Impacted by new Mass
            [0, 1 / self.m]  # Impacted by new Mass
        ], device=device) * self.ts

        # Expand Bi for N agents
        B = torch.kron(torch.eye(self.n_agents, device=device), Bi)

        # Check that B is size (4, 2).
        assert B.shape == (self.state_dim, self.input_dim)

        # --- A Matrix (State -> State map) ---
        Ai = torch.eye(self.state_dim_agent, device=device)  # Start with identity matrix (state stays same).

        # Kinematics: p_new = p + v * ts
        Ai[0, 2] = self.ts
        Ai[1, 3] = self.ts

        # 2. Linear Friction: v_new = v - (b/m)*v*ts  =>  v_new = (1 - b*ts/m) * v
        linear_decay = 1.0 - (self.b * self.ts / self.m)
        Ai[2, 2] = linear_decay
        Ai[3, 3] = linear_decay

        # Expand A for N agents.
        A = torch.kron(torch.eye(self.n_agents, device=device), Ai)

        # Check that A is size (4, 4).
        assert A.shape == (self.state_dim, self.state_dim)

        # 4. Overwrite the buffers
        # Note: We assign directly. Since they are buffers, this works fine.
        self.A = A
        self.B = B

    def non_linear_drag_force(self, x):
        """
        Computes the nonlinear component of the drag force.
        The total drag model is: C(v) = b*v - b2*tanh(v), the linear part is handled by the A matrix (linear decay).

        Args:
            x (torch.Tensor): Current state. Shape = (batch_size, 1, state_dim).
                              Structure: [pos_x, pos_y, vel_x, vel_y]

        Returns:
            torch.Tensor: The state update delta. Shape = (batch_size, 1, state_dim).
                          Structure: [0, 0, delta_vx, delta_vy]
        """
        # Unpack dimensions
        batch_size, _, state_dim = x.shape

        # Reshape to separate agents: (batch, 1, n_agents, state_dim_per_agent)
        # This allows us to easily slice positions vs velocities regardless of batch size
        x_reshaped = x.view(batch_size, 1, self.n_agents, self.state_dim_agent)

        # Extract velocities 'v' (indices 2 and 3 for each agent)
        # Shape: (batch_size, 1, n_agents, 2)
        v = x_reshaped[..., 2:]

        # --- Compute Nonlinear Velocity Delta ---
        # Calculate: delta_v = (Ts / m) * b2 * tanh(v)
        # This term opposes the linear drag in the A matrix, reducing the total friction at higher speeds.
        delta_v = (self.ts / self.m) * self.b2 * torch.tanh(v)

        # Positions are not directly affected by drag forces in this time step
        delta_p = torch.zeros_like(x_reshaped[..., :2])  # shape: (batch_size, 1, n_agents, 2)

        # Concatenate updates (unchanged p and updated v): [delta_p, delta_v] -> [0, 0, dv_x, dv_y]
        delta_state = torch.cat((delta_p, delta_v), dim=-1)  # shape: (batch_size, 1, n_agents, 4)

        # Flatten back to original input shape: (batch_size, 1, state_dim)
        return delta_state.view(batch_size, 1, state_dim)

    def forward(self, u):
        """
        Compute the next state (and output) of the system at time t+1.

        Dynamics: x_{t+1} = A*x_t + B*u_t + Nonlinear_Drag(x_t)

        Args:
            u (torch.Tensor): Control input (Forces) at time t.
                              Shape = (batch_size, 1, input_dim)

        Returns:
            torch.Tensor: The updated state x_{t+1}.
                          Shape = (batch_size, 1, output_dim)
        """
        # Safety check to ensure reset() was called
        if self.x is None:
            raise ValueError("State not initialized. Call `reset()` before using forward().")

        # --- 1. Linear Dynamics ---
        # Calculate the contributions from the linear system matrices.
        x_lin = F.linear(self.x, self.A) + F.linear(u, self.B)

        # --- 2. Nonlinear Dynamics ---
        # Calculate drag perturbation based on the *previous* state x_t
        x_drag = self.non_linear_drag_force(self.x)

        # Add to state: x_{new} = x_{linear} + x_{drag}
        self.x = x_lin + x_drag
        # Return the updated state (OUTPUT = STATE)
        return self.x

    def predict_nominal_next_state(self, x_meas, u):

        # --- 1. Linear Dynamics ---
        # Calculate the contributions from the linear system matrices.
        x_lin = F.linear(x_meas, self.A) + F.linear(u, self.B)

        # --- 2. Nonlinear Dynamics ---
        x_drag = self.non_linear_drag_force(x_meas)

        x_hat = x_lin + x_drag

        return x_hat

    def reset(self, x0, batch_size=1):
        """
        Resets the internal state of the plant.

        Args:
            x0: Initial state. Can be None, scalar (int/float), or Tensor.
            batch_size: Batch size for broadcasting.
        """
        device = self.A.device  # Use system device as reference

        # Case 1: None -> Initialize with zeros
        if x0 is None:
            self.x = torch.zeros(batch_size, 1, self.state_dim, device=device)

        # Case 2: Scalar input (e.g. 0) -> Initialize with that value (broadcasted)
        elif isinstance(x0, (int, float)):
            self.x = torch.full((batch_size, 1, self.state_dim), float(x0), device=device)

        # Case 3: Tensor input
        elif isinstance(x0, torch.Tensor):
            if x0.dim() == 0:  # 0-d tensor (scalar tensor)
                self.x = torch.full((batch_size, 1, self.state_dim), x0.item(), device=device)
            elif x0.dim() == 1:  # Single vector -> Broadcast
                self.x = x0.view(1, 1, -1).expand(batch_size, -1, -1).to(device)
            else:  # Batch input -> Move to device
                self.x = x0.to(device)
        else:
            raise ValueError(f"Unsupported type for x0: {type(x0)}")

    def run(self, x0, u_ext, output_noise=None):

        """
        Simulates the open-loop system for a given initial condition and external signal.

        Args:
            x0 (torch.Tensor): Initial state. If None, defaults to zero. Shape (batch, 1, state_dim).
            u_ext (torch.Tensor): Trajectories of external input signal. Shape = (batch_size, horizon, input_dim).
            output_noise: realizations of output noise Shape (batch, horizon + 1, output_dim).

        Returns:
            torch.Tensor: Trajectories of outputs (batch_size, horizon, output_dim)
        """
        # 1. Setup Dimensions
        batch_size, horizon, _ = u_ext.shape

        # 2. Initialize Internal State
        self.reset(x0, batch_size)

        # 3. Handle Noise
        # Note: We need (horizon + 1) noise samples because we observe y0 ... yT
        if output_noise is None:
            output_noise = torch.zeros(batch_size, horizon + 1, self.output_dim, device=self.A.device)

        # 4. Simulation Loop
        y_traj = []

        # Initial output (t=0)
        # Assuming y = x (full state observation)
        y = self.x + output_noise[:, 0:1, :]

        for t in range(horizon):
            y_traj.append(y)  # Store state at start of step

            # Update state: x_{t+1} = Forward(x_t, u_t)
            # We access noise at t+1 because this is the observation AFTER the step
            self.forward(u_ext[:, t:t + 1, :])
            y = self.x + output_noise[:, t + 1:t + 2, :]

        # Concatenate list of tensors into one large tensor
        return torch.cat(y_traj, dim=1)  # Shape: (batch_size, horizon, output_dim)

    def __call__(self, x0, u_ext, output_noise):
        """

        Args:
            x0 (torch.Tensor): Initial state. If None, defaults to zero. Shape (batch, 1, state_dim).
            u_ext (torch.Tensor): Trajectories of external input signal. Shape = (batch_size, horizon, input_dim).
            output_noise: realizations of output noise Shape (batch, horizon + 1, output_dim).

        Returns:
            torch.Tensor: Trajectories of outputs (batch_size, horizon, output_dim)
        """
        return self.run(x0, u_ext, output_noise)

class ProportionalController(nn.Module):
    def __init__(self, kp=None, y_target=None, n_agents=2):
        """
        Initializes the Proportional Controller.

        This controller is designed to work with the 'RobotPlant' class.
        It implements the control law: u = Kp * (p_target - p)

        Args:
            kp (torch.Tensor, optional): Gain matrix. If None, defaults to identity (spring-like).
            y_target (torch.Tensor, optional): Target state. If None, defaults to origin.
        """
        super().__init__()

        # --- Dimensions (Hardcoded to match RobotPlant) ---
        self.n_agents = n_agents  # Number of agents
        self.input_k_dim = 4 * self.n_agents  # Input to controller (Plant Output): [px, py, vx, vy]
        self.output_k_dim = 2 * self.n_agents  # Output of controller (Plant Input): [Fx, Fy]
        self.kp = kp

        # --- 1. Handle Target State (y_target) ---
        if y_target is None:
            # Default target: Origin [0, 0, 0, 0]
            target_tensor = torch.zeros(self.input_k_dim)
        else:
            target_tensor = y_target

        # Register as a buffer so it automatically moves to GPU/CPU with the model
        self.register_buffer('y_target', target_tensor)

        # --- 2. Handle Gain Matrix (Kp) ---
        if self.kp is None:
            # Default Gain: Acts as a spring (Position Control)
            # We want Force_x = 1.0 * (Target_x - Pos_x)
            # We want Force_y = 1.0 * (Target_y - Pos_y)

            # Base matrix for 1 agent: shape (2, 4)
            # Rows are outputs (Fx, Fy), Columns are inputs (px, py, vx, vy)
            default_kp = torch.tensor([
                [1.0, 0.0, 0.0, 0.0],  # Fx depends on px
                [0.0, 1.0, 0.0, 0.0]  # Fy depends on py
            ], dtype=torch.float32)

            # Expand for N agents (Block Diagonal)
            kp_tensor = torch.kron(torch.eye(self.n_agents), default_kp)
        else:
            kp_tensor = torch.tensor([
                [self.kp, 0.0, 0.0, 0.0],  # Fx depends on px
                [0.0, self.kp, 0.0, 0.0]  # Fy depends on py
            ], dtype=torch.float32)
            kp_tensor = torch.kron(torch.eye(self.n_agents), kp_tensor)


        # Check dimensions: Kp must be (output_dim, input_dim) -> (2, 4)
        expected_shape = (self.output_k_dim, self.input_k_dim)
        assert kp_tensor.shape == expected_shape, f"Kp shape mismatch. Expected {expected_shape}, got {kp_tensor.shape}"

        # Register Kp as a buffer (physics constant, not learned)
        self.register_buffer('kp_tensor', kp_tensor)

    def forward(self, y):
        """
        Computes the control input 'u' based on the plant output 'y'.

        Args:
            y (torch.Tensor): Measured state from the plant.
                              Shape = (batch_size, 1, input_k_dim)

        Returns:
            torch.Tensor: Control input (forces).
                          Shape = (batch_size, 1, output_k_dim)
        """
        batch_size = y.shape[0]

        # 1. Prepare Target Tensor
        # Reshape y_target to match batch: (1, 1, 4) -> (batch, 1, 4)
        target_batch = self.y_target.view(1, 1, self.input_k_dim).expand(batch_size, 1, -1)

        # 2. Compute Error
        error = target_batch - y

        # 3. Compute Control Input
        # u = Kp * error
        u = F.linear(error, self.kp_tensor)

        return u


class StabilizedRobot(nn.Module):
    """
    Simulates the closed-loop system (Plant + Controller).

    Structure:
    - This class 'has a' Plant (system_model) and a Controller.
    - It manages the interaction: Sensor -> Controller -> Actuator -> Plant.
    """

    def __init__(self, system_model, controller):
        super().__init__()
        self.system_model = system_model
        self.controller = controller

    @property
    def x(self):
        """
        Dynamic Property: Always returns the current true internal state of the plant.
        This prevents 'stale reference' bugs where the wrapper holds an old copy of x.
        """
        return self.system_model.x

    def reset(self, x0, batch_size=None):
        """
        Initializes the internal state of the Plant.

        Args:
            x0 (Tensor): Initial state.
            batch_size (int): Optional, inferred from x0 if not provided.
        """
        if batch_size is None:
            # Safely infer batch size, handling the case where x0 might be None
            batch_size = x0.shape[0] if x0 is not None else 1

        # Delegate reset logic entirely to the Plant
        self.system_model.reset(x0, batch_size)

    def forward(self, x_meas, u_ext):
        """
        Performs one timestep of the closed-loop system.

        Args:
            x_meas (torch.Tensor): Current observation (observable states).
            u (torch.Tensor): External input of the closed-loop system.

        Returns:
            u_applied (torch.Tensor): Total input sent to plant (Controller + External).
            x_next (torch.Tensor): The new state after the step.
        """

        # 1. Safety check: Ensure plant is initialized

        if self.x is None:
            raise ValueError("State not initialized. Call `reset()` before using forward().")
        # 2. Compute Control Action
        #    The controller sees 'x_meas' (which might be noisy)
        u_ctrl = self.controller.forward(x_meas)

        # 3. Combine Inputs
        #    u_total = Controller_Action + External_Disturbance/Reference
        u_total = u_ctrl + u_ext

        # 4. Step the Plant
        #    The plant updates its internal state 'x' and returns the new output
        x_next = self.system_model.forward(u_total)

        return x_next

    def predict_nominal_next_state(self, x_meas, u_ext):

        # 2. Compute Control Action
        #    The controller sees 'x_meas' (which might be noisy)
        u_ctrl = self.controller.forward(x_meas)

        # 3. Combine Inputs
        #    u_total = Controller_Action + External_Disturbance/Reference
        u_total = u_ctrl + u_ext

        # --- 1. Linear Dynamics ---
        # Calculate the contributions from the linear system matrices.
        x_hat = self.system_model.predict_nominal_next_state(x_meas, u_total)

        return x_hat

    def run(self, x0, horizon, batch_size, u_ext = None, output_noise=None):
        # 1. Setup Dimensions
        device = self.system_model.A.device

        # 2. Initialize
        self.reset(x0, batch_size)

        # Default noise if None
        if output_noise is None:
            output_dim = self.system_model.output_dim
            output_noise = torch.zeros(batch_size, horizon + 1, output_dim, device=device)
        if u_ext is None:
            input_dim = self.system_model.input_dim
            u_ext = torch.zeros(batch_size, horizon, input_dim, device=device)

        y_traj = []

        # Initial conditions
        y_true = self.x

        # Initial sensor reading (y0 + noise)
        y_sensor = y_true + output_noise[:, 0:1, :]

        # 3. Simulation Loop
        for t in range(horizon):
            # Store the state at the start of the step
            y_traj.append(y_true)

            # Get inputs
            u_ext_t = u_ext[:, t:t + 1, :]

            # Step System (Forward uses y_sensor!)
            y_next_true = self.forward(y_sensor, u_ext_t)

            # Update for next step
            y_sensor = y_next_true + output_noise[:, t + 1:t + 2, :]
            y_true = y_next_true

        y_traj.append(y_true)

        # 4. Return
        # y_traj is now (Batch, Horizon + 1, Dim)

        return torch.cat(y_traj, dim=1)

    def __call__(self, x0, horizon, batch_size, u_ext = None, output_noise=None):
        """

        Args:
            x0 (torch.Tensor): Initial state. If None, defaults to zero. Shape (batch, 1, state_dim).
            u_ext (torch.Tensor): Trajectories of external input signal. Shape = (batch_size, horizon, input_dim).
            output_noise: realizations of output noise Shape (batch, horizon + 1, output_dim).

        Returns:
            torch.Tensor: Trajectories of outputs (batch_size, horizon, output_dim)
        """
        return self.run(x0, horizon, batch_size, u_ext, output_noise)