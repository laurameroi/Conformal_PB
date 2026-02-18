import torch


def generate_random_batch(
    batch_size,
    horizon,
    n_agents=1,
    x0_std=0.2,
    noise_std=0.2,
    device="cpu",
):
    state_dim_per_agent = 4
    state_dim = state_dim_per_agent * n_agents

    if n_agents == 2:
        centers = torch.tensor([-1.0, -1.0, 1.0, -1.0])
    else:
        pos_center_single = torch.tensor([-1.0, -1.0])
        centers = pos_center_single.repeat(n_agents)

    pos_center = centers.view(1, 1, n_agents * 2)
    pos_noise = torch.randn(batch_size, 1, n_agents * 2) * x0_std
    pos = pos_center + pos_noise

    vel = torch.zeros(batch_size, 1, n_agents * 2)

    pos_reshaped = pos.view(batch_size, 1, n_agents, 2)
    vel_reshaped = vel.view(batch_size, 1, n_agents, 2)
    state_reshaped = torch.cat((pos_reshaped, vel_reshaped), dim=-1)
    x0_data = state_reshaped.view(batch_size, 1, state_dim)

    w = noise_std * torch.randn(batch_size, horizon, state_dim)
    w[:, 0, :] = x0_data[:, 0, :]

    return w.to(device)
