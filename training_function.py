import copy
import torch
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm


def train_agent(
        config,
        sim,
        loss_wrapper,
        mode,
        fixed_val_w,
        generate_random_batch,
        plot_results=False,
        plot_kwargs=None
):
    """
    Unified training loop for 4 modes:
    - standard_mse
    - standard_cvar
    - lagrangian_mse
    - lagrangian_cvar
    """
    valid_modes = ["standard_mse", "standard_cvar", "lagrangian_mse", "lagrangian_cvar"]
    if mode not in valid_modes:
        raise ValueError(f"Unknown mode: {mode}. Must be one of {valid_modes}")

    print(f"Starting {mode.upper()} online training on {config.device}...")

    # ==========================================
    # 1. Optimizer Setup
    # ==========================================
    if mode == "standard_mse":
        optimizer = torch.optim.Adam(sim.parameters(), lr=config.lr)

    elif mode == "standard_cvar":
        # Fast Speed for Tau
        optimizer = torch.optim.Adam([
            {'params': sim.parameters(), 'lr': config.lr},
            {'params': [loss_wrapper.tau], 'lr': config.lr * 10.0}
        ])

    elif mode == "lagrangian_mse":
        # No Tau in ERM Lagrangian
        opt_primal = torch.optim.Adam(sim.parameters(), lr=config.lr)
        opt_dual = torch.optim.Adam([loss_wrapper.pre_lambda], lr=config.lr * 0.1, maximize=True)

    elif mode == "lagrangian_cvar":
        opt_primal = torch.optim.Adam([
            {'params': sim.parameters(), 'lr': config.lr},
            {'params': [loss_wrapper.tau], 'lr': config.lr * 10.0}
        ])
        opt_dual = torch.optim.Adam([loss_wrapper.pre_lambda], lr=config.lr * 0.1, maximize=True)

    # ==========================================
    # 2. Tracking Setup & Early Stopping
    # ==========================================
    # ADDED: 'taus' to the history dictionary
    history = {k: [] for k in
               ['train_losses', 'val_losses', 'val_targets', 'val_us', 'val_obss', 'val_perfs', 'val_colls',
                'val_lambdas', 'taus']}

    best_val_metric = float('inf')
    best_model_state = None
    best_tau = None
    patience_counter = 0
    patience_limit = config.early_stopping_patience_limit

    pbar = tqdm(range(config.num_training_steps), desc=f"{mode.replace('_', ' ').title()}")

    # ==========================================
    # 3. Training Loop
    # ==========================================
    for step in pbar:
        sim.train()

        # --- Primal Update ---
        n_inner_steps = config.n_inner_steps if "lagrangian" in mode else 1

        for _ in range(n_inner_steps):
            if "lagrangian" in mode:
                opt_primal.zero_grad()
            else:
                optimizer.zero_grad()

            batch_w = generate_random_batch(config)
            traj_x_train, traj_u_train, _ = sim.run(batch_w)

            if "standard" in mode:
                loss, _, _, _ = loss_wrapper(traj_x_train, traj_u_train)
                loss.backward()
                if config.gradient_clipping is not None:
                    torch.nn.utils.clip_grad_norm_(sim.parameters(), max_norm=config.gradient_clipping)
                optimizer.step()
            elif "lagrangian" in mode:
                loss, _, _, _, _ = loss_wrapper(traj_x_train, traj_u_train)
                loss.backward()
                if config.gradient_clipping is not None:
                    torch.nn.utils.clip_grad_norm_(sim.parameters(), max_norm=config.gradient_clipping)
                opt_primal.step()

        # --- Dual Update (Lambda) ---
        if "lagrangian" in mode:
            opt_dual.zero_grad()
            batch_w_dual = generate_random_batch(config)
            traj_x_dual, traj_u_dual, _ = sim.run(batch_w_dual)
            loss_dual, _, _, _, _ = loss_wrapper(traj_x_dual, traj_u_dual)
            loss_dual.backward()
            opt_dual.step()

        # ==========================================
        # 4. Validation & Early Stopping
        # ==========================================
        if (step + 1) % config.log_interval == 0:
            sim.eval()
            with torch.no_grad():
                traj_x_val, traj_u_val, traj_w_hat_val = sim.run(fixed_val_w)

                if "standard" in mode:
                    val_loss, val_target, val_u, val_obs = loss_wrapper(traj_x_val, traj_u_val)
                    current_metric = val_loss.item()

                    history['train_losses'].append(loss.item())
                    history['val_losses'].append(val_loss.item())
                    history['val_targets'].append(val_target.item())
                    history['val_us'].append(val_u.item())
                    history['val_obss'].append(val_obs.item())

                    pbar.set_postfix({'Val Loss': f"{val_loss.item():.4f}", 'Best': f"{best_val_metric:.4f}"})

                elif "lagrangian" in mode:
                    val_lag, val_perf, val_coll, val_lam, val_viol = loss_wrapper(traj_x_val, traj_u_val)

                    viol = val_viol.item() if hasattr(val_viol, 'item') else val_viol
                    perf = val_perf.item() if hasattr(val_perf, 'item') else val_perf

                    current_metric = perf if viol <= 0 else perf + (1e6 * viol)

                    history['train_losses'].append(loss.item())
                    history['val_losses'].append(val_lag.item())
                    history['val_perfs'].append(perf)
                    history['val_colls'].append(val_coll.item() if hasattr(val_coll, 'item') else val_coll)
                    history['val_lambdas'].append(val_lam.item() if hasattr(val_lam, 'item') else val_lam)

                    pbar.set_postfix({'Perf': f"{perf:.2f}", 'Viol': f"{viol:.4f}", 'Best': f"{best_val_metric:.2f}"})

                # ADDED: Track tau evolution if using CVaR
                if "cvar" in mode and hasattr(loss_wrapper, 'tau'):
                    history['taus'].append(loss_wrapper.tau.item())

            # Early Stopping Check
            if current_metric < best_val_metric:
                best_val_metric = current_metric
                best_model_state = copy.deepcopy(sim.state_dict())
                if "cvar" in mode and hasattr(loss_wrapper, 'tau'):
                    best_tau = loss_wrapper.tau.item()
                if patience_limit is not None:
                    patience_counter = 0
            else:
                if patience_limit is not None:
                    patience_counter += 1

            if patience_limit is not None and patience_counter >= patience_limit:
                print(f"\nEarly stopping triggered! No improvement for {patience_limit * config.log_interval} steps.")
                break

    # ==========================================
    # 5. Restore Best Model & Plotting
    # ==========================================
    if best_model_state is not None:
        sim.load_state_dict(best_model_state)
        if "cvar" in mode and best_tau is not None:
            loss_wrapper.tau.data = torch.tensor(best_tau, device=config.device)
        print(f"\nRestored best model (Metric: {best_val_metric:.4f}).")

    with torch.no_grad():
        final_x, final_u, final_w_hat = sim.run(fixed_val_w)

    if plot_results and plot_kwargs is not None:
        steps = range(config.log_interval, len(history['val_losses']) * config.log_interval + 1, config.log_interval)

        if "standard" in mode:
            # ADDED: Expand to 3 subplots if CVaR is used
            num_plots = 3 if "cvar" in mode else 2
            fig, axes = plt.subplots(1, num_plots, figsize=(8 * num_plots, 5))

            axes[0].plot(steps, history['train_losses'], label='Train Loss', color='blue')
            axes[0].plot(steps, history['val_losses'], label='Val Loss', color='orange', linestyle='--')
            axes[0].set_title(f'{mode.replace("_", " ").title()} Progress')
            axes[0].legend()
            axes[0].grid(True)

            axes[1].plot(steps, history['val_targets'], label='Target Tracking', color='green')
            axes[1].plot(steps, history['val_us'], label='Control Effort', color='purple')
            obs_label = 'CVaR Obstacle Avoidance' if 'cvar' in mode else 'Mean Obstacle Avoidance'
            axes[1].plot(steps, history['val_obss'], label=obs_label, color='red')
            axes[1].set_title('Validation Breakdown')
            axes[1].legend()
            axes[1].grid(True)

            if "cvar" in mode:
                axes[2].plot(steps, history['taus'], label='Tau (VaR Base)', color='brown', linewidth=2)
                axes[2].set_title('Evolution of Tau (Quantile Base)')
                axes[2].legend()
                axes[2].grid(True)

            plt.tight_layout()
            plt.show()

        elif "lagrangian" in mode:
            # ADDED: Expand to 4 subplots if CVaR is used
            num_plots = 4 if "cvar" in mode else 3
            fig, axes = plt.subplots(1, num_plots, figsize=(6 * num_plots, 5))

            axes[0].plot(steps, history['train_losses'], label='Train Lagrangian', color='blue')
            axes[0].plot(steps, history['val_losses'], label='Val Lagrangian', color='orange', linestyle='--')
            axes[0].set_title(f'{mode.replace("_", " ").title()} Min-Max')
            axes[0].legend()
            axes[0].grid(True)

            axes[1].plot(steps, history['val_perfs'], label='Expected Perf', color='green')
            obs_label = 'CVaR Obstacle Avoidance' if 'cvar' in mode else 'Mean Obstacle Avoidance'
            axes[1].plot(steps, history['val_colls'], label=obs_label, color='red')
            if hasattr(loss_wrapper, 'tau_safe_bar'):
                t_bar = loss_wrapper.tau_safe_bar.item() if hasattr(loss_wrapper.tau_safe_bar,
                                                                    'item') else loss_wrapper.tau_safe_bar
                axes[1].axhline(y=t_bar, color='red', linestyle=':', label='Max Safe Target (Tau_bar)')
            axes[1].set_title('Perf vs. Safety')
            axes[1].legend()
            axes[1].grid(True)

            axes[2].plot(steps, history['val_lambdas'], label='Lambda (Dual)', color='purple')
            axes[2].set_title('Dual Variable')
            axes[2].legend()
            axes[2].grid(True)

            if "cvar" in mode:
                axes[3].plot(steps, history['taus'], label='Tau (VaR Base)', color='brown', linewidth=2)
                axes[3].set_title('Evolution of Tau (Quantile Base)')
                axes[3].legend()
                axes[3].grid(True)

            plt.tight_layout()
            plt.show()

        if 'plot_func' in plot_kwargs:
            plot_kwargs['plot_func'](
                traj_x=final_x.cpu(), traj_u=final_u.cpu(), traj_w_hat=final_w_hat.cpu(),
                x_target=plot_kwargs.get('x_target'), obs_centers=plot_kwargs.get('obs_centers'),
                obs_radii=plot_kwargs.get('obs_radii'), obs_radii_safe=plot_kwargs.get('obs_radii_safe'),
                dt=plot_kwargs.get('dt')
            )

    return history, (final_x.cpu(), final_u.cpu(), final_w_hat.cpu())