# src/evaluate.py
import numpy as np
import matplotlib.pyplot as plt
import torch
from src.env import TrafficEnv
from src.model import MLPModel, RNNModel
from src.trainer import collect_trajectory, reinforce_train, dqn_train


def run_controller(env, model, n_episodes=5, mode="mlp", rl=False):
    """
    Execute a policy inside the env and collect metrics.
    mode: "mlp" or "lstm" (for RNNModel)
    rl: if True, the model is used as a policy (sample actions) and we
        accumulate undiscounted returns; otherwise we just run deterministic
        argmax actions (useful for baseline or pretrained policy).
    Returns a dict with average waiting time, queue reduction, throughput.
    """
    total_wait = 0.0
    total_queues = 0.0
    total_throughput = 0.0
    steps = 0
    initial_queues = []

    for ep in range(n_episodes):
        state = env.reset()
        initial_queues.append(np.mean(env.queue))
        done = False
        ep_wait = 0.0
        ep_queues = 0.0
        ep_steps = 0

        while not done:
            # ----- choose action -----
            if rl:
                with torch.no_grad():
                    state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                    output = model(state_t)
                    probs = output if torch.all(output >= 0) and torch.allclose(
                        output.sum(dim=1), torch.ones(1), atol=1e-4
                    ) else torch.softmax(output, dim=1)
                    m = torch.distributions.Categorical(probs)
                    action = m.sample().item()
            else:
                # deterministic: pick phase with higher probability (or argmax)
                with torch.no_grad():
                    state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                    output = model(state_t)
                    probs = output if torch.all(output >= 0) and torch.allclose(
                        output.sum(dim=1), torch.ones(1), atol=1e-4
                    ) else torch.softmax(output, dim=1)
                    action = torch.argmax(probs, dim=1).item()

            # ----- take step in env -----
            next_state, reward, done, info = env.step(action)
            ep_wait += -reward          # reward was negative queue => waiting time proxy
            ep_queues += np.sum(info["queues"])
            ep_steps += 1

            total_wait += -reward
            total_queues += np.sum(info["queues"])
            total_throughput += info.get("departed", 0.0)
            steps += 1

            state = next_state

    avg_wait = total_wait / max(1, steps)
    avg_queue = total_queues / max(1, steps)
    throughput = (total_throughput / max(1, steps)) * 60

    queue_reduction = (np.mean(initial_queues) - avg_queue) / max(
        1e-3, np.mean(initial_queues)
    ) * 100

    return {
        "avg_wait": avg_wait,
        "avg_queue": avg_queue,
        "queue_reduction_pct": queue_reduction,
        "throughput_veh_min": throughput,
        "steps": steps,
    }


def run_baseline(env, n_episodes=5):
    """
    Fixed‑Time controller: toggle phase every 5 steps.
    """
    total_wait = 0.0
    total_queues = 0.0
    total_steps = 0
    initial_queues = []
    total_throughput = 0.0

    for _ in range(n_episodes):
        env.reset()
        initial_queues.append(np.mean(env.queue))
        done = False
        step_cnt = 0
        phase = 0  # start with NS green
        while not done:
            action = phase
            step_cnt += 1
            if step_cnt % 5 == 0:
                phase = 1 - phase

            _, reward, done, info = env.step(action)
            total_wait += -reward
            total_queues += np.sum(info["queues"])
            total_throughput += info.get("departed", 0.0)
            total_steps += 1

    avg_wait = total_wait / max(1, total_steps)
    avg_queue = total_queues / max(1, total_steps)
    queue_reduction = (np.mean(initial_queues) - avg_queue) / max(
        1e-3, np.mean(initial_queues)
    ) * 100
    throughput = (total_throughput / max(1, total_steps)) * 60

    return {
        "avg_wait": avg_wait,
        "avg_queue": avg_queue,
        "queue_reduction_pct": queue_reduction,
        "throughput_veh_min": throughput,
        "steps": total_steps,
    }


def plot_comparison(mlp_results, baseline_results, save_path="results/comparison.png"):
    """
    Bar plot of the three primary metrics.
    """
    metrics = ["avg_wait", "queue_reduction_pct", "throughput_veh_min"]
    mlp_vals = [mlp_results[m] for m in metrics]
    base_vals = [baseline_results[m] for m in metrics]

    fig, ax = plt.subplots(figsize=(9, 5))
    bar_width = 0.35
    indices = np.arange(len(metrics))

    ax.bar(indices - bar_width/2, mlp_vals, bar_width, label="MLP Agent", color="#2c7bb6")
    ax.bar(indices + bar_width/2, base_vals, bar_width, label="Fixed‑Time Baseline", color="#f03b20")

    ax.set_xticks(indices)
    ax.set_xticklabels([m.replace("_", " ").title() for m in metrics])
    ax.set_ylabel("Value")
    ax.set_title("MLP‑based ATMS vs Fixed‑Time Baseline")
    ax.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"[evaluate] Plot saved to {save_path}")


# -----------------------------------------------------------------
# Quick demo: train a policy with REINFORCE and then evaluate
# -----------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Quick RL demo")
    parser.add_argument("--epochs", type=int, default=30, help="REINFORCE epochs")
    parser.add_argument("--steps", type=int, default=100, help="max steps per episode")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    args = parser.parse_args()

    env = TrafficEnv(sumo=False)

    # initialise MLP policy
    policy = MLPModel(input_dim=7, hidden_dims=[32, 16], dropout=0.3)

    # Train with REINFORCE
    policy = reinforce_train(policy, env, n_episodes=args.epochs,
                             max_steps=args.steps, lr=args.lr)

    # Evaluate the learned policy
    mlp_metrics = run_controller(env, policy, n_episodes=5, mode="mlp", rl=True)
    print("\n=== Learned (RL) agent metrics ===")
    for k, v in mlp_metrics.items():
        print(f"  {k}: {v:.3f}")

    # Baseline
    base_metrics = run_baseline(env, n_episodes=5)
    print("\n=== Fixed‑Time Baseline metrics ===")
    for k, v in base_metrics.items():
        print(f"  {k}: {v:.3f}")

    # Plot
    plot_comparison(mlp_metrics, base_metrics, save_path="results/rl_comparison.png")