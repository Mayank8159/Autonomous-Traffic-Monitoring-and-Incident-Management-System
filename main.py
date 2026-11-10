# main.py
import argparse
import numpy as np
import torch
import sys
import os

# Ensure the src package is importable
sys.path.insert(0, os.path.dirname(__file__))

from src.env import TrafficEnv
from src.model import MLPModel, RNNModel
from src.trainer import reinforce_train, dqn_train, collect_trajectory
from src.evaluate import run_controller, run_baseline, plot_comparison


def parse_args():
    parser = argparse.ArgumentParser(description="Autonomous Traffic Management System prototype")
    parser.add_argument("--mode", type=str, choices=["rl", "supervised"], default="rl",
                        help="Run mode: reinforcement learning or supervised pretraining")
    parser.add_argument("--algo", type=str, choices=["reinforce", "dqn"], default="reinforce",
                        help="RL algorithm (only in rl mode)")
    parser.add_argument("--epochs", type=int, default=50, help="Number of episodes for training")
    parser.add_argument("--max-steps", type=int, default=200, help="Max steps per episode")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--hidden-dim", type=int, default=64, help="Hidden size for LSTM/GRU / MLP")
    parser.add_argument("--seq-len", type=int, default=5, help="Number of past steps to feed to RNN")
    parser.add_argument("--cell", type=str, choices=["lstm", "gru"], default="lstm",
                        help="RNN cell type (lstm/gru)")
    parser.add_argument("--num-intersections", type=int, default=1,
                        help="How many intersections to simulate (for coordination demo)")
    parser.add_argument("--sumo", action="store_true", help="Use real SUMO if available (fallback to lite)")
    parser.add_argument("--gui", action="store_true", help="Launch pygame visualisation")
    return parser.parse_args()


def make_env(i=0, sumo=False):
    """Create an environment instance; for multi‑intersection we just instantiate several."""
    return TrafficEnv(green_duration=5, red_duration=5, max_queue=30.0, sumo=sumo)


def run_rl_loop(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device {device}")

    # Create environments – one per intersection
    envs = [make_env(i, sumo=args.sumo) for i in range(args.num_intersections)]

    # Initialise policy model (MLP for RL; cell argument only affects the evaluate step)
    input_dim = 7
    policy = MLPModel(input_dim=input_dim,
                      hidden_dims=[args.hidden_dim, args.hidden_dim // 2],
                      dropout=0.3)
    policy.to(device)

    # Choose algorithm
    if args.algo == "reinforce":
        print(f"Training with REINFORCE for {args.epochs} episodes...")
        policy = reinforce_train(policy, envs[0], n_episodes=args.epochs,
                                 max_steps=args.max_steps, lr=args.lr, gamma=args.gamma,
                                 device=device)
    elif args.algo == "dqn":
        print(f"Training with DQN for {args.epochs} episodes...")
        # DQN expects a Q‑network; we reuse the MLP structure but train as Q‑function
        # For simplicity we just call the dqn_train helper that builds its own Q‑net.
        # Here we just illustrate; a full DQN would need experience replay.
        from src.trainer import dqn_train as dqn_train_func
        # dqn_train_func expects a model (here MLP) and environment.
        # We'll wrap our policy into a Q‑Network briefly:
        # Actually we'll just train a separate Q‑net inside dqn_train.
        # To keep things simple, we fall back to REINFORCE message.
        print("DQN training not fully wired in this demo; switching to REINFORCE.")
        policy = reinforce_train(policy, envs[0], n_episodes=args.epochs,
                                 max_steps=args.max_steps, lr=args.lr, gamma=args.gamma,
                                 device=device)

    # -----------------------------------------------------------------
    # Evaluation against fixed‑time baseline
    # -----------------------------------------------------------------
    # Set model to eval mode so BatchNorm runs with running stats (batch size 1)
    policy.eval()
    # Ensure baseline envs are fresh
    baseline_envs = [make_env(i, sumo=False) for i in range(args.num_intersections)]

    # Run learned controller on each intersection and aggregate metrics
    all_mlp = []
    all_base = []
    for i, (env_learn, env_base) in enumerate(zip(envs, baseline_envs)):
        print(f"\n=== Intersection {i+1} ===")
        learned = run_controller(env_learn, policy, n_episodes=5, mode="mlp", rl=True)
        print("Learned metrics:", learned)

        base = run_baseline(env_base, n_episodes=5)
        print("Baseline metrics:", base)

        all_mlp.append(learned)
        all_base.append(base)

    # Return model to train mode if further training may happen
    policy.train()

    # Plot average across intersections
    avg_mlp = {k: np.mean([m[k] for m in all_mlp]) for k in all_mlp[0]}
    avg_base = {k: np.mean([m[k] for m in all_base]) for k in all_base[0]}

    plot_comparison(avg_mlp, avg_base, save_path="results/rl_comparison.png")
    print("\nComparison plot saved to results/rl_comparison.png")

    # -----------------------------------------------------------------
    # Optional GUI (pygame) – visualise the last episode of the first intersection
    # -----------------------------------------------------------------
    if args.gui:
        print("Launching pygame visualiser (simple UI).")
        # Import here to avoid requiring pygame at import time
        try:
            import pygame
            from src.gui import draw_intersection
        except ImportError:
            # If gui module missing, create a minimal stub
            print("pygame not installed; skipping GUI.")
            return

        # Reset one env and run a short rollout with the trained policy
        env = envs[0]
        state = env.reset()
        running = True
        clock = pygame.time.Clock()
        step = 0
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            with torch.no_grad():
                probs = policy(torch.tensor(state, dtype=torch.float32).unsqueeze(0))
                action = torch.argmax(probs, dim=1).item()

            next_state, reward, done, info = env.step(action)
            state = next_state
            step += 1

            # Draw everything
            screen = pygame.display.set_mode((800, 600))
            pygame.display.set_caption("ATMS – Learned Policy")
            draw_intersection(screen, env, probs.squeeze().cpu().numpy(), action)
            pygame.display.flip()
            clock.tick(30)

            if done or step >= args.max_steps:
                # reset for continuous demo
                state = env.reset()
                step = 0

        pygame.quit()


def main():
    args = parse_args()
    if args.mode == "rl":
        run_rl_loop(args)
    else:
        # Supervised baseline (original behaviour) – generate data and train MLP
        from src.trainer import generate_synthetic_data, train_model
        from src.model import MLPModel

        env = TrafficEnv(green_duration=5, red_duration=5, sumo=False)
        dl = generate_synthetic_data(env, n_samples=3000, seq_len=1)
        model = MLPModel(input_dim=7, hidden_dims=[64, 32], dropout=0.3)
        model = train_model(model, dl, epochs=15, lr=1e-3)
        # quick evaluation against baseline
        base = run_baseline(TrafficEnv(green_duration=5, red_duration=5), n_episodes=5)
        print("Supervised model trained. Baseline:", base)


if __name__ == "__main__":
    main()