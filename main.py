# main.py
import argparse
import json
import numpy as np
import sys
import os
import webbrowser
import subprocess

# Ensure the src package is importable
sys.path.insert(0, os.path.dirname(__file__))



def parse_args():
    parser = argparse.ArgumentParser(description="Autonomous Traffic Management System (ATMS)")
    parser.add_argument("--mode", type=str,
                        choices=["rl", "supervised", "video", "hybrid", "gan", "vae", "cnn-train", "server", "city"],
                        default="rl",
                        help="Run mode: rl, supervised, video, hybrid, gan, vae, cnn-train, server, or city")

    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8000)), help="Port for ATMS web server (mode server)")
    parser.add_argument("--algo", type=str, choices=["reinforce", "dqn"], default="reinforce",
                        help="RL algorithm (only in rl mode)")
    parser.add_argument("--epochs", type=int, default=50, help="Number of episodes/epochs for training")
    parser.add_argument("--max-steps", type=int, default=200, help="Max steps per episode/rollout")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--hidden-dim", type=int, default=64, help="Hidden size for LSTM/GRU / MLP")
    parser.add_argument("--seq-len", type=int, default=5, help="Number of past steps to feed to RNN")
    parser.add_argument("--cell", type=str, choices=["lstm", "gru"], default="lstm",
                        help="RNN cell type (lstm/gru)")
    parser.add_argument("--num-intersections", type=int, default=1,
                        help="How many intersections to simulate (for coordination demo)")
    parser.add_argument("--video-path", type=str, default="results/sample_traffic.mp4",
                        help="Path to real-world traffic video file")
    parser.add_argument("--scenario", type=str, default="rush_hour",
                        choices=["normal", "moderate", "heavy", "rush_hour", "uneven", "surge"],
                        help="Traffic scenario to generate with GAN")
    parser.add_argument("--sumo", action="store_true", help="Use real SUMO if available (fallback to lite)")
    parser.add_argument("--gui", action="store_true", help="Open the web visualizer after exporting rollout data")
    return parser.parse_args()


def make_env(i=0, sumo=False, max_steps=200):
    """Create an environment instance; for multi‑intersection we just instantiate several."""
    from src.env import TrafficEnv
    return TrafficEnv(green_duration=5, red_duration=5, max_queue=30.0,
                      sumo=sumo, max_steps=max_steps)


def run_rl_loop(args):
    import torch
    from src.model import MLPModel
    from src.trainer import reinforce_train, dqn_train
    from src.evaluate import run_controller, run_baseline, plot_comparison

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device {device}")


    # Create environments – one per intersection
    envs = [make_env(i, sumo=args.sumo, max_steps=args.max_steps)
            for i in range(args.num_intersections)]

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
        from src.trainer import QNetwork
        policy = dqn_train(QNetwork(input_dim=input_dim,
                                    hidden_dims=[args.hidden_dim, args.hidden_dim // 2]),
                           envs[0], n_episodes=args.epochs,
                           max_steps=args.max_steps, lr=args.lr, gamma=args.gamma,
                           device=device)

    # -----------------------------------------------------------------
    # Evaluation against fixed‑time baseline
    # -----------------------------------------------------------------
    # Set model to eval mode so BatchNorm runs with running stats (batch size 1)
    policy.eval()
    # Ensure baseline envs are fresh
    baseline_envs = [make_env(i, sumo=False, max_steps=args.max_steps)
                     for i in range(args.num_intersections)]

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
    # Export rollout data for web-based visualizer
    # -----------------------------------------------------------------
    # Reset one env and run a short rollout with the trained policy,
    # saving state, action, reward, probs, and phase at each step
    # so the web visualizer (web_visualizer/visualizer.html) can display it.
    os.makedirs("results", exist_ok=True)

    env = envs[0]
    state = env.reset()
    rollout = []

    policy.eval()  # Ensure BatchNorm uses running stats (helps with batch size 1)
    with torch.no_grad():
        for t in range(args.max_steps):
            state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            output = policy(state_t)
            probs = output if torch.all(output >= 0) and torch.allclose(
                output.sum(dim=1), torch.ones(1), atol=1e-4
            ) else torch.softmax(output, dim=1)
            probs_np = probs.squeeze().cpu().numpy()
            action = int(torch.argmax(probs, dim=1).item())

            next_state, reward, done, info = env.step(action)

            rollout.append({
                "step": t,
                "state": state_t.squeeze().cpu().numpy().tolist(),
                "action": action,
                "probs": probs_np.tolist(),
                "phase": int(env.phase),
                "queues": info["queues"].tolist(),
                "reward": float(reward),
            })

            state = next_state
            if done:
                break

    # Save policy checkpoint for hybrid pipeline
    os.makedirs("models", exist_ok=True)
    try:
        torch.save(policy.state_dict(), "models/rl_policy.pt")
        print("Trained RL policy saved to models/rl_policy.pt")
        from src.model import export_policy_weights_to_js
        export_policy_weights_to_js()
    except Exception as e:
        print(f"Warning saving policy: {e}")

    # Save rollout data for the web visualizer
    with open("results/rl_rollout.json", "w") as f:
        json.dump(rollout, f, indent=2)
    with open("web_visualizer/rollout_data.js", "w") as f:
        f.write("window.ATMS_ROLLOUT = ")
        json.dump(rollout, f)
        f.write(";\n")

    print(f"\nRollout data saved to results/rl_rollout.json ({len(rollout)} steps).")
    print("Open web_visualizer/visualizer.html in a browser to view the visualization.")
    if args.gui:
        project_dir = os.path.abspath(os.path.dirname(__file__))
        subprocess.Popen(
            [sys.executable, "-m", "http.server", "8765", "--bind", "127.0.0.1"],
            cwd=project_dir,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        webbrowser.open("http://127.0.0.1:8765/web_visualizer/visualizer.html")


def main():
    args = parse_args()
    if args.mode == "rl":
        run_rl_loop(args)
    elif args.mode == "supervised":
        # Supervised baseline (original behaviour) – generate data and train MLP
        from src.trainer import generate_synthetic_data, train_model
        from src.model import MLPModel

        env = TrafficEnv(green_duration=5, red_duration=5, sumo=False)
        dl = generate_synthetic_data(env, n_samples=3000, seq_len=1)
        model = MLPModel(input_dim=7, hidden_dims=[64, 32], dropout=0.3)
        model = train_model(model, dl, epochs=15, lr=1e-3)
        model.eval()
        learned = run_controller(TrafficEnv(green_duration=5, red_duration=5),
                                 model, n_episodes=5, mode="mlp")
        base = run_baseline(TrafficEnv(green_duration=5, red_duration=5), n_episodes=5)
        plot_comparison(learned, base, save_path="results/supervised_comparison.png")
        print("Supervised metrics:", learned)
        print("Baseline metrics:", base)
    elif args.mode == "cnn-train":
        from src.cnn_custom import train_custom_cnn
        train_custom_cnn(epochs=args.epochs, batch_size=32, lr=args.lr)
    elif args.mode == "gan":
        from src.gan import train_traffic_gan, generate_scenario
        gen, disc = train_traffic_gan(epochs=args.epochs, lr=args.lr)
        print(f"\nGenerating and validating scenario '{args.scenario}':")
        results = generate_scenario(gen, scenario_type=args.scenario, num_samples=3)
        for i, r in enumerate(results, 1):
            print(f"Sample {i}: Validated={r['validated']} | Mean Queue={r['mean_queue']} | Status: {r['status']}")
    elif args.mode == "vae":
        from src.vae import train_sequence_vae, detect_incident_anomaly
        vae, th_warn, th_crit = train_sequence_vae(epochs=args.epochs, lr=args.lr)
        # Test detection on a test anomaly
        test_seq = np.random.uniform(2.0, 5.0, size=(5, 4)).astype(np.float32)
        test_seq[:, 2] = 28.0  # Simulated blockage on East approach
        alert = detect_incident_anomaly(vae, test_seq, th_warn, th_crit)
        print(f"\nTest Incident Evaluation -> Level: {alert['level']} | Alert: {alert['alert_message']}")
    elif args.mode == "server":
        from src.server import start_server
        start_server(port=args.port, host="0.0.0.0")
    elif args.mode == "city":
        from src.city_network import CityNetworkEnv
        print("Initializing 6-Intersection City Network Simulation...")
        env = CityNetworkEnv(max_steps=args.max_steps)
        env.reset()
        print(f"City Network ready: {len(env.intersections)} Intersections, {len(env.graph)} Nodes.")

        # Simulate an incident at step 10 for demonstration
        rollout_steps = []
        for step in range(args.max_steps):
            if step == 10:
                print("\n[ALERT] Simulating Multi-Vehicle Collision at INT_2 (Downtown Core)...")
                inc = env.trigger_incident("INT_2", "collision")
                print(f"[DISPATCH] Nearest Police: {inc['dispatch']['police']['facility']} (ETA: {inc['dispatch']['police']['eta_sec']}s)")
                print(f"[DISPATCH] Nearest EMS: {inc['dispatch']['ambulance']['facility']} (ETA: {inc['dispatch']['ambulance']['eta_sec']}s)")
                print(f"[PREEMPTION] Green-Wave Preemption corridor: {inc['dispatch']['preemption_corridor']}")

            state, reward, done, info = env.step()
            rollout_steps.append(info)

            if step % 25 == 0:
                print(f"Step {step:03d} | Avg Queue: {info['avg_queue']} | Total Veh: {info['total_vehicles']} | VAE Residual: {info['vae_residual']} | Incidents: {info['active_incidents']}")

            if done:
                break

        os.makedirs("results", exist_ok=True)
        os.makedirs("web_visualizer", exist_ok=True)
        with open("results/city_rollout.json", "w") as f:
            json.dump({"total_steps": len(rollout_steps), "steps": rollout_steps}, f, indent=2)
        with open("web_visualizer/rollout_city.js", "w") as f:
            f.write("window.ATMS_CITY_ROLLOUT = ")
            json.dump({"total_steps": len(rollout_steps), "steps": rollout_steps}, f)
            f.write(";\n")
        print(f"\nCity rollout exported to results/city_rollout.json and web_visualizer/rollout_city.js ({len(rollout_steps)} steps).")


        if args.gui:
            project_dir = os.path.abspath(os.path.dirname(__file__))
            subprocess.Popen(
                [sys.executable, "-m", "http.server", "8765", "--bind", "127.0.0.1"],
                cwd=project_dir,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            webbrowser.open("http://127.0.0.1:8765/web_visualizer/visualizer.html")
    elif args.mode in ["video", "hybrid"]:
        from src.hybrid import run_hybrid_pipeline
        run_hybrid_pipeline(video_path=args.video_path, max_frames=args.max_steps)
        if args.gui:
            project_dir = os.path.abspath(os.path.dirname(__file__))
            subprocess.Popen(
                [sys.executable, "-m", "http.server", "8765", "--bind", "127.0.0.1"],
                cwd=project_dir,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            webbrowser.open("http://127.0.0.1:8765/web_visualizer/visualizer.html")


if __name__ == "__main__":
    main()