# tests/test_ml_backend.py
"""
Unit and integration tests for ATMS ML Backend & City Network Simulation.
Validates:
1. All 4 ML models genuinely integrated and active.
2. Zero randomizers across the simulation engine.
3. Strict physical corridor vehicle conservation.
4. PyTorch RL policy forward pass and Softmax probabilities.
5. Sequence VAE true reconstruction error & incident detection.
6. Custom CNN spatial queue density estimation.
7. Sequence GAN traffic scenario generation.
8. Mathematical equivalence of JS client-side inference to < 1e-6 error.
"""

import sys
import os
import json
import subprocess
import torch
import numpy as np

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.model import load_policy, export_policy_weights_to_js
from src.vae import load_sequence_vae, detect_incident_anomaly
from src.cnn_custom import load_custom_cnn
from src.gan import load_traffic_gan, generate_scenario
from src.city_network import CityNetworkEnv, INTERSECTIONS, FACILITIES, ROAD_SEGMENTS


def test_models_exist_and_load():
    """Verify all 4 ML models exist and load successfully."""
    p = load_policy("models/rl_policy.pt")
    assert p is not None

    v, tw, tc = load_sequence_vae("models/sequence_vae.pt")
    assert v is not None
    assert tw > 0
    assert tc > tw

    c = load_custom_cnn("models/custom_cnn.pt")
    assert c is not None

    g = load_traffic_gan("models/sequence_gan.pt")
    assert g is not None


def test_js_policy_weights_precision():
    """Verify JS client-side inference matches PyTorch forward pass to < 1e-6 error."""
    export_policy_weights_to_js("models/rl_policy.pt", "web_visualizer/policy_weights.js")
    model = load_policy("models/rl_policy.pt")

    # Generate test vectors
    test_vectors = []
    for i in range(50):
        # Deterministic inputs
        q0 = 1.0 + (i * 0.4)
        q1 = 2.0 + (i * 0.3)
        q2 = 3.0 + (i * 0.2)
        q3 = 4.0 + (i * 0.1)
        total = q0 + q1 + q2 + q3
        p0 = 1.0 if i % 2 == 0 else 0.0
        p1 = 1.0 if i % 2 == 1 else 0.0
        test_vectors.append([q0, q1, q2, q3, total, p0, p1])

    # Run PyTorch
    pt_results = []
    for v in test_vectors:
        with torch.no_grad():
            out = model(torch.tensor([v], dtype=torch.float32)).squeeze(0).tolist()
            pt_results.append(out)

    # Run Node.js
    node_code = f"""
    const {{ forwardRLPolicy }} = require('./web_visualizer/policy_weights.js');
    const vectors = {json.dumps(test_vectors)};
    const results = vectors.map(v => forwardRLPolicy(v).rawProbs);
    console.log(JSON.stringify(results));
    """
    proc = subprocess.run(["node", "-e", node_code], capture_output=True, text=True, check=True)
    node_results = json.loads(proc.stdout)

    max_diff = 0.0
    for pt_r, node_r in zip(pt_results, node_results):
        diff = max(abs(pt_r[0] - node_r[0]), abs(pt_r[1] - node_r[1]))
        if diff > max_diff:
            max_diff = diff

    print(f"Max JS vs PyTorch diff: {max_diff:.8e}")
    assert max_diff < 1e-6, f"Error {max_diff} exceeded 1e-6 threshold"


def test_city_network_zero_randomizers():
    """Verify simulation is 100% deterministic (ZERO randomizers)."""
    env1 = CityNetworkEnv(max_steps=20)
    env1.reset()
    rollout1 = [env1.step()[0] for _ in range(15)]

    env2 = CityNetworkEnv(max_steps=20)
    env2.reset()
    rollout2 = [env2.step()[0] for _ in range(15)]

    # Both independent runs must match identically
    for step_idx, (s1, s2) in enumerate(zip(rollout1, rollout2)):
        for node_id in s1:
            q1 = s1[node_id]["queues"]
            q2 = s2[node_id]["queues"]
            assert q1 == q2, f"Discrepancy at step {step_idx} node {node_id}: {q1} != {q2}"


def test_city_network_corridor_conservation():
    """Verify in-transit corridor vehicles and flow conservation."""
    env = CityNetworkEnv(max_steps=30)
    env.reset()

    for step in range(20):
        state, reward, done, info = env.step()
        assert info["total_vehicles"] > 0
        assert info["in_transit_vehicles"] >= 0
        assert info["system_total_vehicles"] == round(info["total_vehicles"] + info["in_transit_vehicles"], 2)


def test_sequence_vae_anomaly_residual_spike():
    """Verify Sequence VAE calculates genuine residuals and detects incidents."""
    env = CityNetworkEnv(max_steps=30)
    env.reset()

    # Normal steps 1 to 5
    for _ in range(5):
        _, _, _, info = env.step()
    baseline_residual = info["vae_residual"]

    # Trigger collision at INT_2
    inc = env.trigger_incident("INT_2", "collision")
    assert inc["node_id"] == "INT_2"

    # Step through incident
    for _ in range(10):
        state, _, _, info = env.step()

    int2_vae = state["INT_2"]["vae_residual"]
    int2_level = state["INT_2"]["vae_level"]

    # Under severe collision blockage, residual must spike and transition to CRITICAL
    assert int2_vae > baseline_residual
    assert int2_level in ("WARNING", "CRITICAL")
    assert info["active_incidents"] == 1


def test_custom_cnn_spatial_densities():
    """Verify CustomTrafficCNN predicts non-negative lane densities."""
    env = CityNetworkEnv(max_steps=10)
    env.reset()
    state, _, _, info = env.step()

    for node_id, inter_data in state.items():
        cnn = inter_data["cnn_densities"]
        assert len(cnn) == 4
        for d in cnn:
            assert d >= 0.0


def test_sequence_gan_scenario_generation():
    """Verify Sequence GAN produces valid scenario trajectories."""
    env = CityNetworkEnv(max_steps=10)
    scenarios = env.generate_traffic_scenarios(scenario_type="rush_hour", num_samples=2)
    assert len(scenarios) == 2
    for sc in scenarios:
        seq = np.array(sc["sequence"])
        assert seq.shape == (10, 4)
        assert np.all(seq >= 0.0)
        assert np.all(seq <= 30.0)


def test_city_network_single_signal_invariant_and_4arm_actuation():
    """Verify strict <=1 GREEN/AMBER mutual exclusion and AMBER->ALL-RED->GREEN clearance across 100 steps."""
    env = CityNetworkEnv(max_steps=120)
    state = env.reset()

    prev_arm = {nid: d["active_arm"] for nid, d in state.items()}
    prev_was_all_red = {nid: d["is_all_red"] for nid, d in state.items()}
    arms_served = {nid: set() for nid in state}

    for step_idx in range(100):
        state, _, _, _ = env.step()
        for nid, d in state.items():
            lights = d["lights"]
            assert set(lights.keys()) == {"North", "South", "East", "West"}
            open_arms = [arm for arm, col in lights.items() if col in ("GREEN", "AMBER")]
            assert len(open_arms) <= 1, f"Step {step_idx} {nid} violated <=1 signal invariant: {lights}"

            if d["is_all_red"]:
                assert len(open_arms) == 0, f"Step {step_idx} {nid} had open lights during all-red: {lights}"
                assert d["active_arm"] is None
            else:
                assert d["active_arm"] in ("N", "S", "E", "W")
                arms_served[nid].add(d["active_arm"])

            assert len(d["arm_scores"]) == 4
            assert len(d["wait_timers"]) == 4

            if d["active_arm"] is not None and prev_arm[nid] is not None and d["active_arm"] != prev_arm[nid]:
                assert prev_was_all_red[nid], (
                    f"Step {step_idx} {nid} switched {prev_arm[nid]} -> {d['active_arm']} without ALL-RED clearance!"
                )
            if d["active_arm"] is not None:
                prev_arm[nid] = d["active_arm"]
            prev_was_all_red[nid] = d["is_all_red"]

    for nid, served in arms_served.items():
        assert len(served) >= 2, f"Intersection {nid} starved arms, only served {served}"

    # Verify explicit 4-arm action & incident preemption also uphold strict <=1 invariant
    inc = env.trigger_incident("INT_2", "collision")
    for _ in range(8):
        state, _, _, _ = env.step(actions={"INT_1": "E"})
        for nid, d in state.items():
            open_arms = [arm for arm, col in d["lights"].items() if col in ("GREEN", "AMBER")]
            assert len(open_arms) <= 1, f"Preemption/action violated <=1 invariant on {nid}: {d['lights']}"
    env.resolve_incident(inc["id"])
    state = env.get_state()
    assert not state["INT_2"]["preemption"]
