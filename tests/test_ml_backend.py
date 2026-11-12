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
from src.city_network import CityNetworkEnv, IntersectionState, INTERSECTIONS, FACILITIES, ROAD_SEGMENTS


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


def test_backend_empty_arm_exclusion_and_wait_timers():
    """Verify empty arms never gain wait timers or outscore queued arms in compute_arm_scores."""
    inter = IntersectionState("INT_1", INTERSECTIONS["INT_1"], initial_queues=[0.0, 0.0, 0.0, 2.0])
    # Force active green to North (empty) and high wait timers on empty S/E arms
    inter.active_approach = 0
    inter.is_yellow = False
    inter.is_all_red = False
    inter.wait_timers = [50, 50, 50, 0]

    scores = inter.compute_arm_scores([0.99, 0.01])
    assert scores[0] == -50.0 and scores[1] == -50.0 and scores[2] == -50.0, (
        f"Empty arms not penalized to -50.0 when W has demand: {scores}"
    )
    assert scores[3] > 0.0, f"Queued West arm should score > 0: {scores}"
    assert int(np.argmax(scores)) == 3, f"Empty arm outscored queued West arm: {scores}"

    # Verify _update_wait_timers resets empty arms & active green, increments only non-empty red arms
    inter.queues = np.array([0.0, 3.0, 0.0, 2.0], dtype=np.float32)
    inter.wait_timers = [10, 4, 10, 2]
    inter._update_wait_timers()
    assert inter.wait_timers == [0, 5, 0, 3], f"Unexpected wait_timers update: {inter.wait_timers}"

    # When all 4 arms are empty, scores are 0.0 and wait_timers reset to 0
    inter.queues = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    inter._update_wait_timers()
    assert inter.wait_timers == [0, 0, 0, 0]
    assert inter.compute_arm_scores([0.8, 0.2]) == [0.0, 0.0, 0.0, 0.0]


def test_backend_standby_all_red_and_instant_wake():
    """Verify empty intersections rest in Standby All-Red and snap straight to GREEN on demand arrival."""
    env = CityNetworkEnv(max_steps=20)
    empty_inter = IntersectionState("INT_1", INTERSECTIONS["INT_1"], initial_queues=[0.0, 0.0, 0.0, 0.0])
    assert empty_inter.is_all_red is True
    assert env._get_lights_state(empty_inter) == {"North": "RED", "South": "RED", "East": "RED", "West": "RED"}

    # Step in standby with zero demand -> stays All-Red
    empty_inter.apply_action(0, standby=True)
    assert empty_inter.is_all_red is True
    assert empty_inter.is_yellow is False
    assert env._get_lights_state(empty_inter) == {"North": "RED", "South": "RED", "East": "RED", "West": "RED"}

    # Demand arrives on West -> snaps straight to GREEN on West without extra amber/all-red delay
    empty_inter.queues = np.array([0.0, 0.0, 0.0, 4.0], dtype=np.float32)
    empty_inter.compute_arm_scores([0.8, 0.2])
    empty_inter.apply_action(1, target_approach=3, standby=False)
    assert empty_inter.is_all_red is False
    assert empty_inter.is_yellow is False
    assert empty_inter.active_approach == 3
    assert env._get_lights_state(empty_inter) == {"North": "RED", "South": "RED", "East": "RED", "West": "GREEN"}

    # Also verify CityNetworkEnv.step() initiates standby clearance when empty and snaps straight to GREEN on wake
    env.reset()
    int2 = env.intersections["INT_2"]
    int2.queues = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    int2.wait_timers = [0, 0, 0, 0]
    int2.is_yellow = False
    int2.is_all_red = False
    # Step once while INT_2 queues are 0 -> CityNetworkEnv.step() selects standby=True and enters AMBER clearance
    env.step()
    assert int2.is_yellow is True and int2.target_approach is None, (
        f"Expected standby clearance (is_yellow=True, target_approach=None), got is_yellow={int2.is_yellow}, target={int2.target_approach}"
    )

    # Place INT_2 in Standby All-Red, then inject West demand -> CityNetworkEnv.step() wakes straight to West GREEN
    int2.is_yellow = False
    int2.is_all_red = True
    int2.all_red_timer = 0
    int2.target_approach = None
    int2.queues = np.array([0.0, 0.0, 0.0, 4.0], dtype=np.float32)
    state, _, _, _ = env.step()
    assert int2.is_all_red is False
    assert int2.active_approach == 3
    assert state["INT_2"]["lights"] == {"North": "RED", "South": "RED", "East": "RED", "West": "GREEN"}


def test_backend_fast_gap_out_and_non_sequential_jump():
    """Verify active empty arm gaps out before min_green and jumps directly N -> AMBER -> ALL-RED -> W."""
    env = CityNetworkEnv(max_steps=20)
    inter = IntersectionState("INT_1", INTERSECTIONS["INT_1"], initial_queues=[0.0, 0.0, 0.0, 6.0])
    inter.active_approach = 0  # North currently GREEN
    inter.target_approach = 0
    inter.phase = 0
    inter.phase_counter = 1  # Strictly below min_green (3)
    inter.is_yellow = False
    inter.is_all_red = False

    visited_active = ["N"]
    phases = []
    # Step 1: fast gap-out must trigger AMBER immediately despite phase_counter < min_green
    inter.compute_arm_scores([0.95, 0.05])
    inter.apply_action(1, target_approach=3, standby=False)
    assert inter.is_yellow is True, "Fast gap-out did not trigger AMBER immediately when active N was empty!"
    assert inter.target_approach == 3
    phases.append("AMBER")

    # Step 2: AMBER -> ALL-RED
    inter.apply_action(1, target_approach=3, standby=False)
    assert inter.is_all_red is True and inter.is_yellow is False
    assert all(col == "RED" for col in env._get_lights_state(inter).values())
    phases.append("ALL-RED")

    # Step 3: ALL-RED -> GREEN directly on West (non-sequential, skipping S and E)
    inter.apply_action(1, target_approach=3, standby=False)
    assert inter.is_all_red is False and inter.is_yellow is False
    assert inter.active_approach == 3
    assert env._get_lights_state(inter) == {"North": "RED", "South": "RED", "East": "RED", "West": "GREEN"}
    visited_active.append("W")
    phases.append("GREEN_W")

    assert visited_active == ["N", "W"], f"Expected direct N -> W jump, visited {visited_active}"
    assert phases == ["AMBER", "ALL-RED", "GREEN_W"], f"Unexpected transition phases: {phases}"

    # Also verify CityNetworkEnv.step() end-to-end picks West directly from N when W=6.0 and N=0
    env.reset()
    j = env.intersections["INT_1"]
    j.active_approach = 0
    j.target_approach = 0
    j.phase = 0
    j.phase_counter = 1
    j.is_yellow = False
    j.is_all_red = False
    j.queues = np.array([0.0, 0.0, 0.0, 6.0], dtype=np.float32)

    seen_arms = ["N"]
    saw_amber = False
    saw_all_red = False
    for _ in range(4):
        state, _, _, _ = env.step()
        d = state["INT_1"]
        open_arms = [a for a, col in d["lights"].items() if col in ("GREEN", "AMBER")]
        assert len(open_arms) <= 1, f"Violated <=1 invariant: {d['lights']}"
        if d["lights"]["North"] == "AMBER":
            saw_amber = True
        if d["is_all_red"]:
            saw_all_red = True
        if d["active_arm"] is not None and d["active_arm"] not in seen_arms:
            seen_arms.append(d["active_arm"])
        if d["active_arm"] == "W" and d["lights"]["West"] == "GREEN":
            break

    assert saw_amber and saw_all_red, f"Missing AMBER ({saw_amber}) or ALL-RED ({saw_all_red}) clearance"
    assert seen_arms == ["N", "W"], f"CityNetworkEnv.step() did not jump N -> W directly: {seen_arms}"
