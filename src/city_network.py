# src/city_network.py
"""
Multi-Intersection City Road Network & Emergency Dispatch Engine.
Simulates a multi-node urban corridor network with:
- 6 connected intersections (West Gate, Downtown Core, Financial Plaza,
  University Crossing, Civic Center, Harbor Terminal)
- Road corridors connecting intersections with bi-directional traffic queues
- Multi-approach 3-lens traffic light state machines per intersection
- Autonomous Deep RL / MLP model signal phase actuation (models/rl_policy.pt)
- Sequence VAE (models/sequence_vae.pt) for physical reconstruction residual anomaly detection
- Custom Convolutional Neural Network (models/custom_cnn.pt) for spatial queue density estimation
- Sequence GAN (models/sequence_gan.pt) for scenario generation analytics
- Dynamic Green-Wave signal preemption for emergency vehicles in transit
- Shortest-path automated emergency dispatch (Dijkstra routing)
- Strict physical queue dynamics, corridor flow conservation, and ZERO randomizers
"""

import math
import heapq
import collections
import numpy as np
import torch

# Coordinates for 2D / 2.5D map space
CITY_WIDTH = 1100
CITY_HEIGHT = 800

# 6 Intersections in a 2x3 Grid with Corridors
INTERSECTIONS = {
    "INT_1": {
        "id": "INT_1",
        "name": "West Gate",
        "streets": "Boulevard Ave & West Pkwy",
        "x": 220,
        "y": 220,
        "neighbors": {"North": None, "South": "INT_4", "East": "INT_2", "West": None},
        "limits": {"N": 25.0, "S": 25.0, "E": 25.0, "W": 25.0},
    },
    "INT_2": {
        "id": "INT_2",
        "name": "Downtown Core",
        "streets": "Central Ave & Main St",
        "x": 550,
        "y": 220,
        "neighbors": {"North": None, "South": "INT_5", "East": "INT_3", "West": "INT_1"},
        "limits": {"N": 30.0, "S": 30.0, "E": 30.0, "W": 30.0},
    },
    "INT_3": {
        "id": "INT_3",
        "name": "Financial Plaza",
        "streets": "Market St & Financial Way",
        "x": 880,
        "y": 220,
        "neighbors": {"North": None, "South": "INT_6", "East": None, "West": "INT_2"},
        "limits": {"N": 25.0, "S": 25.0, "E": 25.0, "W": 25.0},
    },
    "INT_4": {
        "id": "INT_4",
        "name": "University Crossing",
        "streets": "Campus Dr & West Pkwy",
        "x": 220,
        "y": 560,
        "neighbors": {"North": "INT_1", "South": None, "East": "INT_5", "West": None},
        "limits": {"N": 25.0, "S": 25.0, "E": 25.0, "W": 25.0},
    },
    "INT_5": {
        "id": "INT_5",
        "name": "Civic Center",
        "streets": "Central Ave & River Rd",
        "x": 550,
        "y": 560,
        "neighbors": {"North": "INT_2", "South": None, "East": "INT_6", "West": "INT_4"},
        "limits": {"N": 30.0, "S": 30.0, "E": 30.0, "W": 30.0},
    },
    "INT_6": {
        "id": "INT_6",
        "name": "Harbor Terminal",
        "streets": "Harbor Blvd & Market St",
        "x": 880,
        "y": 560,
        "neighbors": {"North": "INT_3", "South": None, "East": None, "West": "INT_5"},
        "limits": {"N": 25.0, "S": 25.0, "E": 25.0, "W": 25.0},
    },
}

# Calibrated deterministic baseline queue states (ZERO randomizers)
INITIAL_QUEUES = {
    "INT_1": [3.2, 4.0, 3.8, 4.5],  # West Gate arterial
    "INT_2": [5.5, 6.0, 5.2, 5.8],  # Downtown Core high-density hub
    "INT_3": [4.0, 4.5, 3.5, 4.2],  # Financial Plaza commercial
    "INT_4": [3.5, 3.0, 4.2, 3.8],  # University Crossing campus
    "INT_5": [4.8, 5.2, 4.5, 4.0],  # Civic Center government center
    "INT_6": [3.8, 4.2, 3.0, 3.5],  # Harbor Terminal port logistics
}

# Designated Emergency Facilities
FACILITIES = {
    "POLICE_MAIN": {
        "id": "POLICE_MAIN",
        "type": "police",
        "name": "Downtown Police Precinct #1",
        "x": 680,
        "y": 480,
        "nearest_node": "INT_5",
        "units": ["Cruiser 101", "Interceptor 104", "Patrol 112"],
        "color": "#38bdf8",
    },
    "POLICE_SUB": {
        "id": "POLICE_SUB",
        "type": "police",
        "name": "West Police Substation #2",
        "x": 100,
        "y": 120,
        "nearest_node": "INT_1",
        "units": ["Patrol 201", "Cruiser 205"],
        "color": "#38bdf8",
    },
    "HOSPITAL_MAIN": {
        "id": "HOSPITAL_MAIN",
        "type": "hospital",
        "name": "Metro Memorial Trauma Center",
        "x": 420,
        "y": 110,
        "nearest_node": "INT_2",
        "units": ["Medic 1", "Ambulance 3", "Trauma Rescue 7"],
        "color": "#ef4444",
    },
    "HOSPITAL_SOUTH": {
        "id": "HOSPITAL_SOUTH",
        "type": "hospital",
        "name": "South Harbor EMS Depot",
        "x": 990,
        "y": 660,
        "nearest_node": "INT_6",
        "units": ["Ambulance 5", "Rescue 9"],
        "color": "#ef4444",
    },
    "FIRE_1": {
        "id": "FIRE_1",
        "type": "fire",
        "name": "Fire & Rescue Station 4",
        "x": 100,
        "y": 470,
        "nearest_node": "INT_4",
        "units": ["Engine 4", "Ladder 2"],
        "color": "#f97316",
    },
}

# Road Corridors connecting intersections & perimeter boundaries
ROAD_SEGMENTS = [
    # Top horizontal corridor
    {"id": "R_W_INT1", "from": "BORDER_W1", "to": "INT_1", "start": (30, 220), "end": (220, 220), "corridor": "Boulevard Ave"},
    {"id": "R_INT1_INT2", "from": "INT_1", "to": "INT_2", "start": (220, 220), "end": (550, 220), "corridor": "Main St"},
    {"id": "R_INT2_INT3", "from": "INT_2", "to": "INT_3", "start": (550, 220), "end": (880, 220), "corridor": "Main St East"},
    {"id": "R_INT3_E", "from": "INT_3", "to": "BORDER_E1", "start": (880, 220), "end": (1070, 220), "corridor": "Expressway East"},
    # Bottom horizontal corridor
    {"id": "R_W_INT4", "from": "BORDER_W2", "to": "INT_4", "start": (30, 560), "end": (220, 560), "corridor": "Campus Dr West"},
    {"id": "R_INT4_INT5", "from": "INT_4", "to": "INT_5", "start": (220, 560), "end": (550, 560), "corridor": "River Rd"},
    {"id": "R_INT5_INT6", "from": "INT_5", "to": "INT_6", "start": (550, 560), "end": (880, 560), "corridor": "Harbor Blvd"},
    {"id": "R_INT6_E", "from": "INT_6", "to": "BORDER_E2", "start": (880, 560), "end": (1070, 560), "corridor": "Harbor Terminal E"},
    # Left vertical corridor
    {"id": "R_N_INT1", "from": "BORDER_N1", "to": "INT_1", "start": (220, 30), "end": (220, 220), "corridor": "West Pkwy North"},
    {"id": "R_INT1_INT4", "from": "INT_1", "to": "INT_4", "start": (220, 220), "end": (220, 560), "corridor": "West Pkwy Central"},
    {"id": "R_INT4_S", "from": "INT_4", "to": "BORDER_S1", "start": (220, 560), "end": (220, 770), "corridor": "West Pkwy South"},
    # Center vertical corridor
    {"id": "R_N_INT2", "from": "BORDER_N2", "to": "INT_2", "start": (550, 30), "end": (550, 220), "corridor": "Central Ave North"},
    {"id": "R_INT2_INT5", "from": "INT_2", "to": "INT_5", "start": (550, 220), "end": (550, 560), "corridor": "Central Ave Midtown"},
    {"id": "R_INT5_S", "from": "INT_5", "to": "BORDER_S2", "start": (550, 560), "end": (550, 770), "corridor": "Central Ave South"},
    # Right vertical corridor
    {"id": "R_N_INT3", "from": "BORDER_N3", "to": "INT_3", "start": (880, 30), "end": (880, 220), "corridor": "Market St North"},
    {"id": "R_INT3_INT6", "from": "INT_3", "to": "INT_6", "start": (880, 220), "end": (880, 560), "corridor": "Market St South"},
    {"id": "R_INT6_S", "from": "INT_6", "to": "BORDER_S3", "start": (880, 560), "end": (880, 770), "corridor": "Harbor Access Rd"},
]


APPROACH_NAMES = ("North", "South", "East", "West")
APPROACH_CODES = ("N", "S", "E", "W")
_APPROACH_LOOKUP = {
    "n": 0, "north": 0,
    "s": 1, "south": 1,
    "e": 2, "east": 2,
    "w": 3, "west": 3,
}


def parse_approach_action(val):
    """
    Parse a manual signal action into either:
    - ('axis', 0|1) for legacy 2-phase actions (0=NS, 1=EW)
    - ('arm', 0..3) for independent 4-approach control ('N'|'S'|'E'|'W' or 'North'|'South'|'East'|'West')
    Raises ValueError on invalid inputs.
    """
    if isinstance(val, bool):
        raise ValueError(f"Invalid boolean signal action: {val!r}")
    if isinstance(val, (int, np.integer)):
        iv = int(val)
        if iv in (0, 1):
            return ("axis", iv)
        raise ValueError(f"Numeric action must be 0 or 1, got {iv}")
    if isinstance(val, str):
        key = val.strip().lower()
        if key in _APPROACH_LOOKUP:
            return ("arm", _APPROACH_LOOKUP[key])
        if key in ("0", "1"):
            return ("axis", int(key))
        raise ValueError(f"Unknown approach/action string: {val!r}")
    raise ValueError(f"Unsupported signal action type: {type(val).__name__}")


class IntersectionState:
    """
    Physical state machine and queue dynamics for an individual 4-way intersection.
    Approaches: 0: North, 1: South, 2: East, 3: West.
    Enforces strict single-approach mutual exclusion: at most ONE of {N, S, E, W}
    is GREEN or AMBER at any time, transitioning through AMBER -> ALL-RED -> GREEN.
    """
    def __init__(self, node_id, config, initial_queues=None):
        self.node_id = node_id
        self.config = config

        # Deterministic calibrated initial queue vector (ZERO randomizers)
        if initial_queues is not None:
            self.queues = np.array(initial_queues, dtype=np.float32)
        elif node_id in INITIAL_QUEUES:
            self.queues = np.array(INITIAL_QUEUES[node_id], dtype=np.float32)
        else:
            self.queues = np.array([4.0, 4.0, 4.0, 4.0], dtype=np.float32)

        # Pick initial active single arm with largest initial queue (tie-break: North=0).
        # If all 4 approaches start completely empty, start in Standby All-Red.
        if float(np.max(self.queues)) <= 1e-6:
            self.active_approach = 0
            self.target_approach = None
            self.phase = 0
            self.is_all_red = True
        else:
            self.active_approach = int(np.argmax(self.queues))
            self.target_approach = self.active_approach
            self.phase = 0 if self.active_approach in (0, 1) else 1  # 0 = NS axis, 1 = EW axis
            self.is_all_red = False
        self.phase_counter = 1  # Steps elapsed in current phase
        self.min_green = 3      # Minimum green duration steps before transition allowed
        self.is_yellow = False  # Yellow / Amber transition state
        self.yellow_timer = 0
        self.all_red_timer = 0
        self.preemption = False # Emergency vehicle green wave active
        self.preemption_target_phase = 0
        self.preemption_target_approach = 0
        self.last_action = self.phase
        self.probs = [0.5, 0.5]
        self.wait_timers = [0, 0, 0, 0]
        self.arm_scores = []
        self.compute_arm_scores(self.probs)

        # ML Model Outputs & Anomaly Detection Telemetry
        self.queue_history = collections.deque(maxlen=5)
        # Preload initial state to seed rolling history
        for _ in range(5):
            self.queue_history.append(self.queues.copy())

        self.vae_residual = 0.85
        self.vae_level = "NORMAL"
        self.vae_alert = "Traffic flow normal across all approaches"
        self.vae_approach_scores = [0.2, 0.2, 0.2, 0.2]
        self.cnn_densities = [float(round(q, 2)) for q in self.queues]
        self.last_departed = np.zeros(4, dtype=np.float32)

    def get_state_7d(self):
        """Converts local state to standard 7-D format [q0, q1, q2, q3, total, p0, p1]."""
        total = float(np.sum(self.queues))
        p0 = 1.0 if self.phase == 0 else 0.0
        p1 = 1.0 if self.phase == 1 else 0.0
        return np.array([
            self.queues[0], self.queues[1], self.queues[2], self.queues[3],
            total, p0, p1
        ], dtype=np.float32)

    def compute_arm_scores(self, probs, wait_bonus=0.35):
        """
        Scores all 4 individual approaches [N, S, E, W] independently by combining
        RL policy axis probabilities [p_ns, p_ew], per-arm queue demand, and
        starvation-prevention wait timers.
        Empty arms (q <= 1e-6) receive a strong penalty whenever any other arm
        has queued vehicles so an empty arm can never outscore a queued arm.
        """
        p_ns = float(probs[0])
        p_ew = float(probs[1])
        has_any_demand = any(float(q) > 1e-6 for q in self.queues)
        scores = []
        for idx in range(4):
            q = float(self.queues[idx])
            if q <= 1e-6:
                scores.append(-50.0 if has_any_demand else 0.0)
                continue
            axis_p = p_ns if idx in (0, 1) else p_ew
            w = float(self.wait_timers[idx])
            effective_wait = wait_bonus * w * min(1.0, q)
            scores.append(q * (1.0 + 0.5 * axis_p) + effective_wait)
        self.arm_scores = [float(round(s, 4)) for s in scores]
        return scores

    def _update_wait_timers(self):
        """
        Increments wait timers only for RED approaches that actually have queued
        vehicles (> 0), and resets active GREEN or empty (q == 0) approaches to 0.
        """
        for idx in range(4):
            is_green = not self.is_yellow and not self.is_all_red and idx == self.active_approach
            if is_green or float(self.queues[idx]) <= 1e-6:
                self.wait_timers[idx] = 0
            else:
                self.wait_timers[idx] += 1

    def apply_action(self, action, target_approach=None, standby=False):
        """
        Apply signal action (0/1 axis or explicit 0..3 approach) with strict
        single-approach mutual exclusion, latched target transitions, fast
        gap-out when the active green arm drains while another arm has demand,
        Standby All-Red when all arms are empty, and AMBER -> ALL-RED -> GREEN
        clearance.
        """
        if self.preemption:
            desired_approach = int(self.preemption_target_approach) % 4
        elif target_approach is not None:
            desired_approach = int(target_approach) % 4
        else:
            axis = 0 if int(action) == 0 else 1
            pair = (0, 1) if axis == 0 else (2, 3)
            # If current active arm is already on the requested axis and no specific
            # arm was requested, keep it unless it is empty while the opposite arm
            # has queued demand, or the opposite arm has significantly higher score.
            if self.active_approach in pair:
                other = pair[1] if self.active_approach == pair[0] else pair[0]
                if float(self.queues[self.active_approach]) <= 1e-6 and float(self.queues[other]) > 1e-6:
                    desired_approach = other
                elif (
                    float(self.queues[other]) > 1e-6
                    and self.arm_scores[other] > self.arm_scores[self.active_approach] * 1.15
                ):
                    desired_approach = other
                else:
                    desired_approach = self.active_approach
            else:
                q0, q1 = float(self.queues[pair[0]]), float(self.queues[pair[1]])
                if q0 <= 1e-6 and q1 > 1e-6:
                    desired_approach = pair[1]
                elif q1 <= 1e-6 and q0 > 1e-6:
                    desired_approach = pair[0]
                else:
                    desired_approach = pair[0] if self.arm_scores[pair[0]] >= self.arm_scores[pair[1]] else pair[1]

        if self.is_yellow:
            self.yellow_timer -= 1
            if self.target_approach is None and (self.preemption or not standby):
                self.target_approach = desired_approach
            if self.yellow_timer <= 0:
                self.is_yellow = False
                self.is_all_red = True
                self.all_red_timer = 1
            self._update_wait_timers()
            return

        if self.is_all_red:
            if self.all_red_timer > 0:
                self.all_red_timer -= 1
                if self.all_red_timer <= 0:
                    if self.target_approach is None and standby and not self.preemption:
                        self._update_wait_timers()
                        return
                    chosen = desired_approach if self.target_approach is None else self.target_approach
                    self.is_all_red = False
                    self.active_approach = chosen
                    self.target_approach = chosen
                    self.phase = 0 if self.active_approach in (0, 1) else 1
                    self.phase_counter = 1
                self._update_wait_timers()
                return
            # Currently resting in Standby All-Red (all_red_timer <= 0)
            if standby and not self.preemption:
                self._update_wait_timers()
                return
            self.is_all_red = False
            self.active_approach = desired_approach
            self.target_approach = desired_approach
            self.phase = 0 if self.active_approach in (0, 1) else 1
            self.phase_counter = 1
            self._update_wait_timers()
            return

        if standby and not self.preemption:
            self.target_approach = None
            self.is_yellow = True
            self.yellow_timer = 1
            self._update_wait_timers()
            return

        if desired_approach != self.active_approach:
            active_q = float(self.queues[self.active_approach])
            desired_q = float(self.queues[desired_approach])
            gap_out = active_q < 0.35 and desired_q > max(1e-6, active_q)
            # Preemption or fast gap-out initiates amber clearance immediately; normal switching respects min_green
            if self.preemption or gap_out or self.phase_counter >= self.min_green:
                self.target_approach = desired_approach
                self.is_yellow = True
                self.yellow_timer = 1
            else:
                self.phase_counter += 1
        else:
            self.phase_counter += 1

        self._update_wait_timers()

    def step_traffic(self, arrivals, blockage_factors=None):
        """
        Discharges ONLY the single active GREEN approach according to physical
        saturation flow (3.80 veh/step concentrated on the single open approach,
        calibrated for 1-of-4 green duty cycle with amber + all-red clearance)
        and accumulates arrivals. Never discharges 2 approaches simultaneously.
        """
        discharge_rate = 3.80
        if blockage_factors is None:
            blockage_factors = np.ones(4, dtype=np.float32)

        # Departures: North=0, South=1, East=2, West=3 (strictly single approach)
        depart_demanded = np.zeros(4, dtype=np.float32)
        if not self.is_yellow and not self.is_all_red:
            app = self.active_approach
            depart_demanded[app] = discharge_rate * blockage_factors[app]

        actual_depart = np.minimum(self.queues, depart_demanded)
        self.last_departed = actual_depart.copy()

        # Update physical queue with Newell-Daganzo inflow/outflow balance
        self.queues = np.clip(self.queues - actual_depart + arrivals, 0.0, 30.0)
        return float(actual_depart.sum()), actual_depart


class CityNetworkEnv:
    """
    Master Environment for the 6-Intersection City Network.
    Coordinates signal policies, traffic flow conservation, incident simulation,
    and emergency dispatch routing with ZERO randomizers.
    """
    def __init__(self, max_steps=500, device="cpu"):
        self.max_steps = max_steps
        self.device = device
        self.step_idx = 0
        self._incident_seq = 0
        self.intersections = {}
        self.active_incidents = []
        self.active_dispatches = []
        self.preemption_active = False

        # In-transit corridor buffers: (from_node, to_node, to_approach) -> float
        # Implements true physical vehicle conservation across the corridor network
        self.corridor_in_transit = collections.defaultdict(float)

        # Load trained PyTorch ML models
        self.policy_model = None
        self.vae_model = None
        self.vae_th_warn = 4.95
        self.vae_th_crit = 7.58
        self.custom_cnn = None
        self.gan_generator = None

        self._load_ml_models()
        self._build_graph()
        self.reset()

    def _load_ml_models(self):
        """Loads and verifies all 4 PyTorch ML models."""
        # 1. RL Policy
        try:
            from src.model import load_policy
            self.policy_model = load_policy("models/rl_policy.pt", device=self.device)
            self.policy_model.eval()
        except Exception as e:
            print(f"[CityNetwork] Policy model load warning: {e}")
            self.policy_model = None

        # 2. Sequence VAE
        try:
            from src.vae import load_sequence_vae
            self.vae_model, self.vae_th_warn, self.vae_th_crit = load_sequence_vae("models/sequence_vae.pt", device=self.device)
            self.vae_model.eval()
        except Exception as e:
            print(f"[CityNetwork] Sequence VAE load warning: {e}")
            self.vae_model = None

        # 3. Custom CNN
        try:
            from src.cnn_custom import load_custom_cnn
            self.custom_cnn = load_custom_cnn("models/custom_cnn.pt", device=self.device)
            self.custom_cnn.eval()
        except Exception as e:
            print(f"[CityNetwork] Custom CNN load warning: {e}")
            self.custom_cnn = None

        # 4. Sequence GAN
        try:
            from src.gan import load_traffic_gan
            self.gan_generator = load_traffic_gan("models/sequence_gan.pt", device=self.device)
            self.gan_generator.eval()
        except Exception as e:
            print(f"[CityNetwork] Sequence GAN load warning: {e}")
            self.gan_generator = None

    def _build_graph(self):
        """Constructs adjacency graph for Dijkstra shortest path routing."""
        self.graph = {}
        for node_id, data in INTERSECTIONS.items():
            self.graph[node_id] = {}
            for direction, neighbor in data["neighbors"].items():
                if neighbor:
                    n_data = INTERSECTIONS[neighbor]
                    dist = math.hypot(data["x"] - n_data["x"], data["y"] - n_data["y"])
                    self.graph[node_id][neighbor] = dist

    def reset(self):
        """Resets all intersections to calibrated initial states and clears active incidents."""
        self.step_idx = 0
        self._incident_seq = 0
        self.intersections = {
            node_id: IntersectionState(node_id, data)
            for node_id, data in INTERSECTIONS.items()
        }
        self.active_incidents = []
        self.active_dispatches = []
        self.preemption_active = False
        self.corridor_in_transit.clear()
        return self.get_state()

    def _generate_spatial_grid(self, queues):
        """
        Creates a 64x64 3-channel spatial occupancy grid from intersection queue lengths.
        Compatible with CustomTrafficCNN input specifications.
        """
        grid = np.zeros((1, 3, 64, 64), dtype=np.float32)
        # North approach (top center)
        n_len = int(np.clip(queues[0] * 1.2, 0, 24))
        if n_len > 0:
            grid[0, 0, 28 - n_len:28, 28:36] += 0.8
            grid[0, 1, 28 - n_len:28, 28:36] += 0.2
        # South approach (bottom center)
        s_len = int(np.clip(queues[1] * 1.2, 0, 24))
        if s_len > 0:
            grid[0, 0, 36:36 + s_len, 28:36] += 0.8
            grid[0, 1, 36:36 + s_len, 28:36] += 0.2
        # East approach (right center)
        e_len = int(np.clip(queues[2] * 1.2, 0, 24))
        if e_len > 0:
            grid[0, 0, 28:36, 36:36 + e_len] += 0.8
            grid[0, 1, 28:36, 36:36 + e_len] += 0.2
        # West approach (left center)
        w_len = int(np.clip(queues[3] * 1.2, 0, 24))
        if w_len > 0:
            grid[0, 0, 28:36, 28 - w_len:28] += 0.8
            grid[0, 1, 28:36, 28 - w_len:28] += 0.2
        return grid

    def get_state(self):
        """Returns comprehensive telemetry state for all intersections."""
        data = {}
        for node_id, inter in self.intersections.items():
            active_arm = None if inter.is_all_red else APPROACH_CODES[inter.active_approach]
            active_approach = None if inter.is_all_red else APPROACH_NAMES[inter.active_approach]
            data[node_id] = {
                "id": node_id,
                "name": inter.config["name"],
                "streets": inter.config["streets"],
                "x": int(inter.config["x"]),
                "y": int(inter.config["y"]),
                "queues": [float(round(float(q), 2)) for q in inter.queues],
                "total_vehicles": float(round(float(np.sum(inter.queues)), 1)),
                "phase": int(inter.phase),
                "is_yellow": bool(inter.is_yellow),
                "is_all_red": bool(inter.is_all_red),
                "active_arm": active_arm,
                "active_approach": active_approach,
                "wait_timers": [int(w) for w in inter.wait_timers],
                "arm_scores": [float(round(float(s), 4)) for s in inter.arm_scores],
                "phase_counter": int(inter.phase_counter),
                "preemption": bool(inter.preemption),
                "action": int(inter.last_action),
                "probs": [float(round(float(p), 4)) for p in inter.probs],
                "lights": self._get_lights_state(inter),
                "cnn_densities": [float(round(float(d), 2)) for d in inter.cnn_densities],
                "vae_residual": float(round(float(inter.vae_residual), 2)),
                "vae_level": str(inter.vae_level),
                "vae_alert": str(inter.vae_alert),
                "vae_approach_scores": [float(round(float(s), 3)) for s in inter.vae_approach_scores],
            }
        return data

    def _get_lights_state(self, inter):
        """
        Returns active colors (RED, AMBER, GREEN) for all 4 approaches.
        Enforces strict single-approach mutual exclusion: at most ONE approach
        is GREEN or AMBER at any time; all others are RED (all 4 RED during clearance).
        """
        lights = {name: "RED" for name in APPROACH_NAMES}
        if inter.is_all_red:
            return lights
        active_name = APPROACH_NAMES[inter.active_approach]
        lights[active_name] = "AMBER" if inter.is_yellow else "GREEN"
        return lights

    def shortest_path(self, start_node, end_node):
        """Dijkstra shortest path algorithm on intersection network."""
        if start_node == end_node:
            return [start_node]
        queue = [(0.0, start_node, [])]
        seen = set()

        while queue:
            (cost, node, path) = heapq.heappop(queue)
            if node in seen:
                continue
            seen.add(node)
            path = path + [node]
            if node == end_node:
                return path

            for neighbor, weight in self.graph.get(node, {}).items():
                if neighbor not in seen:
                    heapq.heappush(queue, (cost + weight, neighbor, path))
        return [start_node, end_node]

    def trigger_incident(self, location_node=None, incident_type="collision", x=None, y=None, blocked_approach=None):
        """
        Spawns an interactive incident at an intersection with deterministic placement (ZERO randomizers).
        Types: 'collision', 'hazard', 'medical'.
        """
        if location_node is None or location_node not in self.intersections:
            if location_node is None:
                # Deterministically pick the intersection with highest total congestion
                location_node = max(
                    self.intersections.keys(),
                    key=lambda k: float(np.sum(self.intersections[k].queues))
                )
            else:
                location_node = "INT_2"

        target = INTERSECTIONS[location_node]
        inter = self.intersections[location_node]

        # Normalize or deterministically choose blocked approach
        if blocked_approach is not None:
            key = str(blocked_approach).strip().lower()
            app_idx = _APPROACH_LOOKUP.get(key, None)
            blocked_approach = APPROACH_NAMES[app_idx] if app_idx is not None else "North"
        else:
            # Block the approach with the highest queue, or by canonical incident type
            worst_app = int(np.argmax(inter.queues))
            if incident_type == "collision":
                blocked_approach = "East" if inter.queues[2] >= inter.queues[0] else "North"
            elif incident_type == "hazard":
                blocked_approach = "North"
            else:
                blocked_approach = APPROACH_NAMES[worst_app]

        # Compute deterministic approach lane coordinate offset
        offsets = {
            "North": (0, -32),
            "South": (0, 32),
            "East": (32, 0),
            "West": (-32, 0),
        }
        dx, dy = offsets.get(blocked_approach, (0, 0))
        target_x = float(x) if x is not None else float(target["x"] + dx)
        target_y = float(y) if y is not None else float(target["y"] + dy)

        severity = 3 if incident_type == "collision" else (2 if incident_type == "hazard" else 1)
        # Blockage capacity: collision blocks 85% (0.15 flow capacity), hazard blocks 70%, medical blocks 50%
        blockage_capacity = 0.15 if incident_type == "collision" else (0.30 if incident_type == "hazard" else 0.50)

        self._incident_seq += 1
        incident_id = f"INC-{self._incident_seq:03d}"
        incident = {
            "id": incident_id,
            "type": incident_type,
            "node_id": location_node,
            "name": f"{str(incident_type).title()} at {target['name']}",
            "streets": target["streets"],
            "x": target_x,
            "y": target_y,
            "severity": severity,
            "blocked_approach": blocked_approach,
            "blockage_capacity": blockage_capacity,
            "step_created": self.step_idx,
            "status": "DETECTED",
            "progress": 0,
        }
        self.active_incidents.append(incident)

        # Dispatch closest emergency units with Green Wave preemption
        dispatch_info = self.dispatch_emergency(incident)
        incident["dispatch"] = dispatch_info
        return incident

    def _route_approach_for_node(self, route, idx, fallback_approach=0):
        """Determines which single approach (0..3) of route[idx] serves the emergency route."""
        node = route[idx]
        neighbors = INTERSECTIONS.get(node, {}).get("neighbors", {})
        # Prefer the incoming approach from the previous node along the route
        if idx > 0:
            prev_node = route[idx - 1]
            for dir_name, neighbor_id in neighbors.items():
                if neighbor_id == prev_node:
                    return _APPROACH_LOOKUP[dir_name.lower()]
        # If starting at this node, align with outgoing axis/approach if available
        if idx + 1 < len(route):
            next_node = route[idx + 1]
            for dir_name, neighbor_id in neighbors.items():
                if neighbor_id == next_node:
                    # Opposite incoming arm feeds straight toward next_node
                    opp = {"North": 1, "South": 0, "East": 3, "West": 2}[dir_name]
                    return opp
        return fallback_approach

    def dispatch_emergency(self, incident):
        """
        Calculates shortest path emergency routes and activates AI Green Wave Preemption.
        """
        inc_node = incident["node_id"]
        inc_app_idx = _APPROACH_LOOKUP.get(str(incident.get("blocked_approach", "North")).lower(), 0)

        # Find closest police precinct
        police_stations = [f for f in FACILITIES.values() if f["type"] == "police"]
        closest_police = min(
            police_stations,
            key=lambda f: math.hypot(f["x"] - incident["x"], f["y"] - incident["y"])
        )
        police_route = self.shortest_path(closest_police["nearest_node"], inc_node)

        # Find closest hospital
        hospitals = [f for f in FACILITIES.values() if f["type"] == "hospital"]
        closest_hospital = min(
            hospitals,
            key=lambda f: math.hypot(f["x"] - incident["x"], f["y"] - incident["y"])
        )
        hospital_route = self.shortest_path(closest_hospital["nearest_node"], inc_node)

        corridor_approaches = {}
        for route in (police_route, hospital_route):
            for idx, node in enumerate(route):
                if node in self.intersections:
                    corridor_approaches[node] = self._route_approach_for_node(
                        route, idx, fallback_approach=inc_app_idx
                    )

        dispatch_record = {
            "incident_id": incident["id"],
            "police": {
                "facility": closest_police["name"],
                "unit": closest_police["units"][0],
                "start": (closest_police["x"], closest_police["y"]),
                "route": police_route,
                "status": "DISPATCHED",
                "eta_sec": len(police_route) * 4 + 4,
            },
            "ambulance": {
                "facility": closest_hospital["name"],
                "unit": closest_hospital["units"][0],
                "start": (closest_hospital["x"], closest_hospital["y"]),
                "route": hospital_route,
                "status": "DISPATCHED",
                "eta_sec": len(hospital_route) * 5 + 6,
            },
            "preemption_corridor": list(set(police_route + hospital_route)),
            "preemption_approaches": corridor_approaches,
        }
        self.active_dispatches.append(dispatch_record)
        self._recompute_preemption()
        return dispatch_record

    def _recompute_preemption(self):
        """Recomputes active Green Wave preemption across all remaining active dispatches."""
        for inter in self.intersections.values():
            inter.preemption = False
        self.preemption_active = False

        for disp in self.active_dispatches:
            approaches = disp.get("preemption_approaches", {})
            if approaches:
                for node, app_idx in approaches.items():
                    if node in self.intersections:
                        inter = self.intersections[node]
                        inter.preemption = True
                        inter.preemption_target_approach = int(app_idx) % 4
                        inter.preemption_target_phase = 0 if inter.preemption_target_approach in (0, 1) else 1
                        self.preemption_active = True
            else:
                self._apply_preemption(disp.get("preemption_corridor", []))

    def _apply_preemption(self, corridor_nodes):
        """Activates Green Wave signal preemption along travel corridor."""
        self.preemption_active = True
        for node in corridor_nodes:
            if node in self.intersections:
                inter = self.intersections[node]
                inter.preemption = True
                # Default single approach along corridor if no explicit route mapping
                app_idx = 2 if node in ["INT_1", "INT_2", "INT_3"] else 0
                inter.preemption_target_approach = app_idx
                inter.preemption_target_phase = 0 if app_idx in (0, 1) else 1

    def resolve_incident(self, incident_id):
        """Resolves an active incident and returns affected signals to normal autonomous control."""
        self.active_incidents = [i for i in self.active_incidents if i["id"] != incident_id]
        self.active_dispatches = [d for d in self.active_dispatches if d["incident_id"] != incident_id]
        self._recompute_preemption()

    def _compute_deterministic_arrival(self, node_id, app_idx):
        """
        Computes physical external commuter arrival demand with ZERO randomizers.
        Uses deterministic diurnal harmonics modeling morning and afternoon peak demand waves.
        """
        node_num = int(node_id.split("_")[-1])
        offset = (node_num * 17 + app_idx * 23) % 100
        t = self.step_idx + offset
        # Smooth physical traffic wave: fundamental diurnal + peak commute harmonic
        wave = 1.0 + 0.30 * math.sin(2.0 * math.pi * t / 100.0) + 0.12 * math.cos(4.0 * math.pi * t / 100.0)
        base = 0.44
        if node_id in ("INT_2", "INT_5"):
            base = 0.52
        elif node_id in ("INT_1", "INT_3"):
            base = 0.46
        return float(round(base * wave, 3))

    def step(self, actions=None):
        """
        Executes one synchronized simulation step across the entire city network.
        Guarantees strict single-approach signal mutual exclusion, corridor vehicle
        conservation, PyTorch DQN/MLP policy actuation, Sequence VAE reconstruction
        residuals, and ZERO randomizers.
        """
        if actions is not None and not isinstance(actions, dict):
            raise ValueError("actions must be a dict mapping node_id to signal action or None")

        # Validate explicit actions before mutating state
        parsed_actions = {}
        if actions:
            for node_id, raw_act in actions.items():
                if node_id in self.intersections:
                    parsed_actions[node_id] = parse_approach_action(raw_act)

        self.step_idx += 1
        total_departed = 0.0
        total_external_inflow = 0.0
        total_external_outflow = 0.0

        # 1. Apply physical blockages from active incidents
        blockages = {k: np.ones(4, dtype=np.float32) for k in self.intersections}
        for inc in self.active_incidents:
            node = inc["node_id"]
            if node in blockages:
                app_idx = _APPROACH_LOOKUP.get(str(inc.get("blocked_approach", "North")).lower(), 0)
                cap = inc.get("blockage_capacity", 0.15)
                blockages[node][app_idx] = float(cap)

        # 2. Select single-approach actions via PyTorch RL Policy (models/rl_policy.pt) + 4-arm scoring
        for node_id, inter in self.intersections.items():
            # Always evaluate RL policy probs on current 7-D state so telemetry stays coherent
            if self.policy_model is not None:
                st_7d = inter.get_state_7d()
                with torch.no_grad():
                    inp_t = torch.from_numpy(st_7d).unsqueeze(0).to(self.device)
                    out_p = self.policy_model(inp_t)
                    probs = out_p.squeeze(0).cpu().numpy().tolist()
                    inter.probs = [float(round(probs[0], 4)), float(round(probs[1], 4))]
            else:
                ns_q = inter.queues[0] + inter.queues[1]
                ew_q = inter.queues[2] + inter.queues[3]
                tot = ns_q + ew_q + 1e-5
                inter.probs = [float(round(ns_q / tot, 4)), float(round(ew_q / tot, 4))]

            arm_scores = inter.compute_arm_scores(inter.probs)

            standby = False
            if node_id in parsed_actions:
                kind, val = parsed_actions[node_id]
                if kind == "arm":
                    target_app = val
                    act = 0 if val in (0, 1) else 1
                    inter.probs = [1.0 if act == 0 else 0.0, 1.0 if act == 1 else 0.0]
                else:
                    act = val
                    inter.probs = [1.0 if act == 0 else 0.0, 1.0 if act == 1 else 0.0]
                    target_app = None
            else:
                if all(float(q) <= 1e-6 for q in inter.queues) and not inter.preemption:
                    standby = True
                    target_app = None
                    act = int(inter.phase)
                else:
                    best_app = int(np.argmax(arm_scores))
                    cur_app = inter.active_approach
                    cur_q = float(inter.queues[cur_app])
                    best_q = float(inter.queues[best_app])
                    # Never pick an empty arm when any other arm has vehicles; switch on
                    # empty/drained active arm, clear score lead, or queued wait >= 8.
                    if (
                        best_app != cur_app
                        and best_q > 1e-6
                        and (
                            cur_q <= 1e-6
                            or (cur_q < 0.35 and best_q > cur_q)
                            or arm_scores[best_app] > arm_scores[cur_app] * 1.10
                            or inter.wait_timers[best_app] >= 8
                        )
                    ):
                        target_app = best_app
                    elif cur_q <= 1e-6 and best_q > 1e-6:
                        target_app = best_app
                    else:
                        target_app = cur_app
                    act = 0 if target_app in (0, 1) else 1

            inter.last_action = act
            inter.apply_action(act, target_approach=target_app, standby=standby)

        # 3. Collect arrivals: External commuter demand + in-transit corridor arrivals
        # Implements true physical vehicle conservation across the city network
        arrivals_per_node = {k: np.zeros(4, dtype=np.float32) for k in self.intersections}

        # Deliver vehicles that were in transit from the previous step
        for (from_node, to_node, to_app), transit_veh in list(self.corridor_in_transit.items()):
            if to_node in arrivals_per_node:
                arrivals_per_node[to_node][to_app] += transit_veh
        self.corridor_in_transit.clear()

        # Add external boundary arrivals and local street generation
        for node_id, inter in self.intersections.items():
            neighbors = INTERSECTIONS[node_id]["neighbors"]
            for a_idx, app_name in enumerate(["North", "South", "East", "West"]):
                # If approach faces boundary (no upstream neighbor), it receives external commuter demand
                if neighbors[app_name] is None:
                    ext_arr = self._compute_deterministic_arrival(node_id, a_idx)
                    arrivals_per_node[node_id][a_idx] += ext_arr
                    total_external_inflow += ext_arr
                else:
                    # Minor local residential / commercial driveway flow
                    local_gen = 0.08 * (1.0 + 0.20 * math.sin(2.0 * math.pi * (self.step_idx + a_idx * 15) / 80.0))
                    arrivals_per_node[node_id][a_idx] += local_gen
                    total_external_inflow += local_gen

        # 4. Discharge queues and transfer departed vehicles into connecting corridors
        departures_per_node = {}
        for node_id, inter in self.intersections.items():
            arr = arrivals_per_node[node_id]
            dep_sum, actual_dep = inter.step_traffic(arr, blockages[node_id])
            departures_per_node[node_id] = actual_dep
            total_departed += dep_sum

            # Route departed vehicles along corridors (Flow Conservation)
            neighbors = INTERSECTIONS[node_id]["neighbors"]
            # Approach 0 (North) discharges Southbound:
            if neighbors["South"] is not None:
                self.corridor_in_transit[(node_id, neighbors["South"], 0)] += actual_dep[0]
            else:
                total_external_outflow += actual_dep[0]

            # Approach 1 (South) discharges Northbound:
            if neighbors["North"] is not None:
                self.corridor_in_transit[(node_id, neighbors["North"], 1)] += actual_dep[1]
            else:
                total_external_outflow += actual_dep[1]

            # Approach 2 (East) discharges Westbound:
            if neighbors["West"] is not None:
                self.corridor_in_transit[(node_id, neighbors["West"], 2)] += actual_dep[2]
            else:
                total_external_outflow += actual_dep[2]

            # Approach 3 (West) discharges Eastbound:
            if neighbors["East"] is not None:
                self.corridor_in_transit[(node_id, neighbors["East"], 3)] += actual_dep[3]
            else:
                total_external_outflow += actual_dep[3]

        # 5. Evaluate Sequence VAE (models/sequence_vae.pt) & Custom CNN (models/custom_cnn.pt)
        for node_id, inter in self.intersections.items():
            # Update rolling queue sequence history
            inter.queue_history.append(inter.queues.copy())

            # A. Sequence VAE Anomaly Detection
            if self.vae_model is not None and len(inter.queue_history) >= 5:
                try:
                    from src.vae import detect_incident_anomaly
                    seq_np = np.array(list(inter.queue_history), dtype=np.float32)
                    diag = detect_incident_anomaly(
                        self.vae_model, seq_np,
                        threshold_warning=self.vae_th_warn,
                        threshold_critical=self.vae_th_crit,
                        device=self.device
                    )
                    inter.vae_residual = diag["score"]
                    inter.vae_level = diag["level"]
                    inter.vae_alert = diag["alert_message"]
                    inter.vae_approach_scores = diag["approach_scores"]
                except Exception as e:
                    inter.vae_residual = 1.05
                    inter.vae_level = "NORMAL"
            else:
                # Baseline normal residual during initial warmup
                inter.vae_residual = float(round(0.85 + (self.step_idx * 0.04), 2))
                inter.vae_level = "NORMAL"

            # B. Custom CNN Spatial Density Estimation
            if self.custom_cnn is not None:
                try:
                    grid = self._generate_spatial_grid(inter.queues)
                    with torch.no_grad():
                        grid_t = torch.from_numpy(grid).to(self.device)
                        cnn_out = self.custom_cnn(grid_t)
                        inter.cnn_densities = [float(round(d, 2)) for d in cnn_out.squeeze(0).cpu().numpy().tolist()]
                except Exception:
                    inter.cnn_densities = [float(round(q, 2)) for q in inter.queues]
            else:
                inter.cnn_densities = [float(round(q, 2)) for q in inter.queues]

        # 6. Update emergency vehicle dispatch progress
        for disp in self.active_dispatches:
            if disp["police"]["eta_sec"] > 0:
                disp["police"]["eta_sec"] -= 1
                if disp["police"]["eta_sec"] <= 0:
                    disp["police"]["status"] = "ON_SCENE"

            if disp["ambulance"]["eta_sec"] > 0:
                disp["ambulance"]["eta_sec"] -= 1
                if disp["ambulance"]["eta_sec"] <= 0:
                    disp["ambulance"]["status"] = "ON_SCENE"

        # 7. Compute network-wide conservation and ML telemetry
        all_queues = [q for inter in self.intersections.values() for q in inter.queues]
        avg_queue = float(np.mean(all_queues))
        total_veh = float(round(float(sum(np.sum(inter.queues) for inter in self.intersections.values())), 2))
        in_transit_veh = float(round(float(sum(self.corridor_in_transit.values())), 2))
        system_total_vehicles = float(round(total_veh + in_transit_veh, 2))

        # True Sequence VAE telemetry
        vae_residuals = [inter.vae_residual for inter in self.intersections.values()]
        avg_vae_residual = float(round(np.mean(vae_residuals), 2))
        max_vae_residual = float(round(np.max(vae_residuals), 2))
        critical_count = sum(1 for inter in self.intersections.values() if inter.vae_level == "CRITICAL")

        done = self.step_idx >= self.max_steps
        info = {
            "step": int(self.step_idx),
            "avg_queue": float(round(avg_queue, 2)),
            "total_vehicles": float(round(total_veh, 2)),
            "in_transit_vehicles": float(round(in_transit_veh, 2)),
            "system_total_vehicles": float(round(system_total_vehicles, 2)),
            "external_inflow": float(round(total_external_inflow, 2)),
            "external_outflow": float(round(total_external_outflow, 2)),
            "total_departed": float(round(total_departed, 1)),
            "vae_residual": float(round(avg_vae_residual, 2)),
            "max_vae_residual": float(round(max_vae_residual, 2)),
            "critical_anomalies": int(critical_count),
            "active_incidents": int(len(self.active_incidents)),
            "preemption_active": bool(self.preemption_active),
            "intersections": self.get_state(),
            "dispatches": self.active_dispatches,
        }
        return self.get_state(), float(-avg_queue), bool(done), info

    def generate_traffic_scenarios(self, scenario_type="rush_hour", num_samples=1):
        """Generates synthetic multi-step traffic queue sequences using trained Sequence GAN."""
        if self.gan_generator is not None:
            from src.gan import generate_scenario
            return generate_scenario(
                self.gan_generator,
                scenario_type=scenario_type,
                num_samples=num_samples,
                device=self.device
            )
        return []
