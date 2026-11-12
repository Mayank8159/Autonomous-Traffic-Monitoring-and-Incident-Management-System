# src/env.py
import numpy as np
import warnings

try:
    import traci
    HAS_SUMO = True
except Exception:  # pragma: no cover
    HAS_SUMO = False


class TrafficEnv:
    """
    Wrapper that can run either a lightweight grid‑based simulation
    or a real SUMO/TraCI simulation.

    State at each time step t contains:
        – queue length per lane (4 lanes: N‑in, N‑out, E‑in, E‑out)
        – total vehicle count in the intersection area
        – one‑hot encoded current signal phase (NS_green, EW_green)
    Action space:
        – 0 = give Green to North‑South traffic
        – 1 = give Green to East‑West traffic
    """

    def __init__(self, green_duration=5, red_duration=5, max_queue=30.0,
                 sumo=False, max_steps=200, min_green=3):
        self.green_duration = green_duration
        self.min_green = min_green if min_green is not None else 3
        self.red_duration = red_duration
        self.max_queue = max_queue
        self.sumo = sumo
        self.max_steps = max_steps

        if sumo and not HAS_SUMO:
            warnings.warn(
                "SUMO is not available; using the lightweight traffic simulation.",
                RuntimeWarning,
            )
            self.sumo = False

        self.reset()

    # -----------------------------------------------------------------
    # Light‑weight mode (identical to the previous implementation)
    # -----------------------------------------------------------------
    def reset_light(self):
        """Reset the internal grid state."""
        self.queue = np.random.uniform(0, 5, size=4).astype(np.float32)
        self.total_vehicles = float(np.clip(np.sum(self.queue) + np.random.randint(0, 5), 0, 200))
        self.phase = 0                # 0 → NS green, 1 → EW green
        self.phase_counter = 0
        self.step_cnt = 0
        return self._get_state()

    # -----------------------------------------------------------------
    # SUMO mode – skeleton; the real TraCI calls are left as placeholders
    # -----------------------------------------------------------------
    def reset_sumo(self):
        """Reset a SUMO simulation. Meant to be overridden by a concrete subclass."""
        raise NotImplementedError("Subclass must implement this method for SUMO mode.")

    # -----------------------------------------------------------------
    # Common state extraction
    # -----------------------------------------------------------------
    def _get_state(self):
        """Return a flat numpy array – the exact format the model expects."""
        phase_oh = np.zeros(2, dtype=np.float32)
        phase_oh[self.phase] = 1.0
        state = np.concatenate([self.queue, [self.total_vehicles], phase_oh])
        return state.copy()

    # -----------------------------------------------------------------
    # Step – lightweight implementation with signal switching & min_green
    # -----------------------------------------------------------------
    def step(self, action):
        """
        Execute one action (0 = NS green, 1 = EW green).
        Returns (next_state, reward, done, info).
        """
        action = int(action)
        # Phase switching logic with configurable minimum green duration:
        if self.phase_counter == 0:
            self.phase = action
            self.phase_counter = 1
        elif action != self.phase:
            if self.phase_counter >= self.min_green:
                self.phase = action
                self.phase_counter = 1
            else:
                self.phase_counter += 1
        else:
            self.phase_counter += 1

        # Inflow arrivals across the 4 approaches (North, South, East, West)
        arrival = np.random.uniform(0, 0.8, size=4).astype(np.float32)

        # Outflow departures (vehicles depart only when their signal is green)
        # 4 approaches: index 0 = North, index 1 = South, index 2 = East, index 3 = West
        discharge_rate = 0.8
        if self.phase == 0:          # North-South green, East-West red
            depart = np.array([discharge_rate, discharge_rate, 0.0, 0.0], dtype=np.float32)
        else:                       # East-West green, North-South red
            depart = np.array([0.0, 0.0, discharge_rate, discharge_rate], dtype=np.float32)

        actual_depart = np.minimum(self.queue, depart)
        self.queue = np.clip(self.queue + arrival - depart, 0.0, self.max_queue)
        departed = float(actual_depart.sum())
        self.total_vehicles = float(np.clip(np.sum(self.queue) + np.random.randint(0, 5), 0, 200))

        # Reward: minimize average queue, with slight penalty on max queue to prevent starvation
        avg_queue = float(np.mean(self.queue))
        max_q = float(np.max(self.queue))
        reward = -avg_queue - 0.05 * max_q

        self.step_cnt += 1
        done = self.step_cnt >= self.max_steps

        next_state = self._get_state()
        info = {
            "queues": self.queue.copy(),
            "phase": self.phase,
            "departed": departed,
            "avg_queue": avg_queue,
        }
        return next_state, reward, done, info

    # -----------------------------------------------------------------
    # Public reset – chooses the appropriate mode
    # -----------------------------------------------------------------
    def reset(self):
        """Reset the environment. sumo=True expects a subclass that implements reset_sumo."""
        if self.sumo:
            return self.reset_sumo()
        return self.reset_light()

    # -----------------------------------------------------------------
    # Convenience method for SUMO‑based subclasses to step via TraCI
    # -----------------------------------------------------------------
    def step_sumo(self, action):
        """
        Step a SUMO simulation using TraCI.
        Returns (next_state, reward, done, info).
        Meant to be overridden by a concrete SUMO subclass.
        """
        raise NotImplementedError("Subclass must implement this method for SUMO mode.")


def get_city_network_env(max_steps=500):
    """Instantiate the 6-crossing multi-intersection city network environment."""
    from src.city_network import CityNetworkEnv
    return CityNetworkEnv(max_steps=max_steps)
