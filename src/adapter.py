# src/adapter.py
"""
Traffic State Adapter for ATMS.
Bridges Computer Vision & real-video detections with the standard 7-dimensional
traffic state vector expected by MLP, LSTM/GRU, REINFORCE, and DQN models:
    s_t = [q_North, q_South, q_East, q_West, N_total, phase_NS_onehot, phase_EW_onehot]
Preserves strict backward compatibility without altering existing decision model interfaces.
"""

import numpy as np
import torch


class VisionToStateAdapter:
    """
    Converts real-video computer vision detections (per-lane vehicle counts,
    bounding boxes, and spatial occupancy grids) into the standard 7-D state format.
    """
    def __init__(self, alpha=0.35, max_queue=30.0):
        self.alpha = alpha          # Smoothing factor for EMA
        self.max_queue = max_queue  # Maximum queue ceiling
        self.smoothed_queues = np.zeros(4, dtype=np.float32)
        self.initialized = False

    def reset(self, initial_queues=None):
        """Reset the internal queue smoothing filter."""
        if initial_queues is not None:
            self.smoothed_queues = np.clip(np.array(initial_queues, dtype=np.float32), 0.0, self.max_queue)
        else:
            self.smoothed_queues = np.zeros(4, dtype=np.float32)
        self.initialized = True

    def counts_to_state(self, lane_counts, current_phase=0):
        """
        Converts lane counts [North, South, East, West] into the standard 7-D vector:
        s_t = [q0, q1, q2, q3, total_vehicles, phase_NS, phase_EW]
        """
        raw_counts = np.array(lane_counts, dtype=np.float32)

        if not self.initialized:
            self.smoothed_queues = np.clip(raw_counts, 0.0, self.max_queue)
            self.initialized = True
        else:
            # Exponential Moving Average smoothing against detection flicker
            self.smoothed_queues = self.alpha * raw_counts + (1.0 - self.alpha) * self.smoothed_queues
            self.smoothed_queues = np.clip(self.smoothed_queues, 0.0, self.max_queue)

        total_vehicles = float(np.sum(self.smoothed_queues))

        # One-hot encoded current phase
        phase_oh = np.zeros(2, dtype=np.float32)
        phase_idx = int(current_phase) % 2
        phase_oh[phase_idx] = 1.0

        state = np.concatenate([self.smoothed_queues, [total_vehicles], phase_oh]).astype(np.float32)
        return state

    def cnn_to_state(self, spatial_grid, cnn_model, current_phase=0, device="cpu"):
        """
        Runs CustomTrafficCNN inference on an occupancy grid tensor (1, 3, H, W)
        and converts predicted densities into the standard 7-D vector.
        """
        cnn_model.eval()
        with torch.no_grad():
            if not isinstance(spatial_grid, torch.Tensor):
                spatial_grid = torch.tensor(spatial_grid, dtype=torch.float32)
            if spatial_grid.ndim == 3:
                spatial_grid = spatial_grid.unsqueeze(0)
            spatial_grid = spatial_grid.to(device)

            predicted_densities = cnn_model(spatial_grid).squeeze(0).cpu().numpy()

        return self.counts_to_state(predicted_densities, current_phase=current_phase)
