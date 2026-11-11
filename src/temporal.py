# src/temporal.py
"""
Module 4: Temporal Traffic Trend Analysis & Short-Term Queue Forecasting.
Extends LSTM & GRU models to analyse historical traffic patterns and forecast
short-term congestion trends (Rising / Falling / Stable) per approach.
"""

import torch
import torch.nn as nn
import numpy as np


class TemporalTrafficPredictor(nn.Module):
    """
    Recurrent sequence predictor implementing Module 4 concepts:
    - Long Short-Term Memory (LSTM) and Gated Recurrent Unit (GRU)
    - Ingests temporal history of queue vectors (batch, seq_len, 4)
    - Forecasts the expected queue delta and trend direction 1-3 steps ahead.
    """
    def __init__(self, input_dim=4, hidden_dim=64, num_layers=2, cell="lstm", seq_len=5):
        super().__init__()
        self.cell_type = cell.lower()
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim

        if self.cell_type == "gru":
            self.rnn = nn.GRU(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=0.1)
        else:
            self.rnn = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=0.1)

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 4)  # Predicts next step queue density for 4 approaches
        )

    def forward(self, x):
        # x: (batch, seq_len, 4)
        out, _ = self.rnn(x)
        last_step = out[:, -1, :]
        predicted_queues = self.head(last_step)
        return predicted_queues


def analyze_traffic_trends(history_sequences, predictor=None, cell="lstm"):
    """
    Analyzes historical queue buffer (seq_len x 4) and determines trend direction.
    Returns: dict with trend classifications and forecasted next-step queues.
    """
    arr = np.asarray(history_sequences, dtype=np.float32)
    if arr.ndim == 2:
        seq = arr
    else:
        seq = arr[-5:]

    approaches = ["North", "South", "East", "West"]
    trends = {}
    deltas = {}

    if len(seq) >= 2:
        # Rate of change over available history
        recent_delta = seq[-1] - seq[0]
        for i, app in enumerate(approaches):
            d = float(recent_delta[i])
            deltas[app] = round(d, 2)
            if d > 1.0:
                trends[app] = "Surging (Heavy Inflow)"
            elif d > 0.3:
                trends[app] = "Rising"
            elif d < -1.0:
                trends[app] = "Rapidly Clearing"
            elif d < -0.3:
                trends[app] = "Decreasing"
            else:
                trends[app] = "Stable"
    else:
        for app in approaches:
            trends[app] = "Stable"
            deltas[app] = 0.0

    # Model forecast if available
    forecast = seq[-1].copy()
    if predictor is not None:
        predictor.eval()
        with torch.no_grad():
            inp = torch.tensor(seq, dtype=torch.float32).unsqueeze(0)
            forecast = predictor(inp).squeeze(0).numpy()

    return {
        "trends": trends,
        "deltas": deltas,
        "forecast_queues": [float(round(q, 2)) for q in forecast],
    }
