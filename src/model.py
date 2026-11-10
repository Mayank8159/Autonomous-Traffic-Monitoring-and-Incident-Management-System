# src/model.py
import torch
import torch.nn as nn
import torch.nn.init as init


# -------------------------------------------------
# Multi‑Layer Perceptron (MLP)
# -------------------------------------------------
class MLPModel(nn.Module):
    """
    Fully‑connected network that maps the current intersection state
    (7 dimensions) to a probability distribution over the two possible
    traffic‑light phases.

    Architecture (all hidden layers use ReLU + He init + BatchNorm + Dropout):
        Input (7) → Linear(64) → ReLU → BatchNorm1d → Dropout → Linear(32) → ReLU → BatchNorm1d → Dropout → Linear(2) → Softmax
    """
    def __init__(self, input_dim=7, hidden_dims=[64, 32], dropout=0.3):
        super().__init__()

        layers = []
        prev = input_dim

        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            init.kaiming_uniform_(layers[-1].weight, nonlinearity='relu')
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.Dropout(dropout))
            prev = h

        # Final output layer (logits; Softmax applied in forward)
        layers.append(nn.Linear(prev, 2))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        logits = self.net(x)
        probs = torch.softmax(logits, dim=-1)
        return probs


# -------------------------------------------------
# LSTM / GRU sequence model
# -------------------------------------------------
class RNNModel(nn.Module):
    """
    Recurrent network that ingests a *sequence* of states
    (e.g., last K time steps) and outputs a phase distribution.

    The cell type is selected by the ``cell`` argument ('lstm' or 'gru').
    This design captures temporal congestion patterns and alleviates
    the vanishing‑gradient problem when traffic patterns are long‑range.

    Input shape (batch, seq_len, input_dim) → RNN cell → Linear(2) → Softmax
    """
    def __init__(self, input_dim=7, hidden_dim=64, num_layers=1,
                 dropout=0.3, seq_len=5, cell='lstm'):
        super().__init__()
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.cell = cell.lower()

        if self.cell == 'lstm':
            self.rnn = nn.LSTM(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0.0,
                batch_first=True,
            )
        elif self.cell == 'gru':
            self.rnn = nn.GRU(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0.0,
                batch_first=True,
            )
        else:
            raise ValueError("cell must be 'lstm' or 'gru'")

        # Read‑out layer
        self.fc = nn.Linear(hidden_dim, 2)
        init.kaiming_uniform_(self.fc.weight, nonlinearity='relu')

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        # We only need the last time‑step's hidden state for the decision.
        rnn_out, _ = self.rnn(x)  # rnn_out: (batch, seq_len, hidden)
        last_timestep = rnn_out[:, -1, :]  # (batch, hidden)
        logits = self.fc(last_timestep)      # (batch, 2)
        probs = torch.softmax(logits, dim=-1)
        return probs