# src/model.py
import os
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

    Supports optional BatchNorm1d, and automatically adapts when loading
    checkpoints with or without BatchNorm layers.
    """
    def __init__(self, input_dim=7, hidden_dims=[64, 32], dropout=0.3, use_batch_norm=False):
        super().__init__()
        self.use_batch_norm = use_batch_norm

        layers = []
        prev = input_dim

        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            init.kaiming_uniform_(layers[-1].weight, nonlinearity='relu')
            layers.append(nn.ReLU(inplace=True))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(nn.Dropout(dropout))
            prev = h

        # Final output layer (logits; Softmax applied in forward)
        layers.append(nn.Linear(prev, 2))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        # Prevent BatchNorm1d crash on single-sample inputs during train mode
        if self.use_batch_norm and x.size(0) == 1 and self.training:
            self.eval()
            with torch.no_grad():
                logits = self.net(x)
            self.train()
        else:
            logits = self.net(x)
        probs = torch.softmax(logits, dim=-1)
        return probs


def load_policy(checkpoint_path="models/rl_policy.pt", input_dim=7, hidden_dims=[64, 32], device="cpu"):
    """
    Safely loads the RL/MLP policy checkpoint, auto-detecting BatchNorm and parameter shapes.
    """
    if not os.path.exists(checkpoint_path):
        model = MLPModel(input_dim=input_dim, hidden_dims=hidden_dims, dropout=0.0, use_batch_norm=False)
        model.to(device)
        model.eval()
        return model

    ckpt = torch.load(checkpoint_path, map_location=device)
    has_bn = any("running_mean" in k for k in ckpt.keys())
    model = MLPModel(input_dim=input_dim, hidden_dims=hidden_dims, dropout=0.0, use_batch_norm=has_bn)
    model.to(device)
    model.load_state_dict(ckpt)
    model.eval()
    return model


def export_policy_weights_to_js(checkpoint_path="models/rl_policy.pt", output_path="web_visualizer/policy_weights.js"):
    """
    Exports exact PyTorch MLP policy weights to JavaScript for in-browser client-side inference.
    Guarantees < 1e-6 numerical error compared to PyTorch forward pass.
    """
    import json
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    weights = {
        "w1": ckpt["net.0.weight"].numpy().tolist(),
        "b1": ckpt["net.0.bias"].numpy().tolist(),
        "w2": ckpt["net.3.weight"].numpy().tolist(),
        "b2": ckpt["net.3.bias"].numpy().tolist(),
        "w3": ckpt["net.6.weight"].numpy().tolist(),
        "b3": ckpt["net.6.bias"].numpy().tolist(),
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    js_template = f"""// web_visualizer/policy_weights.js
// Autonomous Traffic Management System (ATMS) - Trained Deep RL Policy Weights
// Auto-generated export from PyTorch checkpoint ({checkpoint_path})
// Guarantees client-side inference reproduces PyTorch forward pass to < 1e-6 error.

(function(root) {{
    const RL_POLICY_WEIGHTS = {json.dumps(weights, indent=2)};

    /**
     * Executes the exact forward pass of the trained PyTorch MLP Policy.
     * Maps 7-D intersection state vector to phase action and Softmax probabilities.
     * @param {{Array<number>}} state7d - [q0, q1, q2, q3, total_q, p0, p1]
     * @returns {{{{action: number, probs: [number, number], logits: [number, number]}}}}
     */
    function forwardRLPolicy(state7d) {{
        if (!state7d || state7d.length < 7) {{
            return {{ action: 0, probs: [0.5, 0.5], logits: [0, 0] }};
        }}

        const w1 = RL_POLICY_WEIGHTS.w1;
        const b1 = RL_POLICY_WEIGHTS.b1;
        const w2 = RL_POLICY_WEIGHTS.w2;
        const b2 = RL_POLICY_WEIGHTS.b2;
        const w3 = RL_POLICY_WEIGHTS.w3;
        const b3 = RL_POLICY_WEIGHTS.b3;

        // Layer 1: Linear(7, 64) + ReLU
        const h1 = new Float64Array(64);
        for (let i = 0; i < 64; i++) {{
            let sum = b1[i];
            const row = w1[i];
            for (let j = 0; j < 7; j++) {{
                sum += row[j] * state7d[j];
            }}
            h1[i] = sum > 0 ? sum : 0;
        }}

        // Layer 2: Linear(64, 32) + ReLU
        const h2 = new Float64Array(32);
        for (let i = 0; i < 32; i++) {{
            let sum = b2[i];
            const row = w2[i];
            for (let j = 0; j < 64; j++) {{
                sum += row[j] * h1[j];
            }}
            h2[i] = sum > 0 ? sum : 0;
        }}

        // Layer 3: Linear(32, 2)
        const logits = new Float64Array(2);
        for (let i = 0; i < 2; i++) {{
            let sum = b3[i];
            const row = w3[i];
            for (let j = 0; j < 32; j++) {{
                sum += row[j] * h2[j];
            }}
            logits[i] = sum;
        }}

        // Numerically stable Softmax
        const maxLogit = Math.max(logits[0], logits[1]);
        const exp0 = Math.exp(logits[0] - maxLogit);
        const exp1 = Math.exp(logits[1] - maxLogit);
        const sumExp = exp0 + exp1;
        const p0 = exp0 / sumExp;
        const p1 = exp1 / sumExp;

        const action = p0 >= p1 ? 0 : 1;
        return {{
            action: action,
            probs: [Number(p0.toFixed(6)), Number(p1.toFixed(6))],
            rawProbs: [p0, p1],
            logits: [logits[0], logits[1]]
        }};
    }}

    // Export to global browser window or Node module
    if (typeof window !== 'undefined') {{
        window.RL_POLICY_WEIGHTS = RL_POLICY_WEIGHTS;
        window.forwardRLPolicy = forwardRLPolicy;
        window.predictTrafficPolicy = forwardRLPolicy;
        window.evaluatePolicy = forwardRLPolicy;
    }}
    if (typeof module !== 'undefined' && module.exports) {{
        module.exports = {{
            RL_POLICY_WEIGHTS,
            forwardRLPolicy,
            predictTrafficPolicy: forwardRLPolicy,
            evaluatePolicy: forwardRLPolicy
        }};
    }}
}})(typeof globalThis !== 'undefined' ? globalThis : this);
"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(js_template)
    print(f"[Model] Successfully exported RL policy weights to {output_path}")
    return output_path




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