# src/gan.py
"""
Module 5A: Sequence-Based Generative Adversarial Network (GAN) for Traffic Scenario Generation.
Generates realistic multi-step traffic queue sequences (T steps x 4 approaches) across 6 distinct conditions:
- Normal Traffic
- Moderate Traffic
- Heavy Traffic
- Rush-Hour Congestion
- Uneven Traffic Distribution
- Sudden Traffic Surges
Includes a physical validation quality gate to verify generated sequences before data augmentation.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


SCENARIO_TYPES = [
    "normal",
    "moderate",
    "heavy",
    "rush_hour",
    "uneven",
    "surge",
]
NUM_CONDITIONS = len(SCENARIO_TYPES)


class SequenceGenerator(nn.Module):
    """
    Conditional Recurrent Generator producing multi-step queue sequences
    Shape: (batch, seq_len, 4 approaches)
    """
    def __init__(self, latent_dim=32, num_conditions=NUM_CONDITIONS, hidden_dim=128, seq_len=10):
        super().__init__()
        self.seq_len = seq_len
        self.latent_dim = latent_dim
        self.num_conditions = num_conditions

        self.fc_in = nn.Sequential(
            nn.Linear(latent_dim + num_conditions, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.rnn = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.1
        )

        self.fc_out = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64, 4),
            nn.Sigmoid()  # Bound outputs within physical queue capacity
        )

    def forward(self, z, condition_idx):
        # z: (batch, latent_dim)
        batch_size = z.size(0)
        # One-hot encode condition
        c_onehot = torch.zeros(batch_size, self.num_conditions, device=z.device)
        c_onehot.scatter_(1, condition_idx.unsqueeze(1), 1.0)

        zc = torch.cat([z, c_onehot], dim=1)
        hidden_init = self.fc_in(zc)

        # Expand across sequence length
        rnn_in = hidden_init.unsqueeze(1).repeat(1, self.seq_len, 1)
        rnn_out, _ = self.rnn(rnn_in)
        sequences = self.fc_out(rnn_out) * 30.0
        return sequences


class SequenceDiscriminator(nn.Module):
    """
    Conditional Recurrent Discriminator classifying real vs synthetic sequences.
    """
    def __init__(self, num_conditions=NUM_CONDITIONS, hidden_dim=64):
        super().__init__()
        self.num_conditions = num_conditions

        # Input: 4 queue values + num_conditions one-hot
        self.rnn = nn.GRU(
            input_size=4 + num_conditions,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.1
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, seq, condition_idx):
        # seq: (batch, seq_len, 4)
        batch_size, seq_len, _ = seq.size()
        c_onehot = torch.zeros(batch_size, self.num_conditions, device=seq.device)
        c_onehot.scatter_(1, condition_idx.unsqueeze(1), 1.0)
        c_expanded = c_onehot.unsqueeze(1).repeat(1, seq_len, 1)

        x = torch.cat([seq, c_expanded], dim=2)
        rnn_out, _ = self.rnn(x)
        # Global max-pool across sequence time dimension
        pooled = torch.max(rnn_out, dim=1)[0]
        validity = self.classifier(pooled)
        return validity


def generate_real_traffic_sequences(num_samples=1000, seq_len=10):
    """
    Generates realistic ground-truth training sequences adhering to physical traffic dynamics.
    Returns: sequences (num_samples, seq_len, 4), labels (num_samples,)
    """
    sequences = []
    condition_labels = []

    for _ in range(num_samples):
        cond = np.random.randint(0, NUM_CONDITIONS)
        cond_name = SCENARIO_TYPES[cond]
        seq = np.zeros((seq_len, 4), dtype=np.float32)

        # Baseline queue configuration according to scenario
        if cond_name == "normal":
            curr = np.random.uniform(1.0, 5.0, size=4)
            arrival_mean = 0.5
        elif cond_name == "moderate":
            curr = np.random.uniform(4.0, 10.0, size=4)
            arrival_mean = 0.7
        elif cond_name == "heavy":
            curr = np.random.uniform(10.0, 20.0, size=4)
            arrival_mean = 0.9
        elif cond_name == "rush_hour":
            curr = np.random.uniform(18.0, 26.0, size=4)
            arrival_mean = 1.1
        elif cond_name == "uneven":
            # NS approaches heavily congested, EW free
            curr = np.array([np.random.uniform(15, 25), np.random.uniform(15, 25),
                             np.random.uniform(1, 4), np.random.uniform(1, 4)])
            arrival_mean = 0.6
        else:  # surge
            curr = np.random.uniform(2.0, 6.0, size=4)
            arrival_mean = 0.5

        for t in range(seq_len):
            if cond_name == "surge" and t >= seq_len // 2:
                # Sudden surge injection
                surge_lane = np.random.randint(0, 4)
                curr[surge_lane] += np.random.uniform(5.0, 8.0)

            # Evolve with realistic arrival / discharge
            arrivals = np.random.uniform(0.0, arrival_mean * 1.5, size=4)
            # Simulated alternating discharge
            departures = np.zeros(4)
            phase = (t // 3) % 2
            if phase == 0:
                departures[:2] = 0.8
            else:
                departures[2:] = 0.8

            curr = np.clip(curr + arrivals - departures, 0.0, 30.0)
            seq[t] = curr.copy()

        sequences.append(seq)
        condition_labels.append(cond)

    return np.array(sequences, dtype=np.float32), np.array(condition_labels, dtype=np.int64)


def validate_traffic_sequence(sequence, condition_idx=0, max_delta=4.5):
    """
    Physical Validation Quality Gate:
    Validates synthetic sequences before allowing them into data augmentation buffers.
    Checks:
    1. Bounds: 0 <= queue <= 30.0
    2. Physical continuity: |q_{t+1} - q_t| <= max_delta (no teleporting cars)
    3. Scenario characteristic match
    """
    arr = np.asarray(sequence)
    if arr.ndim != 2 or arr.shape[1] != 4:
        return False, "Invalid tensor shape (expected (seq_len, 4))"

    # Bounds check
    if np.any(arr < 0.0) or np.any(arr > 30.0):
        return False, f"Out of bounds: min={arr.min():.2f}, max={arr.max():.2f}"

    # Temporal continuity check
    diffs = np.abs(np.diff(arr, axis=0))
    if np.any(diffs > max_delta):
        return False, f"Physical discontinuity violation: max delta {diffs.max():.2f} > {max_delta}"

    # Scenario check
    cond_name = SCENARIO_TYPES[condition_idx]
    mean_q = np.mean(arr)
    if cond_name == "rush_hour" and mean_q < 8.0:
        return False, f"Rush hour scenario under-congested: mean queue={mean_q:.2f}"
    if cond_name == "normal" and mean_q > 12.0:
        return False, f"Normal scenario over-congested: mean queue={mean_q:.2f}"

    return True, "Validated: physically realistic traffic trajectory"


def train_traffic_gan(epochs=30, batch_size=32, lr=2e-4, seq_len=10, device="cpu", save_path="models/sequence_gan.pt"):
    """
    Trains the Sequence GAN with Wasserstein/BCE minimax objective.
    Saves generator checkpoint to save_path.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    real_seqs, real_labels = generate_real_traffic_sequences(num_samples=1200, seq_len=seq_len)

    real_seqs_t = torch.tensor(real_seqs).to(device)
    real_labels_t = torch.tensor(real_labels).to(device)

    generator = SequenceGenerator(latent_dim=32, hidden_dim=128, seq_len=seq_len).to(device)
    discriminator = SequenceDiscriminator(hidden_dim=64).to(device)

    opt_g = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_d = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
    criterion = nn.BCELoss()

    num_batches = len(real_seqs) // batch_size
    print(f"Training Sequence GAN for {epochs} epochs on device: {device}...")

    for epoch in range(1, epochs + 1):
        perm = torch.randperm(len(real_seqs))
        d_loss_epoch = 0.0
        g_loss_epoch = 0.0

        for b in range(num_batches):
            idx = perm[b * batch_size:(b + 1) * batch_size]
            b_real_seqs = real_seqs_t[idx]
            b_cond = real_labels_t[idx]

            curr_batch_size = len(b_real_seqs)
            real_targets = torch.ones(curr_batch_size, 1, device=device) * 0.9  # Label smoothing
            fake_targets = torch.zeros(curr_batch_size, 1, device=device)

            # ---------------- Train Discriminator ----------------
            opt_d.zero_grad()
            d_real = discriminator(b_real_seqs, b_cond)
            loss_d_real = criterion(d_real, real_targets)

            z = torch.randn(curr_batch_size, 32, device=device)
            fake_seqs = generator(z, b_cond)
            d_fake = discriminator(fake_seqs.detach(), b_cond)
            loss_d_fake = criterion(d_fake, fake_targets)

            loss_d = loss_d_real + loss_d_fake
            loss_d.backward()
            opt_d.step()

            # ---------------- Train Generator ----------------
            opt_g.zero_grad()
            d_fake_for_g = discriminator(fake_seqs, b_cond)
            loss_g = criterion(d_fake_for_g, real_targets)
            loss_g.backward()
            opt_g.step()

            d_loss_epoch += loss_d.item()
            g_loss_epoch += loss_g.item()

        d_loss_epoch /= num_batches
        g_loss_epoch /= num_batches

        if epoch % 5 == 0 or epoch == epochs:
            print(f"Epoch {epoch:02d}/{epochs:02d} | D Loss: {d_loss_epoch:.4f} | G Loss: {g_loss_epoch:.4f}")

    torch.save({
        "generator": generator.state_dict(),
        "discriminator": discriminator.state_dict(),
    }, save_path)
    print(f"Sequence GAN trained and saved to {save_path}")
    return generator, discriminator


def load_traffic_gan(checkpoint_path="models/sequence_gan.pt", seq_len=10, device="cpu"):
    """Loads a trained generator for inference and scenario generation."""
    generator = SequenceGenerator(latent_dim=32, hidden_dim=128, seq_len=seq_len).to(device)
    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
        generator.load_state_dict(ckpt["generator"])
        generator.eval()
    return generator


def generate_scenario(generator, scenario_type="rush_hour", num_samples=3, seq_len=10, device="cpu"):
    """
    Generates and physically validates multi-step traffic queue sequences.
    Returns: list of dicts with {"sequence": (seq_len, 4), "validated": bool, "status": str}
    """
    generator.eval()
    if scenario_type not in SCENARIO_TYPES:
        scenario_type = "normal"
    cond_idx = SCENARIO_TYPES.index(scenario_type)
    cond_t = torch.tensor([cond_idx] * num_samples, dtype=torch.long, device=device)

    with torch.no_grad():
        z = torch.randn(num_samples, 32, device=device)
        fake_seqs = generator(z, cond_t).cpu().numpy()

    results = []
    for i in range(num_samples):
        seq = fake_seqs[i]
        is_valid, msg = validate_traffic_sequence(seq, condition_idx=cond_idx)
        results.append({
            "scenario": scenario_type,
            "sequence": np.round(seq, 2).tolist(),
            "validated": is_valid,
            "status": msg,
            "mean_queue": float(round(np.mean(seq), 2)),
            "max_queue": float(round(np.max(seq), 2)),
        })
    return results
