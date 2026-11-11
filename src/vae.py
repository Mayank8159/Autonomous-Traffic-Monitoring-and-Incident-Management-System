# src/vae.py
"""
Module 5B: Sequence-Based Variational Autoencoder (VAE) for Traffic Incident & Anomaly Detection.
Includes:
- Recurrent Encoder & Latent space (mu, logvar)
- Reparameterization trick (z = mu + sigma * eps)
- Recurrent Decoder
- Combined Reconstruction MSE + KL Divergence Loss
- Validation-based anomaly threshold calibration
- Two-level alerting: Warning vs "Critical Traffic Anomaly — Investigation Required"
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class SequenceVAE(nn.Module):
    """
    Sequence Variational Autoencoder modeling normal multi-step traffic patterns.
    High reconstruction error indicates abnormal traffic patterns, sudden blockages, or incidents.
    """
    def __init__(self, input_dim=4, hidden_dim=64, latent_dim=16, seq_len=5):
        super().__init__()
        self.seq_len = seq_len
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        # Encoder: Sequence of traffic states -> latent distribution
        self.encoder_rnn = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.1
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # Decoder: Latent vector z -> reconstructed sequence
        self.decoder_fc = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.decoder_rnn = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.1
        )
        self.fc_out = nn.Sequential(
            nn.Linear(hidden_dim, input_dim),
            nn.ReLU()  # Queues are non-negative
        )

    def encode(self, x):
        # x: (batch, seq_len, input_dim)
        rnn_out, _ = self.encoder_rnn(x)
        last_hidden = rnn_out[:, -1, :]
        mu = self.fc_mu(last_hidden)
        logvar = self.fc_logvar(last_hidden)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        # z: (batch, latent_dim)
        h = self.decoder_fc(z)
        rnn_in = h.unsqueeze(1).repeat(1, self.seq_len, 1)
        rnn_out, _ = self.decoder_rnn(rnn_in)
        reconstruction = self.fc_out(rnn_out)
        return reconstruction

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decode(z)
        return reconstruction, mu, logvar


def vae_loss_function(recon_x, x, mu, logvar, beta=0.005):
    """
    Combined loss: Reconstruction MSE + beta * KL Divergence.
    """
    mse_loss = nn.functional.mse_loss(recon_x, x, reduction="mean")
    # Analytical KL divergence between N(mu, sigma^2) and N(0, I)
    kld_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return mse_loss + beta * kld_loss, mse_loss, kld_loss


def generate_normal_traffic_data(num_samples=1200, seq_len=5):
    """
    Generates normal, non-incident traffic sequences for training and validation.
    """
    sequences = []
    for _ in range(num_samples):
        seq = np.zeros((seq_len, 4), dtype=np.float32)
        curr = np.random.uniform(2.0, 7.0, size=4)
        for t in range(seq_len):
            arr = np.random.uniform(0.1, 0.7, size=4)
            dep = np.zeros(4)
            phase = (t // 3) % 2
            if phase == 0:
                dep[:2] = 0.8
            else:
                dep[2:] = 0.8
            curr = np.clip(curr + arr - dep, 0.0, 30.0)
            seq[t] = curr.copy()
        sequences.append(seq)
    return np.array(sequences, dtype=np.float32)


def train_sequence_vae(epochs=20, batch_size=32, lr=1e-3, seq_len=5, device="cpu", save_path="models/sequence_vae.pt"):
    """
    Trains the Sequence VAE on normal traffic and calibrates anomaly thresholds on validation data.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    all_data = generate_normal_traffic_data(num_samples=1500, seq_len=seq_len)
    train_data = torch.tensor(all_data[:1200]).to(device)
    val_data = torch.tensor(all_data[1200:]).to(device)

    model = SequenceVAE(input_dim=4, hidden_dim=64, latent_dim=16, seq_len=seq_len).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    num_batches = len(train_data) // batch_size
    print(f"Training Sequence VAE for {epochs} epochs on device: {device}...")

    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(len(train_data))
        total_loss = 0.0

        for b in range(num_batches):
            idx = perm[b * batch_size:(b + 1) * batch_size]
            batch_x = train_data[idx]

            optimizer.zero_grad()
            recon_x, mu, logvar = model(batch_x)
            loss, mse, kld = vae_loss_function(recon_x, batch_x, mu, logvar)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        total_loss /= num_batches

        if epoch % 5 == 0 or epoch == epochs:
            print(f"Epoch {epoch:02d}/{epochs:02d} | VAE Total Loss: {total_loss:.4f} (MSE: {mse.item():.4f}, KLD: {kld.item():.4f})")

    # ---------------- Calibrate Thresholds on Validation Data ----------------
    model.eval()
    val_errors = []
    with torch.no_grad():
        recon_val, _, _ = model(val_data)
        # Per-sequence MSE
        sq_diff = (recon_val - val_data) ** 2
        per_seq_mse = sq_diff.mean(dim=(1, 2)).cpu().numpy()
        val_errors.extend(per_seq_mse)

    mean_err = float(np.mean(val_errors))
    std_err = float(np.std(val_errors))

    # Multi-level calibrated thresholds
    threshold_warning = mean_err + 2.5 * std_err
    threshold_critical = mean_err + 5.0 * std_err

    print(f"Calibrated Thresholds -> Baseline Mean: {mean_err:.4f}, Warning: {threshold_warning:.4f}, Critical: {threshold_critical:.4f}")

    torch.save({
        "model_state": model.state_dict(),
        "threshold_warning": threshold_warning,
        "threshold_critical": threshold_critical,
        "mean_error": mean_err,
        "std_error": std_err,
        "seq_len": seq_len,
    }, save_path)
    print(f"Sequence VAE checkpoint and calibrated thresholds saved to {save_path}")
    return model, threshold_warning, threshold_critical


def load_sequence_vae(checkpoint_path="models/sequence_vae.pt", device="cpu"):
    """Loads Sequence VAE and calibration thresholds."""
    model = SequenceVAE(input_dim=4, hidden_dim=64, latent_dim=16, seq_len=5).to(device)
    threshold_warning = 2.5
    threshold_critical = 5.0

    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        threshold_warning = float(ckpt.get("threshold_warning", 2.5))
        threshold_critical = float(ckpt.get("threshold_critical", 5.0))
        model.eval()
    return model, threshold_warning, threshold_critical


def detect_incident_anomaly(model, sequence, threshold_warning=2.5, threshold_critical=5.0, device="cpu"):
    """
    Evaluates a traffic sequence (seq_len, 4) for incidents or anomalies.
    Returns:
    - level: "NORMAL", "WARNING", "CRITICAL"
    - score: float (reconstruction error)
    - alert_message: str (formatted with exact required phrasing)
    - affected_approach: str ("North", "South", "East", "West")
    """
    model.eval()
    arr = np.asarray(sequence, dtype=np.float32)
    if arr.ndim == 2:
        seq_t = torch.tensor(arr).unsqueeze(0).to(device)
    else:
        seq_t = torch.tensor(arr).to(device)

    with torch.no_grad():
        recon, _, _ = model(seq_t)
        # Residual error across sequence and approaches
        diff = (seq_t - recon).squeeze(0).cpu().numpy()
        approach_errors = np.mean(diff ** 2, axis=0)  # (4,)
        total_error = float(np.mean(approach_errors))

    approaches = ["North", "South", "East", "West"]
    worst_idx = int(np.argmax(approach_errors))
    affected = approaches[worst_idx]

    if total_error >= threshold_critical:
        level = "CRITICAL"
        msg = f"Critical Traffic Anomaly — Investigation Required on approach {affected} (Reconstruction Residual: {total_error:.2f})"
    elif total_error >= threshold_warning:
        level = "WARNING"
        msg = f"Potential Anomaly: Unusual Congestion Surge on approach {affected} (Reconstruction Residual: {total_error:.2f})"
    else:
        level = "NORMAL"
        msg = f"Traffic flow normal across all approaches (Reconstruction Residual: {total_error:.2f})"

    return {
        "level": level,
        "score": float(round(total_error, 3)),
        "alert_message": msg,
        "affected_approach": affected,
        "approach_scores": [float(round(s, 3)) for s in approach_errors],
    }
