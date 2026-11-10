# src/trainer.py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import copy
from torch.utils.data import DataLoader, TensorDataset
from torch.nn import init


def generate_synthetic_data(env, n_samples=3000, seq_len=1):
    """Generate states and heuristic phase labels for supervised pre-training."""
    states = []
    labels = []
    for _ in range(n_samples):
        state = env.reset()
        states.append(state)
        labels.append(int(state[0] + state[3] < state[1] + state[2]))
    x = torch.tensor(np.asarray(states), dtype=torch.float32)
    if seq_len > 1:
        x = x.unsqueeze(1).repeat(1, seq_len, 1)
    dataset = TensorDataset(x, torch.tensor(labels, dtype=torch.long))
    return DataLoader(dataset, batch_size=min(64, n_samples), shuffle=True)


def train_model(model, dataloader, epochs=15, lr=1e-3, device="cpu"):
    """Train a classifier on the synthetic heuristic labels."""
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    model.train()
    for _ in range(epochs):
        for states, labels in dataloader:
            states, labels = states.to(device), labels.to(device)
            if states.ndim == 3:
                states = states[:, -1, :]
            probs = model(states)
            loss = criterion(torch.log(probs.clamp_min(1e-8)), labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model

# -----------------------------------------------------------------
# REINFORCE (policy‑gradient) training loop
# -----------------------------------------------------------------
def compute_return(rewards, gamma=0.99):
    """Compute discounted returns for a list of rewards (scalar)."""
    R = 0
    returns = []
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)
    return returns


def reinforce_train(model, env, n_episodes=200, max_steps=200,
                    lr=1e-3, gamma=0.99, device="cpu"):
    """
    Simple REINFORCE algorithm that directly maximises expected cumulative reward.
    The policy is the model's Softmax output; we sample an action and
    update using the log‑probability * return.

    Returns the trained model.
    """
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.train()
    for ep in range(1, n_episodes + 1):
        state = env.reset()
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)

        log_probs = []
        rewards = []
        entropies = []

        for t in range(max_steps):
            # ---------- forward ----------
            # Use model.eval() so that BatchNorm uses running stats (helps with batch size 1)
            model.eval()
            probs = model(state)                     # (2,) — gradients will be tracked
            model.train()
            # sample action
            m = torch.distributions.Categorical(probs)
            action = m.sample()

            # store log prob and entropy for later
            log_probs.append(m.log_prob(action))
            entropies.append(m.entropy())

            # ---------- env step ----------
            next_state, reward, done, info = env.step(action.item())

            rewards.append(reward)
            state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0).to(device)

            if done:
                break

        # ---------- compute returns ----------
        returns = compute_return(rewards, gamma=gamma)
        returns = torch.tensor(returns, dtype=torch.float32).to(device)
        # normalise returns (optional but helps stability)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # ---------- loss ----------
        # policy gradient = - log_prob * advantage (here advantage = return - baseline)
        # we use the mean return as a simple baseline
        loss = -torch.stack(log_probs).mul(returns).mean()
        loss -= 0.01 * torch.stack(entropies).mean()   # entropy bonus

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if ep % 20 == 0:
            print(f"[Reinforce] Episode {ep:03d} | Return mean {returns.mean().item():.2f} | loss {loss.item():.4f}")

    return model


# -----------------------------------------------------------------
# DQN‑style Q‑learning (tabular‑style, simple MLP head)
# -----------------------------------------------------------------
class QNetwork(nn.Module):
    """
    Simple Q‑value network for discrete actions.
    Takes the current state and outputs Q-values for each phase.
    """
    def __init__(self, input_dim=7, hidden_dims=[64, 32], dropout=0.3):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            init.kaiming_uniform_(layers[-1].weight, nonlinearity='relu')
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, 2))  # Q-values for 2 actions
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def dqn_train(model, env, n_episodes=300, max_steps=200,
              lr=1e-3, gamma=0.99, epsilon_start=1.0, epsilon_end=0.05,
              epsilon_decay=0.995, device="cpu"):
    """
    Simple DQN training loop with epsilon‑greedy exploration.
    Uses Mean‑Squared‑Error loss on the Bellman target.
    """
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    q = model.to(device)
    target_q = copy.deepcopy(q).to(device)

    epsilon = epsilon_start

    for ep in range(1, n_episodes + 1):
        state = env.reset()
        state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        total_reward = 0.0

        for t in range(max_steps):
            # epsilon‑greedy action selection
            if np.random.rand() < epsilon:
                action = np.random.choice(2)
            else:
                with torch.no_grad():
                    q_vals = q(state_t)
                    action = torch.argmax(q_vals).item()

            next_state, reward, done, info = env.step(action)
            next_state_t = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0).to(device)

            # Q‑learning target
            with torch.no_grad():
                max_next = target_q(next_state_t).max(1)[0]
                target = reward + gamma * max_next * (not done)

            # current Q estimate
            q_vals = q(state_t)
            target_q_vals = q_vals.clone()
            target_q_vals[0, action] = target

            loss = criterion(q_vals, target_q_vals)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            state_t = next_state_t
            total_reward += reward

            if done:
                break

        # decay epsilon
        epsilon = max(epsilon_end, epsilon * epsilon_decay)

        if ep % 20 == 0:
            print(f"[DQN] Episode {ep:03d} | Total reward {total_reward:.2f} | eps {epsilon:.3f}")

    return q  # return the trained Q‑network


# -----------------------------------------------------------------
# Helper that runs a short rollout using a policy model and returns
# states, actions, rewards for later training.
# -----------------------------------------------------------------
def collect_trajectory(model, env, max_steps=200, device="cpu"):
    """
    Collect one episode (state, action, reward) tuple list.
    The model's probabilities are used to sample actions (or we can argmax).
    """
    state = env.reset()
    state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)

    states, actions, rewards = [], [], []

    for t in range(max_steps):
        with torch.no_grad():
            probs = model(state)
        m = torch.distributions.Categorical(probs)
        action = m.sample().item()

        next_state, reward, done, info = env.step(action)

        states.append(state.squeeze().detach().cpu())
        actions.append(action)
        rewards.append(reward)

        state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0).to(device)
        if done:
            break

    return states, actions, rewards