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
        # Direction 0 (NS) queue: state[0] + state[1]
        # Direction 1 (EW) queue: state[2] + state[3]
        labels.append(int(state[0] + state[1] < state[2] + state[3]))
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

    for ep in range(1, n_episodes + 1):
        state = env.reset()
        states = []
        actions = []
        rewards = []

        model.eval()
        with torch.no_grad():
            for t in range(max_steps):
                state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
                probs = model(state_t)
                m = torch.distributions.Categorical(probs)
                action = m.sample().item()

                next_state, reward, done, info = env.step(action)

                states.append(state_t)
                actions.append(action)
                rewards.append(reward)

                state = next_state
                if done:
                    break

        returns = compute_return(rewards, gamma=gamma)
        returns = torch.tensor(returns, dtype=torch.float32).to(device)
        total_ep_reward = sum(rewards)
        if len(returns) > 1 and returns.std() > 1e-6:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # Batch forward pass in train mode so BatchNorm and Dropout operate on batch size T
        model.train()
        states_batch = torch.cat(states, dim=0)
        actions_batch = torch.tensor(actions, dtype=torch.long).to(device)

        probs = model(states_batch)
        m = torch.distributions.Categorical(probs)
        log_probs = m.log_prob(actions_batch)
        entropy = m.entropy()

        loss = -torch.mean(log_probs * returns) - 0.01 * torch.mean(entropy)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if ep % 20 == 0:
            print(f"[Reinforce] Episode {ep:03d} | Total reward {total_ep_reward:.1f} | loss {loss.item():.4f}")

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
              epsilon_decay=0.97, target_update_freq=5, device="cpu"):
    """
    Simple DQN training loop with epsilon‑greedy exploration and target network updates.
    Uses Mean‑Squared‑Error loss on the Bellman target.
    """
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    q = model.to(device)
    target_q = copy.deepcopy(q).to(device)
    target_q.eval()

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

        # periodically sync target network
        if ep % target_update_freq == 0:
            target_q.load_state_dict(q.state_dict())

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