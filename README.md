# Autonomous Traffic Management System (ATMS)

## Project Title & Overview

**Autonomous Traffic Management System (ATMS)** is a prototype that demonstrates how deep‑learning models can dynamically control traffic‑signal phases at a single intersection (and, with minor modifications, multiple intersections).

- **Goal:** Minimise average waiting time, reduce queue lengths, and increase throughput by learning an optimal signal‑phase policy from real‑time traffic state (queue lengths, vehicle count, current phase).
- **Tech stack:** Python 3, PyTorch, NumPy, Matplotlib (optional Pygame for visualisation).
- **Two model families:**
  - **MLPModel** – a feed‑forward Multi‑Layer Perceptron that maps the current intersection state to a probability distribution over the two possible phases (NS‑green / EW‑green).
  - **RNNModel** – an LSTM/GRU that ingests a short history of states, capturing temporal congestion patterns and alleviating vanishing‑gradient issues.
- **Training paradigms:**
  - **Supervised pre‑training** on a synthetic dataset generated with a cheap heuristic (used for reference).
  - **Reinforcement‑learning (REINFORCE)** that directly maximises expected cumulative negative‑queue reward.
  - **DQN (Deep Q‑Network)** as a value‑based alternative with ε‑greedy exploration.
- **Baseline:** Fixed‑time controller that toggles phases every 5 steps (standard traffic‑engineering plan).
- **Evaluation:** Average waiting time, queue‑length reduction (%), and throughput (vehicles / minute). Results are visualised with bar‑charts.
- **Visualisation (optional):** Pygame window that draws the intersection, queues, current phase and the agent’s action probabilities.

---

## Architecture & System Design

```mermaid
flowchart TD
    A[Traffic Environment (env.py)] -->|state| B[Deep‑Learning Agent (model.py + trainer.)]
    B -->|action| A
    B -->|metrics| C[Evaluation/Plot (evaluate.py)]
```

1. **Environment (`src/env.py`)** – Light‑weight grid simulation (or placeholder for SUMO/TraCI). At each time‑step `t` it returns:
   - **State** `s_t = [q_{N‑in}, q_{N‑out}, q_{E‑in}, q_{E‑out}, N_{veh}, φ_{one‑hot}]` (7‑dim).
   - **Reward** `r_t = -\frac{1}{4}\sum_{i=1}^{4} q_i` (negative average queue – proxy for waiting time).
   - **Done** after a maximum horizon (200 steps by default).

2. **Neural‑network policy** – two alternatives in `src/model.py`:
   - **MLPModel** – fully‑connected layers with He‑initialisation, ReLU, Batch‑Norm, Dropout and a final Softmax over 2 actions.
   - **RNNModel** – LSTM/GRU cell over a sequence of length `seq_len`; the last hidden state feeds a Linear → Softmax head.

3. **Training loop** – `src/trainer.py` implements:
   - **REINFORCE** (policy‑gradient): sample action, collect rewards, compute discounted returns, maximise `E[∑γᵗ r_t log π(a_t|s_t)]`. Uses Adam, entropy regularisation.
   - **DQN**: ε‑greedy selection, MSE loss on Bellman target `y = r + γ maxₐ' Q(s',a'; θ⁻)`, target‑network update.

4. **Evaluation (`src/evaluate.py`)** – runs a trained policy inside the environment, gathers metrics and produces a grouped bar‑chart comparing the learning agent against the fixed‑time baseline.

5. **Optional GUI** – `src/gui.py` (Pygame) visualises the intersection, queue bars, phase colour and the policy’s action probabilities in real time.

---

## Codebase Mapping Table

| File (relative path) | Function / Class | Course Module & Concept(s) |
|---|---|---|
| **`src/env.py`** | `TrafficEnv` class (methods `reset_light`, `step`, `_get_state`) | **Module 1** – defines the *state space* (queue lengths, vehicle count, one‑hot phase) that the artificial neuron receives.<br>**Module 2** – reward design (`r_t = -avg_queue`) illustrates a *loss‑like* objective for the agent.<br>**Module 4** – optional SUMO/TraCI integration (future extension of sequence data). |
| **`src/model.py`** | `MLPModel` class | **Module 1** – *Multi‑Layer Perceptron*; forward propagation through Linear → ReLU → BatchNorm → Dropout → Softmax.<br>**Module 2** – *He (Kaiming) initialization* for ReLU layers; *Batch Normalisation* and *Dropout* as regularisers.<br>**Module 4** – activation functions: ReLU, Softmax, Sigmoid (used in baseline). |
| **`src/model.py`** | `RNNModel` class (LSTM or GRU cell) | **Module 4** – *Recurrent Neural Networks*; *Backpropagation Through Time (BPTT)* concept (PyTorch automates it).<br>**Module 2** – *Long Short‑Term Memory* and *Gated Recurrent Unit* architectures to capture temporal congestion; *gradient clipping* (implicit via PyTorch). |
| **`src/trainer.py`** | `reinforce_train` function | **Module 2** – *REINFORCE policy‑gradient algorithm*; *Chain rule* for `∇θ log π(a|s)`.<br>**Module 2** – *Adam optimizer* (moment + adaptive LR).<br>**Module 2** – *Loss landscape* navigation via gradient descent; *entropy regularisation*. |
| **`src/trainer.py`** | `dqn_train` function | **Module 2** – *Deep Q‑Network*; *Mean‑Squared‑Error loss* and *Bellman equation* `y = r + γ maxₐ' Q(s',a')`.<br>**Module 2** – *ε‑greedy exploration*, *target network* updates, *experience‑replay* (conceptual). |
| **`src/trainer.py`** | `collect_trajectory` helper | **Module 4** – generates state‑action‑reward tuples for sequence‑based training; illustrates *trajectory sampling* for RNN/BPTT. |
| **`src/evaluate.py`** | `run_controller`, `run_baseline`, `plot_comparison` | **Module 2** – evaluation metrics: *average waiting time*, *queue‑length reduction*, *throughput*.<br>**Module 2** – *baseline comparison* (fixed‑time controller).<br>**Module 4** – *sequence‑aware* evaluation if LSTM policy is used. |
| **`src/gui.py`** | `draw_intersection` function | **Module 1** – visual representation of *activation* (green/red lights) and *decision boundaries* (which phase is active).<br>**Module 4** – real‑time demonstration of *RNN* policy inference. |
| **`main.py`** | Entry‑point script | **All modules** – orchestrates environment, model selection, training algorithm, evaluation and optional GUI; demonstrates *system integration* of the concepts. |

---

## Mathematical Formulations

Below are the key equations used in the project. They are written with LaTeX‑style math delimiters (`$...$` for inline and `$$...$$` for displayed equations) so that they render correctly on GitHub (GitHub uses MathJax/KaTeX for markdown).

### Inline expressions (rendered within text)

- **ReLU activation:** `f_{\text{ReLU}}(x) = \max(0, x)`
- **Sigmoid:** `\sigma(x) = \frac{1}{1+e^{-x}}`
- **Softmax (output layer):** `\hat{y}_i = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}}`

### Displayed equations (stand‑alone blocks)

**ReLU activation**

$$
f_{\text{ReLU}}(x) = \max(0, x)
$$

**Sigmoid**

$$
\sigma(x) = \frac{1}{1+e^{-x}}
$$

**Softmax (output layer)**

$$
\hat{y}_i = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}}
$$

**Cross‑Entropy loss** (supervised pre‑training)

$$
\mathcal{L}_{\text{CE}} = -\sum_{c} y_c \log \hat{y}_c
$$

**REINFORCE policy‑gradient loss**

$$
\mathcal{L}_{\text{RL}} = -\frac{1}{T}\sum_{t=1}^{T} \log \pi(a_t|s_t)\, \hat{A}_t
\qquad\text{with advantage }\hat{A}_t = G_t - b
$$

**Adam optimizer update**

$$
\begin{aligned}
m_t &= \beta_1 m_{t-1} + (1-\beta_1) g_t \\
v_t &= \beta_2 v_{t-1} + (1-\beta_2) g_t^2 \\
\theta_{t+1} &= \theta_t - \alpha \frac{m_t}{\sqrt{v_t} + \epsilon}
\end{aligned}
$$

**He (Kaiming) initialization** for ReLU layers

$$
W \sim \mathcal{N}\!\left(0,\; \frac{2}{n_{\text{in}}}\right)
$$

**Batch Normalisation**

$$
\hat{x}_i = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}, \qquad
y_i = \gamma \hat{x}_i + \beta
$$

**Dropout** (probability `p`)

Randomly zero out activations with probability `p` during training.

**LSTM cell equations** (shown for completeness; PyTorch abstracts them)

$$
\begin{aligned}
i_t &= \sigma(W_i[h_{t-1},x_t] + b_i) \\
f_t &= \sigma(W_f[h_{t-1},x_t] + b_f) \\
g_t &= \tanh(W_c[h_{t-1},x_t] + b_c) \\
o_t &= \sigma(W_o[h_{t-1},x_t] + b_o) \\
c_t &= f_t \odot c_{t-1} + i_t \odot g_t \\
h_t &= o_t \odot \tanh(c_t)
\end{aligned}
$$

**Bellman equation** (DQN loss target)

$$
y = r + \gamma \max_{a'} Q(s',a'; \theta^{-})
$$

*Tip:* If the equations do not render in your viewer, copy the displayed LaTeX code into a local Markdown/LaTeX renderer (e.g., Typora, Obsidian, or a Jupyter notebook) that supports MathJax/KaTeX.

---

## Setup & Execution Guide

### 1. Prerequisites

```bash
# Clone the repository (or ensure you are in the project folder)
git clone <your-repo-url>
cd Autonomous-Traffic-Monitoring-and-Incident-Management-System

# Install required packages
pip install -r requirements.txt
#   • numpy >= 1.20
#   • torch >= 1.8
#   • matplotlib >= 3.3
# (optional) pip install pygame   # for the GUI visualiser
```

### 2. Quick supervised baseline (original code)

```bash
python main.py --mode supervised
```
- Generates a synthetic dataset, trains an `MLPModel` for 15 epochs, evaluates against a fixed‑time controller and saves `results/supervised_comparison.png`.

### 3. Reinforcement‑learning training (REINFORCE)

```bash
python main.py --mode rl --algo reinforce --epochs 50 --max-steps 200
```
- Trains the policy for **50 episodes** (each episode up to 200 steps).  
- After training, the script evaluates the learned controller and produces `results/rl_comparison.png`.

### 4. Deep Q‑Network training

```bash
python main.py --mode rl --algo dqn --epochs 100 --max-steps 200
```
- Runs the DQN loop (ε‑greedy exploration, MSE bellman target).  
- Generates `results/dqn_comparison.png`.

### 5. Launch the Pygame visualiser (after any training run)

```bash
python main.py --mode rl --algo reinforce --epochs 30 --max-steps 100 --gui
```
- A window opens showing the intersection, queue bars, current phase colour, and the agent’s action probabilities.  
- The simulation loops continuously, allowing you to watch the learned policy in action.

### 6. Customising the run

| Argument | Effect |
|---|---|
| `--num-intersections N` | Spawns **N** independent intersections (useful for a simple coordination demo). |
| `--hidden-dim H` | Sets the number of hidden units in MLP or the LSTM/GRU hidden size. |
| `--seq-len L` | Length of the state history fed to the RNN model (only for `RNNModel`). |
| `--cell lstm|gru` | Chooses between LSTM or GRU recurrent cell. |
| `--gamma G` | Discount factor for REINFORCE returns (default 0.99). |
| `--lr α` | Learning rate for the Adam optimiser. |
| `--sumo` | (Placeholder) attempts to load a real SUMO simulation; falls back to the lite mode if SUMO is not installed. |

---

## Further Extensions (ideas for the project)

| Idea | Corresponding Course Topic |
|---|---|
| **Integrate real SUMO/TraCI** – replace the lightweight dynamics block in `env.py` with actual TraCI calls to obtain true queue lengths, waiting times, and vehicle counts. | **Module 1** (state representation) & **Module 4** (sequence data from multiple time steps). |
| **Multi‑agent coordination** – feed neighbour‑intersection states into the policy (concatenate or use attention). | **Module 2** (optimisation) & **Module 4** (RNN/attention). |
| **PPO / A2C** – replace REINFORCE with a proximal‑policy‑optimisation algorithm for more stable learning. | **Module 2** (advanced optimisation, clip‑ratio). |
| **CNN front‑end** – process raw video/images of the intersection as input (e.g., tiny ResNet) instead of tabular state. | **Module 3** (CNN layers, transfer learning). |
| **GAN / VAE** – generate synthetic traffic scenarios for data augmentation. | **Module 4** (generative models). |

---

**Happy coding!** This README provides a complete map from the repository’s source files to the deep‑learning course modules you are studying, the key mathematical formulae implemented, and clear instructions to get the prototype running and visualising results. Feel free to adapt the arguments, add new algorithms, or plug in a real SUMO simulation to extend the system for your final demo.