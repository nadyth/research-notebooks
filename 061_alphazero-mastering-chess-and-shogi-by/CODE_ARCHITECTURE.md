# Code Architecture — AlphaZero Implementation

This document describes the notebook structure, key functions/classes, data flow, and deliberate simplifications versus the original paper.

## Notebook Structure

### 1. Setup & Imports
Installs required packages (`torch`, `numpy`, `matplotlib`) and imports all modules. The implementation is fully self-contained with no external game engine dependencies — the toy game (Connect Four) is implemented from scratch.

### 2. Game Environment: Connect Four
A from-scratch `Connect4` class implementing a 6×7 board:
- **`play(action)`** — drops a piece in column `action`; switches player
- **`get_valid_moves()`** — returns list of non-full columns
- **`is_terminal()` / `get_reward()`** — checks for 4-in-a-row (win=±1) or draw (0)
- **`board_to_tensor()`** — encodes the board as a `(2, 6, 7)` tensor (player 1 pieces, player 2 pieces) for the neural network
- **`clone()`** — deep-copies the state for MCTS simulation

**Why Connect Four?** The paper uses chess/shogi/Go, which require enormous networks and thousands of TPU-hours. Connect Four has a 7-action space, 6×7 board, and games last ~20 moves — perfect for demonstrating AlphaZero's MCTS + neural-network self-play loop on a single GPU in minutes. The algorithm is identical; only the game and network scale differ.

### 3. Neural Network: PolicyValueNet
A single convolutional network with two heads (matching AlphaZero's (p, v) = fθ(s)):
- **Input:** `(batch, 2, 6, 7)` — two binary planes (current player pieces, opponent pieces)
- **Shared trunk:** 2 convolutional layers (32 filters, 3×3, ReLU) — small version of AlphaZero's 19/39-block ResNet
- **Policy head:** 1×1 conv → flatten → FC → `(batch, 7)` logits (one per column)
- **Value head:** 1×1 conv → flatten → FC → FC → `(batch, 1)` tanh (value in [−1, 1])

**Key shapes:**
- Board input: `(1, 2, 6, 7)` for a single position
- Policy logits: `(1, 7)` — 7 possible moves (columns)
- Value: `(1, 1)` — scalar in [−1, 1]

### 4. MCTS Node and Tree Search
An `MCTSNode` class stores:
- `state` — Connect4 board
- `parent` — parent node
- `children` — dict {action: MCTSNode}
- `N` — visit count
- `W` — total value
- `P` — prior probability from the network
- `untried_actions` — actions not yet expanded

The `MCTS` class implements:
- **`search(node, network, num_simulations)`** — runs `num_simulations` iterations:
  1. **Selection:** traverse from root to leaf using PUCT: `a = argmax(Q(s,a) + c_puct · P(s,a) · √N_parent / (1 + N(s,a)))`
  2. **Expansion:** at the leaf, ask the network for (p, v); create child nodes with prior p
  3. **Backup:** propagate v (sign-flipped for the opponent's perspective) up the path to root
- **`get_action_probs(root, temperature)`** — returns policy π proportional to `N_child^(1/temperature)`; temperature=1 for exploration, →0 for greedy

### 5. Self-Play
`self_play_game(network, num_simulations, temperature)`:
1. Reset the game.
2. For each move:
   a. Run MCTS from the current position (800 simulations in the paper; 100 in our simplified version).
   b. Get policy π from visit counts.
   c. Sample action from π (temperature=1 for early moves, →0 after move 10).
   d. Store (state, π, current_player) in training data.
   e. Play the action.
3. At game end, assign z = ±1 to each state based on the player who moved.
4. Return list of (state_tensor, policy_target, value_target).

### 6. Training Loop
`train(network, optimizer, num_iterations, games_per_iter, num_simulations, epochs)`:
For each iteration:
1. **Self-play phase:** generate `games_per_iter` self-play games using the current network.
2. **Training phase:** sample minibatches from the replay buffer:
   - `policy_loss = −Σ π_target · log(policy_probs)` (cross-entropy)
   - `value_loss = (z − v)²` (MSE)
   - `total_loss = value_loss + policy_loss + c·‖θ‖²` (L2 regularization, c=1e-4)
   - Adam optimizer, lr=0.001
3. **Evaluation:** play the current network against a random player; track win rate.
4. **Plotting:** win rate vs. iterations, policy/value losses over training.

### 7. Evaluation & Visualization
- **Win rate tracking:** after each training iteration, play 20 games against a random-move baseline. Track and plot the win rate improvement over iterations.
- **MCTS visualization:** for a sample position, print the visit counts and Q-values at the root to show how the search focuses on promising moves.
- **Loss curves:** plot policy loss, value loss, and total loss over training steps.

### Deliberate Simplifications vs. Full Paper

| Aspect | Paper (AlphaZero) | This Notebook |
|--------|-------------------|---------------|
| Game | Chess / Shogi / Go | Connect Four (6×7) |
| Network | 19/39-block ResNet (256 filters) | 2-layer CNN (32 filters) |
| MCTS simulations/move | 800 | 100 |
| Self-play TPUs | 5,000 first-gen + 64 second-gen | 1 GPU |
| Training time | 24 hours (700k steps) | ~10 minutes (3 iterations × 20 games) |
| Batch size | 4,096 | 64 |
| Exploration noise | Dirichlet α=0.3 (chess), 0.15 (Go) | Dirichlet α=0.3 |
| Evaluation | vs. Stockfish/Elmo (world champions) | vs. random player |
| Continual updates | Yes (no checkpointing) | Iteration-based (simplified) |
| Symmetries | Not used (chess/shogi asymmetric) | Not used |
| c_puct | 1.0 (hard-coded, no per-game tuning) | 1.0 |

The core algorithm — PUCT-based MCTS guided by a neural network, trained by self-play reinforcement learning with combined policy+value loss — is faithfully implemented. The simplifications are purely in scale (game complexity, network size, training budget) to make the algorithm runnable on a single GPU in minutes rather than 5,000 TPUs for 24 hours.
