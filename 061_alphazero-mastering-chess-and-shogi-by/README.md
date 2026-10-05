# Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm (AlphaZero)

**Paper:** [Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm](https://arxiv.org/abs/1712.01815)  
**Authors:** David Silver, Thomas Hubert, Julian Schrittwieser, Ioannis Antonoglou, Matthew Lai, Arthur Guez, Marc Lanctot, Laurent Sifre, Dharshan Kumaran, Thore Graepel, Timothy Lillicrap, Karen Simonyan, Demis Hassabis  
**Published:** arXiv:1712.01815 (5 Dec 2017); Science, 362(6419):1140–1144 (7 Dec 2018)

[![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nadymsazad/alphazero-mastering-chess-and-shogi)

## Summary

AlphaZero is a single, generic reinforcement learning algorithm that achieves superhuman performance in chess, shogi, and Go — starting from random play with no domain knowledge except the game rules. It generalises the earlier AlphaGo Zero approach: a deep neural network (p, v) = fθ(s) takes a board position s and outputs a policy vector p (move probabilities) and a scalar value v (expected outcome), replacing the handcrafted evaluation functions and move-ordering heuristics used by traditional engines. Instead of alpha-beta search, AlphaZero uses Monte-Carlo Tree Search (MCTS) guided by the neural network: each simulation selects moves with high visit counts, high prior probability, and high value. The network is trained entirely by self-play reinforcement learning — games are played by MCTS using the latest network, terminal positions are scored as −1/0/+1, and the loss function combines mean-squared error on the value prediction and cross-entropy on the policy prediction against the MCTS visit-count distribution.

Key differences from AlphaGo Zero: (1) AlphaZero estimates expected outcome including draws, not just binary win/loss; (2) it does not use symmetries (chess/shogi are asymmetric); (3) it maintains a single network updated continually rather than checkpoint-based best-player evaluation; (4) it reuses the same hyperparameters across all games without per-game tuning. Starting from random initialization, AlphaZero surpassed Stockfish in chess after 4 hours, Elmo in shogi after 2 hours, and AlphaGo Zero (3-day) in Go after 8 hours — all using 5,000 first-generation TPUs for self-play generation and 64 second-generation TPUs for training.

Key method details relevant to the code:
- **MCTS with neural network guidance:** each node stores visit count N, total value W, prior probability P, and Q = W/N. Selection uses the PUCT formula: a = argmax(Q(s,a) + c_puct · P(s,a) · √N(s) / (1 + N(s,a))). Expansion adds a leaf, evaluated by the network to get (p, v). Backup propagates v up the path.
- **Self-play data generation:** MCTS runs 800 simulations per move. The search returns a policy π proportional to visit counts at the root (with Dirichlet noise for exploration). Moves are sampled from π (or temperature-annealed).
- **Training loss:** l = (z − v)² − πᵀ log p + c‖θ‖², combining MSE value loss, cross-entropy policy loss, and L2 regularization.
- **Continual updates:** the network is updated continuously from self-play data, unlike AlphaGo Zero's iteration-based best-player checkpointing.

## What Problem Does It Solve

Imagine you want to build a robot that can play any board game — chess, checkers, Go — but instead of programming it with centuries of human strategy knowledge, you just teach it the rules and let it play against itself millions of times until it figures out the best moves on its own. That's the problem AlphaZero solves. Before AlphaZero, the best chess programs (like Stockfish) were essentially giant lookup tables of human-designed rules: "a knight is worth 3 points," "control the center," "don't hang your queen." Engineers spent decades hand-tuning these rules. AlphaZero throws all of that away. You give it the rules of chess, it plays against itself, and within four hours it's better than any chess program ever built by humans. It even rediscovered famous human openings like the Sicilian Defense and the English Opening — purely by trial and error. The big idea is that a single, simple algorithm — "search with a neural network, play against yourself, learn from the results" — can master any game without any human expertise, which is a step toward general AI.

## Influence

AlphaZero is one of the most influential AI papers of the decade:
- **Generalized self-play learning:** proved that a single algorithm can master multiple games without domain-specific engineering, a milestone toward general-purpose AI.
- **Leela Chess Zero:** the open-source community project Leela Chess Zero (lc0) directly implements AlphaZero's approach and competed at the top level against Stockfish in TCEC championships.
- **MuZero (2019):** DeepMind's follow-up extended AlphaZero to learn the game rules themselves (no rules given), playing Atari, chess, shogi, and Go with a single system.
- **AlphaFold:** the planning + neural network paradigm influenced DeepMind's protein-folding breakthrough, which won the 2024 Nobel Prize in Chemistry.
- **MCTS + neural networks in RL:** the combination became a standard paradigm in game-playing RL research, influencing projects from poker (Pluribus) to StarCraft (AlphaStar).
- **Citations:** the paper has accumulated over 3,500 citations and was published in Science (2018).

arXiv link: https://arxiv.org/abs/1712.01815
