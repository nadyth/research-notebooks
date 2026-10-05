# AlphaZero — Press Coverage, Interviews, Misconceptions, Citations

## Verifiable Press & Blog Coverage

1. **The Guardian — "AlphaZero AI beats champion chess program after teaching itself in four hours" (December 7, 2017)**
   Samuel Gibbs reported on AlphaZero's victory over Stockfish, highlighting that the AI learned chess from scratch in four hours of self-play without any human knowledge beyond the rules.
   - https://www.theguardian.com/technology/2017/dec/07/alphazero-google-deepmind-ai-beats-champion-chess-program-teaching-itself-four-hours

2. **Wired — "Alphabet's Latest AI Show Pony Has More Than One Trick" (December 6, 2017)**
   Wired described AlphaZero as "the first multi-skilled AI board-game champ," emphasizing that a single algorithm mastered three different games.
   - https://www.wired.com/story/alphago-zero-ai-master-chess-shogi-go/

3. **MIT Technology Review — "Alpha Zero's 'Alien' Chess Shows the Power, and the Peculiarity, of AI" (December 8, 2017)**
   Will Knight covered AlphaZero's unconventional, "alien" playing style, noting its counterintuitive sacrifices and positional understanding that differed from human chess theory.
   - https://www.technologyreview.com/2017/12/08/105014/alpha-zeros-alien-chess-shows-the-power-and-the-peculiarity-of-ai/

4. **BBC News — "'Superhuman' Google AI claims chess crown" (December 6, 2017)**
   The BBC reported on AlphaZero's 100-game match against Stockfish (28 wins, 0 losses, 72 draws), quoting grandmaster Peter Heine Nielsen comparing AlphaZero's play to "a superior alien species."
   - https://www.bbc.com/news/technology-42263504

5. **Chess.com — "Google's AlphaZero Destroys Stockfish In 100-Game Match" (December 2017)**
   Chess.com provided detailed game analysis and reactions from top grandmasters including Garry Kasparov, who called it "a remarkable achievement, even if we should have expected it after AlphaGo."
   - https://www.chess.com/news/google-alphazero-destroys-stockfish-100-game-match-2017

## Interview Q&A

**Q: How can AlphaZero search a thousand times fewer positions than Stockfish and still win?**
A: AlphaZero uses its neural network to evaluate positions much more intelligently — instead of brute-forcing millions of positions with a handcrafted evaluation function, MCTS with neural network priors focuses the search on the most promising lines. The paper states AlphaZero searches just 80,000 positions/second in chess vs. 70 million for Stockfish, but compensates with "much more selectively on the most promising variations — arguably a more 'human-like' approach to search."

**Q: What did AlphaZero learn about chess that surprised grandmasters?**
A: Demis Hassabis described AlphaZero's style as "alien" — it sometimes wins by offering counterintuitive sacrifices, like sacrificing a queen and bishop to exploit a positional advantage. Danish GM Peter Heine Nielsen said "I've always wondered how it would feel if chess were solved. Now I know." The paper showed AlphaZero independently rediscovered all major human openings (Sicilian, French, English, etc.) during self-play.

**Q: Was the match against Stockfish fair?**
A: This was a subject of significant debate. Stockfish developer Tord Romstad criticized the setup: Stockfish 8 was a year-old version, running with 64 threads and only 1 GB hash (suboptimal), at fixed 1-minute-per-move time controls (negating Stockfish's time management heuristics). GM Hikaru Nakamura argued Stockfish was "basically running on what would be my laptop" while AlphaZero used Google supercomputers. DeepMind addressed these in the Science (2018) version with stronger Stockfish conditions (44 cores, 32 GB hash, 3-hour time control), where AlphaZero still won 155–6–839.

**Q: How does AlphaZero differ from AlphaGo Zero?**
A: The paper lists several differences: (1) AlphaZero handles draws (expected outcome, not binary win/loss); (2) it doesn't use symmetries (chess/shogi are asymmetric, unlike Go); (3) it updates the network continually rather than using checkpoint-based best-player selection; (4) it reuses the same hyperparameters across all games without game-specific tuning.

**Q: Why was AlphaZero never released to the public?**
A: Unlike AlphaGo, DeepMind did not release the AlphaZero program. However, the open-source Leela Chess Zero project implemented the same algorithm and competed successfully against Stockfish, demonstrating that the approach is reproducible without DeepMind's resources.

## Common Misconceptions

1. **"AlphaZero was trained on human chess games."**
   *Correction:* AlphaZero was trained entirely by self-play (tabula rasa), starting from random play. The only domain knowledge given was the rules of chess. It rediscovered human openings independently — the paper explicitly notes it "independently discovered and played frequently" the most common human openings.

2. **"AlphaZero is just a bigger, faster chess engine."**
   *Correction:* AlphaZero is fundamentally different from traditional engines. Stockfish uses alpha-beta search with handcrafted evaluation functions and domain-specific heuristics refined over decades. AlphaZero uses MCTS guided by a neural network trained by reinforcement learning. The search algorithm, evaluation method, and learning approach are all different.

3. **"The 4-hour training means AlphaZero is easy to reproduce."**
   *Correction:* The 4 hours used 5,000 first-generation TPUs for self-play generation and 64 second-generation TPUs for training — equivalent to millions of TPU-hours. The paper notes this was a "computationally intensive" process. Reproducing AlphaZero's full results requires enormous compute, which is why open-source efforts like Leela Chess Zero used distributed community compute over months.

4. **"AlphaZero proved neural networks are inherently superior to alpha-beta search."**
   *Correction:* The paper itself notes this "calling into question the widely held belief that alpha-beta search is inherently superior," but the computer chess community (including Komodo developer Larry Kaufman) argued the advantage was largely due to GPU/TPU hardware. Kaufman predicted the strongest engine would be a hybrid, which has partially come true (Stockfish now incorporates NNUE — a neural network evaluation function).

5. **"AlphaZero can play any game."**
   *Correction:* AlphaZero requires a perfect-information, turn-based game with known rules and a well-defined terminal scoring. It cannot directly handle imperfect-information games (poker), real-time games (StarCraft, though AlphaStar extended the paradigm), or games without clear win/loss/draw outcomes. MuZero (2019) later removed even the requirement for known rules.

## Real Citations

- Silver, D., Hubert, T., Schrittwieser, J., et al. (2018). A general reinforcement learning algorithm that masters chess, shogi, and go through self-play. Science, 362(6419), 1140–1144. doi:10.1126/science.aar6404
- Silver, D., Schrittwieser, J., Hubert, T., et al. (2017). Mastering the game of Go without human knowledge (AlphaGo Zero). Nature, 550, 354–359. (AlphaZero's direct predecessor)
- Silver, D., Huang, A., Maddison, C.J., et al. (2016). Mastering the game of Go with deep neural networks and tree search (AlphaGo). Nature, 529, 484–489. (Original AlphaGo)
- Schrittwieser, J., Antonoglou, I., Hubert, T., et al. (2020). Mastering Atari, Go, chess and shogi by planning with a learned model (MuZero). Nature, 588, 604–609. (Extends AlphaZero to learn the rules)
- Kocsis, L. & Szepesvári, C. (2006). Bandit based Monte-Carlo Planning. ECML. (UCT — the MCTS selection algorithm used by AlphaZero)
- Browne, C., Powley, E., Whitehouse, D., et al. (2012). A Survey of Monte Carlo Tree Search Methods. IEEE TCIAIG, 4(1), 1–43. (MCTS survey)
- Pascutto, G. et al. Leela Zero / Leela Chess Zero. https://lczero.org/ (Open-source AlphaZero implementation)
