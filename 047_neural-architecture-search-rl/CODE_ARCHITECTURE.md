# Code Architecture — Neural Architecture Search with RL

## Overview

The notebook implements a simplified Neural Architecture Search (NAS) pipeline with three main components:
1. An RNN controller that samples CNN architectures
2. A child network builder + trainer
3. A REINFORCE training loop for the controller

## Section-by-Section Breakdown

### 1. Setup & Hyperparameters
- Imports: PyTorch, torchvision (MNIST), numpy, matplotlib
- Key hyperparameters:
  - `NUM_LAYERS = 3` — number of CNN layers per sampled architecture
  - `CONTROLLER_HIDDEN = 64` — LSTM hidden size
  - `CONTROLLER_EMBED_DIM = 16` — embedding for action tokens
  - `NUM_CHILD_EPOCHS = 1` — epochs per child network
  - `NUM_SEARCH_ITERATIONS = 15` — controller update rounds
  - `BASELINE_DECAY = 0.9` — EMA decay for reward baseline
  - `CONTROLLER_LR = 0.001` — controller learning rate
  - `CHILD_LR = 0.01` — child network learning rate

### 2. Search Space Definition
- **Filter sizes:** [1, 3, 5]
- **Number of filters:** [8, 16, 32]
- **Stride:** [1, 2]
- **Pooling:** [none, maxpool] (optional, after each conv layer)
- Each layer: (filter_size, num_filters, stride, pooling) → 4 choices per layer
- Total action space: 4 × NUM_LAYERS categorical decisions
- Encoded as integer indices into choice lists

### 3. Controller (LSTM)
- **Class `NASController(nn.Module)`:**
  - `__init__`: Embedding layer for action tokens, LSTM, linear projection to action spaces
  - `forward(state)`: Autoregressively samples one action at a time. At each layer, outputs 4 distributions (filter_size, num_filters, stride, pooling). Uses softmax sampling. Stores log-probabilities for REINFORCE.
  - `sample()`: Returns a full architecture spec (list of layer configs) and the sum of log-probabilities (for policy gradient).
  - Shape flow: embedding [batch=1, embed_dim] → LSTM [1, 1, hidden] → Linear [1, num_choices]

### 4. Child Network Builder
- **Function `build_child_net(arch_spec, input_channels=1, num_classes=10)`:**
  - Takes architecture spec from the controller
  - Constructs a `nn.Sequential` of Conv2d → BatchNorm → ReLU → (optional MaxPool)
  - Flattens and adds a final Linear layer
  - Returns a `nn.Module`
  - Handles stride/pooling interactions to compute the flattened size dynamically

### 5. Child Network Training
- **Function `train_child(arch_spec, train_loader, val_loader, epochs, device)`:**
  - Builds the child network via `build_child_net`
  - Trains with Adam optimizer, CrossEntropy loss
  - Returns validation accuracy as the reward signal
  - Shape: input [B, 1, 28, 28] → conv layers → flatten → [B, num_classes]

### 6. REINFORCE Update
- After each child is trained:
  - Reward R = validation accuracy
  - Advantage = R − baseline (EMA of past rewards)
  - Loss = −advantage × log_prob (sum of layer-action log-probabilities)
  - Backprop through the controller LSTM
  - Optimizer: Adam

### 7. Search Loop
- For each iteration:
  1. Controller samples an architecture
  2. Train the child network (1 epoch on MNIST subset)
  3. Evaluate validation accuracy
  4. Update baseline EMA
  5. Compute REINFORCE loss and update controller
  6. Log architecture + accuracy

### 8. Results Visualization
- Plot accuracy over search iterations
- Print the best architecture found
- Show a table of sampled architectures and their accuracies

## Data Flow

```
Controller LSTM → sample arch_spec → build_child_net → train on MNIST → val accuracy (reward)
                                                                              ↓
                                                    REINFORCE: loss = -advantage * log_prob
                                                                              ↓
                                                         Update controller weights
```

## Key Shapes

- Controller input: embedding [1, 16] → LSTM hidden [1, 1, 64]
- Controller output: logits [1, num_choices] per action
- Child input: [B, 1, 28, 28] (MNIST)
- Child output: [B, 10] (10 classes)

## Deliberate Simplifications vs Full Paper

| Aspect | Paper | This Notebook |
|--------|-------|---------------|
| Search space | 5+ hyperparameters per layer, skip connections | 4 per layer, no skip connections |
| Dataset | CIFAR-10 | MNIST (faster training) |
| Child epochs | Full training (hundreds of epochs) | 1-2 epochs |
| Controller layers | 2-layer LSTM, 100 hidden | 1-layer LSTM, 64 hidden |
| Parallelism | Asynchronous parameter server | Synchronous, single GPU |
| Number of children | ~8000 | ~15 |
| Skip connections | Anchor-point prediction | Omitted |
| RNN cell search | Also searches RNN cells | CNN search only |
