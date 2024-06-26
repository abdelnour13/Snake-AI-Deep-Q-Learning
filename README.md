# Snake AI With Deep-Q-Learning

This project implements a Snake AI using the Deep Q-Learning algorithm. The goal is to create an intelligent agent capable of playing the Snake game, making decisions based on its understanding of the game environment to maximize its score.

## Overview

Deep Q-Learning combines Q-Learning, a model-free reinforcement learning algorithm, with deep neural networks. This approach enables the AI to learn optimal policies for deciding the next move in the game, considering the current state of the environment.

## Features

- **Deep Q-Network (DQN):** Utilizes a neural network to approximate the Q-value function.

- **Replay Memory:** Implements experience replay to store and reuse past experiences, improving learning efficiency.

- **Epsilon-Greedy Strategy:** Employs an epsilon-greedy strategy for exploration and exploitation, balancing between taking the best known action and exploring new actions.

## Getting Started

To get started with this project, follow these steps:

1. **Clone the Repository:**


```bash
git clone https://github.com/abdelnour13/Snake-AI-Deep-Q-Learning
```

2. **Install Dependencies:**


Ensure you have Python 3.x installed, then run:

```bash
conda create -n snake-ai
conda activate snake-ai
pip install -r requirements.txt
```

3. **Run the project**:

```bash
    python src/main.py [-h] \
        [--max-memory MAX_MEMORY] [--batch-size BATCH_SIZE] \
        [--learning-rate LEARNING_RATE] [--max-games MAX_GAMES] \
        [--max-epsilon MAX_EPSILON] [--min-epsilon MIN_EPSILON] \
        [--gamma GAMMA] [--training TRAINING] [--human HUMAN]
```

if you use linux and got this error : 

```bash
libGL error: MESA-LOADER: failed to open radeonsi: /usr/lib/dri/radeonsi_dri.so: cannot open shared object file: No such file or directory (search paths /usr/lib/x86_64-linux-gnu/dri:\$${ORIGIN}/dri:/usr/lib/dri, suffix _dri)
libGL error: failed to load driver: radeonsi
libGL error: MESA-LOADER: failed to open radeonsi: /usr/lib/dri/radeonsi_dri.so: cannot open shared object file: No such file or directory (search paths /usr/lib/x86_64-linux-gnu/dri:\$${ORIGIN}/dri:/usr/lib/dri, suffix _dri)
libGL error: failed to load driver: radeonsi
libGL error: MESA-LOADER: failed to open swrast: /usr/lib/dri/swrast_dri.so: cannot open shared object file: No such file or directory (search paths /usr/lib/x86_64-linux-gnu/dri:\$${ORIGIN}/dri:/usr/lib/dri, suffix _dri)
libGL error: failed to load driver: swrast
X Error of failed request:  BadValue (integer parameter out of range for operation)
  Major opcode of failed request:  152 (GLX)
  Minor opcode of failed request:  3 (X_GLXCreateContext)
  Value in failed request:  0x0
  Serial number of failed request:  172
  Current serial number in output stream:  173
```

just run this command and re-start the game : 

```bash
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
```

## How It Works

The AI agent observes the state of the game, which includes the position of the snake, the position of the food, and the obstacles. Based on this information, it decides on the next move that maximizes the expected future rewards.

The Deep Q-Network is trained using the Bellman equation, with the goal of minimizing the difference between the predicted Q-values and the target Q-values.