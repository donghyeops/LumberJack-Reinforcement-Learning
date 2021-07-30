from collections import deque
import random

import numpy as np
import torch.nn as nn


class Memory:
    def __init__(self, q_size=128):
        self.q_size = q_size
        self.q = deque()

    def __len__(self):
        return len(self.q)

    def push(self, img, action, next_state, reward):
        if len(self.q) > self.q_size:
            self.q.popleft()
        self.q.append((img, action, next_state, reward))

    def sample(self, batch_size):
        batch = random.sample(self.q, batch_size)
        return np.array(batch)


class DQN(nn.Module):
    def __init__(self, n_actions=2):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=8, stride=4),
            nn.BatchNorm2d(8),
            nn.Conv2d(8, 16, kernel_size=4, stride=2),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 16, kernel_size=3, stride=1),
            nn.BatchNorm2d(16)
        )
        self.mlp = nn.Sequential(
            nn.Linear(1344, 32),
            nn.ReLU(),
            nn.Linear(32, n_actions)
        )

    def forward(self, x):
        x = self.cnn(x)
        x = self.mlp(x.view(x.size(0), -1))
        return x
