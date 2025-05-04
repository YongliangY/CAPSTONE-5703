# file: meta_learner.py
import torch
import torch.nn as nn
import torch.nn.functional as F


# 改进的MetaLearner网络结构
class MetaLearner(nn.Module):
    def __init__(self, input_dim=16, hidden_dims=[64, 32], output_dim=4):
        super().__init__()
        self.fc_layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dims[1], output_dim)
        )

    def forward(self, x):
        return self.fc_layers(x)


