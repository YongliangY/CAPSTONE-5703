# file: models/baseline_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=[128, 64, 32], num_classes=4, dropout_rate=0.4):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, num_classes))
        self.model = nn.Sequential(*layers)


    def forward(self, x):
        # 前向传播: 依次通过所有层 (Forward through all layers defined above)
        return self.model(x)  # 返回logits (Return logits for each class)
