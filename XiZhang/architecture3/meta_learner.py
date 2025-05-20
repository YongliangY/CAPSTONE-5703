import torch
import torch.nn as nn
import torch.nn.functional as F

class MetaLearner(nn.Module):
    """
    MetaLearner 模型：融合多个子模型的预测特征以进行最终分类
    (Meta-learner model: fuses predictions from multiple base models for final classification)
    """
    def __init__(self, input_dim=16, hidden_dims=[64, 32], output_dim=4):
        """
        input_dim: 元模型输入维度 (例如 4 个基模型 * 每个模型输出类别数)
                   (Meta-learner input dimension, e.g. 4 base models * number of classes)
        hidden_dims: 隐藏层维度列表 (List of hidden layer dimensions)
        output_dim: 输出类别数 (Number of output classes)
        """
        super().__init__()
        # 定义前馈全连接层序列 (Define feed-forward fully connected layers sequence)
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
        # 前向传播: 通过全连接层序列 (Forward pass through the sequential layers)
        return self.fc_layers(x)
