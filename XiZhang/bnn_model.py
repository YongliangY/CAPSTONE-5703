# file: models/bnn_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class BayesianLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        # 初始化Bayesian线性层参数 (Initialize Bayesian Linear layer parameters)
        self.in_features = in_features
        self.out_features = out_features
        # 权重参数的均值mu和rho (rho用于计算sigma) (Weight mean and rho for sigma)
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features).normal_(0, 0.1))
        self.weight_rho = nn.Parameter(torch.Tensor(out_features, in_features).uniform_(-5, -4))
        # 偏置参数的均值mu和rho (Bias mean and rho)
        self.bias_mu = nn.Parameter(torch.Tensor(out_features).normal_(0, 0.1))
        self.bias_rho = nn.Parameter(torch.Tensor(out_features).uniform_(-5, -4))

    def forward(self, x, sample=True):
        # 当sample=True时，从分布中采样权重和偏置；否则使用均值 (sample weights if sample=True, else use mean)
        if sample:
            weight_sigma = torch.log1p(torch.exp(self.weight_rho))  # 将rho转换为正的标准差 (rho -> sigma)
            weight = self.weight_mu + weight_sigma * torch.randn_like(weight_sigma)  # 采样权重 (sample weight)
            bias_sigma = torch.log1p(torch.exp(self.bias_rho))
            bias = self.bias_mu + bias_sigma * torch.randn_like(bias_sigma)  # 采样偏置 (sample bias)
        else:
            weight, bias = self.weight_mu, self.bias_mu
        return F.linear(x, weight, bias)

    def kl_loss(self):
        # 计算该层权重和偏置的KL散度 (Compute KL divergence for this layer's weight and bias)
        weight_sigma = torch.log1p(torch.exp(self.weight_rho))
        # KL散度 = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2) (Gaussian KL)&#8203;:contentReference[oaicite:24]{index=24}
        kl = -0.5 * torch.sum(1 + 2 * torch.log(weight_sigma) - self.weight_mu.pow(2) - weight_sigma.pow(2))
        bias_sigma = torch.log1p(torch.exp(self.bias_rho))
        kl += -0.5 * torch.sum(1 + 2 * torch.log(bias_sigma) - self.bias_mu.pow(2) - bias_sigma.pow(2))
        return kl


class BNN(nn.Module):
    """Bayesian Neural Network with configurable structure (贝叶斯神经网络模型)."""

    def __init__(self, input_dim, output_dim, hidden_dims=[64], dropout_rate=0.5):
        super().__init__()
        # 创建指定隐藏层数的BayesianLinear层，并在其后加Dropout (Construct BNN hidden layers)
        self.layers = nn.ModuleList()
        prev_dim = input_dim
        for h_dim in hidden_dims:
            self.layers.append(BayesianLinear(prev_dim, h_dim))
            self.layers.append(nn.Dropout(dropout_rate))
            prev_dim = h_dim
        # 最后一层BayesianLinear输出分类 (Output Bayesian linear layer for classification)
        self.out_layer = BayesianLinear(prev_dim, output_dim)

    def forward(self, x, sample=True):
        # 前向传播: 依次通过BayesianLinear+ReLU+Dropout层，最后输出层 (Forward pass through Bayesian layers)
        for layer in self.layers:
            if isinstance(layer, BayesianLinear):
                x = F.relu(layer(x, sample=sample))
            else:
                x = layer(x)  # Dropout layer
        return self.out_layer(x, sample=sample)  # 输出未经过softmax的logits (raw logits)

    def total_kl_loss(self):
        # 计算所有BayesianLinear层的KL散度总和 (Sum KL loss of all Bayesian layers)
        kl_sum = 0.0
        for layer in self.layers:
            if isinstance(layer, BayesianLinear):
                kl_sum += layer.kl_loss()
        kl_sum += self.out_layer.kl_loss()
        return kl_sum
