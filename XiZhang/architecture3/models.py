import torch
import torch.nn as nn
import torch.nn.functional as F


class BayesLinear(nn.Module):
    """
    自定义 Bayesian Linear 层：权重和偏置服从可学习的高斯分布
    (Custom Bayesian Linear layer: weights and biases are learned as Gaussian distributions)
    """

    def __init__(self, in_features, out_features):
        super(BayesLinear, self).__init__()
        # 参数初始化: mu 初始化为0，rho 初始化为较小的负值 (Parameter init: mu = 0, rho to a small negative for small initial σ)
        self.weight_mu = nn.Parameter(torch.zeros(out_features, in_features))
        self.weight_rho = nn.Parameter(torch.full((out_features, in_features), -4.0))
        self.bias_mu = nn.Parameter(torch.zeros(out_features))
        self.bias_rho = nn.Parameter(torch.full((out_features,), -4.0))

    def forward(self, input):
        # 将 rho 通过 softplus 转换为正的标准差 σ (Convert rho to positive σ via softplus)
        weight_sigma = F.softplus(self.weight_rho)
        bias_sigma = F.softplus(self.bias_rho)
        # 随机采样权重和偏置 (重参数技巧) (Randomly sample weight and bias using reparameterization trick)
        eps_w = torch.randn_like(weight_sigma)
        eps_b = torch.randn_like(bias_sigma)
        weight = self.weight_mu + weight_sigma * eps_w
        bias = self.bias_mu + bias_sigma * eps_b
        # 线性变换 (Linear transformation)
        output = F.linear(input, weight, bias)
        # 计算当前层参数的 KL 散度 (Compute KL divergence for this layer)
        # KL = Σ[ -log(sigma) + 0.5*(σ^2 + μ^2) - 0.5 ]
        kl_weight = torch.sum(-torch.log(weight_sigma) + 0.5 * (weight_sigma ** 2 + self.weight_mu ** 2) - 0.5)
        kl_bias = torch.sum(-torch.log(bias_sigma) + 0.5 * (bias_sigma ** 2 + self.bias_mu ** 2) - 0.5)
        kl = kl_weight + kl_bias
        # 返回输出和 KL 散度 (Return output and KL divergence)
        return output, kl


class BaselineMLP(nn.Module):
    """
    Baseline MLP 模型：两层全连接网络，带残差跳连
    (Baseline MLP model: two-layer fully connected network with a residual skip connection)
    """

    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate=0.3):
        """
        input_dim: 输入维度 (例如 PCA 特征数 21)
                   (Input dimension, e.g. 21 features after PCA)
        hidden_dim: 隐藏层维度 (Hidden layer dimension)
        output_dim: 输出类别数 (例如 4 类)
                    (Number of output classes, e.g. 4)
        dropout_rate: 随机失活率 (Dropout rate)
        """
        super(BaselineMLP, self).__init__()
        # 第一隐藏层 (First hidden layer)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)  # 层归一化确保稳定训练 (LayerNorm for stable training)
        self.dropout = nn.Dropout(dropout_rate)  # Dropout 正则化防止过拟合 (Dropout regularization to prevent overfitting)
        # 输出层 (Output layer)
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        # 跳跃连接: 输入直接映射到输出的线性层 (Skip connection: linear mapping from input directly to output)
        self.fc_skip = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        # 隐藏层前向传播 (Hidden layer forward pass)
        h = self.fc1(x)
        h = self.norm1(h)
        h = F.leaky_relu(h, negative_slope=0.1)  # LeakyReLU 激活，避免死神经元 (LeakyReLU activation to avoid "dead" neurons)
        h = self.dropout(h)
        # 主路径输出 (Main path output)
        out_main = self.fc_out(h)
        # 跳连路径输出 (Skip connection output)
        out_skip = self.fc_skip(x)
        # 残差相加得到最终输出 (Residual add for final output)
        out = out_main + out_skip  # 残差连接改善梯度流动 (Residual connection improves gradient flow)
        return out


class SimpleNet(nn.Module):
    """
    SimpleNet 模型：两层隐藏层 + 残差块
    (SimpleNet model: two hidden layers with a residual block)
    """

    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate=0.3):
        super(SimpleNet, self).__init__()
        # 第一层 (First layer)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)
        # 第二层 (隐藏维度 -> 隐藏维度，用于残差) (Second layer: hidden_dim -> hidden_dim for residual connection)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        # 输出层 (Output layer)
        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # 第一层前向 (First layer forward)
        h1 = self.fc1(x)
        h1 = self.norm1(h1)
        h1 = F.leaky_relu(h1, 0.1)
        h1 = self.dropout(h1)
        # 第二层前向 (未激活，用于残差相加) (Second layer forward, not activated yet for residual add)
        h2 = self.fc2(h1)
        h2 = self.norm2(h2)
        # 残差相加并激活 (Add residual and then activate)
        h_res = h2 + h1  # 将第一层输出直接加到第二层输出 (add first layer output to second layer output)
        h_res = F.leaky_relu(h_res, 0.1)
        # 输出层 (Output layer)
        out = self.fc_out(h_res)
        return out


class ProtoNet(nn.Module):
    """
    ProtoNet 模型：原型网络
    (ProtoNet model: Prototypical Network)
    """

    def __init__(self, input_dim, hidden_dim, embed_dim, output_dim, dropout_rate=0.3):
        """
        embed_dim: 原型和嵌入空间维度 (Dimension of prototypes and embedding space)
        """
        super(ProtoNet, self).__init__()
        # 嵌入网络两层 (Two-layer embedding network)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        # 每个类的原型向量 (Prototype vector for each class, learnable parameter)
        self.prototypes = nn.Parameter(torch.randn(output_dim, embed_dim))
        # 初始化原型参数为标准正态分布 (Initialize prototypes with standard normal distribution)
        nn.init.normal_(self.prototypes, mean=0.0, std=1.0)

    def forward(self, x):
        # 计算嵌入表示 (Compute embedding representation)
        h = self.fc1(x)
        h = self.norm1(h)
        h = F.leaky_relu(h, 0.1)
        h = self.dropout(h)
        embed = self.fc2(h)
        embed = self.norm2(embed)
        embed = F.leaky_relu(embed, 0.1)
        # 计算嵌入与各类原型的距离 (Compute distance between embedding and each class prototype)
        # 使用欧氏距离: dist^2 = ||embed - prototype||^2 (Use squared Euclidean distance)
        batch_size = embed.size(0)
        num_classes = self.prototypes.size(0)
        # 扩展维度以便向量化计算距离 (Expand dimensions for vectorized computation of distances)
        embed_expand = embed.unsqueeze(1).expand(batch_size, num_classes,
                                                 embed.size(1))  # [batch, num_classes, embed_dim]
        proto_expand = self.prototypes.unsqueeze(0).expand(batch_size, num_classes,
                                                           embed.size(1))  # [batch, num_classes, embed_dim]
        # 计算平方距离 (Compute squared distances)
        dist_sq = torch.sum((embed_expand - proto_expand) ** 2, dim=2)  # [batch, num_classes]
        # 返回负的距离作为 logits (Return negative distance as logits for classification)
        logits = -dist_sq
        return logits
        # 注: 在经典 ProtoNet 中，每次推理需要根据全部训练数据计算原型。
        # 本实现中将原型作为可学习参数在训练中更新，相当于隐式学习每类中心，推理时无需重新计算原型。
        # (Note: In a classic ProtoNet, prototypes are computed from training data for each episode.
        # Here, prototypes are learned parameters updated during training (implicit class centers), so no recomputation is needed at inference.)


class BayesianNN(nn.Module):
    """
    BayesianNN 模型：贝叶斯神经网络
    (Bayesian Neural Network model)
    """

    def __init__(self, input_dim, hidden_dim, output_dim):
        super(BayesianNN, self).__init__()
        # Bayesian 线性层 (Bayesian linear layers)
        self.bayes1 = BayesLinear(input_dim, hidden_dim)
        self.bayes_out = BayesLinear(hidden_dim, output_dim)
        self.bayes_skip = BayesLinear(input_dim, output_dim)  # 残差跳连 Bayes 层 (Bayesian layer for skip connection)

    def forward(self, x):
        # 第一层贝叶斯线性 (First Bayesian linear layer)
        h, kl1 = self.bayes1(x)
        h = F.leaky_relu(h, 0.1)
        # 输出层贝叶斯线性 (Output Bayesian linear layer)
        out_main, kl2 = self.bayes_out(h)
        # 跳连输出层 (Bayesian skip connection from input)
        out_skip, kl3 = self.bayes_skip(x)
        # 残差相加得到最终输出 (Residual add for final output)
        out = out_main + out_skip
        # 累积 KL 散度 (Accumulate total KL divergence)
        total_kl = kl1 + kl2 + kl3
        # 返回输出和 KL 散度 (Return output and total KL divergence)
        return out, total_kl
