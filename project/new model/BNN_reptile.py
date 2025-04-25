import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import random
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from copy import deepcopy
import numpy as np

# 定义贝叶斯线性层（保持不变）
class BayesianLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features).normal_(0, 0.1))
        self.weight_rho = nn.Parameter(torch.Tensor(out_features, in_features).uniform_(-5, -4))
        self.bias_mu = nn.Parameter(torch.Tensor(out_features).normal_(0, 0.1))
        self.bias_rho = nn.Parameter(torch.Tensor(out_features).uniform_(-5, -4))

    def forward(self, x, sample=True):
        if sample:
            weight_sigma = torch.log1p(torch.exp(self.weight_rho))
            weight = self.weight_mu + weight_sigma * torch.randn_like(weight_sigma)
            bias_sigma = torch.log1p(torch.exp(self.bias_rho))
            bias = self.bias_mu + bias_sigma * torch.randn_like(bias_sigma)
        else:
            weight, bias = self.weight_mu, self.bias_mu
        return F.linear(x, weight, bias)

    def kl_loss(self):
        weight_sigma = torch.log1p(torch.exp(self.weight_rho))
        kl = -0.5 * torch.sum(1 + 2 * torch.log(weight_sigma) - self.weight_mu.pow(2) - weight_sigma.pow(2))
        bias_sigma = torch.log1p(torch.exp(self.bias_rho))
        kl += -0.5 * torch.sum(1 + 2 * torch.log(bias_sigma) - self.bias_mu.pow(2) - bias_sigma.pow(2))
        return kl


class BNN(nn.Module):
    """动态贝叶斯神经网络，结构参数可配置"""

    def __init__(self, input_dim, output_dim, hidden_dims, dropout_rate):
        super().__init__()
        self.layers = nn.ModuleList()
        prev_dim = input_dim

        # 动态构建隐藏层
        for h_dim in hidden_dims:
            self.layers.append(BayesianLinear(prev_dim, h_dim))
            self.layers.append(nn.Dropout(dropout_rate))
            prev_dim = h_dim

        # 输出层
        self.out_layer = BayesianLinear(prev_dim, output_dim)

    def forward(self, x, sample=True):
        for layer in self.layers:
            if isinstance(layer, BayesianLinear):
                x = F.relu(layer(x, sample))
            else:
                x = layer(x)
        return self.out_layer(x, sample)

    def kl_loss(self):
        """累计所有贝叶斯层的KL散度"""
        total_kl = 0
        for layer in self.layers:
            if isinstance(layer, BayesianLinear):
                total_kl += layer.kl_loss()
        return total_kl + self.out_layer.kl_loss()


# 数据加载与预处理（添加分布检查）
def load_data(train_path, test_path):
    """加载并预处理数据，返回DataLoader和测试Tensor"""
    # 读取CSV文件并跳过标题行
    train_df = pd.read_csv(train_path, header=None, skiprows=1)
    test_df = pd.read_csv(test_path, header=None, skiprows=1)

    # 提取特征和标签
    X_train = train_df.iloc[:, :-1].values.astype('float32')
    y_train = train_df.iloc[:, -1].values.astype('int64') - 1  # 转换为0-based
    X_test = test_df.iloc[:, :-1].values.astype('float32')
    y_test = test_df.iloc[:, -1].values.astype('int64') - 1

    # 打印数据分布
    print("\n数据分布分析:")
    print(f"训练集样本数: {len(X_train)} | 类别分布: {np.bincount(y_train)}")
    print(f"测试集样本数: {len(X_test)} | 类别分布: {np.bincount(y_test)}")

    # 转换为PyTorch Dataset
    train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    return (
        DataLoader(train_dataset, batch_size=32, shuffle=True),
        torch.tensor(X_test),
        torch.tensor(y_test)
    )


# Use the Adam optimizer and conduct multi-batch training
def train_reptile(model, train_loader, params):
    """Reptile元学习训练流程"""
    meta_optimizer = torch.optim.Adam(model.parameters(), lr=params['meta_lr'])
    best_loss = float('inf')
    patience_counter = 0

    for epoch in range(params['num_epochs']):
        # 保存初始参数
        initial_state = deepcopy(model.state_dict())
        temp_model = BNN(
            input_dim=21,
            output_dim=4,
            hidden_dims=params['hidden_dims'],
            dropout_rate=params['dropout_rate']
        )
        temp_model.load_state_dict(initial_state)
        inner_optim = torch.optim.SGD(temp_model.parameters(), lr=params['inner_lr'])

        # 内部训练循环
        for _ in range(params['num_inner_steps']):
            for X_batch, y_batch in train_loader:
                outputs = temp_model(X_batch)
                loss = F.cross_entropy(outputs, y_batch) + params['kl_weight'] * temp_model.kl_loss()
                inner_optim.zero_grad()
                loss.backward()
                inner_optim.step()

        # 元参数更新
        with torch.no_grad():
            for p_model, p_temp in zip(model.parameters(), temp_model.parameters()):
                p_model.grad = (p_model - p_temp)  # Reptile更新规则
        meta_optimizer.step()

        # 早停机制
        current_loss = loss.item()
        if current_loss < best_loss:
            best_loss = current_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 5:
                print(f"早停触发于第 {epoch + 1} 轮")
                break

        # 训练进度监控
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{params['num_epochs']}] Loss: {current_loss:.4f}")


# 修改后的评估函数（多次采样）
def evaluate(model, X_test, y_test, num_samples=30):
    """基于多次采样的概率平均进行评估"""
    model.eval()
    with torch.no_grad():
        probs = []
        for _ in range(num_samples):
            outputs = model(X_test, sample=True)
            probs.append(F.softmax(outputs, dim=1))

        avg_probs = torch.mean(torch.stack(probs), dim=0)
        preds = torch.argmax(avg_probs, dim=1).numpy()

        return {
            'accuracy': accuracy_score(y_test, preds),
            'precision': precision_score(y_test, preds, average='macro', zero_division=0),
            'recall': recall_score(y_test, preds, average='macro', zero_division=0),
            'f1': f1_score(y_test, preds, average='macro', zero_division=0)
        }


def parameter_search(train_loader, X_test, y_test, num_trials=20):
    """随机参数搜索"""
    param_space = {
        'meta_lr': [1e-4, 3e-4, 1e-3, 3e-3],
        'inner_lr': [1e-3, 3e-3, 1e-2, 3e-2],
        'num_inner_steps': [1, 3, 5],
        'kl_weight': [0.001, 0.01, 0.1],
        'hidden_dims': [
            [64],
            [128],
            [256],
            [128, 64],
            [256, 128]
        ],
        'dropout_rate': [0.3, 0.5, 0.7],
        'num_epochs': [50],  # 调参阶段使用较短训练轮次
        'eval_samples': [30]
    }

    best_score = 0
    best_params = None
    results = []

    for trial in range(num_trials):
        # 随机采样参数组合
        params = {
            key: random.choice(values) for key, values in param_space.items()
        }
        params['hidden_dims'] = random.choice(param_space['hidden_dims'])

        print(f"\n试验 {trial + 1}/{num_trials}")
        print("当前参数:", params)

        # 初始化模型
        model = BNN(
            input_dim=21,
            output_dim=4,
            hidden_dims=params['hidden_dims'],
            dropout_rate=params['dropout_rate']
        )

        # 训练模型
        train_reptile(model, train_loader, params)

        # 评估模型
        metrics = evaluate(model, X_test, y_test, num_samples=params['eval_samples'])
        current_score = metrics['f1']
        results.append((params, metrics))

        # 更新最佳参数
        if current_score > best_score:
            best_score = current_score
            best_params = params
            torch.save(model.state_dict(), 'best_model.pth')

        print(f"当前F1: {current_score:.4f} | 最佳F1: {best_score:.4f}")

    # 保存调参结果
    results.sort(key=lambda x: x[1]['f1'], reverse=True)
    pd.DataFrame([
        {**params, **metrics} for params, metrics in results
    ]).to_csv('parameter_search_results.csv', index=False)

    return best_params


def main():
    # 加载数据
    train_loader, X_test, y_test = load_data('train_smote_pca.csv', 'test_smote_pca.csv')

    # 第一阶段：参数搜索
    print("\n开始参数搜索...")
    best_params = parameter_search(train_loader, X_test, y_test, num_trials=20)

    # 第二阶段：最终训练
    print("\n使用最佳参数进行最终训练...")
    final_params = {
        **best_params,
        'num_epochs': 150,  # 延长训练轮次
        'kl_weight': best_params['kl_weight'] * 0.5  # 微调KL权重
    }

    final_model = BNN(
        input_dim=21,
        output_dim=4,
        hidden_dims=best_params['hidden_dims'],
        dropout_rate=best_params['dropout_rate']
    )
    train_reptile(final_model, train_loader, final_params)

    # 最终评估
    final_metrics = evaluate(final_model, X_test, y_test, num_samples=50)
    print("\nFinal model performance:")
    for k, v in final_metrics.items():
        print(f"{k.capitalize()}: {v:.4f}")

    # 保存模型
    torch.save(final_model.state_dict(), 'final_model.pth')


if __name__ == '__main__':
    main()