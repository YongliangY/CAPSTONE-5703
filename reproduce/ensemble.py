"""
神经网络集成方案（以五层神经网络为基础模型）
支持策略：模型平均、多数投票
"""

import numpy as np
import pandas as pd
import torch
from torch import nn
from model import FailurePredictor  # 导入基础模型
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from torch.utils.data import DataLoader, TensorDataset



class EnsembleModel:
    def __init__(self, n_models=5, device='cpu'):
        """
        初始化集成模型
        :param n_models: 基础模型数量（建议奇数）
        :param device: 计算设备
        """
        self.n_models = n_models
        self.device = device
        self.models = [FailurePredictor(input_size=X_train.shape[1], num_classes=num_classes).to(device) for _ in range(n_models)]  # 创建多个独立模型

    def train_models(self, X_train, y_train, epochs=100, batch_size=32):
        """
        并行训练所有基础模型
        """
        criterion = nn.CrossEntropyLoss()

        for i, model in enumerate(self.models):
            print(f"Training model {i + 1}/{self.n_models}")
            optimizer = torch.optim.Adam(model.parameters())

            # 转换为PyTorch Dataset
            dataset = torch.utils.data.TensorDataset(
                torch.FloatTensor(X_train),
                torch.LongTensor(y_train)
            )
            loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

            # 训练单个模型
            model.train()
            for epoch in range(epochs):
                for X_batch, y_batch in loader:
                    X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                    optimizer.zero_grad()
                    outputs = model(X_batch)
                    loss = criterion(outputs, y_batch)
                    loss.backward()
                    optimizer.step()

    def predict(self, X_test, strategy='average'):
        """
        集成预测
        :param strategy: 集成策略 ('average'概率平均 / 'vote'多数投票)
        """
        all_preds = []
        with torch.no_grad():
            for model in self.models:
                model.eval()
                outputs = model(torch.FloatTensor(X_test).to(self.device))
                if strategy == 'average':
                    all_preds.append(torch.softmax(outputs, dim=1).cpu().numpy())
                elif strategy == 'vote':
                    all_preds.append(torch.argmax(outputs, dim=1).cpu().numpy())

        if strategy == 'average':
            avg_probs = np.mean(all_preds, axis=0)
            return np.argmax(avg_probs, axis=1)
        elif strategy == 'vote':
            return np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=np.array(all_preds))

    def evaluate_base_models(self, X_test, y_test):
        """评估所有基础模型"""
        base_f1_scores = []
        y_test = y_test.cpu().numpy()

        for i, model in enumerate(self.models):
            model.eval()
            with torch.no_grad():
                outputs = model(X_test.to(self.device))
                y_pred = torch.argmax(outputs, dim=1).cpu().numpy()
                f1 = f1_score(y_test, y_pred, average='macro')
                base_f1_scores.append(f1)
        return base_f1_scores

def prepare_data(train_path, test_path):
    # 读取数据
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # 特征和标签分离
    X_train = train_df.drop('failure_mode', axis=1).values
    y_train = train_df['failure_mode'].values
    X_test = test_df.drop('failure_mode', axis=1).values
    y_test = test_df['failure_mode'].values

    # 标准化
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # 标签编码
    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    y_test = le.transform(y_test)

    # 转换为Tensor
    X_train = torch.FloatTensor(X_train)
    y_train = torch.LongTensor(y_train)
    X_test = torch.FloatTensor(X_test)
    y_test = torch.LongTensor(y_test)

    # 添加数据校验
    def check_tensor(tensor, name):
        if torch.isnan(tensor).any():
            raise ValueError(f"{name} 包含NaN值")
        if torch.isinf(tensor).any():
            raise ValueError(f"{name} 包含无穷大值")

    check_tensor(X_train, "训练数据")
    check_tensor(X_test, "测试数据")
    check_tensor(y_train, "训练标签")
    check_tensor(y_test, "测试标签")

    return X_train, y_train, X_test, y_test, len(le.classes_)

if __name__ == "__main__":
    # 数据准备
    X_train, y_train, X_test, y_test, num_classes = prepare_data('train.csv', 'test.csv')
    y_test_np = y_test.numpy()

    # 创建集成模型
    ensemble = EnsembleModel(n_models=5, device='cuda' if torch.cuda.is_available() else 'cpu')

    # 训练模型
    ensemble.train_models(X_train, y_train, epochs=100)

    # 评估基础模型
    base_f1_scores = ensemble.evaluate_base_models(X_test, y_test)
    print("\nBase Models Performance:")
    for i, f1 in enumerate(base_f1_scores):
        print(f"Model {i+1}: F1 = {f1:.4f}")
    print(f"Average F1: {np.mean(base_f1_scores):.4f}")

    # 评估集成模型
    print("\nEnsemble Performance:")
    for strategy in ['average', 'vote']:
        y_pred = ensemble.predict(X_test, strategy=strategy)
        acc = accuracy_score(y_test_np, y_pred)
        f1 = f1_score(y_test_np, y_pred, average='macro')
        print(f"{strategy.upper()}策略: 准确率={acc:.4f}, F1分数={f1:.4f}")