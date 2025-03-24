import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from torch.utils.data import DataLoader, TensorDataset


# 数据准备
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


# 神经网络模型
class FailurePredictor(nn.Module):
    def __init__(self, input_size, num_classes):
        super(FailurePredictor, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 4),
            nn.ReLU(),
            nn.Linear(4, num_classes)
        )

    def forward(self, x):
        return self.layers(x)


# 训练参数设置
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 0.001


# 准备数据
X_train, y_train, X_test, y_test, num_classes = prepare_data('train.csv', 'test.csv')

# 创建DataLoader
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# 初始化模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FailurePredictor(input_size=X_train.shape[1], num_classes=num_classes).to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# 训练循环
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # 每个epoch打印训练信息
    avg_loss = running_loss / len(train_loader)
    print(f'Epoch [{epoch + 1}/{EPOCHS}] Loss: {avg_loss:.4f}')

# 测试评估
model.eval()
with torch.no_grad():
    test_outputs = model(X_test.to(device))
    _, predicted = torch.max(test_outputs, 1)
    correct = (predicted == y_test.to(device)).sum().item()
    accuracy = correct / y_test.size(0)

print(f'\n测试集准确率: {accuracy:.4f}')

# 保存模型
torch.save(model.state_dict(), 'failure_model_pytorch.pth')