import pandas as pd
import numpy as np
import torch

# 加载训练和测试数据集
train_df = pd.read_csv('train_smote_pca.csv')
test_df  = pd.read_csv('test_smote_pca.csv')

# 将特征和标签分开
X_train = train_df.drop(columns=['failure mode']).values.astype(np.float32)
y_train = train_df['failure mode'].values.astype(int)
X_test  = test_df.drop(columns=['failure mode']).values.astype(np.float32)
y_test  = test_df['failure mode'].values.astype(int)

# 获取类别列表和特征维度
classes = np.unique(y_train)                   # 所有故障模式的类别
input_dim = X_train.shape[1]                   # PCA降维后的特征维度

# 将训练数据按类别分组，便于Few-Shot抽取
train_data_by_class = {cls: X_train[y_train == cls] for cls in classes}
# 转换每个类别的数据为PyTorch张量 (float32类型)
for cls in train_data_by_class:
    train_data_by_class[cls] = torch.tensor(train_data_by_class[cls], dtype=torch.float32)

print(f"Classes: {classes}, total classes = {len(classes)}")
print(f"Train samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")

import torch.nn as nn

# 定义原型网络的嵌入模型
class EmbeddingNet(nn.Module):
    def __init__(self, input_dim, embedding_dim=16):
        super(EmbeddingNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)    # 全连接层1，将输入映射到64维
        self.fc2 = nn.Linear(64, embedding_dim) # 全连接层2，将64维映射到嵌入维度
        self.relu = nn.ReLU()                  # ReLU激活函数

    def forward(self, x):
        # 前向传播: 输入x -> ReLU激活后的隐藏层 -> 输出嵌入向量
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 初始化模型和优化器
embedding_dim = 16
model = EmbeddingNet(input_dim, embedding_dim)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


import torch.nn.functional as F
import random

# Few-Shot训练: 每次迭代随机抽取部分样本形成一个 episode (支持集 + 查询集)
def train_episode(n_support=5, n_query=5):
    # 随机选择本次episode要使用的类别子集（这里我们使用所有类别，也可以随机选部分类别）
    episode_classes = random.sample(list(classes), k=len(classes))
    support_data, query_data = [], []
    support_labels, query_labels = [], []
    # 为每个选定类别采样支持集和查询集
    for cls in episode_classes:
        all_data = train_data_by_class[cls]
        num_samples = all_data.shape[0]
        # 随机选择支持集和查询集索引（不放回抽样）
        idx = np.random.choice(num_samples, size=n_support+n_query, replace=False)
        support_idx = idx[:n_support]   # 支持集 indices
        query_idx   = idx[n_support:]   # 查询集 indices
        # 收集支持集和查询集的样本数据
        support_data.append(all_data[support_idx])
        query_data.append(all_data[query_idx])
        support_labels += [cls] * n_support
        query_labels   += [cls] * n_query

    # 将支持集和查询集数据整合为tensor
    support_data = torch.cat(support_data, dim=0)
    query_data   = torch.cat(query_data, dim=0)
    support_labels = torch.tensor(support_labels, dtype=torch.long)
    query_labels   = torch.tensor(query_labels, dtype=torch.long)

    # 计算支持集样本的嵌入（通过模型）
    support_embeddings = model(support_data)
    # 计算每个类的原型（支持集中该类嵌入的均值向量）
    prototypes = []
    for cls in episode_classes:
        cls_embed = support_embeddings[support_labels == cls]
        prototypes.append(cls_embed.mean(dim=0))
    prototypes = torch.stack(prototypes)  # 张量形状: [num_classes_episode, embedding_dim]

    # 计算查询集样本的嵌入
    query_embeddings = model(query_data)
    # 计算每个查询样本与每个原型之间的欧氏距离
    # dist_matrix 大小: [num_query_samples, num_episode_classes]
    dist_matrix = ((query_embeddings.unsqueeze(1) - prototypes.unsqueeze(0)) ** 2).sum(dim=2)
    # 将距离转换为分类概率: 距离取负通过softmax得到相似度（距离越小概率越大）
    log_p_y = F.log_softmax(-dist_matrix, dim=1)

    # 查询样本的真实类别在当前episode类列表中的索引
    cls_to_index = {cls: idx for idx, cls in enumerate(episode_classes)}
    target_indices = torch.tensor([cls_to_index[cls.item()] for cls in query_labels], dtype=torch.long)
    # 计算损失（负对数似然损失，等价于交叉熵损失）
    loss = F.nll_loss(log_p_y, target_indices)

    # 反向传播并更新模型参数
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()

# 训练模型（进行多次episode迭代）
num_episodes = 10000
for episode in range(1, num_episodes+1):
    loss = train_episode(n_support=5, n_query=5)
    if episode % 100 == 0:
        print(f"Episode {episode}/{num_episodes} - Loss: {loss:.4f}")


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

# 在测试集上评估模型性能
model.eval()  # 切换模型到评估模式
# 1. 计算每个类的原型（使用整个训练集）
prototypes = []
for cls in classes:
    class_embed = model(train_data_by_class[cls])        # 将该类所有训练样本通过模型得到嵌入
    prototypes.append(class_embed.mean(dim=0))           # 该类嵌入的均值向量
prototypes = torch.stack(prototypes)  # [num_classes, embedding_dim]

# 2. 对测试集样本进行预测
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
test_embeddings = model(X_test_tensor)                   # 得到测试样本的嵌入表示
# 计算每个测试样本与各类别原型的距离，并选择最近的原型对应的类别作为预测
dist_matrix_test = ((test_embeddings.unsqueeze(1) - prototypes.unsqueeze(0)) ** 2).sum(dim=2)
nearest_proto_indices = torch.argmin(dist_matrix_test, dim=1).numpy()  # 每个测试样本最近原型的索引
# 索引转换回实际的类别标签
classes_sorted = np.sort(classes)
y_pred = classes_sorted[nearest_proto_indices]

# 3. 计算评估指标: Accuracy, Precision, Recall, F1-score
acc = accuracy_score(y_test, y_pred)
precision_macro = precision_score(y_test, y_pred, average='macro', zero_division=0)
recall_macro    = recall_score(y_test, y_pred, average='macro', zero_division=0)
f1_macro        = f1_score(y_test, y_pred, average='macro', zero_division=0)
print(f"Accuracy: {acc:.4f}")
print(f"Precision (macro-average): {precision_macro:.4f}")
print(f"Recall (macro-average): {recall_macro:.4f}")
print(f"F1-score (macro-average): {f1_macro:.4f}")

# 输出每个类别的详细分类报告 (精确率,召回率,F1,支持数)
print("\nClassification Report:")
print(classification_report(y_test, y_pred, digits=4))

# 4. 混淆矩阵
cm = confusion_matrix(y_test, y_pred, labels=classes_sorted)
print("Confusion Matrix (counts):")
print(cm)


import matplotlib.pyplot as plt
import seaborn as sns

# 绘制混淆矩阵热力图
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=classes_sorted, yticklabels=classes_sorted)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.show()
