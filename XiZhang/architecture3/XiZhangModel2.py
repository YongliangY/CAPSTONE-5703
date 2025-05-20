import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import seaborn as sns

# 导入自定义模块 (Import custom modules)
from data_loader import RCWallDataset
from models import BaselineMLP, SimpleNet, ProtoNet, BayesianNN
from losses import WeightedFocalLoss
from meta_learner import MetaLearner
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# 设置随机种子确保可重复性 (Set random seed for reproducibility)
torch.manual_seed(42)
np.random.seed(42)

# ===================== 1. 数据集加载与划分 =====================
# 加载训练数据并拆分为基模型训练集和元学习训练集 (Load training data and split into base-model train set and meta-learner train set)
data = np.genfromtxt('train_smote_pca.csv', delimiter=',', skip_header=1)
X_all = data[:, :-1].astype(np.float32)
y_all = data[:, -1].astype(np.int64)
# 将标签从 1-4 转换为 0-3 (Convert labels 1-4 to 0-3 for PyTorch)
y_all -= 1

# 使用 80% 数据训练基模型，其余 20% 用于训练元学习器 (Use 80% of data to train base models, 20% for training the meta-learner)
X_base, X_meta, y_base, y_meta = train_test_split(X_all, y_all, test_size=0.2, stratify=y_all, random_state=42)
print(f"Total training samples: {len(X_all)}, Base training: {len(X_base)}, Meta training: {len(X_meta)}")

# 构建数据集和 DataLoader (Build datasets and data loaders)
base_dataset = RCWallDataset(X_base, y_base, augment=True)    # 基模型训练集 (with augmentation)
meta_dataset = RCWallDataset(X_meta, y_meta, augment=False)   # 元学习训练集 (no augmentation)
batch_size = 32
train_loader = DataLoader(base_dataset, batch_size=batch_size, shuffle=True)

# ===================== 2. 定义加权 Focal Loss =====================
# 定义加权 Focal Loss (将原始类别 2 的权重调高以增强其识别)
# (Define weighted Focal Loss with higher weight for original Class 2 to improve its detection)
criterion = WeightedFocalLoss(class_weights=[1.0, 2.0, 1.0, 1.0], gamma=2.0)

# ===================== 3. 初始化模型和优化器 =====================
input_dim = X_base.shape[1]             # 输入维度 (Input feature dimension, e.g. 21)
output_dim = len(np.unique(y_all))      # 输出类别数 (Number of output classes, e.g. 4)
hidden_dim = 64
embed_dim = 64                          # ProtoNet 嵌入维度 (Embedding dimension for ProtoNet)
# 实例化各基模型 (Instantiate each base model)
baseline_model = BaselineMLP(input_dim, hidden_dim, output_dim)
simple_model   = SimpleNet(input_dim, hidden_dim, output_dim)
proto_model    = ProtoNet(input_dim, hidden_dim, embed_dim, output_dim)
bayes_model    = BayesianNN(input_dim, hidden_dim, output_dim)
# 定义优化器 (Define an optimizer for each model)
learning_rate = 0.001
baseline_opt = torch.optim.Adam(baseline_model.parameters(), lr=learning_rate)
simple_opt   = torch.optim.Adam(simple_model.parameters(), lr=learning_rate)
proto_opt    = torch.optim.Adam(proto_model.parameters(), lr=learning_rate)
bayes_opt    = torch.optim.Adam(bayes_model.parameters(), lr=learning_rate)

# ===================== 4. 训练基模型 =====================
num_epochs = 200  # 增加训练轮数以充分学习少数类 (Increase epochs to fully learn minority classes)
for epoch in range(num_epochs):
    for X_batch, y_batch in train_loader:
        # Baseline MLP 模型训练 (Train Baseline MLP model)
        baseline_opt.zero_grad()
        logits = baseline_model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        baseline_opt.step()
        # SimpleNet 模型训练 (Train SimpleNet model)
        simple_opt.zero_grad()
        logits = simple_model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        simple_opt.step()
        # ProtoNet 模型训练 (Train ProtoNet model)
        proto_opt.zero_grad()
        logits = proto_model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        proto_opt.step()
        # BayesianNN 模型训练 (Train BayesianNN model)
        bayes_opt.zero_grad()
        logits_bnn, kl = bayes_model(X_batch)            # BNN forward 返回 (logits, KL散度)
        base_loss = criterion(logits_bnn, y_batch)
        # 损失 = Focal Loss + KL 正则项 (Loss = focal loss + scaled KL divergence term)
        loss_bnn = base_loss + 0.01 * kl / len(X_all)    # KL 项按总样本数缩放并乘以权重系数 0.01
        loss_bnn.backward()
        bayes_opt.step()
    # 可选: 每轮结束时监控训练情况 (Optional: monitor training progress per epoch)
    # print(f"Epoch {epoch+1}/{num_epochs} completed.")

print("Base models training complete.")

# 切换基模型到评估模式 (Switch base models to evaluation mode)
baseline_model.eval()
simple_model.eval()
proto_model.eval()
bayes_model.eval()

# # ===================== 5. 训练元学习器 =====================
# # 利用训练好的基模型，在元学习训练集上生成训练特征 (Use trained base models to generate features on meta-learning train set)
# X_meta_tensor = torch.tensor(X_meta, dtype=torch.float32)
# with torch.no_grad():
#     # 获得每个基模型对元学习集的预测 (Get predictions of each base model on the meta set)
#     baseline_logits_meta = baseline_model(X_meta_tensor)
#     simple_logits_meta   = simple_model(X_meta_tensor)
#     proto_logits_meta    = proto_model(X_meta_tensor)
#     # BayesianNN: 多次采样取平均以获得稳定预测 (BayesianNN: sample multiple times and average for stable predictions)
#     T = 20  # 抽样次数 (number of stochastic forward passes)
#     bayes_probs_meta = torch.zeros((len(X_meta), output_dim))
#     for t in range(T):
#         logits_sample, _ = bayes_model(X_meta_tensor)
#         prob_sample = F.softmax(logits_sample, dim=1)
#         bayes_probs_meta += prob_sample
#     bayes_probs_meta /= T
#     # 将贝叶斯平均概率转为 logits，用于与其他模型结果一致 (Convert averaged BNN probability to logits for consistency)
#     bayes_logits_meta = torch.log(bayes_probs_meta)
#     # 计算其他模型的预测概率分布 (Compute probability distributions for other models)
#     baseline_probs_meta = F.softmax(baseline_logits_meta, dim=1)
#     simple_probs_meta   = F.softmax(simple_logits_meta, dim=1)
#     proto_probs_meta    = F.softmax(proto_logits_meta, dim=1)
#     # Concatenate all model probabilities to form meta-features
#     meta_features = torch.cat([baseline_probs_meta, simple_probs_meta, proto_probs_meta, bayes_probs_meta], dim=1)
# meta_labels = torch.tensor(y_meta, dtype=torch.long)
#
# # 初始化元学习器模型 (Initialize the meta-learner model)
# meta_input_dim = 4 * output_dim  # 4 个基模型 * 每个输出的类别数 (4 base models * number of classes)
# meta_model = MetaLearner(input_dim=meta_input_dim, hidden_dims=[64, 32], output_dim=output_dim)
# meta_opt = torch.optim.Adam(meta_model.parameters(), lr=learning_rate)
# # 构建元学习训练数据加载器 (Build DataLoader for meta-learner training data)
# meta_dataset_tensor = TensorDataset(meta_features, meta_labels)
# meta_loader = DataLoader(meta_dataset_tensor, batch_size=batch_size, shuffle=True)
#
# # 训练元学习器若干轮次 (Train meta-learner for several epochs)
# meta_epochs = 100  # 训练轮数 (Train for a sufficient number of epochs)
# best_meta_loss = float('inf')
# best_state_dict = None
# for epoch in range(meta_epochs):
#     meta_model.train()
#     running_loss = 0.0
#     for meta_X_batch, meta_y_batch in meta_loader:
#         meta_opt.zero_grad()
#         meta_logits = meta_model(meta_X_batch)
#         loss_meta = criterion(meta_logits, meta_y_batch)  # 使用同样的加权 Focal Loss (use the same weighted focal loss)
#         loss_meta.backward()
#         meta_opt.step()
#         running_loss += loss_meta.item() * meta_X_batch.size(0)
#     avg_loss = running_loss / len(meta_dataset_tensor)
#     # 保存当前最佳模型参数 (Save parameters if current model has lowest loss)
#     if avg_loss < best_meta_loss:
#         best_meta_loss = avg_loss
#         best_state_dict = meta_model.state_dict()
#     # 可选: 打印每轮元学习器训练损失 (Optional: print loss each epoch)
#     # print(f"Meta Epoch {epoch+1}/{meta_epochs}, Loss: {avg_loss:.4f}")
#
# # 加载最佳元学习器参数并保存模型 (Load best meta-learner parameters and save model)
# if best_state_dict is not None:
#     meta_model.load_state_dict(best_state_dict)
# torch.save(meta_model.state_dict(), 'best_meta_model.pth')
# print(f"Meta-learner training complete. Best training loss: {best_meta_loss:.4f}")
#
# # ===================== 6. 测试集评估及集成预测 =====================
# # 加载测试数据 (Load test data)
# data_test = np.genfromtxt('test_smote_pca.csv', delimiter=',', skip_header=1)
# X_test = data_test[:, :-1].astype(np.float32)
# y_test = data_test[:, -1].astype(np.int64)
# y_test -= 1  # 将标签从 1-4 转换为 0-3 (Convert labels 1-4 to 0-3)
# X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
# y_test_tensor = torch.tensor(y_test, dtype=torch.long)
# print(f"Test samples: {len(X_test)}, Feature dimension: {X_test.shape[1]}")
#
# # 基模型在测试集上的预测 (Base model predictions on the test set)
# baseline_model.eval()
# simple_model.eval()
# proto_model.eval()
# bayes_model.eval()
# with torch.no_grad():
#     baseline_logits = baseline_model(X_test_tensor)
#     simple_logits   = simple_model(X_test_tensor)
#     proto_logits    = proto_model(X_test_tensor)
#     # BayesianNN: 多次采样取平均预测 (BNN: sample multiple times and average predictions)
#     T = 20
#     bayes_probs = torch.zeros((len(X_test), output_dim))
#     for t in range(T):
#         logits_sample, _ = bayes_model(X_test_tensor)
#         prob_sample = F.softmax(logits_sample, dim=1)
#         bayes_probs += prob_sample
#     bayes_probs /= T
#     bayes_logits = torch.log(bayes_probs)
#     # 计算每个模型的概率分布 (Compute probability distribution for each model)
#     baseline_probs = F.softmax(baseline_logits, dim=1)
#     simple_probs   = F.softmax(simple_logits, dim=1)
#     proto_probs    = F.softmax(proto_logits, dim=1)
#     # 将所有模型概率拼接为元特征 (Concatenate all model probabilities as meta-features)
#     ensemble_features = torch.cat([baseline_probs, simple_probs, proto_probs, bayes_probs], dim=1)
#
# # 加载训练好的元学习模型并进行最终预测 (Load trained meta-learner and perform final prediction)
# best_meta_model = MetaLearner(input_dim=meta_input_dim, hidden_dims=[64, 32], output_dim=output_dim)
# best_meta_model.load_state_dict(torch.load('best_meta_model.pth'))
# best_meta_model.eval()
# with torch.no_grad():
#     meta_logits_test = best_meta_model(ensemble_features)
#     meta_probs_test = F.softmax(meta_logits_test, dim=1)
#     ensemble_pred = torch.argmax(meta_probs_test, dim=1).numpy()
#
# # 计算并输出性能指标 (Compute and output performance metrics)
# conf_mat = confusion_matrix(y_test, ensemble_pred)
# class2_index = 1  # 如果原始标签 2 对应 Class 2，则索引为 1 (If original label "2" is Class 2, its index after subtracting 1 is 1)
# TP = conf_mat[class2_index, class2_index]
# FN = conf_mat[class2_index].sum() - TP
# FP = conf_mat[:, class2_index].sum() - TP
# precision2 = TP / (TP + FP) if TP + FP > 0 else 0.0
# recall2 = TP / (TP + FN) if TP + FN > 0 else 0.0
# f1_2 = 2 * precision2 * recall2 / (precision2 + recall2) if precision2 + recall2 > 0 else 0.0
# print(f"Class 2 Precision: {precision2:.3f}, Recall: {recall2:.3f}, F1-score: {f1_2:.3f}")
# print(classification_report(y_test, ensemble_pred, target_names=['1','2','3','4'], digits=3))
#
# # 绘制混淆矩阵热力图 (Plot confusion matrix heatmap)
# plt.figure(figsize=(6, 5))
# sns.heatmap(conf_mat, annot=True, fmt="d", cmap="Blues",
#             xticklabels=[1, 2, 3, 4], yticklabels=[1, 2, 3, 4])
# plt.title('Confusion Matrix Heatmap', fontsize=14)
# plt.xlabel('Predicted Class', fontsize=12)
# plt.ylabel('True Class', fontsize=12)
# plt.tight_layout()
# plt.show()


# ===================== 5. 训练元学习器 =====================
# 利用训练好的基模型，在元学习训练集上生成训练特征 (Use trained base models to generate features on meta-learning train set)
X_meta_tensor = torch.tensor(X_meta, dtype=torch.float32)
with torch.no_grad():
    # 获得每个基模型对元学习集的预测 (Get predictions of each base model on the meta set)
    baseline_logits_meta = baseline_model(X_meta_tensor)
    simple_logits_meta   = simple_model(X_meta_tensor)
    proto_logits_meta    = proto_model(X_meta_tensor)
    # BayesianNN: 多次采样取平均以获得稳定预测 (BayesianNN: sample multiple times and average for stable predictions)
    T = 20  # 抽样次数 (number of stochastic forward passes)
    bayes_probs_meta = torch.zeros((len(X_meta), output_dim))
    for t in range(T):
        logits_sample, _ = bayes_model(X_meta_tensor)
        prob_sample = F.softmax(logits_sample, dim=1)
        bayes_probs_meta += prob_sample
    bayes_probs_meta /= T
    # 将贝叶斯平均概率转为 logits，用于与其他模型结果一致 (Convert averaged BNN probability to logits for consistency)
    bayes_logits_meta = torch.log(bayes_probs_meta)
    # 计算其他模型的预测概率分布 (Compute probability distributions for other models)
    baseline_probs_meta = F.softmax(baseline_logits_meta, dim=1)
    simple_probs_meta   = F.softmax(simple_logits_meta, dim=1)
    proto_probs_meta    = F.softmax(proto_logits_meta, dim=1)
    # Concatenate all model probabilities to form meta-features
    meta_features = torch.cat([baseline_probs_meta, simple_probs_meta, proto_probs_meta, bayes_probs_meta], dim=1)
meta_labels = torch.tensor(y_meta, dtype=torch.long)

# 初始化元学习器模型 (Initialize the meta-learner model)
meta_input_dim = 4 * output_dim  # 4 个基模型 * 每个输出的类别数 (4 base models * number of classes)
meta_model = MetaLearner(input_dim=meta_input_dim, hidden_dims=[64, 32], output_dim=output_dim)
meta_opt = torch.optim.Adam(meta_model.parameters(), lr=learning_rate)
# 构建元学习训练数据加载器 (Build DataLoader for meta-learner training data)
meta_dataset_tensor = TensorDataset(meta_features, meta_labels)
meta_loader = DataLoader(meta_dataset_tensor, batch_size=batch_size, shuffle=True)

# 训练元学习器若干轮次 (Train meta-learner for several epochs)
meta_epochs = 100  # 训练轮数 (Train for a sufficient number of epochs)
best_meta_loss = float('inf')
best_state_dict = None
# 初始化记录列表：训练损失和准确率 (Initialize lists to record training loss and accuracy)
train_loss_history = []
train_acc_history = []
for epoch in range(meta_epochs):
    meta_model.train()
    running_loss = 0.0
    running_corrects = 0  # 初始化本轮正确预测计数 (Initialize correct prediction count for this epoch)
    for meta_X_batch, meta_y_batch in meta_loader:
        meta_opt.zero_grad()
        meta_logits = meta_model(meta_X_batch)
        loss_meta = criterion(meta_logits, meta_y_batch)  # 使用同样的加权 Focal Loss (use the same weighted focal loss)
        preds = torch.argmax(meta_logits, dim=1)
        running_corrects += (preds == meta_y_batch).sum().item()
        loss_meta.backward()
        meta_opt.step()
        running_loss += loss_meta.item() * meta_X_batch.size(0)
    avg_loss = running_loss / len(meta_dataset_tensor)
    epoch_acc = running_corrects / len(meta_dataset_tensor)
    train_loss_history.append(avg_loss)
    train_acc_history.append(epoch_acc)
    # 保存当前最佳模型参数 (Save parameters if current model has lowest loss)
    if avg_loss < best_meta_loss:
        best_meta_loss = avg_loss
        best_state_dict = meta_model.state_dict()
    # 可选: 打印每轮元学习器训练损失和准确率 (Optional: print training loss and accuracy each epoch)
    # print(f"Meta Epoch {epoch+1}/{meta_epochs}, Loss: {avg_loss:.4f}, Acc: {epoch_acc:.4f}")

# 加载最佳元学习器参数并保存模型 (Load best meta-learner parameters and save model)
if best_state_dict is not None:
    meta_model.load_state_dict(best_state_dict)
torch.save(meta_model.state_dict(), 'best_meta_model.pth')
print(f"Meta-learner training complete. Best training loss: {best_meta_loss:.4f}")

# ===================== 6. 测试集评估及集成预测 =====================
# 加载测试数据 (Load test data)
data_test = np.genfromtxt('test_smote_pca.csv', delimiter=',', skip_header=1)
X_test = data_test[:, :-1].astype(np.float32)
y_test = data_test[:, -1].astype(np.int64)
y_test -= 1  # 将标签从 1-4 转换为 0-3 (Convert labels 1-4 to 0-3)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)
print(f"Test samples: {len(X_test)}, Feature dimension: {X_test.shape[1]}")

# 基模型在测试集上的预测 (Base model predictions on the test set)
baseline_model.eval()
simple_model.eval()
proto_model.eval()
bayes_model.eval()
with torch.no_grad():
    baseline_logits = baseline_model(X_test_tensor)
    simple_logits   = simple_model(X_test_tensor)
    proto_logits    = proto_model(X_test_tensor)
    # BayesianNN: 多次采样取平均预测 (BNN: sample multiple times and average predictions)
    T = 20
    bayes_probs = torch.zeros((len(X_test), output_dim))
    for t in range(T):
        logits_sample, _ = bayes_model(X_test_tensor)
        prob_sample = F.softmax(logits_sample, dim=1)
        bayes_probs += prob_sample
    bayes_probs /= T
    bayes_logits = torch.log(bayes_probs)
    # 计算每个模型的概率分布 (Compute probability distribution for each model)
    baseline_probs = F.softmax(baseline_logits, dim=1)
    simple_probs   = F.softmax(simple_logits, dim=1)
    proto_probs    = F.softmax(proto_logits, dim=1)
    # 将所有模型概率拼接为元特征 (Concatenate all model probabilities as meta-features)
    ensemble_features = torch.cat([baseline_probs, simple_probs, proto_probs, bayes_probs], dim=1)

# 加载训练好的元学习模型并进行最终预测 (Load trained meta-learner and perform final prediction)
best_meta_model = MetaLearner(input_dim=meta_input_dim, hidden_dims=[64, 32], output_dim=output_dim)
best_meta_model.load_state_dict(torch.load('best_meta_model.pth'))
best_meta_model.eval()
with torch.no_grad():
    meta_logits_test = best_meta_model(ensemble_features)
    meta_probs_test = F.softmax(meta_logits_test, dim=1)
    ensemble_pred = torch.argmax(meta_probs_test, dim=1).numpy()

# 计算并输出性能指标 (Compute and output performance metrics)
conf_mat = confusion_matrix(y_test, ensemble_pred)
class2_index = 1  # 如果原始标签 2 对应 Class 2，则索引为 1 (If original label "2" is Class 2, its index after subtracting 1 is 1)
TP = conf_mat[class2_index, class2_index]
FN = conf_mat[class2_index].sum() - TP
FP = conf_mat[:, class2_index].sum() - TP
precision2 = TP / (TP + FP) if TP + FP > 0 else 0.0
recall2 = TP / (TP + FN) if TP + FN > 0 else 0.0
f1_2 = 2 * precision2 * recall2 / (precision2 + recall2) if precision2 + recall2 > 0 else 0.0
print(f"Class 2 Precision: {precision2:.3f}, Recall: {recall2:.3f}, F1-score: {f1_2:.3f}")
print(classification_report(y_test, ensemble_pred, target_names=['1','2','3','4'], digits=3))

# 绘制混淆矩阵热力图 (Plot confusion matrix heatmap)
plt.figure(figsize=(6, 5))
sns.heatmap(conf_mat, annot=True, fmt="d", cmap="Blues",
            xticklabels=[1, 2, 3, 4], yticklabels=[1, 2, 3, 4])
plt.title('Confusion Matrix Heatmap', fontsize=14)
plt.xlabel('Predicted Class', fontsize=12)
plt.ylabel('True Class', fontsize=12)
plt.tight_layout()
plt.show()

# 绘制训练损失曲线 (Plot training loss curve)
# 该曲线显示随训练轮次增加，元学习器的训练损失如何变化 (This curve shows how the meta-learner's training loss changes with epochs)
# 损失越低表示模型预测误差越小，损失曲线下降表明模型性能在改善 (Lower loss means smaller prediction error, and a descending loss curve indicates improving model performance)
plt.figure()
plt.plot(train_loss_history, label='Training Loss')
plt.title('Meta-Learner Training Loss Curve')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.show()

# 绘制训练准确率曲线 (Plot training accuracy curve)
# 该曲线显示随训练轮次增加，元学习器的训练准确率变化 (This curve shows how the meta-learner's training accuracy changes with epochs)
# 准确率越接近1.0（100%）表示模型在训练集上预测越准确，上升的准确率曲线表明模型性能在提高 (Accuracy closer to 1.0 (100%) means more accurate predictions on the training set; an upward accuracy curve indicates improving model performance)
plt.figure()
plt.plot(train_acc_history, label='Training Accuracy')
plt.title('Meta-Learner Training Accuracy Curve')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.tight_layout()
plt.show()

