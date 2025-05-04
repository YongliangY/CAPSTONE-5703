# file: train.py
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from bnn_model import BNN
from maml_model import SimpleNet
from proto_model import ProtoNet
from baseline_model import DeepMLP
from meta_learner import MetaLearner
import torch.nn.functional as F

# 设置随机种子以确保可重复性 (Set random seed for reproducibility)
torch.manual_seed(42)
np.random.seed(42)

# 1. 加载训练数据 (Load training data)
data = np.genfromtxt('train_smote_pca.csv', delimiter=',', skip_header=1)
X_train = data[:, :-1].astype(np.float32)
y_train = data[:, -1].astype(np.int64)
# 将标签从1-4转换为0-3 (Convert labels 1-4 to 0-3 for PyTorch)
y_train -= 1

# 转换为Tensor并创建DataLoader (Convert to Tensors and create DataLoader)
X_train_tensor = torch.tensor(X_train)
y_train_tensor = torch.tensor(y_train)
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# 获取输入特征维度和类别数 (Determine feature dimension and number of classes)
input_dim = X_train.shape[1]  # 21 features
num_classes = len(np.unique(y_train))  # should be 4
print(f"Training samples: {len(X_train)}, Feature dimension: {input_dim}, Classes: {num_classes}")

# 2. 初始化并训练每个子模型 (Initialize and train each sub-model)

# 2.1 初始化子模型 (BNN, MAML/SimpleNet, ProtoNet, DeepMLP)
bnn_model = BNN(input_dim=input_dim, output_dim=num_classes, hidden_dims=[64], dropout_rate=0.5)
maml_model = SimpleNet(input_dim=input_dim, hidden_dims=[128, 64], num_classes=num_classes)
proto_model = ProtoNet(input_dim=input_dim, embedding_dim=16, num_classes=num_classes)
baseline_model = DeepMLP(input_dim=input_dim, hidden_dims=[64, 32], num_classes=num_classes, dropout_rate=0.5)

# 2.2 为每个模型选择优化器 (Choose optimizer for each model)
bnn_optimizer = torch.optim.Adam(bnn_model.parameters(), lr=0.001)
maml_optimizer = torch.optim.Adam(maml_model.parameters(), lr=0.001)
proto_optimizer = torch.optim.Adam(proto_model.parameters(), lr=0.001)
baseline_optimizer = torch.optim.Adam(baseline_model.parameters(), lr=0.001)

# 2.3 定义损失函数 (Define loss function)
criterion = torch.nn.CrossEntropyLoss()

# 2.4 训练 Bayesian Neural Network (Train BNN model)
epochs_bnn = 50
kl_weight = 0.01  # KL散度损失权重 (weight for KL divergence term)
bnn_model.train()
for epoch in range(1, epochs_bnn+1):
    total_loss = 0.0
    for X_batch, y_batch in train_loader:
        # 前向传播 (Forward pass with weight sampling)
        outputs = bnn_model(X_batch, sample=True)  # sample Bayesian weights
        ce_loss = criterion(outputs, y_batch)      # 交叉熵损失 (cross-entropy loss)
        kl_loss = bnn_model.total_kl_loss()        # KL散度损失 (KL divergence loss)
        loss = ce_loss + kl_weight * kl_loss       # 总损失 = CE + 权重*KL (total loss)
        # 反向传播和优化 (Backward and optimize)
        bnn_optimizer.zero_grad()
        loss.backward()
        bnn_optimizer.step()
        total_loss += loss.item()
    # 每10轮输出一次损失 (Print loss every 10 epochs)
    if epoch % 10 == 0:
        avg_loss = total_loss / len(train_loader)
        print(f"BNN Epoch {epoch}/{epochs_bnn}, Loss: {avg_loss:.4f}")

# 2.5 训练 MAML模型 (其实就是SimpleNet的普通训练) (Train MAML/SimpleNet model normally)
epochs_maml = 50
maml_model.train()
for epoch in range(1, epochs_maml+1):
    total_loss = 0.0
    for X_batch, y_batch in train_loader:
        outputs = maml_model(X_batch)
        loss = criterion(outputs, y_batch)
        maml_optimizer.zero_grad()
        loss.backward()
        maml_optimizer.step()
        total_loss += loss.item()
    if epoch % 10 == 0:
        avg_loss = total_loss / len(train_loader)
        print(f"SimpleNet Epoch {epoch}/{epochs_maml}, Loss: {avg_loss:.4f}")

# 2.6 训练 Prototypical Network (训练embedding网络，通过few-shot episode) (Train ProtoNet with episodic training)
episodes = 2000
proto_model.train()
# 先将训练数据按类划分方便抽取 (Group training samples by class for episode sampling)
indices_by_class = {c: np.where(y_train == c)[0] for c in range(num_classes)}
for episode in range(1, episodes+1):
    # 每个episode: 为每个类别随机选择support和query样本索引 (sample support/query for each class)
    support_indices = []
    query_indices = []
    n_support = 5  # 每类支持样本数 (support shots per class)
    n_query = 5    # 每类查询样本数 (query shots per class)
    for c in range(num_classes):
        # 如果某类样本不足 n_support+n_query，则随机重采样有放回 (Handle class with fewer samples than needed)
        idx_all = indices_by_class[c]
        if len(idx_all) < n_support + n_query:
            idx = np.random.choice(idx_all, size=n_support+n_query, replace=True)
        else:
            idx = np.random.choice(idx_all, size=n_support+n_query, replace=False)
        support_indices.extend(idx[:n_support])
        query_indices.extend(idx[n_support:])
    # 构造支持集和查询集 (Build support and query sets)
    support_x = X_train_tensor[support_indices]
    support_y = torch.tensor(y_train[support_indices], dtype=torch.long)
    query_x = X_train_tensor[query_indices]
    query_y = torch.tensor(y_train[query_indices], dtype=torch.long)
    # 计算支持集的嵌入并得到原型 (Compute embeddings for support set and prototypes)
    support_emb = proto_model.embedding_net(support_x)
    prototypes = []
    for c in range(num_classes):
        class_embeds = support_emb[support_y == c]
        prototypes.append(class_embeds.mean(dim=0))
    prototypes = torch.stack(prototypes)  # tensor shape: (num_classes, embed_dim)
    # 计算查询集样本到每个原型的距离并计算loss (Compute distance of query embeddings to prototypes)
    query_emb = proto_model.embedding_net(query_x)
    # Negative squared distance as logits
    dist_sq = ((query_emb.unsqueeze(1) - prototypes.unsqueeze(0))**2).sum(dim=2)
    logits = -dist_sq  # use negative distance as logits
    loss = criterion(logits, query_y)
    # 更新embedding网络参数 (Update embedding network parameters)
    proto_optimizer.zero_grad()
    loss.backward()
    proto_optimizer.step()
    # 打印进度 (Print progress every 100 episodes)
    if episode % 500 == 0:
        print(f"ProtoNet Episode {episode}/{episodes}, Loss: {loss.item():.4f}")

# 在训练集上计算最终的原型用于以后的预测 (Compute final prototypes on full training set for ProtoNet)
proto_model.set_prototypes(X_train_tensor, y_train_tensor)
# （注：ProtoNet在train阶段通过episodic训练学习embedding，在eval阶段使用全训练集的原型进行实际分类）
# (Note: The ProtoNet's embedding network has been trained. We set prototypes from all training data for use in evaluation.)

# 2.7 训练 Baseline DeepMLP 模型 (Train baseline DeepMLP model)
epochs_base = 50
baseline_model.train()
for epoch in range(1, epochs_base+1):
    total_loss = 0.0
    for X_batch, y_batch in train_loader:
        outputs = baseline_model(X_batch)
        loss = criterion(outputs, y_batch)
        baseline_optimizer.zero_grad()
        loss.backward()
        baseline_optimizer.step()
        total_loss += loss.item()
    if epoch % 10 == 0:
        avg_loss = total_loss / len(train_loader)
        print(f"Baseline Epoch {epoch}/{epochs_base}, Loss: {avg_loss:.4f}")

# 3. Meta-Learner训练 (Train the Meta-Learner on sub-models' outputs)
# 首先，将所有子模型切换到评估模式，以固定其参数并获得稳定的输出 (Freeze sub-models for stable predictions)
bnn_model.eval()
maml_model.eval()
proto_model.eval()
baseline_model.eval()

# 准备元学习器训练数据 (Prepare meta-learner training data)
with torch.no_grad():
    # 获取每个子模型对训练集中每个样本的预测概率 (Get each model's prediction probabilities for training samples)
    # For BNN, we take the mean prediction by doing one forward pass with sample=False (using mean weights)
    bnn_logits = bnn_model(X_train_tensor, sample=False)
    bnn_probs = F.softmax(bnn_logits, dim=1)
    maml_logits = maml_model(X_train_tensor)
    maml_probs = F.softmax(maml_logits, dim=1)
    proto_logits = proto_model(X_train_tensor)  # ProtoNet forward returns logits (negative dist)
    proto_probs = F.softmax(proto_logits, dim=1)
    base_logits = baseline_model(X_train_tensor)
    base_probs = F.softmax(base_logits, dim=1)
# 将所有概率拼接作为元学习特征 (Concatenate all model probabilities to form meta features)
# Shape of each probs: (N_train, 4), we concatenate along feature dim -> (N_train, 16)
meta_features = torch.cat([bnn_probs, maml_probs, proto_probs, base_probs], dim=1)
meta_targets = y_train_tensor  # same labels as training set

# 引入类别权重
from sklearn.utils.class_weight import compute_class_weight
weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
weights = torch.tensor(weights, dtype=torch.float32)

criterion = torch.nn.CrossEntropyLoss(weight=weights)

# 改进后的MetaLearner模型
meta_model = MetaLearner(input_dim=4*num_classes, hidden_dims=[64, 32], output_dim=num_classes)
meta_optimizer = torch.optim.Adam(meta_model.parameters(), lr=0.001)

# Early stopping 参数
best_loss = float('inf')
patience = 10
counter = 0

meta_features_tensor = meta_features  # 假设之前已计算好
meta_targets_tensor = meta_targets

epochs_meta = 100  # 增大Epochs允许模型充分训练
for epoch in range(1, epochs_meta + 1):
    meta_model.train()
    permutation = torch.randperm(meta_features_tensor.size(0))
    shuffled_feats = meta_features_tensor[permutation]
    shuffled_tgts = meta_targets_tensor[permutation]

    outputs = meta_model(shuffled_feats)
    loss = criterion(outputs, shuffled_tgts)
    meta_optimizer.zero_grad()
    loss.backward()
    meta_optimizer.step()

    if epoch % 10 == 0:
        print(f"MetaLearner Epoch {epoch}/{epochs_meta}, Loss: {loss.item():.4f}")

    # Early Stopping Check
    if loss.item() < best_loss:
        best_loss = loss.item()
        counter = 0
        torch.save(meta_model.state_dict(), "best_meta_model.pth")
    else:
        counter += 1
        if counter >= patience:
            print(f"Early stopping triggered at epoch {epoch}")
            meta_model.load_state_dict(torch.load("best_meta_model.pth"))
            break

# 初始化元学习器并训练 (Initialize MetaLearner and train it)
# meta_model = MetaLearner(input_dim=4* num_classes, hidden_dims=[64, 32], output_dim=num_classes)
# meta_optimizer = torch.optim.Adam(meta_model.parameters(), lr=0.001)
meta_model.train()
epochs_meta = 50
for epoch in range(1, epochs_meta+1):
    # 因为meta_features已包含整个训练集，这里直接用整体 (Since meta_features is the whole training set, treat it as one batch or mini-batch)
    # We can shuffle the indices for each epoch for robustness
    permutation = torch.randperm(meta_features.size(0))
    shuffled_feats = meta_features[permutation]
    shuffled_tgts = meta_targets[permutation]
    # We can split into mini-batches if desired; here, we'll do full batch for simplicity
    outputs = meta_model(shuffled_feats)
    loss = criterion(outputs, shuffled_tgts)
    meta_optimizer.zero_grad()
    loss.backward()
    meta_optimizer.step()
    if epoch % 10 == 0:
        print(f"MetaLearner Epoch {epoch}/{epochs_meta}, Loss: {loss.item():.4f}")

# 4. 保存所有模型参数 (Save all model weights for later evaluation)
torch.save(bnn_model.state_dict(), "bnn_model.pth")
torch.save(maml_model.state_dict(), "maml_model.pth")
torch.save(proto_model.state_dict(), "proto_model.pth")
torch.save(baseline_model.state_dict(), "baseline_model.pth")
torch.save(meta_model.state_dict(), "meta_model.pth")
print("Training complete. Model weights saved.")
