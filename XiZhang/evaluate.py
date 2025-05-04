# file: evaluate.py
import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from bnn_model import BNN
from maml_model import SimpleNet
from proto_model import ProtoNet
from baseline_model import DeepMLP
from meta_learner import MetaLearner

# 1. 加载测试数据 (Load test data)
data_test = np.genfromtxt('test_smote_pca.csv', delimiter=',', skip_header=1)
X_test = data_test[:, :-1].astype(np.float32)
y_test = data_test[:, -1].astype(np.int64)
y_test -= 1  # 将标签从1-4转换为0-3 (convert labels to 0-indexed)
X_test_tensor = torch.tensor(X_test)
y_test_tensor = torch.tensor(y_test)
num_samples = X_test.shape[0]
print(f"Test samples: {num_samples}, Feature dimension: {X_test.shape[1]}")

# 2. 加载训练好的模型参数 (Load trained model weights)
input_dim = X_test.shape[1]
num_classes = 4
# Instantiate model objects (structure must match training)
bnn_model = BNN(input_dim=input_dim, output_dim=num_classes, hidden_dims=[64], dropout_rate=0.5)
maml_model = SimpleNet(input_dim=input_dim, hidden_dims=[128, 64], num_classes=num_classes)
proto_model = ProtoNet(input_dim=input_dim, embedding_dim=16, num_classes=num_classes)
baseline_model = DeepMLP(input_dim=input_dim, hidden_dims=[64, 32], num_classes=num_classes, dropout_rate=0.5)
meta_model = MetaLearner(input_dim=16, hidden_dims=[64,32], output_dim=num_classes)

# Load weights
bnn_model.load_state_dict(torch.load("bnn_model.pth"))
maml_model.load_state_dict(torch.load("maml_model.pth"))
proto_model.load_state_dict(torch.load("proto_model.pth"))
baseline_model.load_state_dict(torch.load("baseline_model.pth"))
meta_model.load_state_dict(torch.load("best_meta_model.pth"))
# Set to eval mode
bnn_model.eval()
maml_model.eval()
proto_model.eval()
baseline_model.eval()
meta_model.eval()

# **Note**: For ProtoNet, we need to recompute prototypes using training data before using it.
# However, since we saved proto_model after calling set_prototypes in training, the prototypes should be in state dict.
# If not, we would recompute prototypes here by loading training data again. We assume prototypes saved.

# 3. 使用每个子模型对测试集进行预测，并通过meta-learner融合 (Predict test samples using each sub-model and combine via meta-learner)
with torch.no_grad():
    # Get sub-model probabilities for each test sample
    # BNN: use sample=False (mean weights) for deterministic output
    bnn_logits_test = bnn_model(X_test_tensor, sample=False)
    bnn_probs_test = torch.softmax(bnn_logits_test, dim=1)
    maml_logits_test = maml_model(X_test_tensor)
    maml_probs_test = torch.softmax(maml_logits_test, dim=1)
    proto_logits_test = proto_model(X_test_tensor)  # ProtoNet forward (uses stored prototypes)
    proto_probs_test = torch.softmax(proto_logits_test, dim=1)
    base_logits_test = baseline_model(X_test_tensor)
    base_probs_test = torch.softmax(base_logits_test, dim=1)
    # Concatenate all sub-model probabilities
    meta_features_test = torch.cat([bnn_probs_test, maml_probs_test, proto_probs_test, base_probs_test], dim=1)
    # Meta-learner prediction
    meta_logits_test = meta_model(meta_features_test)
    meta_pred_probs = torch.softmax(meta_logits_test, dim=1)
    pred_classes = torch.argmax(meta_pred_probs, dim=1).numpy()

# 4. 计算指标 (Calculate evaluation metrics)
y_true = y_test  # true labels as numpy array
y_pred = pred_classes
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
print(f"Ensemble Accuracy: {accuracy:.4f}")
print(f"Ensemble Precision (macro): {precision:.4f}")
print(f"Ensemble Recall (macro): {recall:.4f}")
print(f"Ensemble F1-score (macro): {f1:.4f}")

# 5. 混淆矩阵 (Confusion matrix)
cm = confusion_matrix(y_true, y_pred, labels=[0,1,2,3])
print("Confusion Matrix (counts):\n", cm)
# 绘制混淆矩阵热力图 (Plot confusion matrix heatmap)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap="Blues",
            xticklabels=[1,2,3,4], yticklabels=[1,2,3,4])  # label classes as 1-4 for clarity
plt.title('Confusion Matrix - ISE Ensemble')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.savefig("confusion_matrix.png")  # 保存混淆矩阵图像 (save the confusion matrix figure)
plt.show()
