# file: models/proto_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class EmbeddingNet(nn.Module):
    def __init__(self, input_dim, embedding_dim=16):
        super(EmbeddingNet, self).__init__()
        # 嵌入层网络: 输入 -> 64维隐藏层 -> 嵌入维度 (Embedding network: input -> 64 -> embedding_dim)
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, embedding_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        # 前向传播: ReLU(FC1(x)) -> FC2 -> 输出嵌入向量 (Compute embedding for input)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x  # 返回样本的嵌入表示 (Return embedding vector)


class ProtoNet(nn.Module):
    def __init__(self, input_dim, embedding_dim=16, num_classes=4):
        super(ProtoNet, self).__init__()
        self.embedding_net = EmbeddingNet(input_dim, embedding_dim)
        self.num_classes = num_classes
        # 用于保存每个类别的原型向量 (Prototype vectors for each class)
        self.register_buffer('prototypes', torch.zeros(num_classes, embedding_dim))

    def set_prototypes(self, X_ref, y_ref):
        """
        根据参考数据计算并设置每个类别的原型 (Calculate class prototypes from reference data).
        X_ref: Tensor of shape (N, input_dim), y_ref: Tensor of shape (N,) with class labels.
        """
        self.eval()  # 确保模型在评估模式 (no dropout, etc.)
        with torch.no_grad():
            embeddings = self.embedding_net(X_ref)  # 计算参考集所有样本的嵌入 (embed all reference samples)
        # 对每个类别计算嵌入均值作为原型 (Compute mean embedding per class)
        prototypes = []
        for c in range(self.num_classes):
            class_embeds = embeddings[y_ref == c]
            if class_embeds.shape[0] == 0:
                # 若某类别在参考集中无样本，则原型设为0 (Handle class with no samples if any)
                prototypes.append(torch.zeros(self.prototypes.shape[1]))
            else:
                prototypes.append(class_embeds.mean(dim=0))
        prototypes = torch.stack(prototypes)
        self.prototypes.copy_(prototypes)  # 保存到ProtoNet的缓冲区 (store prototypes)

    def forward(self, x):
        # 计算输入x到每个类原型的平方距离 (Compute squared distance from input to each prototype)
        emb = self.embedding_net(x)
        # 扩展维度以便计算所有样本与所有原型的距离 (broadcasting subtraction)
        diff = emb.unsqueeze(1) - self.prototypes.unsqueeze(0)  # shape: (batch, num_classes, embed_dim)
        dist_sq = torch.sum(diff * diff, dim=2)  # shape: (batch, num_classes), each entry = ||emb - prototype||^2
        # 将负的距离作为logits（距离越小logit越大） (Use negative distance as logits for classification)
        logits = -dist_sq
        return logits  # 返回每个类别的未归一化得分 (Return unnormalized score for each class)
