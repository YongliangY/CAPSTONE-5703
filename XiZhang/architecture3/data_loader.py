import numpy as np
import torch

class RCWallDataset(torch.utils.data.Dataset):
    def __init__(self, X, y, augment=False):
        """
        X: 特征矩阵 (numpy 数组或 torch 张量)
           Feature matrix (numpy array or torch tensor)
        y: 样本标签 (numpy 数组或 torch 张量)
           Sample labels (numpy array or torch tensor)
        augment: 是否对特征进行扰动增强 (whether to apply augmentation by adding noise to features)
        """
        # 将数据转换为 torch 张量 (Convert data to torch tensors)
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        self.augment = augment
        if augment:
            # 预先计算每个特征的标准差, 用于噪声扰动 (Compute std of each feature for adding noise)
            # 按列计算标准差 (axis=0 ensures we compute std per feature dimension)
            self.feature_std = self.X.std(dim=0)
        else:
            self.feature_std = None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # 获取第 idx 个样本 (Get the idx-th sample)
        x = self.X[idx]
        y = self.y[idx]
        if self.augment:
            # 用高斯噪声扰动特征 (Add Gaussian noise for augmentation)
            # 噪声强度为每个特征 std 的 10% (noise std is 10% of each feature's std)
            noise = torch.randn_like(x) * self.feature_std * 0.1
            x = x + noise
        return x, y
