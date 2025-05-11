import pandas as pd
import numpy as np
import random
from sklearn.preprocessing import LabelEncoder
import argparse

def mixup_data(X, y, alpha=0.4):
    lam = np.random.beta(alpha, alpha)
    index = np.random.permutation(X.shape[0])
    X_mix = lam * X + (1 - lam) * X[index]
    y_mix = lam * y + (1 - lam) * y[index]
    return X_mix, y_mix

def generate_mixup_augmented_csv(input_csv, output_csv, num_mixup=3, alpha=0.4):
    # 读取原始数据
    df = pd.read_csv(input_csv)
    X = df.iloc[:, :-1].values.astype(np.float32)
    y = df.iloc[:, -1].values

    # 编码标签为 one-hot
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    num_classes = len(np.unique(y_encoded))
    y_onehot = np.eye(num_classes)[y_encoded]

    # 保留原始数据
    X_all = [X]
    y_all = [y_onehot]

    # 多次生成 mixup
    for _ in range(num_mixup):
        X_mix, y_mix = mixup_data(X, y_onehot, alpha=alpha)
        X_all.append(X_mix)
        y_all.append(y_mix)

    # 合并数据
    X_combined = np.vstack(X_all)
    y_combined = np.vstack(y_all)

    # 构造 DataFrame
    df_out = pd.DataFrame(X_combined, columns=[f"feature_{i}" for i in range(X.shape[1])])
    for i in range(num_classes):
        df_out[f"label_{i}"] = y_combined[:, i]

    # 保存
    df_out.to_csv(output_csv, index=False)
    print(f"✅ Augmented data saved to {output_csv} (original + {num_mixup}x mixup)")

if __name__ == "__main__":
    random.seed(34)
    input_csv = "train_original.csv"          # 原始数据路径
    output_csv = "train_augmented.csv"        # 增强后数据保存路径
    mixup_times = 4                            # Mixup 次数
    alpha = 0.4                                # Beta 分布参数

    generate_mixup_augmented_csv(
        input_csv=input_csv,
        output_csv=output_csv,
        num_mixup=mixup_times,
        alpha=alpha
    )
