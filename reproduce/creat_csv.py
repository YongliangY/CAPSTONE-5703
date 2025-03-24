import pandas as pd
from sklearn.model_selection import train_test_split

# 读取CSV文件
df = pd.read_csv('/Users/wenyu/Downloads/dataset_1.csv')
df = df.head(498)


df = df.dropna()
print("\n清洗后数据:")
print(f"剩余行数: {len(df)}")
print(f"剩余缺失值总数: {df.isna().sum().sum()}")
# 设置随机种子保证可重复性
SEED = 34

# 按1:9比例分割数据集（测试集10%，训练集90%）
train_df, test_df = train_test_split(
    df,
    test_size=0.1,
    random_state=SEED,
    # 如果需要进行分层抽样（适用于分类任务），取消下面注释
    # stratify=df['failure_mode']
)

# 保存分割后的数据集
train_df.to_csv('train.csv', index=False)
test_df.to_csv('test.csv', index=False)

print(f"训练集样本数: {len(train_df)}")
print(f"测试集样本数: {len(test_df)}")
print("分割已完成，文件已保存为 train.csv 和 test.csv")
print(df.head(5))

