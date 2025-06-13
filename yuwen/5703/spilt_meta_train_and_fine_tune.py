import pandas as pd
from sklearn.model_selection import train_test_split
import os


df_train = pd.read_csv('train_smote_augmented.csv')
df_test = pd.read_csv('test_original.csv')


label_cols = [col for col in df_train.columns if col.startswith("label_")]
X_train = df_train.drop(columns=label_cols)
y_train = df_train[label_cols]

X_test = df_test.drop(columns='failure mode')
y_test = df_test['failure mode']

output_dir = 'splits'
os.makedirs(output_dir, exist_ok=True)

for i in range(5):
    X_meta_train, X_fine_tune, y_meta_train, y_fine_tune = train_test_split(
        X_train,
        y_train,
        test_size=1/3,
        stratify=None,
        random_state=i * 10
    )

    df_meta_train = pd.concat([X_meta_train, y_meta_train], axis=1)
    df_fine_tune = pd.concat([X_fine_tune, y_fine_tune], axis=1)

    meta_train_filename = os.path.join(output_dir, f'meta_train_{i+1}_mixup.csv')
    fine_tune_filename = os.path.join(output_dir, f'fine_tune_{i+1}_mixup.csv')

    df_meta_train.to_csv(meta_train_filename, index=False)
    df_fine_tune.to_csv(fine_tune_filename, index=False)

    print(f"\nthe {i+1} times：")
    print(f"  meta：{meta_train_filename}，numbers：{len(df_meta_train)}")
    print(f"  fine tuen：{fine_tune_filename}，numbers：{len(df_fine_tune)}")
