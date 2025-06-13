import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import os
import random
from copy import deepcopy
import higher
from sklearn.model_selection import train_test_split

def set_seed(seed=34):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def soft_cross_entropy(preds, soft_targets):
    return torch.mean(torch.sum(-soft_targets * F.log_softmax(preds, dim=1), dim=1))

class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout_rate=0.1):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out += identity
        out = self.relu(out)
        return out

class MLPBackbone(nn.Module):
    def __init__(self, input_dim=69, hidden_dim=128, output_dim=4, num_blocks=4):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )
        self.blocks = nn.Sequential(*[ResidualBlock(hidden_dim) for _ in range(num_blocks)])
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.blocks(x)
        x = self.output_layer(x)
        return x

def load_csv_dataset(path):
    df = pd.read_csv(path)
    if "failure mode" in df.columns:
        X = df.drop(columns=["failure mode"]).values.astype(np.float32)
        y = df["failure mode"].values
        y = torch.tensor(y - 1, dtype=torch.long)
    else:
        label_cols = [col for col in df.columns if col.startswith("label_")]
        X = df.drop(columns=label_cols).values.astype(np.float32)
        y = df[label_cols].values.astype(np.float32)
        y = torch.tensor(y, dtype=torch.float32)
    X = torch.tensor(X, dtype=torch.float32)
    return X, y

def create_tasks(X, y, num_tasks=150, k_shot=5, q_query=10, num_classes=4, device='cpu'):
    tasks = []
    is_soft_label = y.ndim == 2
    insufficient_classes = set()

    for _ in range(num_tasks):
        support_x, support_y, query_x, query_y = [], [], [], []
        for cls in range(num_classes):
            if is_soft_label:
                cls_idx = (y[:, cls] > 0.5).nonzero(as_tuple=True)[0]
            else:
                cls_idx = torch.where(y == cls)[0]

            if len(cls_idx) < k_shot + q_query:
                insufficient_classes.add(cls)
                continue

            perm = torch.randperm(len(cls_idx))
            support_idx = cls_idx[perm[:k_shot]]
            query_idx = cls_idx[perm[k_shot:k_shot + q_query]]
            support_x.append(X[support_idx])
            support_y.append(y[support_idx])
            query_x.append(X[query_idx])
            query_y.append(y[query_idx])

        if len(support_x) == num_classes:
            tasks.append((
                torch.cat(support_x).to(device),
                torch.cat(support_y).to(device),
                torch.cat(query_x).to(device),
                torch.cat(query_y).to(device)
            ))

    if insufficient_classes:
        print(f"⚠️ Warning: Class(es) with insufficient samples for k={k_shot}, q={q_query}: {sorted(insufficient_classes)}")

    if not tasks:
        raise ValueError(
            "create tasks() failed: No tasks were constructed"
        )
    return tasks


def maml_train(model, train_tasks, val_tasks, meta_lr=2e-3, inner_lr=1e-2, inner_steps=3, meta_epochs=100, patience=25, model_save_path=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"📦 Using device for MAML training: {device}")

    train_tasks = [(sx.to(device), sy.to(device), qx.to(device), qy.to(device)) for (sx, sy, qx, qy) in train_tasks]
    val_tasks = [(sx.to(device), sy.to(device), qx.to(device), qy.to(device)) for (sx, sy, qx, qy) in val_tasks]

    meta_optimizer = torch.optim.Adam(model.parameters(), lr=meta_lr)
    meta_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        meta_optimizer, mode='min', factor=0.5, patience=5
    )

    best_val_loss = float("inf")
    best_model_state = deepcopy(model.state_dict())
    no_improve_epochs = 0

    for epoch in range(meta_epochs):
        model.train()
        meta_optimizer.zero_grad()
        total_meta_loss = 0.0

        for support_x, support_y, query_x, query_y in train_tasks:
            inner_optimizer = torch.optim.SGD(model.parameters(), lr=inner_lr)
            with higher.innerloop_ctx(model, inner_optimizer, copy_initial_weights=False, device=device) as (fmodel, diffopt):
                for _ in range(inner_steps):
                    support_loss = soft_cross_entropy(fmodel(support_x), support_y) if support_y.ndim == 2 else F.cross_entropy(fmodel(support_x), support_y)
                    diffopt.step(support_loss)
                query_loss = soft_cross_entropy(fmodel(query_x), query_y) if query_y.ndim == 2 else F.cross_entropy(fmodel(query_x), query_y)
                query_loss.backward()
                total_meta_loss += query_loss.item()

        meta_optimizer.step()
        avg_meta_loss = total_meta_loss / len(train_tasks)

        model.eval()
        val_loss_total = 0.0
        for sx, sy, qx, qy in val_tasks:
            inner_optimizer = torch.optim.SGD(model.parameters(), lr=inner_lr)
            with higher.innerloop_ctx(model, inner_optimizer, copy_initial_weights=False, device=device) as (fmodel, diffopt):
                for _ in range(inner_steps):
                    s_loss = soft_cross_entropy(fmodel(sx), sy) if sy.ndim == 2 else F.cross_entropy(fmodel(sx), sy)
                    diffopt.step(s_loss)
                q_loss = soft_cross_entropy(fmodel(qx), qy) if qy.ndim == 2 else F.cross_entropy(fmodel(qx), qy)
                val_loss_total += q_loss.item()

        avg_val_loss = val_loss_total / len(val_tasks)

        print(f"[Epoch {epoch}] Meta Loss: {avg_meta_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        meta_scheduler.step(avg_val_loss)
        current_lr = meta_optimizer.param_groups[0]['lr']
        print(f" Meta LR: {current_lr:.6f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = deepcopy(model.state_dict())
            no_improve_epochs = 0
            if model_save_path:
                os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
                torch.save(best_model_state, model_save_path)
                print(f"   🔥 Saved best meta model to {model_save_path}")
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= patience:
                print(f"⏹️ Early stopping at epoch {epoch}")
                break

    model.load_state_dict(best_model_state)

def fine_tune(model, support_x, support_y, lr=1e-2, steps=1000, val_ratio=0.2, patience=50, save_path="models/finetuned_model.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"️ Using device for fine-tuning: {device}")

    support_x = support_x.to(device)
    support_y = support_y.to(device)

    if support_y.ndim == 2:
        support_y = support_y.argmax(dim=1)

    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(
        support_x.cpu().numpy(), support_y.cpu().numpy(), test_size=val_ratio, random_state=34
    )
    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    X_val = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.long).to(device)
    y_val = torch.tensor(y_val, dtype=torch.long).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=20
    )

    best_val_loss = float('inf')
    best_model_state = deepcopy(model.state_dict())
    no_improve_epochs = 0

    for step in range(steps):
        model.train()
        preds = model(X_train)
        loss = F.cross_entropy(preds, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_preds = model(X_val)
            val_loss = F.cross_entropy(val_preds, y_val)

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"[Step {step+1}] Train Loss: {loss.item():.4f} | Val Loss: {val_loss.item():.4f} | LR: {current_lr:.6f}")

        if val_loss.item() < best_val_loss:
            best_val_loss = val_loss.item()
            best_model_state = deepcopy(model.state_dict())
            no_improve_epochs = 0
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(best_model_state, save_path)
            print(f"   🔥 Saved best fine-tuned model to {save_path}")
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= patience:
                print(f"⏹️ Early stopping at step {step+1}")
                break

    model.load_state_dict(best_model_state)


if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    set_seed(34)
    print("CUDA available:", torch.cuda.is_available())
    device = torch.device("cpu")
    print(f" Running on device: {device}")

    for i in range(1, 6):
        print(f"\n Training model {i} using meta_train_{i}_mixup.csv & fine_tune_{i}_mixup.csv")

        meta_path = f"splits/meta_train_{i}_mixup.csv"
        fine_path = f"splits/fine_tune_{i}_mixup.csv"

        X_meta, y_meta = load_csv_dataset(meta_path)
        X_fine, y_fine = load_csv_dataset(fine_path)

        X_train, X_val, y_train, y_val = train_test_split(X_meta, y_meta, test_size=0.2, stratify=None, random_state=34)
        train_tasks = create_tasks(X_train, y_train, num_tasks=150)
        val_tasks = create_tasks(X_val, y_val, num_tasks=15)

        model = MLPBackbone(input_dim=69, output_dim=4).to(device)

        train_tasks = create_tasks(X_train, y_train, num_tasks=300, device=device)
        val_tasks = create_tasks(X_val, y_val, num_tasks=30, device=device)
        save_path = f"models/maml_model_{i}.pth"
        save_path1 = f"models/fine_model_{i}.pth"

        maml_train(model, train_tasks, val_tasks, meta_epochs=250, patience=25, model_save_path=save_path)

        fine_tune(model, X_fine, y_fine, steps=2000, save_path=save_path1)
