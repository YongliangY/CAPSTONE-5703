import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import os
from copy import deepcopy
import higher
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score


def set_global_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


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
    X = df.drop(columns=["failure mode"]).values
    y = df["failure mode"].values
    y = torch.tensor(y - 1, dtype=torch.long)
    X = torch.tensor(X, dtype=torch.float32)
    return X, y


def create_tasks(X, y, num_tasks=40, k_shot=2, q_query=3, num_classes=4, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    tasks = []
    for _ in range(num_tasks):
        support_x, support_y, query_x, query_y = [], [], [], []
        for cls in range(num_classes):
            idx = torch.where(y == cls)[0]
            if len(idx) < k_shot + q_query:
                continue
            perm = torch.randperm(len(idx))
            support_idx = idx[perm[:k_shot]]
            query_idx = idx[perm[k_shot:k_shot+q_query]]
            support_x.append(X[support_idx])
            support_y.append(y[support_idx])
            query_x.append(X[query_idx])
            query_y.append(y[query_idx])
        if len(support_x) == num_classes:
            tasks.append((
                torch.cat(support_x), torch.cat(support_y),
                torch.cat(query_x), torch.cat(query_y)
            ))
    return tasks


def maml_train(model, train_tasks, val_tasks, meta_lr=2e-3, inner_lr=1e-2, inner_steps=3, meta_epochs=300, patience=30):
    meta_optimizer = torch.optim.Adam(model.parameters(), lr=meta_lr)
    best_val_loss = float("inf")
    best_model_state = deepcopy(model.state_dict())
    no_improve_epochs = 0

    for epoch in range(meta_epochs):
        model.train()
        meta_optimizer.zero_grad()
        total_meta_loss = 0.0

        for support_x, support_y, query_x, query_y in train_tasks:
            inner_optimizer = torch.optim.SGD(model.parameters(), lr=inner_lr)
            with higher.innerloop_ctx(model, inner_optimizer, copy_initial_weights=False) as (fmodel, diffopt):
                fmodel.train()  # Ensure training mode for inner loop
                for _ in range(inner_steps):
                    support_loss = F.cross_entropy(fmodel(support_x), support_y)
                    diffopt.step(support_loss)
                fmodel.eval()  # Use eval mode for meta update
                query_loss = F.cross_entropy(fmodel(query_x), query_y)
                query_loss.backward()
                total_meta_loss += query_loss.item()

        meta_optimizer.step()
        avg_meta_loss = total_meta_loss / len(train_tasks)

        model.eval()
        val_loss_total = 0.0
        for sx, sy, qx, qy in val_tasks:
            inner_optimizer = torch.optim.SGD(model.parameters(), lr=inner_lr)
            with higher.innerloop_ctx(model, inner_optimizer, copy_initial_weights=False) as (fmodel, diffopt):
                fmodel.train()
                for _ in range(inner_steps):
                    s_loss = F.cross_entropy(fmodel(sx), sy)
                    diffopt.step(s_loss)
                fmodel.eval()
                q_loss = F.cross_entropy(fmodel(qx), qy)
                val_loss_total += q_loss.item()
        avg_val_loss = val_loss_total / len(val_tasks)

        print(f"[Epoch {epoch}] Meta Loss: {avg_meta_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = deepcopy(model.state_dict())
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= patience:
                print(f" Early stopping at epoch {epoch}")
                break

    model.load_state_dict(best_model_state)


def fine_tune_with_early_stopping(model, support_x, support_y, val_x, val_y, lr=1e-3, max_steps=500, patience=40):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_model_state = deepcopy(model.state_dict())
    best_val_loss = float('inf')
    no_improve_steps = 0

    for step in range(max_steps):
        model.train()
        loss = F.cross_entropy(model(support_x), support_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_preds = model(val_x)
            val_loss = F.cross_entropy(val_preds, val_y).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = deepcopy(model.state_dict())
            no_improve_steps = 0
        else:
            no_improve_steps += 1
            if no_improve_steps >= patience:
                print(f" Early stopped fine-tuning at step {step}, best val loss = {best_val_loss:.4f}")
                break

    model.load_state_dict(best_model_state)


def evaluate(model, test_x, test_y):
    model.eval()
    with torch.no_grad():
        preds = model(test_x)
        pred_labels = preds.argmax(dim=1) + 1
        true_labels = test_y + 1
        acc = (pred_labels == true_labels).float().mean().item()
        f1 = f1_score(true_labels.numpy(), pred_labels.numpy(), average='macro')
        print(f" Test Accuracy: {acc:.4f} | F1 Score: {f1:.4f}")


if __name__ == "__main__":
    set_global_seed(34)
    os.makedirs("models", exist_ok=True)

    for i in range(1, 6):
        print(f"\n Training model {i} using meta_train_{i}.csv & fine_tune_{i}.csv")

        meta_path = f"splits/meta_train_{i}.csv"
        fine_path = f"splits/fine_tune_{i}.csv"

        X_meta, y_meta = load_csv_dataset(meta_path)
        X_fine, y_fine = load_csv_dataset(fine_path)

        X_train, X_val, y_train, y_val = train_test_split(X_meta, y_meta, test_size=0.3, stratify=y_meta, random_state=34)
        train_tasks = create_tasks(X_train, y_train, num_tasks=50, seed=100)
        val_tasks = create_tasks(X_val, y_val, num_tasks=10, seed=200)

        model = MLPBackbone(input_dim=69, output_dim=4)

        maml_train(model, train_tasks, val_tasks, meta_epochs=500, patience=20)

        X_support, X_val_fine, y_support, y_val_fine = train_test_split(X_fine, y_fine, test_size=0.2, stratify=y_fine, random_state=34)
        fine_tune_with_early_stopping(model, X_support, y_support, X_val_fine, y_val_fine)

        torch.save(model.state_dict(), f"models/fine_tuned_model_{i}.pth")
        print(f" Saved fine-tuned model: models/fine_tuned_model_{i}.pth")

        evaluate(model, y_support, y_val_fine)
