import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix


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
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)
    return X, y


def predict_with_model(model, test_x):
    model.eval()
    with torch.no_grad():
        logits = model(test_x)
        probs = F.softmax(logits, dim=1)
        pred_labels = torch.argmax(probs, dim=1) + 1  # Convert to 1-based labels
    return pred_labels.numpy(), probs.numpy()


def vote_ensemble(predictions_list):
    predictions = np.stack(predictions_list, axis=0)
    votes = []
    for i in range(predictions.shape[1]):
        vote = np.bincount(predictions[:, i], minlength=5)
        votes.append(np.argmax(vote[:4]))
    return np.array(votes)


def average_ensemble(prob_list):
    avg_probs = np.mean(np.stack(prob_list, axis=0), axis=0)
    return np.argmax(avg_probs, axis=1) + 1


def plot_confusion(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred, labels=[1, 2, 3, 4])
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=[1, 2, 3, 4], yticklabels=[1, 2, 3, 4])
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(title)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    print("\n loading test_original.csv ...")
    test_x, test_y = load_csv_dataset("test_original.csv")
    test_y_np = test_y.numpy()

    model_preds = []
    model_probs = []

    print("\n result of singal model：")
    for i in range(1, 6):
        model = MLPBackbone(input_dim=69, output_dim=4)
        model.load_state_dict(torch.load(f"models/maml_model_{i}.pth", map_location=torch.device('cpu')))
        preds, probs = predict_with_model(model, test_x)

        acc = accuracy_score(test_y_np, preds)
        f1 = f1_score(test_y_np, preds, average='weighted')
        print(f"   Model {i}: Accuracy = {acc:.4f}, F1 = {f1:.4f}")

        model_preds.append(preds)
        model_probs.append(probs)

    # -------------------- Vote Ensemble --------------------
    print("\n️ Vote Ensemble ：")
    vote_preds = vote_ensemble(model_preds)
    vote_acc = accuracy_score(test_y_np, vote_preds)
    vote_f1 = f1_score(test_y_np, vote_preds, average='weighted')
    print(f" Accuracy: {vote_acc:.4f}, weighted F1: {vote_f1:.4f}")
    print(" report:\n", classification_report(test_y_np, vote_preds, labels=[1, 2, 3, 4], digits=4))
    plot_confusion(test_y_np, vote_preds, "Vote Ensemble Confusion Matrix")

    # -------------------- Average Ensemble --------------------
    print("\n Average Ensemble result：")
    avg_preds = average_ensemble(model_probs)
    avg_acc = accuracy_score(test_y_np, avg_preds)
    avg_f1 = f1_score(test_y_np, avg_preds, average='weighted')
    print(f" Accuracy: {avg_acc:.4f}, weighted F1: {avg_f1:.4f}")
    print(" report:\n", classification_report(test_y_np, avg_preds, labels=[1, 2, 3, 4], digits=4))
    plot_confusion(test_y_np, avg_preds, "Average Ensemble Confusion Matrix")
