import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import random
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,confusion_matrix
from copy import deepcopy
import numpy as np

# Define the Bayesian linear layer
class BayesianLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features).normal_(0, 0.1))
        self.weight_rho = nn.Parameter(torch.Tensor(out_features, in_features).uniform_(-5, -4))
        self.bias_mu = nn.Parameter(torch.Tensor(out_features).normal_(0, 0.1))
        self.bias_rho = nn.Parameter(torch.Tensor(out_features).uniform_(-5, -4))

    def forward(self, x, sample=True):
        if sample:
            weight_sigma = torch.log1p(torch.exp(self.weight_rho))
            weight = self.weight_mu + weight_sigma * torch.randn_like(weight_sigma)
            bias_sigma = torch.log1p(torch.exp(self.bias_rho))
            bias = self.bias_mu + bias_sigma * torch.randn_like(bias_sigma)
        else:
            weight, bias = self.weight_mu, self.bias_mu
        return F.linear(x, weight, bias)

    def kl_loss(self):
        weight_sigma = torch.log1p(torch.exp(self.weight_rho))
        kl = -0.5 * torch.sum(1 + 2 * torch.log(weight_sigma) - self.weight_mu.pow(2) - weight_sigma.pow(2))
        bias_sigma = torch.log1p(torch.exp(self.bias_rho))
        kl += -0.5 * torch.sum(1 + 2 * torch.log(bias_sigma) - self.bias_mu.pow(2) - bias_sigma.pow(2))
        return kl

# Bayesian Neural Network
class BNN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims, dropout_rate):
        super().__init__()
        self.layers = nn.ModuleList()
        prev_dim = input_dim

        # the hidden layer
        for h_dim in hidden_dims:
            self.layers.append(BayesianLinear(prev_dim, h_dim))
            self.layers.append(nn.Dropout(dropout_rate))
            prev_dim = h_dim

        # output layer
        self.out_layer = BayesianLinear(prev_dim, output_dim)

    def forward(self, x, sample=True):
        for layer in self.layers:
            if isinstance(layer, BayesianLinear):
                x = F.relu(layer(x, sample))
            else:
                x = layer(x)
        return self.out_layer(x, sample)

    def kl_loss(self):
        # Accumulate the KL divergence of all Bayesian layers
        total_kl = 0
        for layer in self.layers:
            if isinstance(layer, BayesianLinear):
                total_kl += layer.kl_loss()
        return total_kl + self.out_layer.kl_loss()

# Data loading and preprocessing
def load_data(train_path, test_path):
    train_df = pd.read_csv(train_path, header=None, skiprows=1)
    test_df = pd.read_csv(test_path, header=None, skiprows=1)

    # Extract features and labels
    X_train = train_df.iloc[:, :-1].values.astype('float32')
    y_train = train_df.iloc[:, -1].values.astype('int64') - 1
    X_test = test_df.iloc[:, :-1].values.astype('float32')
    y_test = test_df.iloc[:, -1].values.astype('int64') - 1
    print("\nData distribution:")
    print(f"number of training set samples: {len(X_train)} | categorical distribution: {np.bincount(y_train)}")
    print(f"number of test set samples: {len(X_test)} | categorical distribution: {np.bincount(y_test)}")

    # convert to PyTorch Dataset
    train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    return (
        DataLoader(train_dataset, batch_size=32, shuffle=True),
        torch.tensor(X_test),
        torch.tensor(y_test)
    )

# Use the Adam optimizer and conduct multi-batch training
def train_reptile(model, train_loader, params):
    #Reptile Meta-learning training
    meta_optimizer = torch.optim.Adam(model.parameters(), lr=params['meta_lr'])
    best_loss = float('inf')
    patience_counter = 0

    for epoch in range(params['num_epochs']):
        # Save the initial parameters
        initial_state = deepcopy(model.state_dict())
        temp_model = BNN(
            input_dim=21,
            output_dim=4,
            hidden_dims=params['hidden_dims'],
            dropout_rate=params['dropout_rate']
        )
        temp_model.load_state_dict(initial_state)
        inner_optim = torch.optim.SGD(temp_model.parameters(), lr=params['inner_lr'])

        # Internal training cycle
        for _ in range(params['num_inner_steps']):
            for X_batch, y_batch in train_loader:
                outputs = temp_model(X_batch)
                loss = F.cross_entropy(outputs, y_batch) + params['kl_weight'] * temp_model.kl_loss()
                inner_optim.zero_grad()
                loss.backward()
                inner_optim.step()

        # Meta-parameter update
        with torch.no_grad():
            for p_model, p_temp in zip(model.parameters(), temp_model.parameters()):
                p_model.grad = (p_model - p_temp)  # Reptile update rule
        meta_optimizer.step()

        # Early stop
        current_loss = loss.item()
        if current_loss < best_loss:
            best_loss = current_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 5:
                print(f"Early stop is triggered in the {epoch + 1} round")
                break

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{params['num_epochs']}] Loss: {current_loss:.4f}")


# Evaluation function (multiple sampling)
def evaluate(model, X_test, y_test, num_samples=30):
    #Evaluation is conducted based on the average probability of multiple samplings
    model.eval()
    with torch.no_grad():
        probs = []
        for _ in range(num_samples):
            outputs = model(X_test, sample=True)
            probs.append(F.softmax(outputs, dim=1))

        avg_probs = torch.mean(torch.stack(probs), dim=0)
        preds = torch.argmax(avg_probs, dim=1).numpy()
        y_test_np = y_test.numpy()
        return {
            'accuracy': accuracy_score(y_test, preds),
            'precision': precision_score(y_test, preds, average='macro', zero_division=0),
            'recall': recall_score(y_test, preds, average='macro', zero_division=0),
            'f1': f1_score(y_test, preds, average='macro', zero_division=0),
            'confusion_matrix': confusion_matrix(y_test, preds)
        }


def parameter_search(train_loader, X_test, y_test, num_trials=20):
    # Random parameter search
    param_space = {
        'meta_lr': [1e-4, 3e-4, 1e-3, 3e-3],
        'inner_lr': [1e-3, 3e-3, 1e-2, 3e-2],
        'num_inner_steps': [1, 3, 5],
        'kl_weight': [0.001, 0.01, 0.1],
        'hidden_dims': [
            [64],
            [128],
            [256],
            [128, 64],
            [256, 128]
        ],
        'dropout_rate': [0.3, 0.5, 0.7],
        'num_epochs': [50],
        'eval_samples': [30]
    }

    best_score = 0
    best_params = None
    results = []

    for trial in range(num_trials):
        params = {
            key: random.choice(values) for key, values in param_space.items()
        }
        params['hidden_dims'] = random.choice(param_space['hidden_dims'])

        print(f"\ntrial {trial + 1}/{num_trials}")
        print("Current parameters:", params)

        # Initialize the model
        model = BNN(
            input_dim=21,
            output_dim=4,
            hidden_dims=params['hidden_dims'],
            dropout_rate=params['dropout_rate']
        )

        # training model
        train_reptile(model, train_loader, params)

        # Evaluate model
        metrics = evaluate(model, X_test, y_test, num_samples=params['eval_samples'])
        current_score = metrics['f1']
        results.append((params, metrics))

        # Update the best parameters
        if current_score > best_score:
            best_score = current_score
            best_params = params
            torch.save(model.state_dict(), 'best_model.pth')

        print(f"Current F1: {current_score:.4f} | Best F1: {best_score:.4f}")

    results.sort(key=lambda x: x[1]['f1'], reverse=True)
    pd.DataFrame([
        {**params, **metrics} for params, metrics in results
    ]).to_csv('parameter_search_results.csv', index=False)

    return best_params


def main():
    train_loader, X_test, y_test = load_data('train_smote_pca.csv', 'test_smote_pca.csv')

    # Parameter Search
    print("\nStart parameter search...")
    best_params = parameter_search(train_loader, X_test, y_test, num_trials=40)

    # Final training
    print("\nConduct the final training using the best parameters...")
    final_params = {
        **best_params,
        'num_epochs': 150,  # Extend the training rounds
        'kl_weight': best_params['kl_weight'] * 0.5  # Fine-tune the KL weight
    }

    final_model = BNN(
        input_dim=21,
        output_dim=4,
        hidden_dims=best_params['hidden_dims'],
        dropout_rate=best_params['dropout_rate']
    )
    train_reptile(final_model, train_loader, final_params)

    # Final evaluation
    final_metrics = evaluate(final_model, X_test, y_test, num_samples=50)
    print("\nFinal model performance:")
    print("Confusion Matrix:")
    print(final_metrics['confusion_matrix'])
    for metric in ['accuracy', 'precision', 'recall', 'f1']:
        print(f"{metric.capitalize()}: {final_metrics[metric]:.4f}")

    torch.save(final_model.state_dict(), 'final_model.pth')


if __name__ == '__main__':
    main()