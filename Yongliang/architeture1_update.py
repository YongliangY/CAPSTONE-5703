# -*- coding: utf-8 -*-
"""architeture1_update.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/10yVgsK3oRp3hqY1NaC1-2IQJUL430PD7
"""

#!/usr/bin/env python3
"""
Ensemble Learning Code: Base models (Logistic, KNN, SVM, RF, XGBoost, LightGBM, ExtraTrees, NB),
hyperparameter tuning via GridSearchCV, calibration, stacking, and evaluation.
"""

import numpy as np
import pandas as pd
import random

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, precision_score, f1_score

# Set random seed for reproducibility
seed = 42
np.random.seed(seed)
random.seed(seed)

# Try to import LightGBM; skip if not available
try:
    from lightgbm import LGBMClassifier
    lightgbm_available = True
except ImportError:
    lightgbm_available = False
    print("Warning: lightgbm is not installed. LightGBM model will be skipped.")

def load_data(train_path, test_path):
    """
    Load training and test data from CSV files.
    Assumes target column is named 'failure mode'.
    """
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    # Separate features and target
    X_train = train_df.drop('failure mode', axis=1).values
    y_train = train_df['failure mode'].values
    X_test = test_df.drop('failure mode', axis=1).values
    y_test = test_df['failure mode'].values
    return X_train, y_train, X_test, y_test

def define_models_and_grids():
    """
    Define base models and their hyperparameter grids.
    Returns:
        models: dict of model name to scikit-learn Pipeline or estimator.
        param_grids: dict of model name to hyperparameter grid.
    """
    models = {}
    param_grids = {}

    # Logistic Regression (with StandardScaler)
    lr_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(random_state=seed, max_iter=1000))
    ])
    models['Logistic Regression'] = lr_pipeline
    param_grids['Logistic Regression'] = {
        'clf__C': [0.1, 1, 10],
        'clf__penalty': ['l2'],
        'clf__solver': ['lbfgs'],
        'clf__multi_class': ['auto']
    }

    # K-Nearest Neighbors (with StandardScaler)
    knn_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', KNeighborsClassifier())
    ])
    models['K-Nearest Neighbors'] = knn_pipeline
    param_grids['K-Nearest Neighbors'] = {
        'clf__n_neighbors': [3, 5, 7],
        'clf__weights': ['uniform', 'distance']
    }

    # Support Vector Machine (RBF kernel, with StandardScaler)
    svm_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', SVC(kernel='rbf', probability=True, random_state=seed))
    ])
    models['SVM (RBF)'] = svm_pipeline
    param_grids['SVM (RBF)'] = {
        'clf__C': [0.1, 1, 10],
        'clf__gamma': ['scale', 'auto']
    }

    # Random Forest (no scaling needed)
    rf_pipeline = Pipeline([
        ('clf', RandomForestClassifier(random_state=seed))
    ])
    models['Random Forest'] = rf_pipeline
    param_grids['Random Forest'] = {
        'clf__n_estimators': [100, 200],
        'clf__max_depth': [None, 10, 20],
        'clf__min_samples_split': [2, 5]
    }

    # XGBoost (no scaling needed)
    xgb_pipeline = Pipeline([
        ('clf', XGBClassifier(random_state=seed, use_label_encoder=False, eval_metric='mlogloss'))
    ])
    models['XGBoost'] = xgb_pipeline
    param_grids['XGBoost'] = {
        'clf__n_estimators': [100, 200],
        'clf__max_depth': [3, 6],
        'clf__learning_rate': [0.01, 0.1],
        'clf__subsample': [0.8, 1.0]
    }

    # LightGBM (no scaling needed), if available
    if lightgbm_available:
        lgbm_pipeline = Pipeline([
            ('clf', LGBMClassifier(random_state=seed))
        ])
        models['LightGBM'] = lgbm_pipeline
        param_grids['LightGBM'] = {
            'clf__n_estimators': [100, 200],
            'clf__num_leaves': [31, 50],
            'clf__learning_rate': [0.01, 0.1]
        }

    # Extra Trees (no scaling needed)
    et_pipeline = Pipeline([
        ('clf', ExtraTreesClassifier(random_state=seed))
    ])
    models['Extra Trees'] = et_pipeline
    param_grids['Extra Trees'] = {
        'clf__n_estimators': [100, 200],
        'clf__max_depth': [None, 10, 20],
        'clf__min_samples_split': [2, 5]
    }

    # Gaussian Naive Bayes (no scaling needed)
    nb_pipeline = Pipeline([
        ('clf', GaussianNB())
    ])
    models['Gaussian NB'] = nb_pipeline
    param_grids['Gaussian NB'] = {
        'clf__var_smoothing': [1e-9, 1e-8, 1e-7]
    }

    return models, param_grids

def tune_models(X, y, models, param_grids, cv):
    """
    Perform GridSearchCV to tune hyperparameters for each model.
    Returns a dictionary of best-estimator models.
    """
    best_models = {}
    for name, model in models.items():
        print(f"Tuning {name}...")
        grid = GridSearchCV(model, param_grids[name], cv=cv, n_jobs=-1)
        grid.fit(X, y)
        best_models[name] = grid.best_estimator_
        print(f"Best params for {name}: {grid.best_params_}")
    return best_models

def calibrate_models(best_models, X, y, cv):
    """
    Calibrate each tuned model using CalibratedClassifierCV (Platt scaling).
    Returns a dictionary of calibrated classifier models.
    """
    calibrated_models = {}
    for name, model in best_models.items():
        print(f"Calibrating {name}...")
        # Use cv on calibration to avoid data leakage; clones base estimator internally
        calibrator = CalibratedClassifierCV(estimator=model, method='isotonic',cv=cv)
        #calibrator = CalibratedClassifierCV(estimator=model, method='sigmoid', cv=cv)
        calibrator.fit(X, y)
        calibrated_models[name] = calibrator
    return calibrated_models

def generate_oof_predictions(best_models, X, y, cv):
    """
    Generate out-of-fold (OOF) predictions (probabilities) for each base model.
    Returns a 2D array of shape (n_samples, n_models * n_classes).
    """
    oof_list = []
    for name, model in best_models.items():
        print(f"Generating OOF predictions for {name}...")
        probs = cross_val_predict(model, X, y, cv=cv, method='predict_proba', n_jobs=-1)
        oof_list.append(probs)
    # Concatenate horizontally to form meta-features
    # Each element in oof_list is (n_samples, n_classes)
    X_meta = np.hstack(oof_list)
    return X_meta

def train_meta_learner(X_meta, y):
    """
    Train a Logistic Regression meta-learner on OOF predictions.
    """
    print("Training meta-learner (Logistic Regression)...")
    meta = LogisticRegression(random_state=seed, max_iter=1000)
    meta.fit(X_meta, y)
    return meta

def evaluate_models(models, X_test, y_test):
    """
    Evaluate given models on test data.
    Returns a list of tuples: (model_name, accuracy, macro_precision, macro_f1).
    """
    results = []
    for name, model in models.items():
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, average='macro', zero_division=0)
        f1 = f1_score(y_test, preds, average='macro', zero_division=0)
        results.append((name, acc, prec, f1))
    return results

def main():
    # Load data
    X_train, y_train, X_test, y_test = load_data('train_smote.csv', 'test_original.csv')


    from sklearn.preprocessing import LabelEncoder

    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    y_test = le.fit_transform(y_test)


    # Stratified CV setup
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    # Define models and hyperparameter grids
    models, param_grids = define_models_and_grids()

    # Hyperparameter tuning
    best_models = tune_models(X_train, y_train, models, param_grids, cv)

    # Calibrate tuned models
    calibrated_models = calibrate_models(best_models, X_train, y_train, cv)

    # Generate OOF predictions for stacking
    X_meta_train = generate_oof_predictions(best_models, X_train, y_train, cv)

    # Train meta-learner
    meta_model = train_meta_learner(X_meta_train, y_train)

    # Evaluate base models on test data
    base_results = []
    for name in best_models:
        # Use calibrated model for final predictions
        calibrator = calibrated_models[name]
        y_pred = calibrator.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
        base_results.append((name, acc, prec, f1))

    # Soft voting ensemble (average probabilities)
    print("Generating Soft Voting ensemble predictions...")
    probs_list = []
    for name in calibrated_models:
        probs = calibrated_models[name].predict_proba(X_test)
        probs_list.append(probs)
    avg_probs = np.mean(probs_list, axis=0)
    classes = calibrated_models[list(calibrated_models.keys())[0]].classes_
    soft_preds = classes[np.argmax(avg_probs, axis=1)]
    acc = accuracy_score(y_test, soft_preds)
    prec = precision_score(y_test, soft_preds, average='macro', zero_division=0)
    f1 = f1_score(y_test, soft_preds, average='macro', zero_division=0)
    base_results.append(("Soft Voting", acc, prec, f1))

    # Stacked ensemble (meta-learner)
    print("Generating Stacked ensemble predictions...")
    meta_features_test = []
    for name in calibrated_models:
        probs = calibrated_models[name].predict_proba(X_test)
        meta_features_test.append(probs)
    X_meta_test = np.hstack(meta_features_test)
    stacked_preds = meta_model.predict(X_meta_test)
    acc = accuracy_score(y_test, stacked_preds)
    prec = precision_score(y_test, stacked_preds, average='macro', zero_division=0)
    f1 = f1_score(y_test, stacked_preds, average='macro', zero_division=0)
    base_results.append(("Stacked Ensemble", acc, prec, f1))

    # Present results in a clean table
    results_df = pd.DataFrame(base_results, columns=['Model', 'Accuracy', 'Macro Precision', 'Macro F1'])
    print("\nFinal Evaluation Results:")
    print(results_df.to_string(index=False, float_format='%.4f'))

    from sklearn.metrics import classification_report, confusion_matrix

    stacked_preds = meta_model.predict(X_meta_test)  # or re-generate from base models
    print("Stacked Ensemble Classification Report:")
    print(classification_report(y_test, stacked_preds, digits=4))

    cm = confusion_matrix(y_test, stacked_preds)

    class_acc = cm.diagonal() / cm.sum(axis=1)
    for i, acc in enumerate(class_acc):
        print(f"Class {i} accuracy: {acc:.4f}")

if __name__ == "__main__":
    main()

Stacked Ensemble Classification Report:
              precision    recall  f1-score   support

           0     0.8810    1.0000    0.9367        37
           1     0.7500    0.5000    0.6000         6
           2     0.9722    0.9211    0.9459        38
           3     1.0000    0.8000    0.8889         5

    accuracy                         0.9186        86
   macro avg     0.9008    0.8053    0.8429        86
weighted avg     0.9191    0.9186    0.9145        86

Class 0 accuracy: 1.0000
Class 1 accuracy: 0.5000
Class 2 accuracy: 0.9211
Class 3 accuracy: 0.8000

#wihtout

# Stacking Ensemble Model Training
#
# 1. Data Loading
def load_data(train_path, test_path, target_col='failure mode'):
    """Load training and testing data from CSV files."""
    import pandas as pd
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    # Separate features and target
    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col]
    X_test = test_df.drop(columns=[target_col])
    y_test = test_df[target_col]
    return X_train, X_test, y_train, y_test

# 2. Define Base Models and Hyperparameter Grids
def define_models_and_parameters(random_seed=42):
    """Define diverse base models and their hyperparameter grids."""
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
    from sklearn.naive_bayes import GaussianNB
    models = []
    # Logistic Regression (scaled)
    lr_pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(max_iter=1000, random_state=random_seed))
    ])
    lr_param_grid = {'clf__C': [0.1, 1.0, 10.0]}
    models.append(('Logistic Regression', lr_pipe, lr_param_grid))
    # K-Nearest Neighbors (scaled)
    knn_pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', KNeighborsClassifier())
    ])
    knn_param_grid = {
        'clf__n_neighbors': [3, 5, 7, 9],
        'clf__weights': ['uniform', 'distance']
    }
    models.append(('KNN', knn_pipe, knn_param_grid))
    # SVM with RBF kernel (scaled)
    svm_pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', SVC(kernel='rbf', probability=True, random_state=random_seed))
    ])
    svm_param_grid = {
        'clf__C': [0.1, 1.0, 10.0],
        'clf__gamma': [0.01, 0.1, 1.0]
    }
    models.append(('SVM (RBF)', svm_pipe, svm_param_grid))
    # Random Forest
    rf_pipe = Pipeline([
        ('clf', RandomForestClassifier(random_state=random_seed))
    ])
    rf_param_grid = {
        'clf__n_estimators': [100, 300],
        'clf__max_depth': [None, 5, 10]
    }
    models.append(('Random Forest', rf_pipe, rf_param_grid))
    # XGBoost (if available)
    try:
        from xgboost import XGBClassifier
    except ImportError:
        XGBClassifier = None
    if XGBClassifier is not None:
        xgb_pipe = Pipeline([
            ('clf', XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=random_seed))
        ])
        xgb_param_grid = {
            'clf__n_estimators': [100, 300],
            'clf__max_depth': [3, 6],
            'clf__learning_rate': [0.1, 0.01]
        }
        models.append(('XGBoost', xgb_pipe, xgb_param_grid))
    # LightGBM (if available)
    try:
        from lightgbm import LGBMClassifier
    except ImportError:
        LGBMClassifier = None
    if LGBMClassifier is not None:
        lgb_pipe = Pipeline([
            ('clf', LGBMClassifier(random_state=random_seed))
        ])
        lgb_param_grid = {
            'clf__n_estimators': [100, 300],
            'clf__num_leaves': [31, 63],
            'clf__learning_rate': [0.1, 0.01]
        }
        models.append(('LightGBM', lgb_pipe, lgb_param_grid))
    # Extra Trees
    #et_pipe = Pipeline([
    #    ('clf', ExtraTreesClassifier(random_state=random_seed))
    #])
    #et_param_grid = {
    #    'clf__n_estimators': [100, 300],
    #    'clf__max_depth': [None, 5, 10]
    #}
    #models.append(('Extra Trees', et_pipe, et_param_grid))
    # Naive Bayes (GaussianNB)
    #nb_pipe = Pipeline([
    #    ('clf', GaussianNB())
    #])
    #nb_param_grid = {
    #    'clf__var_smoothing': [1e-9, 1e-8, 1e-7]
    #}
    #models.append(('Naive Bayes', nb_pipe, nb_param_grid))
    return models

# 3. Hyperparameter Tuning with GridSearchCV
def tune_models(models, X_train, y_train, random_seed=42):
    """Tune each model using 5-fold Stratified CV and return the best estimators."""
    from sklearn.model_selection import StratifiedKFold, GridSearchCV
    best_models = {}
    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_seed)
    for name, pipeline, param_grid in models:
        grid = GridSearchCV(pipeline, param_grid, cv=cv_strategy, scoring='accuracy', n_jobs=-1, refit=True)
        grid.fit(X_train, y_train)
        best_models[name] = grid.best_estimator_
    return best_models

# 4. Generate Out-of-Fold Predictions for Stacking
def get_oof_predictions(best_models, X_train, y_train, random_seed=42):
    """Generate out-of-fold training set predictions (probabilities) for each base model."""
    import numpy as np
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_seed)
    oof_preds = {}
    for name, model in best_models.items():
        oof_prob = cross_val_predict(model, X_train, y_train, cv=cv_strategy, method='predict_proba')
        oof_preds[name] = oof_prob
    return oof_preds

# 5. Train Meta-Learner on OOF Predictions
def train_meta_learner(oof_preds, y_train, random_seed=42):
    """Train a Logistic Regression meta-learner on the concatenated OOF predictions."""
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    meta_features = np.hstack([oof_preds[name] for name in oof_preds.keys()])
    meta_clf = LogisticRegression(max_iter=1000, random_state=random_seed)
    meta_clf.fit(meta_features, y_train)
    return meta_clf

# 6. Final Predictions and Evaluation
def predict_and_evaluate(base_models, meta_clf, X_test, y_test):
    """Evaluate base models, soft-voting ensemble, and stacked ensemble on the test set."""
    import numpy as np
    import pandas as pd
    from sklearn.metrics import accuracy_score, precision_score, f1_score
    results = []
    # Base models
    for name, model in base_models.items():
        y_pred = model.predict(X_test)
        results.append({
            'Model': name,
            'Accuracy': accuracy_score(y_test, y_pred),
            'Macro Precision': precision_score(y_test, y_pred, average='macro', zero_division=0),
            'Macro F1': f1_score(y_test, y_pred, average='macro')
        })
    # Soft voting ensemble (average probabilities)
    prob_sum = None
    for model in base_models.values():
        probs = model.predict_proba(X_test)
        prob_sum = probs if prob_sum is None else prob_sum + probs
    avg_proba = prob_sum / len(base_models)
    ensemble_pred = base_models[next(iter(base_models))].classes_[np.argmax(avg_proba, axis=1)]
    results.append({
        'Model': 'Soft Voting Ensemble',
        'Accuracy': accuracy_score(y_test, ensemble_pred),
        'Macro Precision': precision_score(y_test, ensemble_pred, average='macro', zero_division=0),
        'Macro F1': f1_score(y_test, ensemble_pred, average='macro')
    })
    # Stacked ensemble (meta-learner)
    base_prob_test = np.hstack([model.predict_proba(X_test) for model in base_models.values()])
    stack_pred = meta_clf.predict(base_prob_test)
    results.append({
        'Model': 'Stacked Ensemble',
        'Accuracy': accuracy_score(y_test, stack_pred),
        'Macro Precision': precision_score(y_test, stack_pred, average='macro', zero_division=0),
        'Macro F1': f1_score(y_test, stack_pred, average='macro')
    })
    results_df = pd.DataFrame(results)
    results_df[['Accuracy', 'Macro Precision', 'Macro F1']] = results_df[['Accuracy', 'Macro Precision', 'Macro F1']].round(4)
    return results_df

# 7. Main Execution
def main():
    # Paths to the preprocessed CSV files
    train_path = 'train_smote.csv'
    test_path = 'test_original.csv'
    # Load data
    X_train, X_test, y_train, y_test = load_data(train_path, test_path)


    from sklearn.preprocessing import LabelEncoder

    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    y_test = le.fit_transform(y_test)
    # Define base models and hyperparameter grids
    models = define_models_and_parameters(random_seed=42)
    # Tune models and get best estimators
    best_models = tune_models(models, X_train, y_train, random_seed=42)
    # Generate out-of-fold predictions from base models
    oof_preds = get_oof_predictions(best_models, X_train, y_train, random_seed=42)
    # Train meta-learner on OOF predictions
    meta_clf = train_meta_learner(oof_preds, y_train, random_seed=42)
    # Evaluate on test data
    results_df = predict_and_evaluate(best_models, meta_clf, X_test, y_test)
    # Print final results table
    print(results_df.set_index('Model'))

# Run the main routine
if __name__ == "__main__":
    main()

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_predict
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, precision_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.naive_bayes import GaussianNB

# Optional imports for XGBoost and LightGBM if available
try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None
try:
    from lightgbm import LGBMClassifier
except ImportError:
    LGBMClassifier = None

def load_data(train_path, test_path, target_col='failure mode'):
    """Load train and test data, separate features and target."""
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    X_train = train.drop(target_col, axis=1)
    y_train = train[target_col]
    X_test = test.drop(target_col, axis=1)
    y_test = test[target_col]
    return X_train, y_train, X_test, y_test

def get_base_models():
    """Define base models and their hyperparameter grids."""
    random_state = 42
    models = []
    # Logistic Regression with scaling
    models.append(('LogisticRegression',
                   Pipeline([
                       ('scaler', StandardScaler()),
                       ('clf', LogisticRegression(random_state=random_state, max_iter=1000, multi_class='auto', solver='lbfgs'))
                   ]),
                   {'clf__C': [0.01, 0.1, 1, 10]}
                  ))
    # K-Nearest Neighbors with scaling
    models.append(('KNeighbors',
                   Pipeline([
                       ('scaler', StandardScaler()),
                       ('clf', KNeighborsClassifier())
                   ]),
                   {'clf__n_neighbors': [3, 5, 7],
                    'clf__weights': ['uniform', 'distance']}
                  ))
    # SVM RBF with scaling
    models.append(('SVM',
                   Pipeline([
                       ('scaler', StandardScaler()),
                       ('clf', SVC(kernel='rbf', probability=True, random_state=random_state))
                   ]),
                   {'clf__C': [0.1, 1, 10],
                    'clf__gamma': ['scale', 'auto']}
                  ))
    # Random Forest
    models.append(('RandomForest',
                   RandomForestClassifier(random_state=random_state),
                   {'n_estimators': [100, 200],
                    'max_depth': [None, 10, 20]}
                  ))
    # Extra Trees Classifier
    #models.append(('ExtraTrees',
    #               ExtraTreesClassifier(random_state=random_state),
    #               {'n_estimators': [100, 200],
    #                'max_depth': [None, 10, 20]}
    #              ))
    # Gaussian Naive Bayes
    #models.append(('GaussianNB',
     #              GaussianNB(),
      #             {'var_smoothing': [1e-9, 1e-8, 1e-7]}
       #           ))
    # XGBoost if available
    if XGBClassifier is not None:
        models.append(('XGBoost',
                       XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=random_state),
                       {'n_estimators': [100, 200],
                        'max_depth': [3, 6],
                        'learning_rate': [0.01, 0.1]}
                      ))
    # LightGBM if available
    if LGBMClassifier is not None:
        models.append(('LightGBM',
                       LGBMClassifier(random_state=random_state),
                       {'n_estimators': [100, 200],
                        'num_leaves': [31, 61],
                        'learning_rate': [0.01, 0.1]}
                      ))
    return models

def tune_models(models, X, y, cv):
    """Tune hyperparameters for each model using GridSearchCV."""
    best_estimators = {}
    for name, estimator, params in models:
        grid = GridSearchCV(estimator, params, cv=cv, scoring='accuracy', n_jobs=-1)
        grid.fit(X, y)
        best_estimators[name] = grid.best_estimator_
        print(f"{name} best params: {grid.best_params_}")
    return best_estimators

def get_oof_predictions(models, X, y, cv):
    """
    Generate out-of-fold predictions (probabilities) for each model after calibration.
    Returns a dictionary of OOF probability arrays for each model.
    """
    oof_preds = {}
    for name, model in models.items():
        # Calibrated classifier with Platt scaling (sigmoid)
        calibrator = CalibratedClassifierCV(estimator=model, method='sigmoid', cv=3)
        # Generate OOF probabilities via cross_val_predict
        oof_proba = cross_val_predict(calibrator, X, y, cv=cv, method='predict_proba', n_jobs=-1)
        oof_preds[name] = oof_proba
        print(f"Generated OOF predictions for {name}")
    return oof_preds

def refit_calibrated_models(models, X, y):
    """Refit each base model with calibration on full training data."""
    calibrated_models = {}
    for name, model in models.items():
        calibrator = CalibratedClassifierCV(estimator=model, method='sigmoid', cv=3)
        calibrator.fit(X, y)
        calibrated_models[name] = calibrator
    return calibrated_models

def evaluate_model(name, y_true, y_pred):
    """Calculate accuracy, macro precision, and macro F1."""
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    return acc, prec, f1

def main():
    # Load data
    X_train, y_train, X_test, y_test = load_data('train_smote.csv', 'test_original.csv', target_col='failure mode')


    from sklearn.preprocessing import LabelEncoder

    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    y_test = le.fit_transform(y_test)
    # Define Stratified K-Fold cross-validator
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Define base models and hyperparameter grids
    base_models = get_base_models()

    # Tune base models
    print("Tuning base models...")
    best_base_models = tune_models(base_models, X_train, y_train, cv=skf)

    # Generate OOF predictions for stacking
    print("\nGenerating out-of-fold predictions...")
    oof_preds = get_oof_predictions(best_base_models, X_train, y_train, cv=skf)

    # Prepare features for meta-learner (concatenate probabilities)
    X_meta = np.hstack([oof_preds[name] for name in oof_preds])

    # Train Logistic Regression as meta-learner on OOF predictions
    meta_clf = LogisticRegression(random_state=42, max_iter=1000)
    meta_clf.fit(X_meta, y_train)

    # Refit calibrated base models on full training data
    print("\nRefitting base models on full training data with calibration...")
    calibrated_models = refit_calibrated_models(best_base_models, X_train, y_train)

    # Make predictions on test set
    print("\nMaking predictions on test set...")
    test_probs = {}
    test_preds = {}
    for name, model in calibrated_models.items():
        prob = model.predict_proba(X_test)
        preds = np.argmax(prob, axis=1) + 1  # classes are 1-indexed (1,2,3,4)
        # If classes in model are not starting at 1, adjust accordingly:
        classes = model.classes_
        # Map predicted indices to original class labels
        preds = classes[np.argmax(prob, axis=1)]
        test_probs[name] = prob
        test_preds[name] = preds

    # Soft voting ensemble: average of probabilities
    all_probs = np.array(list(test_probs.values()))
    avg_prob = np.mean(all_probs, axis=0)
    avg_pred = np.argmax(avg_prob, axis=1)
    # If classes list exists, get actual class labels for each index
    classes = list(calibrated_models.values())[0].classes_
    avg_pred_labels = classes[avg_pred]

    # Stacked ensemble: use meta-learner on base probabilities
    X_test_meta = np.hstack([test_probs[name] for name in oof_preds])
    stacked_pred = meta_clf.predict(X_test_meta)

    # Evaluate all models
    results = []
    for name, preds in test_preds.items():
        acc, prec, f1 = evaluate_model(name, y_test, preds)
        results.append((name, acc, prec, f1))
    # Ensemble results
    acc, prec, f1 = evaluate_model('SoftVoting', y_test, avg_pred_labels)
    results.append(('SoftVoting', acc, prec, f1))
    acc, prec, f1 = evaluate_model('Stacked', y_test, stacked_pred)
    results.append(('Stacked', acc, prec, f1))

    # Print results in a table
    results_df = pd.DataFrame(results, columns=['Model', 'Accuracy', 'Macro Precision', 'Macro F1'])
    print("\nFinal evaluation results:")
    print(results_df.to_string(index=False))

if __name__ == "__main__":
    main()

Accuracy  Macro Precision  Macro F1
Model
Logistic Regression     0.8023           0.6676    0.6820
KNN                     0.8488           0.7161    0.7538
SVM (RBF)               0.8488           0.7187    0.7379
Random Forest           0.8721           0.7361    0.7664
XGBoost                 0.8372           0.6487    0.6618
LightGBM                0.8372           0.6485    0.6725
Soft Voting Ensemble    0.8372           0.6747    0.6926
Stacked Ensemble        0.8953           0.8323    0.7866