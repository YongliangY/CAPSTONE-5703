import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import xgboost as xgb
import joblib 


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, Flatten, LSTM
from tensorflow.keras.optimizers import Adam


data = pd.read_excel("database.xlsx", sheet_name="Maintable (M1) (CORRECTED)")
for col in ["Paper No", "Specimen"]:
    if col in data.columns:
        data.drop(col, axis=1, inplace=True)

data['failure_mode'] = data['failure_mode'].astype(int)
feature_cols = [col for col in data.columns if col != "failure_mode"]
X = data[feature_cols]
y = data["failure_mode"]

X.fillna(X.mean(), inplace=True)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)


def evaluate_traditional_model(model, model_name):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"==== {model_name} ====")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    print("\n")


rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
evaluate_traditional_model(rf_model, "Random Forest")

xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
evaluate_traditional_model(xgb_model, "XGBoost")

svm_model = SVC(kernel='rbf', probability=True, random_state=42)
evaluate_traditional_model(svm_model, "Support Vector Machine")

lr_model = LogisticRegression(max_iter=1000, random_state=42)
evaluate_traditional_model(lr_model, "Logistic Regression")


joblib.dump(rf_model, "rf_model.pkl")
joblib.dump(xgb_model, "xgb_model.pkl")
joblib.dump(svm_model, "svm_model.pkl")
joblib.dump(lr_model, "lr_model.pkl")

# FFNN
def create_ffnn_model(input_dim):
    # Create a simple Feedforward Neural Network model
    model = Sequential()
    model.add(Dense(128, input_dim=input_dim, activation='relu'))  # Input layer and first hidden layer
    model.add(Dense(64, activation='relu'))  # Second hidden layer
    model.add(Dense(32, activation='relu'))  # Third hidden layer
    model.add(Dense(3, activation='softmax'))  # Output layer
    return model

ffnn_model = create_ffnn_model(X_train.shape[1])
ffnn_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
print("Training FFNN...")
ffnn_model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)
ffnn_model.save("ffnn_model.h5")

# CNN
X_train_cnn = np.expand_dims(X_train, axis=-1)
X_test_cnn = np.expand_dims(X_test, axis=-1)

def create_cnn_model(input_shape):
    # Create a simple CNN model using 1D convolution for structured data
    model = Sequential()
    model.add(Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=input_shape))  # Convolution layer
    model.add(MaxPooling1D(pool_size=2))  # Max pooling layer
    model.add(Flatten())  # Flatten layer
    model.add(Dense(64, activation='relu'))  # Dense layer
    model.add(Dense(3, activation='softmax'))  # Output layer
    return model

cnn_model = create_cnn_model((X_train_cnn.shape[1], X_train_cnn.shape[2]))
cnn_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
print("Training CNN...")
cnn_model.fit(X_train_cnn, y_train, epochs=50, batch_size=32, validation_split=0.2)
cnn_model.save("cnn_model.h5")

# LSTM
# 
X_train_lstm = np.expand_dims(X_train, axis=1)  # 每个样本作为一个时间步长
X_test_lstm = np.expand_dims(X_test, axis=1)

def create_lstm_model(input_shape):
    # Create a simple LSTM model for structured data
    model = Sequential()
    model.add(LSTM(64, input_shape=input_shape, return_sequences=True))  # First LSTM layer with return sequences
    model.add(LSTM(32))  # Second LSTM layer
    model.add(Dense(3, activation='softmax'))  # Output layer
    return model

lstm_model = create_lstm_model((X_train_lstm.shape[1], X_train_lstm.shape[2]))
lstm_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
print("Training LSTM...")
lstm_model.fit(X_train_lstm, y_train, epochs=50, batch_size=32, validation_split=0.2)
lstm_model.save("lstm_model.h5")
