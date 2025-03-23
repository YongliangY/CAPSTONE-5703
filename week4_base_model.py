import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load data from "database.xlsx", sheet "Maintable (M1) (CORRECTED)"
data = pd.read_excel("database.xlsx", sheet_name="Maintable (M1) (CORRECTED)")

# Drop non-numeric columns if present
for col in ["Paper No", "Specimen"]:
    if col in data.columns:
        data.drop(col, axis=1, inplace=True)

# Ensure the target column 'failure_mode' is integer type
data['failure_mode'] = data['failure_mode'].astype(int)

# Define features and target
feature_cols = [col for col in data.columns if col != "failure_mode"]
X = data[feature_cols]
y = data["failure_mode"]

# Fill missing values and scale features (assuming normalization to roughly [-1, 1])
X.fillna(X.mean(), inplace=True)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Define functions to build the five sub-models as described in the paper
# All models use the 'tanh' activation, Adam optimizer with learning rate 0.01, and are compiled for 4-class classification.

def build_submodel_1(input_dim):
    model = Sequential()
    model.add(Dense(40, input_dim=input_dim, activation='tanh'))
    model.add(Dropout(0.02))
    model.add(Dense(80, activation='tanh'))
    model.add(Dense(4, activation='softmax'))
    optimizer = Adam(learning_rate=0.01)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def build_submodel_2(input_dim):
    model = Sequential()
    model.add(Dense(25, input_dim=input_dim, activation='tanh'))
    model.add(Dropout(0.01))
    model.add(Dense(80, activation='tanh'))
    model.add(Dense(20, activation='tanh'))
    model.add(Dense(4, activation='softmax'))
    optimizer = Adam(learning_rate=0.01)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def build_submodel_3(input_dim):
    model = Sequential()
    model.add(Dense(40, input_dim=input_dim, activation='tanh'))
    model.add(Dropout(0.01))
    model.add(Dense(60, activation='tanh'))
    model.add(Dropout(0.01))
    model.add(Dense(60, activation='tanh'))
    model.add(Dense(15, activation='tanh'))
    model.add(Dense(4, activation='softmax'))
    optimizer = Adam(learning_rate=0.01)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def build_submodel_4(input_dim):
    model = Sequential()
    model.add(Dense(25, input_dim=input_dim, activation='tanh'))
    model.add(Dense(20, activation='tanh'))
    model.add(Dense(80, activation='tanh'))
    model.add(Dense(80, activation='tanh'))
    model.add(Dense(40, activation='tanh'))
    model.add(Dense(4, activation='softmax'))
    optimizer = Adam(learning_rate=0.01)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def build_submodel_5(input_dim):
    model = Sequential()
    model.add(Dense(20, input_dim=input_dim, activation='tanh'))
    model.add(Dense(50, activation='tanh'))
    model.add(Dense(30, activation='tanh'))
    model.add(Dense(60, activation='tanh'))
    model.add(Dense(80, activation='tanh'))
    model.add(Dense(25, activation='tanh'))
    model.add(Dense(4, activation='softmax'))
    optimizer = Adam(learning_rate=0.01)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Build all sub-models using the input dimension of our dataset
input_dim = X_train.shape[1]
submodel1 = build_submodel_1(input_dim)
submodel2 = build_submodel_2(input_dim)
submodel3 = build_submodel_3(input_dim)
submodel4 = build_submodel_4(input_dim)
submodel5 = build_submodel_5(input_dim)

# Print model summaries to verify architecture
print("Submodel 1 Summary:")
submodel1.summary()
print("\nSubmodel 2 Summary:")
submodel2.summary()
print("\nSubmodel 3 Summary:")
submodel3.summary()
print("\nSubmodel 4 Summary:")
submodel4.summary()
print("\nSubmodel 5 Summary:")
submodel5.summary()

# Optionally, train each sub-model (example for submodel1):
# history1 = submodel1.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)
