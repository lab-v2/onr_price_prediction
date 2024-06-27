from sklearn.preprocessing import StandardScaler
import numpy as np

def prepare_features_and_target(df, feature_columns, target_column):
    X = df[feature_columns].values
    y = df[target_column].values
    return X, y

# Feature scaling using StandardScaler, redoing this so we scale train / test separately to avoid leakage
def scale_features(X_train, X_test, X_val):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_val_scaled = scaler.fit_transform(X_val)
    return X_train_scaled, X_test_scaled, X_val_scaled

def scale_features_no_val(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled

# Function to create sequences from the scaled data
def create_sequences(X_scaled, y, window_size):
    X_sequences, y_sequences = [], []
    for i in range(len(X_scaled) - window_size + 1):
        X_sequences.append(X_scaled[i:i + window_size, :])
        y_sequences.append(y[i + window_size - 1])
    return np.array(X_sequences), np.array(y_sequences)

