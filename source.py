import pandas as pd
import os
import requests

url = 'https://raw.githubusercontent.com/dcleres/Parkinson_Disease_ML/refs/heads/master/pd_speech_features.csv'
file_name = 'pd_speech_features.csv'

# Download the file using requests
try:
    response = requests.get(url)
    response.raise_for_status()  # Check if the request was successful
    with open(file_name, 'wb') as file:
        file.write(response.content)
    print(f"File '{file_name}' downloaded successfully!")
except requests.exceptions.RequestException as e:
    print(f"Error: Failed to download or find the file '{file_name}'. Exception: {e}")

# Load the CSV
try:
    pd_speech_features = pd.read_csv(file_name)
    print("CSV loaded successfully!")
except Exception as e:
    print(f"An error occurred while loading the CSV: {e}")


header = pd_speech_features.iloc[0]
pd_speech_features = pd_speech_features[1:] # removing header row
pd_speech_features.columns = header # Set column header
pd_speech_features.head()


pd_speech_features.describe()

print(pd_speech_features.groupby('class').size()/3)

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


# Step 1.2: Separate important columns
# Convert to correct types
pd_speech_features = pd_speech_features.apply(pd.to_numeric)

pd_speech_features =  pd_speech_features.astype(float) #per default all floats
pd_speech_features[['id', 'numPulses', 'numPeriodsPulses']] = pd_speech_features[['id', 'numPulses', 'numPeriodsPulses']].astype(int)
pd_speech_features[['gender', 'class']] = pd_speech_features[['gender', 'class']].astype('category')
pd_speech_features.dtypes


labels = pd_speech_features['class'].astype(int)   # 0 or 1 (Healthy or PD)
person_ids = pd_speech_features['id'].astype(int)  # Each person's ID for LOPO
all_features = pd_speech_features.drop(['id', 'gender', 'class'], axis=1)


# Step 1.3: Normalize features (MinMax scaling)
scaler = MinMaxScaler()
all_features_scaled = pd.DataFrame(scaler.fit_transform(all_features), columns=all_features.columns)

print("All Features shape:", all_features_scaled.shape)
print("Labels shape:", labels.shape)
print("Person IDs shape:", person_ids.shape)

from sklearn.model_selection import LeaveOneGroupOut

# Step 2.1: Setup LOPO-CV
logo = LeaveOneGroupOut()

# Step 2.2: Create splits
splits = logo.split(all_features_scaled, labels, groups=person_ids)

# Check how many folds
print(f"Total folds (equal to total unique persons): {len(np.unique(person_ids))}")


import tensorflow as tf

# ✅ GPU Check (Insert this once after imports)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print("✅ GPU(s) detected:", gpus)
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)  # Optional: prevent memory hogging
    except RuntimeError as e:
        print("Memory growth setting failed:", e)
else:
    print("⚠️ No GPU found. Training will run on CPU.")


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout
from tensorflow.keras.regularizers import l2

# Step 3.1: Define a simple 9-layer CNN for feature-level combination
def build_feature_level_cnn(input_shape):
    model = Sequential()

    # First Conv block
    model.add(Conv1D(32, kernel_size=8, activation='relu', input_shape=input_shape, kernel_regularizer=l2(0.001)))
    model.add(Conv1D(32, kernel_size=8, activation='relu', kernel_regularizer=l2(0.001)))
    model.add(MaxPooling1D(pool_size=2))

    # Second Conv block
    model.add(Conv1D(64, kernel_size=8, activation='relu', kernel_regularizer=l2(0.001)))
    model.add(Conv1D(64, kernel_size=8, activation='relu', kernel_regularizer=l2(0.001)))
    model.add(MaxPooling1D(pool_size=2))

    # Third Conv block
    model.add(Conv1D(128, kernel_size=8, activation='relu', kernel_regularizer=l2(0.001)))
    model.add(Conv1D(128, kernel_size=8, activation='relu', kernel_regularizer=l2(0.001)))
    model.add(MaxPooling1D(pool_size=2))

    # Dense and output
    model.add(Flatten())
    model.add(Dropout(0.3))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))  # Binary output

    # Compile
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model


from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from sklearn.model_selection import LeaveOneGroupOut

# Prepare LOPO
logo = LeaveOneGroupOut()
splits = logo.split(all_features_scaled, labels, groups=person_ids)

# Create full list of splits once
all_splits = list(splits)
total_folds = len(all_splits)

print(f"Total LOPO folds: {total_folds}")

# Initialize trackers
all_preds = []
all_labels = []

 # Initialize trackers
all_preds = []
all_labels = []

# Run first half
for fold_counter, (train_idx, test_idx) in enumerate(all_splits[:total_folds // 2], start=1):
    print(f"\n🔄 Fold {fold_counter} / {total_folds // 2}: Training...")

    X_train, X_test = all_features_scaled.iloc[train_idx], all_features_scaled.iloc[test_idx]
    y_train, y_test = labels.iloc[train_idx], labels.iloc[test_idx]

    X_train = np.expand_dims(X_train.values, axis=2)
    X_test = np.expand_dims(X_test.values, axis=2)

    model = build_feature_level_cnn(input_shape=(X_train.shape[1], 1))
    model.fit(X_train, y_train, epochs=200, batch_size=16, verbose=0)

    y_pred = (model.predict(X_test).flatten() > 0.5).astype(int)

    all_preds.extend(y_pred)
    all_labels.extend(y_test)

    print(f"✅ Fold {fold_counter}: Accuracy = {accuracy_score(y_test, y_pred):.4f}")


# Evaluate half results
accuracy = accuracy_score(all_labels, all_preds)
f1 = f1_score(all_labels, all_preds)
mcc = matthews_corrcoef(all_labels, all_preds)

print(f"\n🚩 First Half LOPO Results:")
print(f"🔹 Accuracy : {accuracy:.4f}")
print(f"🔹 F1-Score : {f1:.4f}")
print(f"🔹 MCC      : {mcc:.4f}")


# Continue from fold total_folds // 2
for fold_counter, (train_idx, test_idx) in enumerate(all_splits[total_folds // 2:], start=(total_folds // 2) + 1):
    print(f"\n🔄 Fold {fold_counter} / {total_folds}: Training...")

    X_train, X_test = all_features_scaled.iloc[train_idx], all_features_scaled.iloc[test_idx]
    y_train, y_test = labels.iloc[train_idx], labels.iloc[test_idx]

    X_train = np.expand_dims(X_train.values, axis=2)
    X_test = np.expand_dims(X_test.values, axis=2)

    model = build_feature_level_cnn(input_shape=(X_train.shape[1], 1))
    model.fit(X_train, y_train, epochs=200, batch_size=16, verbose=0)

    y_pred = (model.predict(X_test).flatten() > 0.5).astype(int)

    all_preds.extend(y_pred)
    all_labels.extend(y_test)

    print(f"✅ Fold {fold_counter}: Accuracy = {accuracy_score(y_test, y_pred):.4f}")


# Final full metrics
accuracy = accuracy_score(all_labels, all_preds)
f1 = f1_score(all_labels, all_preds)
mcc = matthews_corrcoef(all_labels, all_preds)

print(f"\n🏁 Full LOPO Results After All Folds:")
print(f"🔹 Accuracy : {accuracy:.4f}")
print(f"🔹 F1-Score : {f1:.4f}")
print(f"🔹 MCC      : {mcc:.4f}")