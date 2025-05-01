import pandas as pd
import numpy as np
import os
import requests
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Dense, Dropout, concatenate
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam

# Download dataset
url = 'https://raw.githubusercontent.com/dcleres/Parkinson_Disease_ML/refs/heads/master/pd_speech_features.csv'
file_name = 'pd_speech_features.csv'

try:
    response = requests.get(url)
    response.raise_for_status()
    with open(file_name, 'wb') as file:
        file.write(response.content)
    print(f"File '{file_name}' downloaded successfully!")
except requests.exceptions.RequestException as e:
    print(f"Error: Failed to download or find the file '{file_name}'. Exception: {e}")

try:
    pd_speech_features = pd.read_csv(file_name)
    print("CSV downloaded and loaded successfully!")
except FileNotFoundError:
    print(f"Error: Failed to download or find the file '{file_name}'")
except Exception as e:
    print(f"An error occurred: {e}")

header = pd_speech_features.iloc[0]
pd_speech_features = pd_speech_features[1:]
pd_speech_features.columns = header
pd_speech_features = pd_speech_features.apply(pd.to_numeric)
pd_speech_features = pd_speech_features.astype(float)
pd_speech_features[['id', 'numPulses', 'numPeriodsPulses']] = pd_speech_features[['id', 'numPulses', 'numPeriodsPulses']].astype(int)
pd_speech_features[['gender', 'class']] = pd_speech_features[['gender', 'class']].astype('category')

labels = pd_speech_features['class'].astype(int)
person_ids = pd_speech_features['id'].astype(int)

import tensorflow as tf
from tensorflow.keras import mixed_precision

# ✅ Enable mixed precision (speeds up RTX 3050)
mixed_precision.set_global_policy('mixed_float16')

# ✅ GPU Check
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print("✅ GPU(s) detected:", gpus)
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print("Memory growth setting failed:", e)
else:
    print("⚠️ No GPU found. Training will run on CPU.")

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

# Feature sets
mfccs = pd_speech_features.iloc[:, 22:84].astype(float)
wavelets = pd_speech_features.iloc[:, 84:148].astype(float)
tqwts = pd_speech_features.iloc[:, 148:].astype(float)

# Normalize
scaler = MinMaxScaler()
mfccs_scaled = scaler.fit_transform(mfccs)
wavelets_scaled = scaler.fit_transform(wavelets)
tqwts_scaled = scaler.fit_transform(tqwts)

# Setup LOPO-CV
logo = LeaveOneGroupOut()
splits = list(logo.split(mfccs_scaled, labels, groups=person_ids))
print(f"Total folds (equal to total unique persons): {len(np.unique(person_ids))}")

def build_model_level_cnn(input_shapes):
    inputs, branches = [], []
    for shape in input_shapes:
        inp = Input(shape=shape)
        x = Conv1D(32, kernel_size=3, activation='relu', kernel_regularizer=l2(0.001))(inp)
        x = Conv1D(32, kernel_size=3, activation='relu', kernel_regularizer=l2(0.001))(x)
        x = MaxPooling1D(pool_size=2)(x)
        x = Conv1D(64, kernel_size=3, activation='relu', kernel_regularizer=l2(0.001))(x)
        x = Conv1D(64, kernel_size=3, activation='relu', kernel_regularizer=l2(0.001))(x)
        x = MaxPooling1D(pool_size=2)(x)
        x = Conv1D(128, kernel_size=3, activation='relu', kernel_regularizer=l2(0.001))(x)
        x = Conv1D(128, kernel_size=3, activation='relu', kernel_regularizer=l2(0.001))(x)
        x = MaxPooling1D(pool_size=2)(x)
        x = Flatten()(x)
        inputs.append(inp)
        branches.append(x)

    merged = concatenate(branches)
    merged = Dropout(0.3)(merged)
    merged = Dense(64, activation='relu')(merged)
    output = Dense(1, activation='sigmoid', dtype='float32')(merged)  # float32 because of mixed precision

    model = Model(inputs=inputs, outputs=output)
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    return model

all_preds, all_labels = [], []
total_folds = len(splits)

for fold_counter, (train_idx, test_idx) in enumerate(splits[:total_folds // 2], start=1):
    print(f"\n🔄 Fold {fold_counter} / {total_folds // 2}: Training...")

    X_train_mfcc = mfccs_scaled[train_idx]
    X_test_mfcc = mfccs_scaled[test_idx]
    X_train_wavelet = wavelets_scaled[train_idx]
    X_test_wavelet = wavelets_scaled[test_idx]
    X_train_tqwt = tqwts_scaled[train_idx]
    X_test_tqwt = tqwts_scaled[test_idx]

    X_train = [np.expand_dims(X_train_mfcc, axis=2),
               np.expand_dims(X_train_wavelet, axis=2),
               np.expand_dims(X_train_tqwt, axis=2)]
    X_test = [np.expand_dims(X_test_mfcc, axis=2),
              np.expand_dims(X_test_wavelet, axis=2),
              np.expand_dims(X_test_tqwt, axis=2)]
    y_train, y_test = labels.iloc[train_idx], labels.iloc[test_idx]

    # ✅ Use tf.data pipeline
    train_dataset = tf.data.Dataset.from_tensor_slices((tuple(X_train), y_train))
    train_dataset = train_dataset.shuffle(1024).batch(64).prefetch(tf.data.AUTOTUNE)

    test_dataset = tf.data.Dataset.from_tensor_slices((tuple(X_test), y_test))
    test_dataset = test_dataset.batch(64).prefetch(tf.data.AUTOTUNE)

    model = build_model_level_cnn([x.shape[1:] for x in X_train])
    model.fit(train_dataset, epochs=200, verbose=0)

    y_pred = (model.predict(test_dataset).flatten() > 0.5).astype(int)
    all_preds.extend(y_pred)
    all_labels.extend(y_test)

    print(f"✅ Fold {fold_counter}: Accuracy = {accuracy_score(y_test, y_pred):.4f}")

accuracy = accuracy_score(all_labels, all_preds)
f1 = f1_score(all_labels, all_preds)
mcc = matthews_corrcoef(all_labels, all_preds)

print(f"\n🏁 Full LOPO Results After All Folds:")
print(f"🔹 Accuracy : {accuracy:.4f}")
print(f"🔹 F1-Score : {f1:.4f}")
print(f"🔹 MCC      : {mcc:.4f}")