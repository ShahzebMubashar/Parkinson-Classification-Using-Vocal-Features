import pandas as pd
import numpy as np
import requests
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
import tensorflow as tf
from tensorflow.keras import mixed_precision
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Dense, Dropout, concatenate
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam

# ✅ Enable mixed precision for RTX 3050
mixed_precision.set_global_policy('mixed_float16')

# ✅ GPU check and config
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

# ✅ Download dataset
url = 'https://raw.githubusercontent.com/dcleres/Parkinson_Disease_ML/refs/heads/master/pd_speech_features.csv'
file_name = 'pd_speech_features.csv'
response = requests.get(url)
with open(file_name, 'wb') as file:
    file.write(response.content)
print(f"✅ File '{file_name}' downloaded successfully!")

# ✅ Load and preprocess dataset
df = pd.read_csv(file_name)
header = df.iloc[0]
df = df[1:]
df.columns = header
df = df.apply(pd.to_numeric).astype(float)
df[['id', 'numPulses', 'numPeriodsPulses']] = df[['id', 'numPulses', 'numPeriodsPulses']].astype(int)
df[['gender', 'class']] = df[['gender', 'class']].astype('category')

labels = df['class'].astype(int)
person_ids = df['id'].astype(int)

mfccs = df.iloc[:, 22:84].astype(float)
wavelets = df.iloc[:, 84:148].astype(float)
tqwts = df.iloc[:, 148:].astype(float)

# ✅ Normalize and convert to float32
scaler = MinMaxScaler()
mfccs_scaled = scaler.fit_transform(mfccs).astype(np.float32)
wavelets_scaled = scaler.fit_transform(wavelets).astype(np.float32)
tqwts_scaled = scaler.fit_transform(tqwts).astype(np.float32)

# ✅ Setup LOPO-CV
logo = LeaveOneGroupOut()
splits = list(logo.split(mfccs_scaled, labels, groups=person_ids))
print(f"✅ Total folds (unique persons): {len(splits)}")

# ✅ Build model
def build_model_level_cnn(input_shapes):
    inputs, branches = [], []
    for shape in input_shapes:
        inp = Input(shape=shape)
        x = Conv1D(32, 3, activation='relu', kernel_regularizer=l2(0.001))(inp)
        x = Conv1D(32, 3, activation='relu', kernel_regularizer=l2(0.001))(x)
        x = MaxPooling1D(2)(x)
        x = Conv1D(64, 3, activation='relu', kernel_regularizer=l2(0.001))(x)
        x = Conv1D(64, 3, activation='relu', kernel_regularizer=l2(0.001))(x)
        x = MaxPooling1D(2)(x)
        x = Conv1D(128, 3, activation='relu', kernel_regularizer=l2(0.001))(x)
        x = Conv1D(128, 3, activation='relu', kernel_regularizer=l2(0.001))(x)
        x = MaxPooling1D(2)(x)
        x = Flatten()(x)
        inputs.append(inp)
        branches.append(x)

    merged = concatenate(branches)
    merged = Dropout(0.3)(merged)
    merged = Dense(64, activation='relu')(merged)
    output = Dense(1, activation='sigmoid', dtype='float32')(merged)  # float32 for mixed precision

    model = Model(inputs, output)
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    return model

# ✅ Training over all folds
all_preds, all_labels = [], []
for fold_counter, (train_idx, test_idx) in enumerate(splits, start=1):
    print(f"\n🔄 Fold {fold_counter} / {len(splits)}: Training...")

    X_train_mfcc, X_test_mfcc = mfccs_scaled[train_idx], mfccs_scaled[test_idx]
    X_train_wavelet, X_test_wavelet = wavelets_scaled[train_idx], wavelets_scaled[test_idx]
    X_train_tqwt, X_test_tqwt = tqwts_scaled[train_idx], tqwts_scaled[test_idx]

    X_train = [np.expand_dims(X_train_mfcc, 2),
               np.expand_dims(X_train_wavelet, 2),
               np.expand_dims(X_train_tqwt, 2)]
    X_test = [np.expand_dims(X_test_mfcc, 2),
              np.expand_dims(X_test_wavelet, 2),
              np.expand_dims(X_test_tqwt, 2)]
    y_train, y_test = labels.iloc[train_idx], labels.iloc[test_idx]

    # ✅ Use tf.data with smaller batch size (16) and GPU prefetching
    train_ds = tf.data.Dataset.from_tensor_slices((tuple(X_train), y_train))
    train_ds = train_ds.shuffle(1024).batch(16).prefetch(tf.data.AUTOTUNE)
    test_ds = tf.data.Dataset.from_tensor_slices((tuple(X_test), y_test))
    test_ds = test_ds.batch(16).prefetch(tf.data.AUTOTUNE)

    model = build_model_level_cnn([x.shape[1:] for x in X_train])
    model.fit(train_ds, epochs=100, verbose=0)  # ✅ reduce to 100 epochs to speed up

    y_pred = (model.predict(test_ds, verbose=0).flatten() > 0.5).astype(int)
    all_preds.extend(y_pred)
    all_labels.extend(y_test)

    print(f"✅ Fold {fold_counter}: Accuracy = {accuracy_score(y_test, y_pred):.4f}")

# ✅ Final metrics
accuracy = accuracy_score(all_labels, all_preds)
f1 = f1_score(all_labels, all_preds)
mcc = matthews_corrcoef(all_labels, all_preds)

print(f"\n🏁 Final LOPO Results After All 256 Folds:")
print(f"🔹 Accuracy : {accuracy:.4f}")
print(f"🔹 F1-Score : {f1:.4f}")
print(f"🔹 MCC      : {mcc:.4f}")
