import pandas as pd
import numpy as np
import requests
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from sklearn.feature_selection import SelectKBest, chi2

# Step 1: Download dataset
url = 'https://raw.githubusercontent.com/dcleres/Parkinson_Disease_ML/refs/heads/master/pd_speech_features.csv'
file_name = 'pd_speech_features.csv'

response = requests.get(url)
with open(file_name, 'wb') as file:
    file.write(response.content)

# Step 2: Load and preprocess data
df = pd.read_csv(file_name)
header = df.iloc[0]
df = df[1:]
df.columns = header
df = df.apply(pd.to_numeric)
df = df.astype(float)

# Separate labels and person IDs
labels = df['class'].astype(int)
person_ids = df['id'].astype(int)
all_features = df.drop(['id', 'gender', 'class'], axis=1)

# Normalize features
scaler = MinMaxScaler()
all_features_scaled = pd.DataFrame(scaler.fit_transform(all_features), columns=all_features.columns)

# Step 3: Setup LOPO-CV
logo = LeaveOneGroupOut()
splits = list(logo.split(all_features_scaled, labels, groups=person_ids))
total_folds = len(splits)

# Step 4: SVM + Chi-square feature selection
all_preds = []
all_labels = []

for fold_counter, (train_idx, test_idx) in enumerate(splits, start=1):
    print(f"\n🔄 Fold {fold_counter} / {total_folds}: Training...")

    X_train, X_test = all_features_scaled.iloc[train_idx], all_features_scaled.iloc[test_idx]
    y_train, y_test = labels.iloc[train_idx], labels.iloc[test_idx]

    # Select top 25% features using chi-square
    selector = SelectKBest(chi2, k=int(0.25 * X_train.shape[1]))
    X_train_selected = selector.fit_transform(X_train, y_train)
    X_test_selected = selector.transform(X_test)

    # Train SVM (RBF kernel, like in many PD studies)
    svm = SVC(kernel='rbf', C=1.0, gamma='scale')
    svm.fit(X_train_selected, y_train)

    y_pred = svm.predict(X_test_selected)

    all_preds.extend(y_pred)
    all_labels.extend(y_test)

    print(f"✅ Fold {fold_counter}: Accuracy = {accuracy_score(y_test, y_pred):.4f}")

# Step 5: Final metrics
accuracy = accuracy_score(all_labels, all_preds)
f1 = f1_score(all_labels, all_preds)
mcc = matthews_corrcoef(all_labels, all_preds)

print(f"\n🏁 Final SVM Benchmark Results:")
print(f"🔹 Accuracy : {accuracy:.4f}")
print(f"🔹 F1-Score : {f1:.4f}")
print(f"🔹 MCC      : {mcc:.4f}")
