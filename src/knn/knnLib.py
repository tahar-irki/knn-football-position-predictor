import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt


PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
)

DATA_DIR = os.path.join(PROJECT_ROOT, "data")

INPUT_FILE = os.path.join(DATA_DIR, "Squad_PlayerStats__stats_standard.csv")
OUTPUT_FILE = os.path.join(DATA_DIR, "dataPlCleaned4Cknn.csv")


df = pd.read_csv(INPUT_FILE)


df["Pos"] = df["Pos"].str[:2]

df = df.drop(
    columns=["Matches", "Rk", "Player", "Nation", "Born", "Squad"],
    errors="ignore"
)

X = df.drop(columns=["Pos"]).copy()
y = df["Pos"].copy()

numeric_cols = []
categorical_cols = []


for col in X.columns:
    try:
        X[col] = pd.to_numeric(X[col], errors="raise")
        numeric_cols.append(col)
    except Exception:
        categorical_cols.append(col)


for col in numeric_cols:
    X[col] = pd.to_numeric(X[col], errors="coerce")


X = X.dropna()
y = y.loc[X.index]

df_cleaned = X.copy()
df_cleaned["Pos"] = y
df_cleaned.to_csv(OUTPUT_FILE, index=False)

print(f"Cleaned data saved to: {OUTPUT_FILE}")
print(f"Rows after cleaning: {len(df_cleaned)}")


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


scaler = StandardScaler()

X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()

X_train_scaled[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
X_test_scaled[numeric_cols] = scaler.transform(X_test[numeric_cols])

model = KNeighborsClassifier(
    n_neighbors=5,
    metric="minkowski",  
    p=2
)

model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)

def average_specificity(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    specs = []

    for i in range(len(cm)):
        tn = cm.sum() - (cm[i].sum() + cm[:, i].sum() - cm[i, i])
        fp = cm[:, i].sum() - cm[i, i]
        specs.append(tn / (tn + fp) if (tn + fp) else 0)

    return np.mean(specs)

print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"Average Specificity: {average_specificity(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))


cm = confusion_matrix(y_test, y_pred)
labels = np.unique(y_test)

plt.figure(figsize=(8, 6))
plt.imshow(cm, cmap="Blues")
plt.xticks(range(len(labels)), labels, rotation=45)
plt.yticks(range(len(labels)), labels)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.colorbar()

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, cm[i, j], ha="center", va="center", color="red")

plt.tight_layout()
plt.show()