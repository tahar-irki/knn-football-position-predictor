import numpy as np
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import matplotlib.pyplot as plt


class FootballKNN:
    def __init__(self, k=20, cat_cols_indices=None):
        self.k = k
        self.cat_cols_indices = cat_cols_indices if cat_cols_indices else []

    def fit(self, X, y):
        self.X_train = np.array(X, dtype=object)
        self.y_train = np.array(y)

    def predict(self, X):
        X = np.array(X, dtype=object)
        return np.array([self._predict_single(x) for x in X])

    def _predict_single(self, x):
        distances = []

        for xt in self.X_train:
            dist = 0.0
            for i in range(len(x)):
                if i in self.cat_cols_indices:
                    dist += 0 if x[i] == xt[i] else 1
                else:
                    dist += abs(float(x[i]) - float(xt[i]))
            distances.append(dist)

        k_idx = np.argsort(distances)[:self.k]
        return Counter(self.y_train[k_idx]).most_common(1)[0][0]

INPUT_FILE = "Squad_PlayerStats__stats_standard.csv"
OUTPUT_FILE = "dataPlCleaned.csv"

df = pd.read_csv(INPUT_FILE)


df["Pos"] = df["Pos"].str[:2]


df = df.drop(columns=["Matches","Rk","Player","Nation","Born","Squad"], errors="ignore")
X = df.drop(columns=["Pos"]).copy()
y = df["Pos"].copy()
numeric_cols = []
categorical_cols = []
for col in X.columns:
    try:
        X[col] = pd.to_numeric(X[col], errors="raise")
        numeric_cols.append(col)
    except:
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
cat_indices = [X.columns.get_loc(c) for c in categorical_cols]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()

X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()

X_train_scaled[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
X_test_scaled[numeric_cols] = scaler.transform(X_test[numeric_cols])

model = FootballKNN(k=5, cat_cols_indices=cat_indices)
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

print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"Average Specificity: {average_specificity(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))


cm = confusion_matrix(y_test, y_pred)
labels = np.unique(y_test)

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, output_dict=True)
report_df = pd.DataFrame(report).transpose()

print("\n==============================")
print(f"Accuracy: {accuracy:.4f}")
print("==============================\n")
print("Classification Report (DataFrame):")
print(report_df)

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