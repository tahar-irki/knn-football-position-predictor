import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    classification_report
)
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


def load_data(file_path):
    """
    Load dataset and separate features from target.
    """
    df = pd.read_csv(file_path)


    df = df.drop(columns=["Student_ID"], errors="ignore")

    if "Dropout" not in df.columns:
        raise ValueError("Target column 'Dropout' not found.")

    X = df.drop(columns=["Dropout"])
    y = df["Dropout"]

    return X, y




def detect_column_types(X):

    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

    return numeric_cols, categorical_cols




def split_and_preprocess(X, y, numeric_cols, categorical_cols,
                         test_size=0.2, random_state=42):

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Numeric pipeline
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())
    ])

    # Categorical pipeline
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(drop="first", handle_unknown="ignore"))
    ])

    # Combine both
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_cols),
            ("cat", categorical_pipeline, categorical_cols)
        ]
    )

    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    return X_train_processed, X_test_processed, y_train, y_test




def train_model(X_train, y_train):
    """
    Train Gaussian Naive Bayes.
    """
    model = GaussianNB()
    model.fit(X_train, y_train)
    return model




def average_specificity(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    specs = []

    for i in range(len(cm)):
        tn = cm.sum() - (cm[i].sum() + cm[:, i].sum() - cm[i, i])
        fp = cm[:, i].sum() - cm[i, i]
        specs.append(tn / (tn + fp) if (tn + fp) else 0)

    return np.mean(specs)


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    spec = average_specificity(y_test, y_pred)

    print(f"\nAccuracy: {acc:.4f}")
    print(f"Average Specificity: {spec:.4f}")
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))

    plot_confusion_matrix(y_test, y_pred)




def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    labels = np.unique(y_true)

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, cmap="Blues")
    plt.xticks(range(len(labels)), labels, rotation=45)
    plt.yticks(range(len(labels)), labels)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix (Gaussian Naive Bayes)")
    plt.colorbar()

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j],
                     ha="center", va="center", color="red")

    plt.tight_layout()
    plt.show()



def main():

    PROJECT_ROOT = os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))
        )
    )

    DATA_DIR = os.path.join(PROJECT_ROOT, "data")
    INPUT_FILE = os.path.join(DATA_DIR, "student_dropout_dataset_v3.csv")


    X, y = load_data(INPUT_FILE)

    print("Dataset loaded successfully.")
    print(f"Total samples: {len(X)}")


    numeric_cols, categorical_cols = detect_column_types(X)

    print(f"Numeric features: {len(numeric_cols)}")
    print(f"Categorical features: {len(categorical_cols)}")


    X_train, X_test, y_train, y_test = split_and_preprocess(
        X, y, numeric_cols, categorical_cols
    )


    model = train_model(X_train, y_train)


    evaluate_model(model, X_test, y_test)


if __name__ == "__main__":
    main()