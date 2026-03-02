import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    classification_report
)
import matplotlib.pyplot as plt


class StudentDropoutDataLoader:
    def __init__(self, input_path):
        self.input_path = input_path

    def load_and_clean(self):
        df = pd.read_csv(self.input_path)

        df = df.drop(columns=["Student_ID"], errors="ignore")

        if "Dropout" not in df.columns:
            raise ValueError("Target column 'Dropout' not found.")

        X = df.drop(columns=["Dropout"]).copy()
        y = df["Dropout"].copy()

    
        numeric_cols = X.select_dtypes(include=np.number).columns.tolist()
        categorical_cols = X.select_dtypes(exclude=np.number).columns.tolist()

        X = X.dropna()
        y = y.loc[X.index]

        return X, y, numeric_cols, categorical_cols




class MixedNaiveBayes:

    def __init__(self):
        self.classes_ = None
        self.class_priors_ = None
        self.numeric_cols_ = None
        self.categorical_cols_ = None

        self.means_ = {}
        self.vars_ = {}

        self.cat_prob_ = {}
        self.cat_levels_ = {}

        self.eps_ = 1e-9

    def fit(self, X: pd.DataFrame, y: pd.Series):

        self.classes_ = np.unique(y)
        n_samples = len(y)

        self.numeric_cols_ = X.select_dtypes(include=np.number).columns.tolist()
        self.categorical_cols_ = X.select_dtypes(exclude=np.number).columns.tolist()


        self.class_priors_ = {
            c: np.sum(y == c) / n_samples
            for c in self.classes_
        }

        # Numeric (Gaussian)
        for c in self.classes_:
            X_c = X[y == c]
            self.means_[c] = X_c[self.numeric_cols_].mean()
            self.vars_[c] = X_c[self.numeric_cols_].var() + self.eps_

        #  Categorical 
        for col in self.categorical_cols_:
            self.cat_levels_[col] = X[col].unique()
            self.cat_prob_[col] = {}

            for c in self.classes_:
                X_c = X[y == c][col]
                counts = X_c.value_counts()

                total = len(X_c)
                k = len(self.cat_levels_[col])

                probs = {}
                for level in self.cat_levels_[col]:
                    count = counts.get(level, 0)
                    probs[level] = (count + 1) / (total + k)

                self.cat_prob_[col][c] = probs

        return self

    def _log_gaussian(self, x, mean, var):
        return -0.5 * np.log(2 * np.pi * var) - ((x - mean) ** 2) / (2 * var)

    def predict(self, X: pd.DataFrame):

        predictions = []

        for _, row in X.iterrows():

            class_scores = {}

            for c in self.classes_:

                log_prob = np.log(self.class_priors_[c])

                # Numeric contribution
                for col in self.numeric_cols_:
                    mean = self.means_[c][col]
                    var = self.vars_[c][col]
                    log_prob += self._log_gaussian(row[col], mean, var)

                # Categorical contribution
                for col in self.categorical_cols_:
                    x_val = row[col]

                    if x_val in self.cat_prob_[col][c]:
                        log_prob += np.log(self.cat_prob_[col][c][x_val])
                    else:
                        k = len(self.cat_levels_[col])
                        log_prob += np.log(1 / k)

                class_scores[c] = log_prob

            predictions.append(max(class_scores, key=class_scores.get))

        return np.array(predictions)

    def score(self, X, y):
        return accuracy_score(y, self.predict(X))



class ModelEvaluator:

    @staticmethod
    def average_specificity(y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)
        specs = []

        for i in range(len(cm)):
            tn = cm.sum() - (cm[i].sum() + cm[:, i].sum() - cm[i, i])
            fp = cm[:, i].sum() - cm[i, i]
            specs.append(tn / (tn + fp) if (tn + fp) else 0)

        return np.mean(specs)

    @staticmethod
    def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix"):
        cm = confusion_matrix(y_true, y_pred)
        labels = np.unique(y_true)

        plt.figure(figsize=(8, 6))
        plt.imshow(cm, cmap="Blues")
        plt.xticks(range(len(labels)), labels, rotation=45)
        plt.yticks(range(len(labels)), labels)
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.title(title)
        plt.colorbar()

        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, cm[i, j],
                         ha="center", va="center", color="red")

        plt.tight_layout()
        plt.show()




if __name__ == "__main__":

    PROJECT_ROOT = os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))
        )
    )

    DATA_DIR = os.path.join(PROJECT_ROOT, "data")
    INPUT_FILE = os.path.join(DATA_DIR, "student_dropout_dataset_v3.csv")

    loader = StudentDropoutDataLoader(INPUT_FILE)
    X, y, numeric_cols, categorical_cols = loader.load_and_clean()


    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )


    scaler = StandardScaler()
    X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])


    model = MixedNaiveBayes()
    model.fit(X_train, y_train)


    y_pred = model.predict(X_test)


    print("\nAccuracy:", accuracy_score(y_test, y_pred))
    print("Average Specificity:",
          ModelEvaluator.average_specificity(y_test, y_pred))

    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))

    ModelEvaluator.plot_confusion_matrix(
        y_test, y_pred,
        title="Confusion Matrix (Mixed Naive Bayes)"
    )