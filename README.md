# ⚽ Football Player Position Prediction using Custom KNN

This project implements a **custom K-Nearest Neighbors (KNN)** classifier to predict football player positions (DF, MF, FW, GK) using Premier League statistics.

The dataset is automatically downloaded from Kaggle and cleaned before training.

---

## 📌 Features

- Custom KNN implementation (no sklearn KNN used)
- Handles both numeric and categorical features
- Manhattan distance for numeric features
- 0/1 distance for categorical features
- Data scaling using StandardScaler
- Performance evaluation:
  - Accuracy
  - Specificity
  - Precision
  - Recall
  - F1-score
- Confusion Matrix visualization

---

## 📂 Dataset

Dataset source:

Premier League 2024–2025 Data  
Kaggle Dataset: https://www.kaggle.com/datasets/furkanark/premier-league-2024-2025-data

Downloaded automatically using `kagglehub`.

---

## 🧠 Model Description

The custom `FootballKNN` class:

- Stores training data
- Computes distances manually
- Selects K nearest neighbors
- Uses majority voting for classification

Distance calculation:

- Numeric features → Manhattan Distance  
- Categorical features → 0 if equal, 1 if different  

---

## 🚀 How to Run

### 1️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 2️⃣ Download dataset there is a problem with the auto download so the dataset is in the data folder it is directlly downloaded when you clone the repo

```bash
python src/fetchData.py
```

### 3️⃣ Run the model

```bash
python src/knnAlgo.py
```

---

## 📊 Output

The program prints:

- Accuracy
- Average Specificity
- Full Classification Report
- Confusion Matrix Plot

---

## 📈 Example Metrics

```
Accuracy: 0.82
Average Specificity: 0.90
```

(Results may vary depending on dataset updates.)

---

## 🛠 Technologies Used

- Python
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- KaggleHub

---

## 🎓 Academic Purpose

This project was built for educational purposes to demonstrate:

- Distance-based classification
- Feature preprocessing
- Performance evaluation
- Custom ML implementation

---

## 👤 Author

Irki Tahar