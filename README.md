# 🚀 Multi-Algorithm Machine Learning Sandbox

A structured Machine Learning project implementing algorithms from **two complementary perspectives**:

- 🧮 **From-Scratch Mathematical Implementations**
- ⚙️ **Production-Ready Library Pipelines**

This repository explores real-world datasets to bridge the gap between:

- 📐 Mathematical theory  
- 🧠 Algorithmic reasoning  
- 🏗 Practical ML engineering  

---

# 📂 Project Structure

```
├── data/                         # Datasets 
│ 
│
├── src/                          # Algorithm implementations
│   ├── knn/                      # K-Nearest Neighbors
│   │   ├── knnAlgo.py            # Custom KNN (from scratch)
│   │   ├── knnFetchData.py       # Data acquisition script
│   │   └── knnLib.py             # scikit-learn benchmark version
│   │
│   ├── naive bayes/              # Naive Bayes classifiers
│   │   ├── nBayesAlgo.py         # Custom Mixed Naive Bayes
│   │   ├── nBayesDataFetcher.py  # Dataset loader
│   │   └── nBayesLib.py          # Production pipeline version
│   │
│   └── next algorithm/           # Future implementations
│       └── dataFetcher.py
│
├── requirements.txt              # Dependencies
└── README.md                     # Project documentation
```

---

# 🧠 Core Implementations

## 1️⃣ K-Nearest Neighbors (KNN)

### 🔹 From-Scratch Implementation

A manual implementation of the KNN algorithm including:

- Custom distance computation
- Hybrid metric design:
  - Manhattan distance → Numeric features
  - Hamming distance (0/1) → Categorical features
- Majority voting classification logic

This version demonstrates deep understanding of:

- Distance-based learning  
- Feature engineering  
- Algorithmic complexity  

---

### 🔹 Library-Based Benchmark

Implemented using **scikit-learn’s KNeighborsClassifier** for performance comparison against the custom logic.

This allows benchmarking between:

- Mathematical implementation  
- Optimized production model  

---

### 🎯 Domain: Sports Analytics

The dataset focuses on predicting football player positions:

- DF → Defender  
- MF → Midfielder  
- FW → Forward  
- GK → Goalkeeper  

Predictions are based on real performance statistics.

---

## 2️⃣ Naive Bayes

### 🔹 Custom Mixed Naive Bayes

A from-scratch implementation supporting mixed feature types:

- Gaussian probability distributions → Continuous academic features  
- Laplace-smoothed categorical probabilities → Demographic features  

This implementation demonstrates understanding of:

- Bayes Theorem  
- Conditional independence assumption  
- Likelihood estimation  
- Numerical stability  

---

### 🔹 Production Pipeline Version

Built using scikit-learn with:

- Pipeline  
- ColumnTransformer  
- SimpleImputer  
- StandardScaler  
- OneHotEncoder  

Capabilities include:

- Automatic missing value handling  
- Feature scaling  
- Categorical encoding  
- Data leakage prevention  
- Clean train/test separation  

---

### 🎯 Domain: Educational Data

The model predicts student dropout risk using:

- Academic performance metrics  
- Socio-demographic variables  

---

# 📊 Model Evaluation Strategy

Each algorithm (custom and library-based) is evaluated using:

### ✔ Accuracy  
Overall prediction correctness.

### ✔ Average Specificity  
Measures the ability to correctly identify negative classes across multiple labels.

### ✔ Classification Report  
Includes:
- Precision  
- Recall  
- F1-Score  
- Support  

### ✔ Confusion Matrix  
Visualized using matplotlib to analyze misclassification patterns.

This ensures consistent and fair comparison between implementations.

---

# 🛠 Setup & Usage

## 1️⃣ Installation

```bash
git clone https://github.com/tahar-irki/ml-algorithms-implementation.git
cd ml-algorithms-implementation
pip install -r requirements.txt
```

---

## 2️⃣ Running Algorithms

### Run Custom KNN
```bash
python src/knn/knnAlgo.py
```

### Run Naive Bayes Production Pipeline
```bash
python "src/naive bayes/nBayesLib.py"
```

---



# 👤 Author

**tahar irki**  
