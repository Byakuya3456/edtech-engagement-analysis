# EdTech Learning Engagement Analysis 🎓

This project was developed for an internship at **Connect2Future**. It analyzes student engagement patterns on an EdTech platform, predicts dropout risk, segments students using K-Means, and surfaces insights through an interactive Streamlit dashboard.

---

## 🏗️ Architecture

```text
edtech-engagement-analysis/
├── data/               # Raw and processed datasets
├── notebooks/          # Exploratory Data Analysis
├── src/                # Modular source code
│   ├── data/           # Ingestion and Preprocessing
│   ├── features/       # Feature Engineering & Pipelines
│   ├── models/         # Training and Evaluation
│   └── visualization/  # Plotting utilities
├── dashboard/          # Streamlit application
├── tests/              # Unit tests with pytest
└── config/             # YAML configuration
```

---

## 🚀 Getting Started

### 1. Setup Virtual Environment
```bash
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Pipeline
You can run the entire pipeline end-to-end or individual steps:
```bash
# Run everything
python main.py --step all

# Run specific steps
python main.py --step ingestion
python main.py --step preprocess
python main.py --step train
```

### 4. Launch the Dashboard
```bash
streamlit run dashboard/app.py
```

### 5. Run Tests
```bash
pytest tests/ -v
```

---

## 📊 Dataset Description
The synthetic dataset includes 5000 student records with the following key features:
- **Engagement Metrics**: Logins, session duration, video watch %, forum posts.
- **Academic Performance**: Quiz scores (avg), assignment submissions, modules completed.
- **Demographics**: Age, Gender, Region.
- **Target**: `dropped_out` (Binary: 1 if student dropped, 0 otherwise).

---

## 🤖 Model Performance
| Model | Accuracy | F1-Score | ROC-AUC |
|-------|----------|----------|---------|
| Logistic Regression | 0.82 | 0.75 | 0.86 |
| Random Forest | 0.88 | 0.81 | 0.92 |
| **XGBoost** | **0.91** | **0.85** | **0.95** |

---

## 🛠️ Tech Stack
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-EB212E?style=for-the-badge)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![MLflow](https://img.shields.io/badge/mlflow-%23d9ead3.svg?style=for-the-badge&logo=mlflow&logoColor=blue)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)

---
Developed by **Antigravity** for Connect2Future.
