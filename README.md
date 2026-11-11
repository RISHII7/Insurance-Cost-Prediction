# Insurance Cost Prediction

An end-to-end Machine Learning project to predict individual medical insurance costs based on demographic and lifestyle factors such as age, BMI, smoking status, number of children, and geographic region.

---

## 📌 Project Overview

Health insurance premiums are influenced by multiple risk factors. This project aims to build regression models to accurately forecast medical expenses billed by health insurance providers, helping both individuals and insurers understand cost drivers.

Key models explored:
- **Linear Regression** (Baseline model)
- **Random Forest Regressor** (Ensemble bagging)
- **Gradient Boosting Regressor** (Ensemble boosting)

Evaluation metrics:
- $R^2$ Score
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE) / Root Mean Squared Error (RMSE)

---

## 📊 Dataset Description

The repository includes `medical_insurance.csv` with the following attributes:

| Feature | Type | Description |
|---|---|---|
| `age` | Integer | Age of the primary beneficiary |
| `sex` | Categorical | Insurance contractor gender (`female`, `male`) |
| `bmi` | Float | Body Mass Index ($kg/m^2$), objective indicator of body weight |
| `children` | Integer | Number of children / dependents covered by health insurance |
| `smoker` | Categorical | Smoking status (`yes`, `no`) |
| `region` | Categorical | Beneficiary's residential area in the US (`northeast`, `southeast`, `southwest`, `northwest`) |
| `charges` | Float | **Target variable**: Individual medical costs billed by health insurance |

---

## 📁 Repository Structure

```text
├── .gitignore               # Ignored files (virtual environments, caches)
├── medical_insurance.csv    # Medical insurance dataset
├── main.ipynb               # Jupyter Notebook with data exploration, preprocessing & modeling
├── requirements.txt         # Project dependencies and pinned library versions
└── README.md                # Project documentation
```

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/RISHII7/Insurance-Cost-Prediction.git
cd Insurance-Cost-Prediction
```

### 2. Set Up a Virtual Environment
```bash
# Windows
python -m venv .venv
.\.venv\Scripts\activate

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Notebook
Open `main.ipynb` in your preferred editor (VS Code, JupyterLab, or Jupyter Notebook) and select the `.venv` kernel to run the cells.

---

## 🛠️ Tech Stack & Libraries

- **Data Manipulation**: `pandas`, `numpy`
- **Visualization**: `matplotlib`, `seaborn`
- **Machine Learning**: `scikit-learn`
- **Model Persistence**: `joblib`

