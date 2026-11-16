# 🧠 Student Social Media & Mental Health Score Prediction

An end-to-end Machine Learning project analyzing the impact of social media usage, sleep patterns, study habits, and stress levels on student mental health, featuring automated preprocessing pipelines and predictive modeling using Scikit-Learn.

---

## 📌 Project Overview

With the rapid expansion of digital connectivity, excessive social media consumption and screen time have become critical factors influencing students' psychological well-being. This project applies statistical analysis, exploratory data analysis (EDA), and machine learning regression models to predict a student's **Mental Health Score** based on demographic data, digital habits, and lifestyle metrics.

### Key Objectives
- **Analyze Behavioral Patterns:** Uncover correlations between daily screen time, device unlocks, physical activity, sleep duration, and mental health.
- **Robust Feature Engineering:** Build end-to-end preprocessing pipelines using Scikit-Learn's `ColumnTransformer` to handle skewed numerical features, ordinal ranks, and categorical indicators.
- **Predictive Regression Modeling:** Train and benchmark multiple regressors (Linear Regression vs. Random Forest) and optimize hyperparameters using `RandomizedSearchCV`.
- **Model Serialization:** Export the trained pipeline into a reusable serialized model (`Mental_Health_Model.pkl`) ready for downstream inference.

---

## 📊 Dataset Overview

The dataset consists of **5,000 student survey responses** with 13 distinct features capturing academic level, daily digital device usage, and lifestyle habits.

| Feature Name | Type | Description |
| :--- | :--- | :--- |
| `Age` | Numerical | Age of the student (18 to 24+ years) |
| `Gender` | Categorical | Gender identity (`Male`, `Female`, `Other`) |
| `Country` | Categorical | Student country of residence (grouped into top 10 + `Other`) |
| `Academic_Level` | Categorical | Current education level (`High School`, `Undergraduate`, `Graduate`) |
| `Most_Used_Platform` | Categorical | Primary social media platform (e.g., `Instagram`, `TikTok`, `YouTube`, `Facebook`, etc.) |
| `Purpose_Of_Use` | Categorical | Primary intent (`Entertainment`, `Networking`, `Education`, etc.) |
| `Avg_Daily_Usage_Hours`| Numerical | Average hours spent daily on social media |
| `Daily_Unlocks` | Numerical | Frequency of phone unlocks per day |
| `Study_Hours` | Numerical | Average daily academic study hours (right-skewed) |
| `Physical_Activity_Hours`| Numerical | Daily exercise/activity hours (clipped to non-negative values) |
| `Sleep_Hours_Per_Night`| Numerical | Average nightly sleep hours |
| `Stress_Level` | Ordinal | Self-reported stress level (`Low`, `Medium`, `High`, `Very High`) |
| **`Mental_Health_Score`** | **Target (Numerical)** | Continuous target score reflecting psychological well-being |

---

## 🔍 Exploratory Data Analysis & Preprocessing

### 1. Data Cleaning & Integrity
- **Deduplication:** Identified and eliminated duplicate entries to prevent data leakage.
- **Boundary Corrections:** Clipped negative/unrealistic values in `Physical_Activity_Hours` at a baseline of `0.0`.
- **Outlier Assessment:** Applied the Interquartile Range (IQR) method to analyze anomalous extreme values across numerical metrics.

### 2. Feature Transformations
- **Skewness Handling:** Identified right-skew in `Study_Hours` and applied a logarithmic transformation (`np.log1p`) followed by standard scaling.
- **Dimensionality Reduction for Country:** Grouped less frequent countries outside the top 10 into an `'Other'` category (`Grouped_country`) to avoid high-cardinality explosion.
- **ColumnTransformer Pipeline:**
  - **Skewed Numeric:** `FunctionTransformer(np.log1p)` → `StandardScaler()`
  - **Standard Numeric:** `StandardScaler()` for `Age`, `Avg_Daily_Usage_Hours`, `Daily_Unlocks`, `Physical_Activity_Hours`, and `Sleep_Hours_Per_Night`
  - **Ordinal Features:** `OrdinalEncoder` mapping `['Low', 'Medium', 'High', 'Very High']`
  - **Nominal Features:** `OneHotEncoder(handle_unknown='ignore')` for multi-class categorical variables

---

## ⚙️ Model Development & Tuning

The dataset was partitioned into an **70% training** and **30% testing** split (`random_state=42`).

Three modeling approaches were developed and compared:
1. **Baseline Model:** Ordinary Least Squares `LinearRegression` pipeline.
2. **Ensemble Regressor:** Default `RandomForestRegressor(random_state=42)`.
3. **Hyperparameter Tuned Model:** `RandomForestRegressor` tuned using `RandomizedSearchCV` across tree depth, estimator count, and split/leaf constraints:
   - `n_estimators`: `200`
   - `max_depth`: `15`
   - `min_samples_split`: `5`
   - `min_samples_leaf`: `2`

---

## 📈 Evaluation & Results

The models were evaluated using $R^2$ Score, Mean Absolute Error (MAE), and Root Mean Squared Error (RMSE):

| Model | Test $R^2$ | Training $R^2$ | MAE | RMSE | Remarks |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Linear Regression** | `0.7398` | `0.7237` | `0.5362` | `0.6760` | Reliable linear baseline |
| **Random Forest (Default)** | **`0.8776`** | `0.9808` | **`0.3472`** | **`0.4637`** | **Top testing performance** |
| **Random Forest (Tuned)** | `0.8650` | `0.9547` | `0.3689` | `0.4869` | Balanced generalization |

![Model Evaluation](Model%20Evaluation.png)

> **Conclusion:** The ensemble Random Forest pipeline captured complex non-linear interactions between sleep deprivation, excessive daily device usage, and heightened stress levels, outperforming the linear baseline by over **13.7% in $R^2$ score**.

---

## 📁 Repository Structure

```text
├── Mental_Health_Score.ipynb               # Complete Jupyter notebook (EDA, preprocessing, modeling)
├── Mental_Health_Model.pkl                 # Serialized production pipeline (preprocessor + model)
├── Student Social Media And Mental Health Impact.csv  # Dataset
├── Model Evaluation.png                    # Comparative performance visualization
└── README.md                               # Project documentation
```

---

## 🚀 Getting Started

### Prerequisites
Make sure you have Python 3.8+ installed along with the required scientific libraries:

```bash
pip install numpy pandas scikit-learn matplotlib seaborn joblib
```

### Running the Notebook
Launch Jupyter Notebook or JupyterLab to interact with the code:

```bash
jupyter notebook Mental_Health_Score.ipynb
```

---

## 💡 Quick Inference Example

Use the exported pipeline to make predictions on new student profiles without manual preprocessing:

```python
import joblib
import pandas as pd

# Load the saved model pipeline
pipeline = joblib.load('Mental_Health_Model.pkl')

# Define sample input data
sample_student = pd.DataFrame([{
    'Study_Hours': 3.5,
    'Age': 21,
    'Avg_Daily_Usage_Hours': 6.0,
    'Daily_Unlocks': 150,
    'Physical_Activity_Hours': 1.0,
    'Sleep_Hours_Per_Night': 6.0,
    'Stress_Level': 'High',
    'Gender': 'Male',
    'Academic_Level': 'Undergraduate',
    'Most_Used_Platform': 'Instagram',
    'Purpose_Of_Use': 'Entertainment',
    'Grouped_country': 'USA'
}])

# Predict mental health score
predicted_score = pipeline.predict(sample_student)
print(f"Predicted Mental Health Score: {predicted_score[0]:.2f}")
```

---

## 🌟 Acknowledgements & Learning Journey

This project was developed as part of an applied data science and machine learning learning journey, guided and inspired by structured hands-on educational walkthroughs and technical tutorials shared on YouTube. Sincere gratitude to the online developer and data science educator community for providing open-access knowledge and end-to-end workflow demonstrations that make self-directed learning possible.
