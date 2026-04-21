# 💧 AquaCheck — Water Potability Prediction

An AI-powered web app that predicts whether water is safe to drink based on 9 chemical parameters. Built with XGBoost and Streamlit.

---

## 🚀 Live App

👉 **[https://water-safety-guardian.streamlit.app](https://water-safety-guardian.streamlit.app)**

---

## 📁 Project Structure

```
├── app.py                        # Home page
├── ui_utils.py                   # Shared UI components & CSS
├── pages/
│   ├── 1_Predict.py              # Water safety prediction page
│   ├── 2_Insights.py             # Model insights & feature importance
│   └── 3_About.py                # About the project
├── models/
│   ├── XGBoost.py                # XGBoost training script
│   ├── ann_water_potability.py   # ANN training script
│   └── mymodel.pkl               # Trained XGBoost model + artifacts (generated)
├── data/
│   └── water_potability.csv      # Dataset (3,276 samples)
├── notebooks/
│   ├── dataset.ipynb             # Jupyter notebook analysis
│   └── water_potability_analysis.py  # Exploratory data analysis
├── outputs_xgb/
│   ├── xgb_results.json          # XGBoost evaluation results
│   └── plots/                    # Generated visualisation plots
├── outputs_ann/
│   ├── ann_results.json          # ANN evaluation results
│   ├── ann_artifacts.pkl         # ANN scaler & threshold (generated)
│   └── ann_water_potability.keras# Trained ANN model (generated)
├── .streamlit/
│   └── config.toml               # Streamlit theme config
├── logo.svg                      # App logo
└── requirements.txt              # Python dependencies
```

---

## 🧪 Dataset

- **Source:** [Water Potability Dataset — Kaggle](https://www.kaggle.com/datasets/adityakadiwal/water-potability)
- **Samples:** 3,276 water samples
- **Features:** pH, Hardness, Solids, Chloramines, Sulfate, Conductivity, Organic Carbon, Trihalomethanes, Turbidity
- **Target:** Potability (1 = safe, 0 = not safe)

---

## 🤖 Models

| Model | Accuracy | ROC-AUC |
|-------|----------|---------|
| XGBoost ✅ | **80.3%** | 0.872 |
| ANN | 65% | 0.687 |

XGBoost is used as the deployed prediction model. ANN was built and optimised as part of the deep learning component.

---

## ⚙️ How to Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/<your-username>/aquacheck.git
cd aquacheck

# 2. Create virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # Mac/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run app.py
```

To retrain the models:
```bash
python models/XGBoost.py              # Train XGBoost (saves models/mymodel.pkl)
python models/ann_water_potability.py # Train ANN
```

---

## 🏗️ Tech Stack

- **ML Models:** XGBoost, TensorFlow/Keras ANN
- **App Framework:** Streamlit
- **Data Processing:** Pandas, NumPy, Scikit-learn
- **Visualisation:** Matplotlib, Seaborn
- **Imbalance Handling:** imbalanced-learn (SMOTE)

---

## ⚠️ Disclaimer

This app is built for **educational purposes only**. Do not use it as the sole basis for any health or safety decision. Always consult a certified water testing laboratory for official results.
