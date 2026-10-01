# 🏡 California House Price Prediction

A machine-learning web application that predicts California median house prices
using the built-in scikit-learn California Housing dataset, a Random Forest
regression model, and a Streamlit frontend.

---

## Features

- **Interactive sliders** for all 8 dataset features
- **One-click prediction** with a formatted dollar-value result
- **Feature importance chart** rendered after each prediction
- **Live map** that plots the selected latitude / longitude
- **Model performance panel** showing R² and Mean Absolute Error

---

## Project Structure

```
house-price-prediction/
├── train.py          # Data loading, model training, evaluation & model save
├── app.py            # Streamlit frontend
├── requirements.txt  # Python dependencies
├── README.md         # This file
└── model.joblib      # Generated at runtime — NOT committed to source control
```

---

## Quick Start

### 1. Clone / open the project

```bash
cd house-price-prediction
```

### 2. Create and activate a virtual environment (recommended)

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Train the model

```bash
python train.py
```

Expected output:

```
Loading California Housing dataset...
Training samples : 16512
Test samples     : 4128

Training RandomForestRegressor (n_estimators=100)...

── Model Performance on Test Set ──
  R² Score             : 0.8050
  Mean Absolute Error  : $32,500  (in dollars)

Model saved to model.joblib
```

> The exact numbers may vary slightly between scikit-learn versions.

### 5. Launch the Streamlit app

```bash
streamlit run app.py
```

Then open the URL printed in the terminal (usually `http://localhost:8501`).

---

## Dataset

The **California Housing** dataset is built into scikit-learn and requires no
separate download.  It was derived from the 1990 US Census and contains
20,640 block-group records with the following features:

| Feature | Description |
|---|---|
| `MedInc` | Median income (tens of thousands USD) |
| `HouseAge` | Median age of houses |
| `AveRooms` | Average rooms per household |
| `AveBedrms` | Average bedrooms per household |
| `Population` | Block group population |
| `AveOccup` | Average household occupancy |
| `Latitude` | Geographic latitude |
| `Longitude` | Geographic longitude |

Target variable: **median house value** in $100,000 units.

---

## Model

| Setting | Value |
|---|---|
| Algorithm | `RandomForestRegressor` |
| Trees | 100 |
| Preprocessing | `StandardScaler` |
| Train / test split | 80 % / 20 % |
| Random seed | 42 |

---

## Requirements

| Package | Minimum version |
|---|---|
| scikit-learn | 1.3.0 |
| joblib | 1.3.0 |
| streamlit | 1.35.0 |
| pandas | 2.0.0 |
| numpy | 1.24.0 |

---

## License

This project uses a publicly available dataset included with scikit-learn.
The code is released under the MIT License.

## About This Project

This project was built as part of the **IBM SkillsBuild - AICTE - BharatCares (CSRBOX) Machine Learning & Applied AI Internship Program 2026**.

It was developed using **IBM Bob**, an AI-powered coding assistant, in VS Code. IBM Bob helped plan the project, generate the training script and the Streamlit frontend, and fix errors along the way.

### How to Run

1. Install the dependencies: `pip install -r requirements.txt`
2. Train the model (this creates `model.joblib`): `python train.py`
3. Start the app: `python -m streamlit run app.py`

### Author

Kashish - [GitHub](https://github.com/Kashish7714)
