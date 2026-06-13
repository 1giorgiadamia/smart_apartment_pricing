<p align="center">
  <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/6/69/Airbnb_Logo_B%C3%A9lo.svg/1280px-Airbnb_Logo_B%C3%A9lo.svg.png" width="140" alt="Airbnb">
</p>

<h1 align="center">NYC Airbnb Price Prediction</h1>
<p align="center">
  End-to-end machine learning pipeline for predicting nightly rental prices on Airbnb&nbsp;NYC
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12-blue?logo=python" alt="Python">
  <img src="https://img.shields.io/badge/LightGBM-4.6.0-brightgreen" alt="LightGBM">
  <img src="https://img.shields.io/badge/scikit--learn-1.8.0-orange" alt="sklearn">
  <img src="https://img.shields.io/badge/dataset-Kaggle-20BEFF?logo=kaggle" alt="Kaggle">
</p>

---

## Table of Contents

1. [Overview](#overview)
2. [Dataset](#dataset)
3. [Pipeline](#pipeline)
4. [Feature Engineering](#feature-engineering)
5. [Model Results](#model-results)
6. [Project Structure](#project-structure)
7. [Installation](#installation)
8. [Usage](#usage)
9. [Key Findings](#key-findings)
10. [Limitations & Next Steps](#limitations--next-steps)

---

## Overview

New York City is one of Airbnb's most competitive markets globally. For a first-time host, pricing a listing correctly is the difference between a fully booked calendar and months of vacancy

This project builds a full supervised regression pipeline from raw CSV to a tuned, evaluated model that predicts the nightly price of an NYC Airbnb listing given its characteristics

---

## Dataset

| Property | Value |
|---|---|
| **Source** | [NYC Airbnb Open Data 2019](https://www.kaggle.com/datasets/dgomonov/new-york-city-airbnb-open-data) — Kaggle |
| **Rows** | 48,895 listings |
| **Columns** | 16 features |
| **Target** | price - nightly rate in USD |
| **Coverage** | All 5 NYC boroughs, snapshot as of 2019 |

### Column Reference

| Column | Type | Description |
|---|---|---|
| `neighbourhood_group` | str | Borough (Manhattan, Brooklyn, Queens, Bronx, Staten Island) |
| `neighbourhood` | str | Specific neighbourhood (~220 unique values) |
| `latitude` / `longitude` | float | GPS coordinates |
| `room_type` | str | Entire home/apt · Private room · Shared room |
| `price` | int | Nightly rate (USD) (target variable) |
| `minimum_nights` | int | Minimum booking length |
| `number_of_reviews` | int | Cumulative review count |
| `reviews_per_month` | float | Monthly review frequency |
| `calculated_host_listings_count` | int | Total active listings by this host |
| `availability_365` | int | Days available per year |

---

## Pipeline

```
Raw CSV (48,895 rows)
  1.Data Cleaning - remove zero-price rows, cap outliers (IQR fence), fill nulls
  2.EDA - price distributions, borough/room-type analysis, review patterns
  3.Feature Engineering - dist_to_center, neighborhood_tier, has_reviews
  4.Correlation Analysis - heatmap, pairplot, feature importance pre-training
  5.Data Preparation - log-transform target, OHE categoricals, 80/20 split, StandardScaler
  6.Model Comparison - Linear Regression · Ridge · Random Forest · XGBoost · LightGBM
  7.Hyperparameter Tuning  - GridSearchCV 5-fold CV on winning model
  8.Evaluation - MAE, RMSE, R^2 on test set
```

## Feature Engineering

Three derived features were created to enrich the model:

| Feature | Formula / Logic | Why |
|---|---|---|
| `dist_to_center` | Haversine distance to Times Square (km) | Continuous, geographically meaningful proxy for centrality |
| `neighborhood_tier` | Median nightly price of the listing's neighbourhood | Compresses 220+ neighbourhoods into one numeric "prestige" signal without OHE explosion |
| `has_reviews` | 1 if `reviews_per_month > 0`, else 0 | Separates active listings from new/unreviewed ones |

> **Result:** `neighborhood_tier` achieved the strongest correlation with price of any feature (r around 0.50), while all raw numerical features had |r| < 0.20.

---

## Model Results

Five regressors were trained and evaluated on the same 20% hold-out test set. The target was log-transformed (`log1p`) during training and inverted (`expm1`) for evaluation in USD.

| Model | MAE ↓ | RMSE ↓ | R² ↑ |
|---|---|---|---|
| Linear Regression | $33.04 | $47.21 | 0.508 |
| Ridge Regression | $33.04 | $47.21 | 0.508 |
| Random Forest | $30.74 | $44.18 | 0.569 |
| XGBoost | $30.41 | $43.74 | 0.578 |
| **LightGBM** | **$30.38** | **$43.73** | **0.578** |

### After GridSearchCV tuning (LightGBM)

| | Baseline | Tuned |
|---|---|---|
| **MAE** | $30.38 | **$30.15** |
| **RMSE** | $43.73 | **$43.41** |
| **R^2** | 0.5776 | **0.5838** |

**Best hyperparameters:** `n_estimators=400`, `learning_rate=0.05`, `num_leaves=63`

---

## Project Structure

```
smart_apartment_pricing/
│
├── data/
│   └── AirBnB_NYC_2019.csv        # Raw dataset (Kaggle)
│
├── venv/                          # Python virtual environment
│
├── model.ipynb                    # Main notebook — full pipeline
├── presentation_script.docx       # 10-minute presentation script (English)
├── create_doc.py                  # Helper script used to generate the .docx
└── README.md                      # This file
```

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/1giorgiadamia/smart_apartment_pricing.git
cd smart_apartment_pricing
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the dataset

Place `AirBnB_NYC_2019.csv` into the `data/` folder.  
Download from Kaggle: [NYC Airbnb Open Data](https://www.kaggle.com/datasets/dgomonov/new-york-city-airbnb-open-data)

---

## Key Findings

1. The engineered `neighborhood_tier` feature of median price of the listing's neighbourhood became the single most important predictor, outperforming raw coordinates and borough dummies

2. Entire homes command a consistent twice premium over private rooms in every borough. This "privacy premium" is multiplicative with location, not additive. The dollar gap is much wider in Manhattan than in the Bronx

3. All raw numerical features have |r| < 0.20 with price. Linear and Ridge regression are solid baselines but are substantially outperformed by tree-based ensembles that capture non-linear feature interactions

4. Unreviewed listings carry a slightly higher median price, review accumulation correlates with price normalisation

5. Listings with 100+ reviews concentrate in the $50–$100 range, confirming that price competitiveness drives booking frequency