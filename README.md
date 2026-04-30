# 📈 KOTAKBANK Stock Price Forecasting — LSTM & GRU

<div align="center">

**Multi-output time-series forecasting on Kotak Mahindra Bank stock data using LSTM and GRU neural networks built entirely in PyTorch.**

*CA 24 · Theory Assignment 1 · MCA Programme*

</div>

---

## 🔗 Quick Links

- 📁 [Repository](https://github.com/mozammil-raja/Multi-Output-Time-series-Forecasting)
- 📊 [KOTAKBANK.csv — Dataset](https://github.com/mozammil-raja/Multi-Output-Time-series-Forecasting/blob/main/KOTAKBANK.csv)
- 📓 [LTSM_GRU_KOTAKBANK.ipynb — Main Notebook](https://github.com/mozammil-raja/Multi-Output-Time-series-Forecasting/blob/main/LTSM_GRU_KOTAKBANK.ipynb)
- 📓 [LSTM_KOTAKBANK.ipynb — LSTM Only Notebook](https://github.com/mozammil-raja/Multi-Output-Time-series-Forecasting/blob/main/LSTM_KOTAKBANK.ipynb)
- 🌐 [NIFTY-50 Dataset Source — Kaggle](https://www.kaggle.com/datasets/rohanrao/nifty50-stock-market-data/data)
- 📖 [PyTorch nn.LSTM Docs](https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html)
- 📖 [PyTorch nn.GRU Docs](https://pytorch.org/docs/stable/generated/torch.nn.GRU.html)

---

## 👤 Student Details

| Field | Details |
|---|---|
| **Name** | Md Mozammil Raja |
| **Faculty No.** | 25MCA029 |
| **Student ID** | 202511284 |
| **Roll No.** | 29 |
| **Dataset** | KOTAKBANK.csv (Roll 29 → serial #29 in NIFTY-50 list) |

---

## 📂 Repository Structure

```
📦 KOTAKBANK-Forecasting
 ┣ 📓 LSTM_KOTAKBANK.ipynb          # LSTM-only implementation (with train/val loss tracking)
 ┣ 📓 LTSM_GRU_KOTAKBANK.ipynb      # Combined LSTM + GRU comparison (main submission)
 ┣ 📄 README.md                     # This file
 ┗ 📊 KOTAKBANK.csv                 # Dataset (download from Kaggle — see below)
```

---

## 🗂️ Dataset

| Property | Details |
|---|---|
| **Source** | [NIFTY-50 Stock Market Data (2000–2021) — Kaggle](https://www.kaggle.com/datasets/rohanrao/nifty50-stock-market-data/data) |
| **File** | [`KOTAKBANK.csv`](https://github.com/mozammil-raja/Multi-Output-Time-series-Forecasting/blob/main/KOTAKBANK.csv) (Serial #29 in Version 15 of the dataset) |
| **Coverage** | 1st January 2000 → 30th April 2021 |
| **Type** | Daily OHLCV time-series |

### 📋 Features Used (11 total)

| # | Column | Description |
|---|---|---|
| 1 | `Prev Close` | Previous day's closing price |
| 2 | `Open` | Opening price |
| 3 | `High` | Day's highest price |
| 4 | `Low` | Day's lowest price |
| 5 | `Last` | Last traded price |
| 6 | `Close` | Closing price |
| 7 | `VWAP` | Volume Weighted Average Price |
| 8 | `Volume` | Total volume traded |
| 9 | `Turnover` | Total turnover (in Rs. Crore) |
| 10 | `Trades` | Number of trades |
| 11 | `Deliverable Volume` | Deliverable quantity |

> **Download:** [`KOTAKBANK.csv`](https://github.com/mozammil-raja/Multi-Output-Time-series-Forecasting/blob/main/KOTAKBANK.csv) is available directly in the repository. Clone the repo or download the file and place it in `/content/` if running on Google Colab.

---

## 🧪 Notebooks Overview

This repository contains **two notebooks** — an LSTM-only version and a full LSTM + GRU comparison.

---

### 📓 Notebook 1 — `LSTM_KOTAKBANK.ipynb`

A clean, focused implementation of an LSTM model with **train and validation loss tracking**.

#### Hyperparameters

| Parameter | Value |
|---|---|
| Input window | 10 days |
| Output horizon | 5 days |
| Overlap / Stride | 5 days / 5 days |
| Hidden dim | 64 |
| Num layers | 1 |
| Dropout | 0.4 |
| Batch size | 64 |
| Epochs | 50 |
| Learning rate | 1e-3 |
| Optimiser | Adam |
| Scheduler | ReduceLROnPlateau (patience=5, factor=0.5) |

#### Key Features
- ✅ Both **training loss** and **validation loss** tracked per epoch
- ✅ Helps visually detect overfitting/underfitting via loss curve
- ✅ Gradient clipping (`max_norm=1.0`) for stable training
- ✅ Per-feature Relative RMSE bar chart
- ✅ Date-wise True vs LSTM prediction plot for all 11 features
- ✅ Dedicated Close price focus plot

---

### 📓 Notebook 2 — `LTSM_GRU_KOTAKBANK.ipynb` *(Main Submission)*

A comprehensive side-by-side comparison of **LSTM vs GRU** on the same dataset, with both train and validation loss tracked per epoch.

#### Hyperparameters

| Parameter | Value |
|---|---|
| Input window | 10 days |
| Output horizon | 5 days |
| Overlap / Stride | 5 days / 5 days |
| Hidden dim | 64 |
| Num layers | 2 |
| Dropout | 0.4 |
| Batch size | 64 |
| Epochs | 50 |
| Learning rate | 1e-3 |
| Optimiser | Adam |
| Scheduler | ReduceLROnPlateau (patience=5, factor=0.5) |

#### Key Features
- ✅ **Both LSTM and GRU** trained and evaluated on identical data splits
- ✅ **Train and validation loss** tracked per epoch for both models
- ✅ Separate Train vs Val loss curve for LSTM and GRU (2 subplots)
- ✅ Side-by-side per-feature Relative RMSE bar chart (LSTM vs GRU)
- ✅ Date-wise True vs LSTM vs GRU prediction for all 11 features
- ✅ Dedicated Close price comparison plot (True vs LSTM vs GRU)
- ✅ Final metrics table comparing both models

---

## 🏗️ ML Pipelines

Each notebook has its own pipeline reflecting its scope and model configuration.

---

### Pipeline A — `LTSM_GRU_KOTAKBANK.ipynb` (LSTM + GRU)

```
KOTAKBANK.csv
      │
      ▼
1. DATA CLEANING
   ├── Parse & sort dates chronologically
   ├── Remove exact duplicate rows
   ├── Forward-fill then back-fill missing/NaN values
   └── Coerce all feature columns to numeric (pd.to_numeric)
      │
      ▼
2. FEATURE PREPARATION
   └── Select 11 feature columns matched against actual CSV columns
       [Prev Close, Open, High, Low, Last, Close, VWAP,
        Volume, Turnover, Trades, Deliverable Volume]
      │
      ▼
3. TRAIN / TEST SPLIT  (80% / 20% — chronological, no shuffle)
      │
      ▼
4. NORMALISATION
   └── MinMaxScaler fitted on TRAIN only → transform both splits
       (no data leakage)
      │
      ▼
5. SLIDING WINDOW SEQUENCING
   ├── Input:  10 days × 11 features
   ├── Output: 5 days  × 11 features
   └── Stride: 5 days  (overlap = 5)
      │
      ▼
6. MODEL DEFINITION  (PyTorch — MultiOutputRNN)
   ├── LSTM: nn.LSTM(hidden=64, layers=2, dropout=0.4)
   │         → nn.Linear → reshape (B, 5, 11)
   └── GRU:  nn.GRU (hidden=64, layers=2, dropout=0.4)
             → nn.Linear → reshape (B, 5, 11)
      │
      ▼
7. TRAINING  (both models trained independently, with validation loop)
   ├── Loss:      MSELoss
   ├── Optimiser: Adam (lr=1e-3)
   ├── Scheduler: ReduceLROnPlateau (patience=5, factor=0.5)
   ├── Gradient clipping: max_norm=1.0
   └── Epochs: 50  |  Batch size: 64
       [Both train loss AND validation loss tracked per epoch]
      │
      ▼
8. EVALUATION & VISUALISATION
   ├── Metrics:  MSE, RMSE, MAE (original scale) for LSTM and GRU
   ├── Plot 1:   Train vs Val loss curve — separate subplot per model (LSTM | GRU)
   ├── Plot 2:   Per-feature Relative RMSE bar chart — LSTM vs GRU side-by-side
   ├── Plot 3:   Date-wise True vs LSTM vs GRU (all 11 features, 4×3 grid)
   └── Plot 4:   Close price focus — True vs LSTM vs GRU
```

---

### Pipeline B — `LSTM_KOTAKBANK.ipynb` (LSTM only)

```
KOTAKBANK.csv
      │
      ▼
1. DATA CLEANING
   ├── Parse & sort dates chronologically
   ├── Remove exact duplicate rows
   ├── Forward-fill then back-fill missing/NaN values
   └── Coerce all feature columns to numeric (pd.to_numeric)
      │
      ▼
2. FEATURE PREPARATION
   └── Select 11 feature columns matched against actual CSV columns
       [Prev Close, Open, High, Low, Last, Close, VWAP,
        Volume, Turnover, Trades, Deliverable Volume]
      │
      ▼
3. TRAIN / TEST SPLIT  (80% / 20% — chronological, no shuffle)
      │
      ▼
4. NORMALISATION
   └── MinMaxScaler fitted on TRAIN only → transform both splits
       (no data leakage)
      │
      ▼
5. SLIDING WINDOW SEQUENCING
   ├── Input:  10 days × 11 features
   ├── Output: 5 days  × 11 features
   └── Stride: 5 days  (overlap = 5)
      │
      ▼
6. MODEL DEFINITION  (PyTorch — MultiOutputRNN, LSTM only)
   └── LSTM: nn.LSTM(hidden=64, layers=1, dropout=0.4)
             → nn.Linear → reshape (B, 5, 11)
      │
      ▼
7. TRAINING  (with validation loop)
   ├── Loss:      MSELoss
   ├── Optimiser: Adam (lr=1e-3)
   ├── Scheduler: ReduceLROnPlateau (patience=5, factor=0.5)
   ├── Gradient clipping: max_norm=1.0
   └── Epochs: 50  |  Batch size: 64
       [Both train loss AND validation loss tracked per epoch]
      │
      ▼
8. EVALUATION & VISUALISATION
   ├── Metrics:  MSE, RMSE, MAE (original scale) for LSTM
   ├── Plot 1:   Training vs Validation loss curve (overfitting check)
   ├── Plot 2:   Per-feature Relative RMSE bar chart — LSTM only
   ├── Plot 3:   Date-wise True vs LSTM prediction (all 11 features, 4×3 grid)
   └── Plot 4:   Close price focus — True vs LSTM
```

---

## 🧠 Model Architecture

Both LSTM and GRU use the `MultiOutputRNN` class — a shared encoder-projector design:

```python
class MultiOutputRNN(nn.Module):
    """
    Sequence encoder using LSTM or GRU, followed by a single linear
    projection from the final hidden state to all forecast timesteps × features.

    Input:  (B, INPUT_LEN, num_features)
    Output: (B, OUTPUT_LEN, num_features)
    """
    def __init__(self, cell_type, input_size, hidden_dim, num_layers,
                 output_len, output_size, dropout):
        ...
        self.rnn = nn.LSTM / nn.GRU(...)
        self.fc  = nn.Linear(hidden_dim, output_len * output_size)

    def forward(self, x):
        _, hidden = self.rnn(x)
        h = hidden[0][-1]          # top-layer hidden state → (B, hidden_dim)
        return self.fc(h).view(B, output_len, output_size)
```

### Why this design?
- **No decoder RNN** — the entire forecast is generated in a single linear projection from the final hidden state, keeping the model lightweight and fast.
- **Shared architecture** — identical class for LSTM and GRU via `cell_type` argument, ensuring a fair comparison.
- **Gradient clipping** — prevents exploding gradients common in deep RNN stacks.

---

## 📊 Evaluation Metrics

All metrics are computed on **inverse-transformed (original scale)** predictions:

| Metric | Formula | What it measures |
|---|---|---|
| **MSE** | mean((y - ŷ)²) | Average squared error; penalises large errors heavily |
| **RMSE** | √MSE | Same unit as target; interpretable as typical error magnitude |
| **MAE** | mean(\|y - ŷ\|) | Average absolute error; robust to outliers |
| **Relative RMSE** | (RMSE / mean(\|y\|)) × 100 | Per-feature normalised error as % of feature mean |

---

## 📈 Visualisations Produced

| Plot | Notebook 1 (LSTM only) | Notebook 2 (LSTM + GRU) |
|---|---|---|
| Training loss curve | Train **+** Validation | Train **+** Val per model (2 subplots: LSTM \| GRU) |
| Per-feature RMSE bar chart | LSTM bars | LSTM **+** GRU side-by-side |
| Date-wise predictions (all features) | True vs LSTM | True vs LSTM vs GRU |
| Close price focus plot | True vs LSTM | True vs LSTM vs GRU |

---

## ⚙️ Setup & Reproduction

### Prerequisites

```bash
pip install torch torchvision numpy pandas matplotlib scikit-learn
```

### Running on Google Colab (recommended)

1. Open the notebook in Colab.
2. Upload `KOTAKBANK.csv` to `/content/` or mount your Google Drive.
3. Verify `CSV_PATH = r"/content/KOTAKBANK.csv"` at the top of the notebook.
4. **Runtime → Run All**.

### Running locally

```bash
git clone https://github.com/mozammil-raja/Multi-Output-Time-series-Forecasting.git
cd Multi-Output-Time-series-Forecasting

# Place KOTAKBANK.csv in the project root, then update CSV_PATH in the notebook
jupyter notebook LTSM_GRU_KOTAKBANK.ipynb
```

---

## 📋 Assignment Compliance Checklist

- [x] Correct dataset selected — Roll 29 → `KOTAKBANK.csv`
- [x] PyTorch used exclusively (no TensorFlow/Keras)
- [x] LSTM implemented and trained
- [x] GRU implemented and trained
- [x] Data cleaning — date parsing & sorting
- [x] Data cleaning — duplicate removal
- [x] Data cleaning — missing/NaN handling (ffill + bfill)
- [x] Data cleaning — incorrect format correction (pd.to_numeric)
- [x] Exactly 11 feature columns used
- [x] Chronological 80/20 train/test split (no shuffle)
- [x] MinMaxScaler fitted on train only (no data leakage)
- [x] Sliding window with overlap (INPUT=10, OUTPUT=5, OVERLAP=5)
- [x] Multi-output shape verified: (N, 5, 11)
- [x] MSE, RMSE, MAE reported
- [x] Training loss curve plotted
- [x] Per-feature RMSE bar chart plotted
- [x] Date-wise True vs Predicted plotted on same graph

---

## 📚 References

- [NIFTY-50 Stock Market Data — Kaggle](https://www.kaggle.com/datasets/rohanrao/nifty50-stock-market-data/data)
- [PyTorch Documentation — nn.LSTM](https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html)
- [PyTorch Documentation — nn.GRU](https://pytorch.org/docs/stable/generated/torch.nn.GRU.html)
- Hochreiter, S. & Schmidhuber, J. (1997). *Long Short-Term Memory.* Neural Computation, 9(8), 1735–1780.
- Cho, K. et al. (2014). *Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation.* arXiv:1406.1078.

---

<div align="center">

Made with ❤️ for CA 24 · Theory Assignment 1

**Md Mozammil Raja · 25MCA029 · Roll No. 29**

</div>
