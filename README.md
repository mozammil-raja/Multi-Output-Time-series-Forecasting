
# 📈 KOTAKBANK Stock Price Prediction — LSTM vs GRU

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Assignment Metadata](#-assignment-metadata)
- [Dataset](#-dataset)
- [Pipeline Architecture](#-pipeline-architecture)
- [Data Cleaning](#-data-cleaning)
- [Feature Engineering](#-feature-engineering)
- [Model Architecture](#-model-architecture)
- [Hyperparameters](#-hyperparameters)
- [Training Strategy](#-training-strategy)
- [Evaluation Metrics](#-evaluation-metrics)
- [Visualizations](#-visualizations)
- [Project Structure](#-project-structure)
- [Setup & Installation](#-setup--installation)
- [How to Run](#-how-to-run)
- [Results Summary](#-results-summary)
- [Key Design Decisions](#-key-design-decisions)
- [Author](#-author)

---

## 🔍 Project Overview

This project implements **multi-step, multi-feature time series forecasting** on NSE-listed **Kotak Mahindra Bank (KOTAKBANK)** historical stock data using two recurrent neural network architectures:

| Model | Full Name | Key Strength |
|-------|-----------|-------------|
| **LSTM** | Long Short-Term Memory | Captures long-range temporal dependencies via gating mechanisms |
| **GRU** | Gated Recurrent Unit | Faster to train, fewer parameters, competitive accuracy |

Both models share an identical `MultiOutputRNN` wrapper, enabling a **fair, apples-to-apples comparison** of the two architectures. The models are trained on a 10-day look-back window to predict the next **5 days** across **11 stock features** simultaneously.

---

## 🎓 Assignment Metadata

| Field | Details |
|-------|---------|
| **Course** | CA 24 — Theory Assignment 1 |
| **Student** | Md Mozammil Raja |
| **Faculty No.** | 25MCA029 |
| **Student ID** | 202511284 |
| **Dataset** | KOTAKBANK.csv (Roll No. 29 → KOTAKBANK) |
| **Task** | Implement LSTM/GRU for time series prediction |

---

## 📊 Dataset

**Source:** NSE (National Stock Exchange of India) — KOTAKBANK historical OHLCV data  
**File:** `KOTAKBANK.csv`  
**Format:** CSV with a `Date` index column + 11 numerical feature columns

### Features Used (11 total)

| # | Column | Description |
|---|--------|-------------|
| 1 | `Prev Close` | Previous day's closing price (₹) |
| 2 | `Open` | Opening price (₹) |
| 3 | `High` | Intraday high price (₹) |
| 4 | `Low` | Intraday low price (₹) |
| 5 | `Last` | Last traded price (₹) |
| 6 | `Close` | Closing price (₹) — primary target for visual analysis |
| 7 | `VWAP` | Volume-Weighted Average Price (₹) |
| 8 | `Volume` | Number of shares traded |
| 9 | `Turnover` | Total trade value (₹) |
| 10 | `Trades` | Number of individual trades |
| 11 | `Deliverable Volume` | Shares resulting in actual delivery |

> **Note:** The feature list is dynamically built at runtime from available columns, making the notebook robust to minor CSV schema differences.

---

## 🔄 Pipeline Architecture

```
Raw CSV
   │
   ▼
┌──────────────────────────────┐
│   1. Data Loading            │  pd.read_csv → inspect shape & dtypes
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│   2. Data Cleaning           │  Parse dates → sort → dedup → NaN fill → type coerce
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│   3. Train/Test Split        │  Chronological 80/20, no shuffling
│      + Normalization         │  MinMaxScaler fit on train only
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│   4. Sliding Window          │  INPUT_LEN=10, OUTPUT_LEN=5, OVERLAP=5
│      Sequence Generation     │  Stride = INPUT_LEN − OVERLAP = 5
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│   5. PyTorch DataLoaders     │  TensorDataset → DataLoader (batch=64)
└──────────────┬───────────────┘
               │
        ┌──────┴──────┐
        ▼             ▼
  ┌──────────┐  ┌──────────┐
  │   LSTM   │  │   GRU    │    MultiOutputRNN (shared architecture)
  │  Model   │  │  Model   │    2 layers, hidden=128, dropout=0.2
  └─────┬────┘  └────┬─────┘
        │             │
        ▼             ▼
┌──────────────────────────────┐
│   6. Training Loop           │  Adam optimizer + ReduceLROnPlateau
│                              │  MSE loss + gradient clipping (max_norm=1.0)
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│   7. Evaluation & Plots      │  MSE, RMSE, MAE + 4 visualization types
└──────────────────────────────┘
```

---

## 🧹 Data Cleaning

The cleaning pipeline follows 5 structured steps:

```
Step 1 │ Parse & Sort Dates
       │  df["Date"] = pd.to_datetime(df["Date"])
       │  df.sort_values("Date") — ensures strict chronological order

Step 2 │ Remove Duplicate Rows
       │  df.drop_duplicates() — exact row duplicates removed

Step 3 │ Detect Feature Columns Dynamically
       │  Cross-checks 11 candidate columns against actual CSV schema
       │  Builds FEATURE_COLS from what exists → robust to schema drift

Step 4 │ Handle Missing / NaN Values
       │  df.ffill().bfill() — forward-fill (carry last known), then
       │  back-fill to cover leading NaNs at the start of series

Step 5 │ Ensure Numeric Types
       │  pd.to_numeric(errors="coerce") + dropna() on FEATURE_COLS
       │  Protects downstream PyTorch tensors from type errors
```

---

## ⚙️ Feature Engineering

### Sliding Window Sequences

The raw time series is converted into **supervised learning samples** using an overlapping sliding window:

```
Timeline:  d1  d2  d3  d4  d5  d6  d7  d8  d9  d10 | d11 d12 d13 d14 d15
           ◄──────────── INPUT (10 days) ────────────► ◄── OUTPUT (5 days) ──►

Window 1:  [d1..d10]  →  [d11..d15]
Window 2:  [d6..d15]  →  [d16..d20]   ← stride = 5 (overlap = 5)
Window 3:  [d11..d20] →  [d21..d25]
...
```

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `INPUT_LEN` | 10 | Days of history given as context |
| `OUTPUT_LEN` | 5 | Days to forecast |
| `OVERLAP` | 5 | Days shared between consecutive windows |
| `stride` | 5 | `INPUT_LEN − OVERLAP` — window step size |

**Output tensor shapes:**
```
X_train : (N_train, 10, 11)   # (samples, time steps, features)
y_train : (N_train,  5, 11)
X_test  : (N_test,  10, 11)
y_test  : (N_test,   5, 11)
```

### Normalization

- **Scaler:** `MinMaxScaler(feature_range=(0, 1))`
- **Fit:** on training set only — **prevents data leakage**
- **Transform:** applied to both train and test sets
- **Inverse transform:** applied at evaluation to recover original ₹ scale

---

## 🧠 Model Architecture

Both LSTM and GRU share the same `MultiOutputRNN` class:

```python
class MultiOutputRNN(nn.Module):
    """
    RNN encoder → linear projection → multi-step, multi-feature output.
    
    Input  : (Batch, INPUT_LEN, num_features)
    Output : (Batch, OUTPUT_LEN, num_features)
    """
    def __init__(self, cell_type, input_size, hidden_dim,
                 num_layers, output_len, output_size, dropout):
        ...
        self.rnn = nn.LSTM / nn.GRU(...)   # swapped by cell_type flag
        self.fc  = nn.Linear(hidden_dim, output_len * output_size)

    def forward(self, x):
        _, hidden = self.rnn(x)
        h = hidden[0][-1]  # LSTM: (h, c) → take h; GRU: take last layer
        return self.fc(h).view(batch, OUTPUT_LEN, num_features)
```

### Architecture Comparison

| Component | LSTM | GRU |
|-----------|------|-----|
| Gates | Input, Forget, Output, Cell | Reset, Update |
| Hidden state | `(h_n, c_n)` — separate cell state | `h_n` only |
| Parameters | More (~4× gates) | Fewer (~3× gates) |
| Strengths | Better for very long sequences | Faster training, fewer params |
| Forward pass extract | `hidden[0][-1]` (h from top layer) | `hidden[-1]` (top layer) |
| Layers | 2 | 2 |
| Dropout | 0.2 (between layers) | 0.2 (between layers) |

> Dropout is applied between stacked layers only — disabled when `num_layers=1` to avoid PyTorch warning.

---

## ⚡ Hyperparameters

| Hyperparameter | Value | Rationale |
|----------------|-------|-----------|
| `INPUT_LEN` | 10 | 2-week trading history as context |
| `OUTPUT_LEN` | 5 | 1-week forecast horizon |
| `OVERLAP` | 5 | Dense sampling, stride of 5 |
| `BATCH_SIZE` | 64 | Stable gradient estimates |
| `EPOCHS` | 50 | Sufficient convergence |
| `LR` | 1e-3 | Adam default, scheduler adjusts |
| `HIDDEN_DIM` | 128 | Capacity to model 11 correlated features |
| `NUM_LAYERS` | 2 | Stacked RNN for hierarchical patterns |
| `DROPOUT` | 0.2 | Regularization between stacked layers |
| `TEST_RATIO` | 0.2 | 80/20 chronological split |

---

## 🏋️ Training Strategy

### Optimizer: Adam
- Initial LR: `1e-3`
- Adaptive moment estimation — well-suited to RNN training

### Learning Rate Scheduler: ReduceLROnPlateau
```python
scheduler = ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
```
- Halves the LR if training loss doesn't improve for 5 consecutive epochs
- Stabilises convergence in later epochs

### Gradient Clipping
```python
nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```
- Prevents **exploding gradients** — critical for RNN stability
- Applied every batch before the optimizer step

### Loss Function
- **MSE Loss** (`nn.MSELoss`) — penalises large deviations heavily
- Computed across all 5 output timesteps and all 11 features simultaneously

---

## 📏 Evaluation Metrics

All metrics are computed in the **original (₹) scale** after inverse-transforming predictions.

| Metric | Formula | What It Measures |
|--------|---------|-----------------|
| **MSE** | $\frac{1}{n}\sum(y_i - \hat{y}_i)^2$ | Average squared error — penalises outliers |
| **RMSE** | $\sqrt{\text{MSE}}$ | Same unit as target (₹) — interpretable |
| **MAE** | $\frac{1}{n}\sum\|y_i - \hat{y}_i\|$ | Average absolute error — robust to outliers |

```python
# Per-feature RMSE — identify which stock attributes are hardest to predict
def per_feature_rmse(true_inv, pred_inv, feature_names):
    # Computed independently for each of the 11 columns
    ...
```

---

## 📊 Visualizations

The notebook generates **4 publication-quality plots**:

### 1️⃣ Training Loss Curve
- Dual-line chart: LSTM (solid) vs GRU (dashed) over 50 epochs
- MSE on scaled data — reveals convergence speed and stability differences

### 2️⃣ Per-Feature RMSE Bar Chart
- Grouped bar chart: one pair of bars (LSTM / GRU) per feature column
- Shows **which features** (e.g., Volume vs Close) each model predicts best
- X-axis rotated for readability; grid on Y-axis for comparison

### 3️⃣ Date-Wise Prediction vs True Values (All Features)
- `ceil(11/3) × 3` subplot grid (4 rows × 3 cols)
- Each subplot: True (steelblue) vs LSTM Pred (tomato dashed) vs GRU Pred (seagreen dotted)
- X-axis: quarterly date labels; overlapping windows averaged per date via `flatten_windowed()`

### 4️⃣ Close Price Focus Plot
- Single clean full-width chart for the `Close` price column
- Clearest signal for financial interpretation: True vs LSTM vs GRU

---

## 📁 Project Structure

```
📦 KOTAKBANK-LSTM-GRU/
 ┣ 📓 LTSM_GRU_KOTAKBANK.ipynb   ← Main notebook (all code, outputs, plots)
 ┣ 📄 KOTAKBANK.csv               ← Raw NSE stock data (place here before running)
 ┗ 📄 README.md                   ← This file
```

---

## 🛠️ Setup & Installation

### Prerequisites

- Python 3.9+
- pip or conda

### Install Dependencies

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install pandas numpy scikit-learn matplotlib
```

Or all at once via requirements:

```bash
# requirements.txt
torch>=2.0.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
```

```bash
pip install -r requirements.txt
```

### Google Colab (Recommended — zero setup)

All libraries (PyTorch, scikit-learn, matplotlib, pandas) are pre-installed in Colab. Just upload the notebook and the CSV:

```
Runtime → Run all
```

---

## ▶️ How to Run

### Step 1 — Place the dataset
```
KOTAKBANK.csv  →  /content/KOTAKBANK.csv   (Colab)
                  or update CSV_PATH in cell 2 (local)
```

### Step 2 — Configure (optional)
Edit the constants block at the top of the notebook:
```python
INPUT_LEN  = 10    # look-back window
OUTPUT_LEN = 5     # forecast horizon
EPOCHS     = 50    # training epochs
HIDDEN_DIM = 128   # RNN hidden size
```

### Step 3 — Run all cells
```
Kernel → Restart & Run All   (Jupyter)
Runtime → Run all            (Colab)
```

### Expected Console Output
```
Using device: cuda
Loaded /content/KOTAKBANK.csv  ->  shape: (XXXX, 15)
Duplicate rows removed: 0
Using 11 feature columns: ['Prev Close', 'Open', ...]
Train samples: XXXX  |  Test samples: XXXX
X_train: (N, 10, 11)   y_train: (N, 5, 11)

[LSTM] Epoch   1/50  Loss: 0.031456
[LSTM] Epoch  10/50  Loss: 0.008932
...
[GRU]  Epoch  50/50  Loss: 0.003XXX

 Model       MSE     RMSE      MAE
  LSTM   XXX.XX   XX.XXX   XX.XXX
   GRU   XXX.XX   XX.XXX   XX.XXX
```

---

## 📈 Results Summary

> *Exact metric values depend on the full dataset — run the notebook to reproduce.*

### What to Expect

| Observation | Detail |
|-------------|--------|
| **Loss convergence** | Both models converge within ~30–40 epochs; GRU typically slightly faster |
| **Overall accuracy** | RMSE in ₹ scale — lower is better; both models track the Close price trend well |
| **Feature difficulty** | `Volume`, `Trades`, `Turnover` tend to have higher RMSE than price columns |
| **Close price tracking** | Both models capture the general trend; GRU may generalise better with fewer params |
| **Window averaging** | Overlapping windows are averaged per date — smooths the evaluation curve |

---

## 🎯 Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **Chronological split (no shuffle)** | Prevents temporal data leakage — realistic evaluation |
| **Scaler fit on train only** | Strictly avoids leaking test distribution into normalization |
| **Gradient clipping** | Prevents exploding gradients, a common RNN failure mode |
| **ReduceLROnPlateau** | Adaptive LR without manual tuning — stabilises later-epoch loss |
| **Shared `MultiOutputRNN` class** | Guarantees architectural parity; only cell type differs |
| **Dynamic feature detection** | Notebook works even if CSV has minor column schema differences |
| **Window averaging for plots** | Overlapping windows create duplicate dates — averaging gives a clean curve |
| **Inverse transform before metrics** | Metrics in ₹ are interpretable; scaled metrics would be misleading |

---

## 👤 Author

| Field | Value |
|-------|-------|
| **Name** | Md Mozammil Raja |
| **Faculty No.** | 25MCA029 |
| **Student ID** | 202511284 |
| **Course** | CA 24 — Theory Assignment 1 |
| **Dataset** | KOTAKBANK.csv |

---

<div align="center">

Made with ❤️ using PyTorch · pandas · scikit-learn · matplotlib

*"Predicting the market is hard. Understanding the model should not be."*

</div>
