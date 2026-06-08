# Crypto Market Analyzer & Quant Trading Pipeline

An end-to-end Quantitative Trading Pipeline built to forecast Bitcoin (BTC) price movements using Machine Learning (Random Forest, XGBoost) and Deep Learning (LSTM Networks). The project focuses on a highly realistic approach to market friction, strictly accounting for exchange fees and slippage, and uses vectorized backtesting for true evaluation.

## 📌 The Problem

In financial time-series forecasting, especially in crypto, achieving high accuracy (e.g., 55% accuracy on predicting "Up" or "Down") does not guarantee profitability. The primary reasons are **Data Leakage** (using future information implicitly during training) and **Market Frictions** (ignoring the cost of trading).

Many models suffer from the **"Predict Zero" problem**: when fees are introduced, the most rational mathematical decision for the model to minimize loss is to stop trading entirely (always predict 0), because random market noise will slowly bleed the account through fees.

### Our Solution: A Realistic Target
Instead of asking the model to predict if the price will go up, we ask it to predict if the price will go up **enough to cover all trading costs**. 

```python
# The target is only 1 if the expected return overcomes the fees
FEE_RATE = 0.0015 # 0.15% Maker/Taker fee
SLIPPAGE = 0.0005 # 0.05% Slippage

Cost = FEE_RATE * 2 + SLIPPAGE # Round-trip cost
df['Realistic_Target'] = (df['Expected_Return'] > Cost).astype(int)
```
This forces the model to only find entries with high expected momentum, entirely filtering out low-volatility chop.

## 🚀 Architecture & Pipeline

1. **Data Engineering (`src/data.py` & `src/features.py`)**: Fetches data from Yahoo Finance and calculates 20+ technical indicators (RSI, MACD, Bollinger Bands, ATR, etc.) focusing on momentum and volatility.
2. **Baseline Models (`src/train_base.py`)**: Implementation of Random Forest and XGBoost. To prevent Data Leakage, we rigorously use `TimeSeriesSplit` (Walk-Forward Validation) to simulate chronological trading.
3. **Deep Learning (`src/train_dl.py`)**: Sequence modeling using LSTM networks to capture temporal dependencies. 
4. **Vectorized Backtesting (`src/backtest.py`)**: High-performance simulation using `vectorbt`, processing Out-of-Sample predictions (2024 - 2026) to validate the strategy.

## 🏆 Backtesting Results

The best performing model was the **LSTM Network with Dynamic Thresholding**. Rather than using a static 0.5 probability cutoff, the model dynamically calculated the optimal probability threshold during cross-validation (typically `> 0.35`).

### Out-of-Sample Performance (Jan 2024 - Jun 2026)
* **Initial Capital:** $10,000
* **Final Capital:** $12,956 (+29.56% Return)
* **Total Trades:** 14 (Extremely patient execution)
* **Win Rate:** 30.76%
* **Avg Winning Trade:** +33.71%
* **Avg Losing Trade:** -5.24%
* **Profit Factor:** 1.89

### Key Takeaway
The model operates as a pure **Trend-Follower**. Its win rate is low (~30%), but the asymmetry of the strategy is massive. When the model is wrong, it cuts losses immediately (-5.2%). When the model is right, it rides the trend for huge gains (+33.7%). This results in a robust, profitable system that survives the brutal frictions of the crypto market.

## 🛠️ Usage

### Installation
The project relies on `uv` for ultra-fast dependency management.
```bash
# Install dependencies
uv sync
```

### Running the Pipeline
You can run the entire pipeline sequentially:
```bash
# 1. Download raw data
uv run python src/data.py

# 2. Engineer features
uv run python src/features.py

# 3. Train Baseline (RF & XGBoost)
uv run python src/train_base.py

# 4. Train Deep Learning (LSTM)
uv run python src/train_dl.py

# 5. Run Backtest
uv run python src/backtest.py
```

The backtester will generate a detailed statistical report (`backtest_report.txt`) and a highly interactive HTML equity curve in the `results/` directory.

### Visualizing Experiments (MLflow)
This project uses MLflow for experiment tracking. To visualize the results and metrics of your runs, execute the following command in your terminal:
```bash
mlflow server --host localhost --port 5000
```
Then, open your browser and navigate to [http://localhost:5000](http://localhost:5000).

## 🔮 Future Iterations

The pipeline is built with modularity in mind. To transition from Daily (`1d`) to Hourly (`1h`) trading:
1. Modify `INTERVAL = "1h"` in `src/config.py`.
2. Ensure the historical data provider supports the granularity.
3. The feature engineering and sliding-window logic will automatically adapt.
