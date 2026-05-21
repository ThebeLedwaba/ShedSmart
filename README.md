# ShedSmart — LSTM-Based Load-Shedding Stage Forecasting

## Overview

This project develops an LSTM (Long Short-Term Memory) neural network model to forecast South African load-shedding stages (0-6) using historical Eskom grid metrics and stage event data from 2018-2023.

## Key Results

| Model | RMSE | Accuracy | MAPE |
|-------|------|----------|------|
| Persistence Baseline | 0.658 | 91.5% | 14.4% |
| Logistic Regression | 0.973 | 72.4% | 29.9% |
| LSTM (ShedSmart) | 1.398 | 52.4% | 53.9% |

Key Insight: The LSTM underperformed due to distribution shift — trained on 2018-2021 (80% Stage 0), tested on 2023 (34% non-zero stages). This highlights the challenge of forecasting changing system behavior.

## Project Structure

```
├── README.md                                          # This file
├── LoadShedding_LSTM_Forecast.ipynb                  # Main Jupyter notebook with LSTM model
├── ESK2033.csv                                        # Eskom hourly grid metrics (2018-2023)
├── EskomSePush_history.csv                           # Historical stage events (2014-2025)
├── tuning_log.csv                                     # Hyperparameter tuning results (12 combinations)
├── model_results.csv                                  # Model comparison results
├── eda_analysis.png                                   # Stage distribution and seasonality plots
├── training_curves.png                                # LSTM loss and accuracy over epochs
├── confusion_matrix.png                               # Prediction error patterns
├── actual_vs_predicted.png                            # 30-day forecast vs reality
├── error_analysis.png                                 # Error by season and stage
├── tuning_results.png                                 # All 12 hyperparameter combinations
└── requirements.txt                                   # Python dependencies
```

## Dataset

**Files:**
- `ESK2033.csv` - Eskom hourly grid metrics (Residual Demand, UCLF, Thermal Generation, etc.)
- `EskomSePush_history.csv` - Load-shedding stage events resampled to hourly frequency

**Time Range:** 2018-2023 (43,824 hourly observations)

## Model Architecture

| Layer | Type | Parameters |
|-------|------|------------|
| 1 | LSTM | 128 units, return sequences |
| 2 | Dropout | 0.2 |
| 3 | LSTM | 64 units |
| 4 | Dropout | 0.2 |
| 5 | Dense | 32 units (ReLU) |
| 6 | Dense | 5 units (Softmax) |

## Feature Engineering (16 features)

- Lag features: stage at t-1, t-24, t-48 hours
- Cyclical time encodings: sin/cos for hour, day-of-week, month
- Rolling statistics: 12-hour rolling mean and standard deviation
- Eskom grid metrics: Residual Demand, Total UCLF, Thermal Generation, Manual Load Reduction, Dispatchable Generation

## Hyperparameter Tuning

Grid search over 12 combinations using 3-fold TimeSeriesSplit:

| Hyperparameter | Values Tested | Best Value |
|----------------|---------------|-------------|
| LSTM units | 32, 64, 128 | 128 |
| Dropout rate | 0.2, 0.3 | 0.2 |
| Learning rate | 0.001, 0.0005 | 0.001 |

## Visualisations

| File | Description |
|------|-------------|
| eda_analysis.png | Stage distribution, monthly averages, seasonality, hourly patterns |
| training_curves.png | Training and validation loss and accuracy |
| confusion_matrix.png | Raw and normalised prediction error counts |
| actual_vs_predicted.png | 30-day and 7-day forecast vs actual stages |
| error_analysis.png | Mean absolute error by season and by true stage |
| tuning_results.png | All 12 hyperparameter combinations ranked by validation MAE |

## Getting Started

### Requirements

```
python >= 3.8
jupyter
tensorflow
pandas
numpy
matplotlib
seaborn
scikit-learn
```

### Installation

```bash
pip install -r requirements.txt
```

### Running the Notebook

```bash
jupyter notebook LoadShedding_LSTM_Forecast.ipynb
```

## Limitations

- **Distribution shift:** Model trained on 2018-2021 (low severity), tested on 2023 (high severity)
- **Limited high-stage examples:** Only 52 Stage 3 and 77 Stage 4 examples in training
- **No exogenous variables:** Weather forecasts, maintenance schedules, public holidays not included
- **Fixed window size:** 24-hour window only; other sizes not explored

## Future Work

- Retrain on 2022-2024 data (balanced Stage 4-6 examples)
- Hybrid persistence-LSTM ensemble
- Hierarchical classification (binary stage 0 vs non-zero, then specific stage)
- Add exogenous features (weather, holidays, maintenance)
- Deploy as real-time API

## Author

Thebe Ledwaba — Student ID: 219119007

## License

Educational Project - 2026

---

**Note:** Load shedding patterns are influenced by multiple factors including demand, generation capacity, maintenance schedules, and unexpected failures. This model provides probabilistic forecasts based on historical patterns.
