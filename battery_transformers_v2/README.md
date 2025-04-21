# 🔋 Battery SOH Prediction and Model Analysis

This project explores machine learning and deep learning approaches for predicting the **State of Health (SOH)** of batteries. Accurate SOH prediction is vital for Battery Management Systems (BMS), especially in domains such as electric vehicles (EVs) and energy storage systems, ensuring safety, reliability, and performance.

It includes implementations of multiple time-series models, a robust data processing pipeline, and tools for monitoring hardware performance during model training and inference.

---

## 🚀 Features

- **SOH Prediction Models:**
  - Long Short-Term Memory (LSTM)
  - Temporal Convolutional Network (TCN)
  - Transformer (including an HDMR variant)
  - Convolutional Neural Network (CNN)
  - XGBoost

- **Data Handling:**
  - CSV-based data loading with flexible configuration
  - Standard and literature-adapted preprocessing pipelines
  - Sequence generation for time-series input formats

- **Hardware Monitoring:**
  - Real-time monitoring of GPU and CPU usage
  - Calculates differential usage based on idle baselines
  - Requires `nvidia-smi` for GPU metrics

- **Experiment Framework:**
  - Configuration-driven via YAML files
  - Modular codebase: data loading, models, training, evaluation
  - Output logging (metrics, models, visualizations)

- **Evaluation Metrics:**
  - Standard: RMSE, MAE, R²
  - Time-series: Pearson Correlation Coefficient (PCC), Mean Directional Accuracy (MDA)

- **Visualization:**
  - Plot predictions vs actual values
  - Jupyter Notebook for comparison of results & hardware usage

---

## 🧠 Models Implemented

- `src/models/lstm.py`
- `src/models/tcn.py`
- `src/models/transformer.py` (including HDMR variant)
- `src/models/cnn.py`
- `src/models/xgboost_model.py`

---

## 📁 Dataset Structure

You’ll need time-series cycle data in CSV format.

- Suggested structure:
  ```
  data/
    ├── XJTU_data/
    └── battery/
  ```

- Expected columns: features, targets, timestamp (configurable)
- Also compatible with fuel cell datasets (see `fc_transformer.py`)

---

## 🧱 Project Structure

```
├── config/               # YAML experiment configs
├── data/                 # Dataset directory (user-created)
├── notebooks/            # Jupyter notebooks
├── outputs/              # Logs, models, plots
├── src/                  # Core source code
│   ├── data_loader.py
│   ├── evaluator.py
│   ├── monitoring.py
│   ├── preprocess.py
│   ├── trainer.py
│   └── models/
│       ├── cnn.py
│       ├── lstm.py
│       ├── tcn.py
│       ├── transformer.py
│       └── xgboost_model.py
├── main.py               # Main experiment runner
├── requirements.txt
└── README.md             # You are here!
```

---

## ⚙️ Setup

1. **Clone the Repository**
   ```bash
   git clone <your-repo-url>
   cd <your-repo-name>
   ```

2. **Create a Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Linux/macOS
   .\venv\Scripts\activate   # On Windows
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```
   > *Note: For GPU acceleration, ensure CUDA-compatible PyTorch is installed.*

4. **Prepare Data**
   ```bash
   mkdir -p data/XJTU_data
   # Place your CSV files inside the appropriate directory
   ```

---

## 🧪 Running Experiments

Use `main.py` with a config file:

```bash
# LSTM Experiment
python main.py --config config/lstm_config.yaml

# XGBoost Experiment
python main.py --config config/xgb_config.yaml

# Transformer with custom preprocessor
python main.py --config config/transformer_config.yaml
```

> All results will be saved under `outputs/`

---

## ⚙️ Configuration (YAML)

Each YAML config controls:
- `data`: dataset path, feature selection, sequence length
- `model`: type, architecture-specific settings
- `training`: batch size, epochs, optimizer, learning rate
- `monitoring`: enable logging, idle baseline duration
- `outputs`: save paths for logs/models/plots

---

## 📊 Visualization

Use the provided notebook to compare multiple runs:

```bash
cd notebooks
jupyter notebook visualize_results.ipynb
```

Generates visual summaries using logs and hardware metrics.

---

## 🧭 Monitoring Internals

The monitoring system:
- Measures idle GPU/CPU usage first
- Logs usage periodically during training
- Reports the difference vs idle to isolate model resource demands

---

## 🔧 TODO / Extensions

- [ ] Outlier detection improvements
- [ ] Integration with hyperparameter optimization (Optuna, Ray Tune)
- [ ] Model optimization (pruning, distillation, quantization)
- [ ] More accurate energy calculations
- [ ] Unit and integration tests

---

## 📄 License

[MIT License or specify accordingly]

---

## 🙌 Acknowledgements

- [Optional: Acknowledge any public datasets, frameworks, or papers referenced.]
