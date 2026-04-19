# FutureBoosting

> The official code implementation of **Regression Models Meet Foundation Models: A Hybrid-AI Approach to Practical Electricity Price Forecasting**


---

## Overview

FutureBoosting is a two-stage hybrid forecasting framework for electricity price forecasting (EPF). It bridges two complementary paradigms — **time series foundation models (TSFMs)** and **regression models** — to overcome the fundamental limitations of each.

**Stage 1 — TSFM Feature Augmentation**: A frozen, pre-trained TSFM runs in zero-shot mode to generate forecasts of future-unavailable variables (e.g., system load, renewable generation, thermal generation) over the target trading day. These forecasts encode forward-looking temporal expectations that are otherwise inaccessible to regression models at planning time.

**Stage 2 — Cross-Variate Regression**: An enriched feature set is assembled from the TSFM forecasts, domain-knowledge-constructed factors, and future-available exogenous variables. A lightweight downstream regressor (LightGBM or linear model) is trained on this enriched set to predict the final electricity price curve.

---

## Real-World Deployment

- **Users & Evaluators:** FutureBoosting is evaluated by data analysts from Suzhou Industrial Park Xingzhixun CS Co., Ltd., who co-authored this work.

- **Deployment Pattern:** FutureBoosting delivers day‑ahead and real‑time electricity price forecasts for day D+1 on day D to support trading decisions. Its lightweight regression model is updated monthly to adapt to rapid distribution shifts in the electricity market; users may increase the update frequency for improved performance.

- **Deployment Stats:** FutureBoosting has been deployed since December 2025. We provide specific month-level online deployment statistics below. An additional business-related metric, ACC (1-WAPE), is added for online validation.

- **Estimated Deployment Value:** Based on user feedback, FutureBoosting is estimated to reduce electricity costs by approximately 0.001–0.003 RMB per kWh. For a typical small trading firm with an annual volume of one billion kWh, this translates to roughly 2 million RMB yearly savings under similar trading strategies. This estimate assumes ideal conditions and may vary with policy shifts, strategy adjustments, and trading scale.

- **Deployment Technical Details:** FutureBoosting is deployed as an online forecasting service inside AINode, an AI inference engine of IoTDB, used by Xingzhixun's market traders.

---

## Supported Models

### Time Series Foundation Models

| Key | Model      |
|-----|------------|
| `chronos2` | Chronos2   |
| `moirai2` | Moirai2    |
| `timerxl` | TimerXL    |
| `timesfm` | TimesFM2.5 |
| `tirex` | TiRex      |
| `sundial` | Sundial    |
| `tabpfn` | TabPFN     |

### Second-Stage Regressors

| Key | Description |
|-----|-------------|
| `lgbm` | LightGBM gradient boosting |
| `linear` | Scikit-learn linear model (Ridge / Lasso / ElasticNet) |
| `lgbm+linear` | Train both and output results for each |

---

## Features

- TSFM rollout with Parquet-based prediction caching (avoids redundant inference)
- Zero-shot evaluation of TSFMs on held-out test splits
- Feature engineering and train / validation / test splitting for two dataset styles:
  - **Shanxi-style**: Time-indexed electricity market data with workday filtering
  - **Standard sliding-window** (enabled via `--is_std`): EPF, REALE, and similar benchmark datasets
- LightGBM with early stopping and extensive hyperparameter control
- Linear regression with Ridge, Lasso, or ElasticNet
- Unified evaluation: RMSE, MAE, MAPE, R², plus interactive Plotly plots
- SHAP global feature importance (beeswarm, bar, dependence, waterfall, interaction heatmaps)
- SHAP casebook generation for low / high prediction examples
- Per-stage efficiency profiling (wall-clock time, CPU RSS, GPU memory)

---

## Directory Structure

```text
FutureBoosting/
├── run_pipeline.py             # Main entry point
├── pyproject.toml              # Dependencies (managed by uv)
├── configs/
│   ├── columns/                # JSON column specifications (covariates + target)
│   │   ├── shanxi/
│   │   ├── EPF/
│   │   └── REALE/
│   └── time/                   # JSON data-split configurations
│       ├── data_split_1Y_1M/   # Rolling monthly splits
│       ├── data_split_REALE/
│       └── PVPF/
├── data_provider/
│   └── data_loader.py          # CovariateDatasetBenchmark (sliding-window datasets)
├── exp/
│   ├── exp_pipeline.py         # Pipeline orchestrator
│   └── pipeline/
│       ├── tsfm_infer.py       # TSFM inference & caching
│       ├── feature_select.py   # Feature matrix construction & splitting
│       ├── regressor.py        # LightGBM & linear regressors
│       ├── evaluator.py        # Metrics, CSV summaries, plots
│       ├── shap_explain.py     # SHAP global explanations
│       ├── shap_case.py        # SHAP casebook generation
│       └── eff_profile.py      # Timing & memory profiling
├── ts_models/
│   ├── base.py                 # Abstract TSFM interface
│   ├── factory.py              # Model factory
│   └── adapters/               # One adapter per TSFM
├── scripts/
│   ├── shanxi/
│   │   ├── dayahead/
│   │   └── realtime/
│   └── realE/
│       ├── DE/
│       └── FR/
└── results/                    # Auto-created experiment outputs
```

---

## Environment

**Requirements**

- Python >= 3.12
- CUDA-capable GPU (recommended)
- Pre-downloaded TSFM checkpoints for the models you intend to use

**Install**

```bash
uv sync
```

> The project uses [uv](https://github.com/astral-sh/uv) for dependency management. All dependencies are declared in `pyproject.toml`.

---

## Running Experiments

All experiments are run from the project root via pre-written shell scripts.

### Shanxi electricity market

```bash
# Day-ahead price forecasting with Chronos2
bash scripts/shanxi/dayahead/pipeline_ic94_tsfm_ic27_lgbm_chronos2.sh

# Real-time market (example)
bash scripts/shanxi/realtime/<script_name>.sh
```

### European benchmark (REALE)

```bash
bash scripts/realE/FR/FR_ic16_ic15_chronos2.sh
bash scripts/realE/DE/<script_name>.sh
```

### Script structure

Each script declares variables at the top and then loops over a list of data-split config files:

```bash
# --- paths ---
data_path="/path/to/dataset.csv"
regress_cols_path="configs/columns/shanxi/dayahead/regress_ic27.json"
tsfm_cols_path="configs/columns/shanxi/dayahead/tsfm_ic94.json"

# --- TSFM ---
tsfm_models="chronos2"
seq_len=2048        # history length (timesteps)
pred_len=96         # forecast horizon (96 × 15 min = 1 day)
tsfm_num_samples=20

# --- regression ---
regression_model="lgbm+linear"
linear_method="ridge"
linear_alpha=0.001

# --- LightGBM ---
num_boost_round=30000
early_stopping_rounds=1000
lgbm_learning_rate=0.05
num_leaves=63
feature_fraction=0.9

for data_split_path in "${data_list[@]}"; do
  tag=${data_split_path##*/}
  save_dir=./results/${exp_name}/${model_name}/${tag}
  mkdir -p "$save_dir"
  python run_pipeline.py --data_split_path "$data_split_path" --tag "$tag" --save_dir "$save_dir" ...
done
```

### Switching regressors

Change the variables near the top of any script:

```bash
# LightGBM only
regression_model="lgbm"

# ElasticNet only
regression_model="linear"
linear_method="elasticnet"
linear_alpha=0.001
linear_l1_ratio=0.3

# Both (default)
regression_model="lgbm+linear"
linear_method="ridge"
```

---

## TSFM Checkpoint Configuration

Set the `tsfm_model_paths` variable in your script to point to local checkpoint directories:

```bash
tsfm_model_paths="$(cat <<'JSON'
{
  "sundial":  "/path/to/Sundial",
  "timerxl":  "/path/to/TimerXL",
  "chronos2": "/path/to/Chronos2",
  "tirex":    "/path/to/TiRex",
  "moirai2":  "/path/to/Moirai2",
  "timesfm":  "/path/to/TimesFM",
  "tabpfn":   "/path/to/TabPFN"
}
JSON
)"
```

Only the models listed in `--tsfm_models` need their paths set.

---

## Key CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--device` | `cuda` | Compute device (`cuda` or `cpu`) |
| `--data_path` | — | Path to raw dataset (CSV or Parquet) |
| `--data_split_path` | — | JSON file defining train/val/test boundaries |
| `--regress_cols_path` | — | JSON file listing covariate + target columns for regression |
| `--tsfm_cols_path` | — | JSON file listing variables for TSFM rollout |
| `--tsfm_models` | — | Comma-separated list of TSFM keys to use |
| `--tsfm_model_paths` | — | JSON mapping of model key → checkpoint path |
| `--seq_len` | `2048` | Context length fed to the TSFM |
| `--pred_len` | `96` | Forecast horizon |
| `--tsfm_num_samples` | `20` | Number of samples drawn from TSFM |
| `--tsfm_cache_path` | — | Directory for caching TSFM predictions |
| `--enable_tsfm` | `1` | Set to `0` to skip TSFM inference (use cache only) |
| `--regression_model` | `lgbm` | Regressor: `lgbm`, `linear`, or `lgbm+linear` |
| `--linear_method` | `ridge` | Linear model type: `ridge`, `lasso`, `elasticnet` |
| `--linear_alpha` | `0.001` | Regularization strength for linear models |
| `--num_leaves` | `63` | LightGBM `num_leaves` |
| `--learning_rate` | `0.05` | LightGBM learning rate |
| `--early_stopping_rounds` | `1000` | LightGBM early stopping patience |
| `--disable_shap` | `False` | Skip SHAP analysis |
| `--shap_topk` | `50` | Number of top features shown in SHAP plots |
| `--is_std` | `False` | Use standard sliding-window dataset mode |
| `--save_dir` | — | Root directory for all outputs |
| `--tag` | — | Experiment tag (used for naming sub-directories) |

---

## Output Structure

After a run, results are organized under `--save_dir`:

```text
results/
└── <exp_name>/<model_name>/<tag>/
    ├── lgbm/
    │   ├── metrics_test.json          # RMSE, MAE, MAPE, R² on test set
    │   ├── plots/
    │   │   ├── test_point_series.html # Interactive forecast vs. actual plot
    │   │   └── ...
    │   └── shap/
    │       ├── shap_values_test.npy   # Raw SHAP values (NumPy array)
    │       ├── feature_importance_test.csv
    │       ├── beeswarm_test.png
    │       ├── bar_test.png
    │       └── casebook_low_high_png/
    │           └── ...                # Per-case SHAP waterfall images
    └── linear_ridge/
        └── ...                        # Same structure for linear model

metrics_all.csv                        # Aggregated metrics across all splits
metrics_efficiency_cache.csv           # Per-stage timing & memory profile
tsfm_cache/
└── <model>_<tag>.parquet              # Cached TSFM predictions
```

| File | Description |
|------|-------------|
| `metrics_test.json` | Evaluation metrics for the test split |
| `metrics_all.csv` | Summary table across all date splits |
| `plots/` | Interactive Plotly and static Matplotlib visualizations |
| `shap/` | SHAP values, importance CSVs, and explanation plots |
| `tsfm_cache/*.parquet` | Cached TSFM predictions (reused across runs) |
| `metrics_efficiency_cache.csv` | Wall-clock time and CPU/GPU memory per pipeline stage |
