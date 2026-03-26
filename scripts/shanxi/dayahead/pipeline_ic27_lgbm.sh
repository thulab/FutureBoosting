#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=7

# task
device="cuda"


# data configs (monthly json list)
# data files
data_path="/data/libinzhu/research/dataset/shanxi_spot_data.csv"
regress_cols_path="configs/columns/shanxi/dayahead/regress_ic27.json"
tsfm_cols_path="configs/columns/shanxi/dayahead/tsfm_ic94.json"

data_list_dir=configs/time/data_split_1Y_1M
data_list=(
  "${data_list_dir}/SX-dayahead-20250101_20250201.json"
  "${data_list_dir}/SX-dayahead-20250201_20250301.json"
  "${data_list_dir}/SX-dayahead-20250301_20250401.json"
  "${data_list_dir}/SX-dayahead-20250401_20250501.json"
  "${data_list_dir}/SX-dayahead-20250501_20250601.json"
  "${data_list_dir}/SX-dayahead-20250601_20250701.json"
  "${data_list_dir}/SX-dayahead-20250701_20250801.json"
  "${data_list_dir}/SX-dayahead-20250801_20250901.json"
  "${data_list_dir}/SX-dayahead-20250901_20251001.json"
  "${data_list_dir}/SX-dayahead-20251001_20251101.json"
  "${data_list_dir}/SX-dayahead-20251101_20251201.json"
  "${data_list_dir}/SX-dayahead-20251201_20260101.json"
)


# -------- pipeline params --------
target_hour=0
enable_tsfm=0
pred_len=96
seq_len=2048
batch_size=8
tsfm_num_samples=20
tsfm_use_future_covariates=0

# ---- Regressor params ----
regression_model="lgbm+linear"
linear_method="ridge"
linear_alpha=0.001
linear_l1_ratio=0.5
linear_fit_intercept=0
linear_max_iter=20000
linear_tol=1e-4
linear_plot_train=0
linear_topk_feat=30


# model list
tsfm_models="chronos2"

# ---- TSFM model paths (JSON) ----
tsfm_model_paths="$(cat <<'JSON'
{
  "sundial":  "/share/workspace/qiuyunzhong/CKPT/Sundial-base-128m-transformers4.56.2",
  "timerxl":  "/share/workspace/qiuyunzhong/CKPT/Timerxl-base-transformers4.56.2",
  "chronos2": "/share/workspace/qiuyunzhong/CKPT/chronos-ckpt/chronos2",
  "tirex":    "/share/workspace/qiuyunzhong/CKPT/TiRex",
  "moirai2":  "/share/workspace/qiuyunzhong/CKPT/moirai-ckpt/moirai-2.0-R-small",
  "timesfm":  "/share/workspace/qiuyunzhong/CKPT/TimesFM/timesfm-2.5-200m-pytorch",
  "tabpfn":   "/share/workspace/qiuyunzhong/CKPT/TabPFN/TabPFN-v2-reg/tabpfn-v2-regressor-v2_default.ckpt"
}
JSON
)"

regression_tag="$regression_model"
if [[ "$regression_model" == "linear" ]]; then
  regression_tag="linear_${linear_method}"
elif [[ "$regression_model" == "lgbm+linear" ]]; then
  regression_tag="lgbm_linear_${linear_method}"
fi
model_name="pipeline_$(basename "$regress_cols_path" .json)_${regression_tag}"
exp_name="shanxi/dayahead/regress_ic27"

# ---- LGBM params ----
num_boost_round=30000
early_stopping_rounds=1000
lgbm_learning_rate=0.05
num_leaves=63
feature_fraction=0.9
bagging_fraction=0.8
min_gain_to_split=0.08
bagging_freq=5
reg_alpha=0.0
reg_lambda=0.0
n_jobs=32
log_period=200
seed=42

LOG_DIR=./results/${exp_name}/${model_name}
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/run.log"

for data_split_path in "${data_list[@]}"; do
  tag=${data_split_path##*/}
  tag=${tag%.json}
  save_dir=./results/${exp_name}/${model_name}/${tag}
  tsfm_cache_path="./results/${exp_name}/tsfm_cache"
  mkdir -p "$save_dir"
  mkdir -p "$tsfm_cache_path"

  echo "=== Running ${model_name} on ${tag} ===" | tee -a "$LOG_FILE"
  echo "Start time: $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$LOG_FILE"
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}" | tee -a "$LOG_FILE"
  echo "Save dir: $save_dir" | tee -a "$LOG_FILE"
  echo "Regression: $regression_model (linear_method=$linear_method)" | tee -a "$LOG_FILE"
  echo "----------------------------------------" | tee -a "$LOG_FILE"

  linear_args=(
    --regression_model "$regression_model"
    --linear_method "$linear_method"
    --linear_alpha "$linear_alpha"
    --linear_l1_ratio "$linear_l1_ratio"
    --linear_max_iter "$linear_max_iter"
    --linear_tol "$linear_tol"
    --linear_topk_feat "$linear_topk_feat"
  )
  if [[ "$linear_fit_intercept" == "1" ]]; then
    linear_args+=(--linear_fit_intercept)
  fi
  if [[ "$linear_plot_train" == "1" ]]; then
    linear_args+=(--linear_plot_train)
  fi

  python run_pipeline.py \
    --device "$device" \
    --model_name $model_name \
    --seed $seed \
    --data_path "$data_path" \
    --regress_cols_path "$regress_cols_path" \
    --tsfm_cols_path "$tsfm_cols_path" \
    --data_split_path $data_split_path \
    --tag "$tag" \
    --save_dir "$save_dir" \
    --target_hour $target_hour \
    --seq_len $seq_len \
    --batch_size $batch_size \
    --enable_tsfm $enable_tsfm \
    --pred_len $pred_len \
    --tsfm_num_samples $tsfm_num_samples \
    --tsfm_cache_path "$tsfm_cache_path" \
    --tsfm_models "$tsfm_models" \
    --tsfm_model_paths "$tsfm_model_paths" \
    --tsfm_use_future_covariates $tsfm_use_future_covariates \
    --num_boost_round $num_boost_round \
    --early_stopping_rounds $early_stopping_rounds \
    --lgbm_learning_rate $lgbm_learning_rate \
    --num_leaves $num_leaves \
    --feature_fraction $feature_fraction \
    --bagging_fraction $bagging_fraction \
    --min_gain_to_split $min_gain_to_split \
    --bagging_freq $bagging_freq \
    --reg_alpha $reg_alpha \
    --reg_lambda $reg_lambda \
    --n_jobs $n_jobs \
    --log_period $log_period \
    --disable_shap \
    --tsfm_force_current_rerun \
    "${linear_args[@]}" \
    2>&1 | tee -a "$LOG_FILE"

  echo "End time: $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$LOG_FILE"
  echo "========================================" | tee -a "$LOG_FILE"
  echo "" | tee -a "$LOG_FILE"
done