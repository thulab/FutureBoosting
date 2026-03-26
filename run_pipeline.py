#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import random
import numpy as np
import torch
from datetime import datetime

from exp.exp_pipeline import Exp_Pipeline

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EPFLab")

    # basic
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--task_name", type=str, default="pipeline")
    parser.add_argument("--model_name", type=str, default="pipeline-lgbm")
    parser.add_argument("--seed", type=int, default=42)

    # data & io
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True, help="Raw dataset path (.csv/.parquet).")
    parser.add_argument("--data_split_path", type=str, required=True, help="Path to data split json (only data_split).")
    parser.add_argument("--regress_cols_path", type=str, required=True, help="Path to regression columns json (LGBM covariates + target).")
    parser.add_argument("--tsfm_cols_path", type=str, required=True, help="Path to TSFM rollout columns json (variables to forecast).")
    parser.add_argument("--tag", type=str, required=True, help="experiment tag, e.g. SX-dayahead-20240701_20240801")

    # pipeline
    parser.add_argument("--target_hour", type=int, default=0)
    parser.add_argument("--is_std", action="store_true", default=False, help="whether the dataset is standard (e.g. EPF) or shanxi style")
    parser.add_argument("--std_freq", type=str, default=None, help="standard dataset time frequency, e.g. 'h' for hourly, '15min' for 15-minutely")

    # ---- TSFM rollout ----
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--enable_tsfm", type=int, default=0, help="0: disable TSFM rollout, 1: enable TSFM rollout")
    parser.add_argument("--seq_len", type=int, default=1440)  # history length (points)
    parser.add_argument("--pred_len", type=int, default=96)  # forecast horizon
    parser.add_argument("--tsfm_num_samples", type=int, default=20)
    parser.add_argument("--tsfm_cache_path", type=str, default=None)
    parser.add_argument("--tsfm_models", type=str, default="")
    parser.add_argument("--tsfm_model_paths", type=str, default="{}")
    parser.add_argument("--scale", action="store_true", default=False, help="whether to scale the data before TSFM rollout")
    parser.add_argument("--tsfm_use_future_covariates", type=int, default=0)
    parser.add_argument("--tsfm_force_current_rerun", action="store_true", default=False, help="force rerun TSFM rollout for current tag keys and update cache")

    # ---- Regressor selection ----
    parser.add_argument("--regression_model", type=str, default="lgbm", choices=["lgbm", "linear", "lgbm+linear"], help="regression backend: lgbm, linear, or lgbm+linear")
    parser.add_argument("--linear_method", type=str, default="ridge", choices=["ridge", "lasso", "elasticnet"], help="linear regressor type when --regression_model=linear")
    parser.add_argument("--linear_alpha", type=float, default=0.001, help="alpha for linear regression")
    parser.add_argument("--linear_l1_ratio", type=float, default=0.5, help="l1_ratio for elasticnet when --regression_model=linear")
    parser.add_argument("--linear_fit_intercept", action="store_true", default=False, help="fit intercept for linear regression")
    parser.add_argument("--linear_max_iter", type=int, default=20000, help="max_iter for lasso/elasticnet")
    parser.add_argument("--linear_tol", type=float, default=1e-4, help="tol for linear regression")
    parser.add_argument("--linear_plot_train", action="store_true", default=False, help="also plot train fit curve for linear regression")
    parser.add_argument("--linear_topk_feat", type=int, default=30, help="top-k features in linear coefficient plot")

    # ---- LGBM params (pipeline) ----
    parser.add_argument("--num_boost_round", type=int, default=30000)
    parser.add_argument("--early_stopping_rounds", type=int, default=1000)
    parser.add_argument("--lgbm_learning_rate", type=float, default=0.05)
    parser.add_argument("--num_leaves", type=int, default=63)
    parser.add_argument("--feature_fraction", type=float, default=0.9)
    parser.add_argument("--bagging_fraction", type=float, default=0.8)
    parser.add_argument("--min_gain_to_split", type=float, default=0.08)
    parser.add_argument("--bagging_freq", type=int, default=5)
    parser.add_argument("--reg_alpha", type=float, default=0.0)
    parser.add_argument("--reg_lambda", type=float, default=0.0)
    parser.add_argument("--n_jobs", type=int, default=32)
    parser.add_argument("--log_period", type=int, default=200)
    parser.add_argument("--max_depth", type=int, default=-1)
    parser.add_argument("--min_data_in_leaf", type=int, default=20)
    parser.add_argument("--min_sum_hessian_in_leaf", type=float, default=1e-3)
    parser.add_argument("--linear_tree", action="store_true", default=False, help="whether to use linear tree in LightGBM")

    # ===== SHAP explainability =====
    parser.add_argument("--disable_shap", action="store_true", default=False, help="disable SHAP explainability step")
    parser.add_argument("--shap_no_sample", action="store_true", default=False, help="disable SHAP subsampling (use all rows)")
    parser.add_argument("--shap_max_samples", type=int, default=5000, help="max rows used to compute SHAP (subsampled)")
    parser.add_argument("--shap_topk", type=int, default=50, help="max displayed features in SHAP plots")

    args = parser.parse_args()
    args.task_name = "pipeline"

    # seed
    fix_seed = args.seed
    random.seed(fix_seed)
    np.random.seed(fix_seed)
    torch.manual_seed(fix_seed)

    os.makedirs(args.save_dir, exist_ok=True)

    setting = datetime.now().strftime("%y-%m-%d_%H-%M-%S") + f"_{args.tag}_{args.model_name}"

    exp = Exp_Pipeline(args)

    print(f">>>>>>> start pipeline : {setting} >>>>>>>>>>>>>>>>>>>>>>>>>>")
    exp.run(setting)

    torch.cuda.empty_cache()
