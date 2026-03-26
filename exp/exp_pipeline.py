from __future__ import annotations

import json
import os
from pathlib import Path
import numpy as np

from .pipeline.tsfm_infer import get_tsfm_target_col, load_compare_series_from_cache, run_tsfm_rollout
from .pipeline.feature_select import build_features
from .pipeline.regressor import lgbm_regression, Linear_regression
from .pipeline.evaluator import evaluate
from .pipeline.eff_profile import profile_stage
from .pipeline.shap_explain import run_model_shap_explain
from .pipeline.shap_case import export_shap_casebook_low_high

class Exp_Pipeline:
    def __init__(self, args):
        self.args = args

    def run(self, setting=None):
        os.makedirs(self.args.save_dir, exist_ok=True)
        eff_csv = Path(self.args.save_dir).parent / "metrics_efficiency_cache.csv"

        with profile_stage(stage="tsfm_rollout", args=self.args, eff_csv=eff_csv):
            pred_table = run_tsfm_rollout(args=self.args)

        with profile_stage(stage="build_features", args=self.args, eff_csv=eff_csv):
            X_tr, y_tr, N_tr, X_va, y_va, N_va, X_te, y_te, N_te, meta = build_features(
                args=self.args,
                pred_table=pred_table,
            )

        regression_model = str(getattr(self.args, "regression_model", "lgbm")).strip().lower()
        is_std = bool(getattr(self.args, "is_std", False))
        std_step = getattr(self.args, "std_freq", None) if is_std else None
        plot_target_mean = meta.get("target_mean") if is_std and bool(meta.get("target_scaled", False)) else None
        plot_target_std = meta.get("target_std") if is_std and bool(meta.get("target_scaled", False)) else None
        compare_series = {}
        compare_series = load_compare_series_from_cache(
            args=self.args,
            time_points=meta["te_time"],
        )

        if regression_model == "lgbm+linear":
            run_modes = ["lgbm", "linear"]
        elif regression_model in {"lgbm", "linear"}:
            run_modes = [regression_model]
        else:
            raise ValueError(
                f"Unknown regression_model={regression_model}, expected one of: lgbm, linear, lgbm+linear"
            )

        summary_csv = os.path.join(os.path.dirname(self.args.save_dir), "metrics_all.csv")
        linear_method = str(getattr(self.args, "linear_method", "ridge")).strip().lower()

        for run_mode in run_modes:
            if len(run_modes) == 1:
                run_save_dir = self.args.save_dir
            elif run_mode == "lgbm":
                run_save_dir = str(Path(self.args.save_dir) / "lgbm")
            else:
                run_save_dir = str(Path(self.args.save_dir) / f"linear_{linear_method}")

            os.makedirs(run_save_dir, exist_ok=True)

            if run_mode == "lgbm":
                with profile_stage(stage="train_regressor_lgbm", args=self.args, eff_csv=eff_csv):
                    model = lgbm_regression(X_tr, y_tr, X_va, y_va, self.args)

                with profile_stage(stage="test_predict_eval_lgbm", args=self.args, eff_csv=eff_csv):
                    y_te_pred = model.predict(X_te, num_iteration=model.best_iteration)
                    test_metrics = evaluate(
                        y_point_pred=y_te_pred,
                        y_point_true=y_te,
                        N=N_te,
                        L=meta["L"],
                        split="test",
                        tag=self.args.tag,
                        save_dir=run_save_dir,
                        best_iteration=int(model.best_iteration),
                        time_points=meta["te_time"],
                        summary_csv=summary_csv,
                        model=model,
                        feature_names=meta["feature_names"],
                        topk=20,
                        compare_series=compare_series,
                        is_std=is_std,
                        std_step=std_step,
                        regression_model="lgbm",
                        summary_meta={"model_name": getattr(self.args, "model_name", "pipeline-lgbm")},
                        plot_target_mean=plot_target_mean,
                        plot_target_std=plot_target_std,
                        compare_series_is_raw=True,
                    )
                    print("[test][lgbm]", test_metrics)
            else:
                with profile_stage(stage=f"train_regressor_linear_{linear_method}", args=self.args, eff_csv=eff_csv):
                    model = Linear_regression(
                        X_tr=X_tr,
                        y_tr=y_tr,
                        method=linear_method,
                        alpha=float(getattr(self.args, "linear_alpha", 0.001)),
                        l1_ratio=float(getattr(self.args, "linear_l1_ratio", 0.5)),
                        fit_intercept=bool(getattr(self.args, "linear_fit_intercept", False)),
                        max_iter=int(getattr(self.args, "linear_max_iter", 20000)),
                        tol=float(getattr(self.args, "linear_tol", 1e-4)),
                        random_state=int(getattr(self.args, "seed", 0)),
                    )
                with profile_stage(stage=f"test_predict_eval_linear_{linear_method}", args=self.args, eff_csv=eff_csv):
                    if (not np.isfinite(X_te).all()) or (not np.isfinite(y_te).all()):
                        raise ValueError("[linear pipeline] test contains NaN/Inf.")
                    y_te_pred = model.predict(X_te)
                    y_tr_pred = model.predict(X_tr) if bool(getattr(self.args, "linear_plot_train", False)) else None
                    summary_meta = {
                        "alpha": float(getattr(self.args, "linear_alpha", 0.001)),
                        "fit_intercept": bool(getattr(self.args, "linear_fit_intercept", False)),
                    }
                    if linear_method == "elasticnet":
                        summary_meta["l1_ratio"] = float(getattr(self.args, "linear_l1_ratio", 0.5))

                    test_metrics = evaluate(
                        y_point_pred=y_te_pred,
                        y_point_true=y_te,
                        N=N_te,
                        L=meta["L"],
                        split="test",
                        tag=self.args.tag,
                        save_dir=run_save_dir,
                        best_iteration=None,
                        time_points=meta["te_time"],
                        summary_csv=summary_csv,
                        model=model,
                        feature_names=meta["feature_names"],
                        topk=int(getattr(self.args, "linear_topk_feat", 30)),
                        compare_series=compare_series,
                        is_std=is_std,
                        std_step=std_step,
                        regression_model="linear",
                        method=linear_method,
                        summary_meta=summary_meta,
                        plot_train=bool(getattr(self.args, "linear_plot_train", False)),
                        train_point_pred=y_tr_pred,
                        train_point_true=y_tr if y_tr_pred is not None else None,
                        train_time_points=meta.get("tr_time") if y_tr_pred is not None else None,
                        train_N=meta.get("N_tr") if y_tr_pred is not None else None,
                        plot_target_mean=plot_target_mean,
                        plot_target_std=plot_target_std,
                        compare_series_is_raw=True,
                    )
                    print(f"[test][{linear_method}]", test_metrics)

            if not getattr(self.args, "disable_shap", False):
                with profile_stage(stage=f"shap_explain_{run_mode}", args=self.args, eff_csv=eff_csv):
                    try:
                        run_model_shap_explain(
                            model=model,
                            X=X_te,
                            feature_names=meta.get("feature_names", []),
                            save_dir=run_save_dir,
                            split="test",
                            tag=getattr(self.args, "tag", "pipeline"),
                            max_samples=int(getattr(self.args, "shap_max_samples", 5000)),
                            seed=int(getattr(self.args, "seed", 2021)),
                            topk=int(getattr(self.args, "shap_topk", 10)),
                            n_dependence_plots=int(getattr(self.args, "shap_n_dependence", 2)),
                            n_waterfall_samples=int(getattr(self.args, "shap_n_waterfall", 0)),
                            n_decision_samples=int(getattr(self.args, "shap_n_decision", 0)),
                            enable_interaction=bool(getattr(self.args, "shap_enable_interaction", False)),
                            enable_heatmap=bool(getattr(self.args, "shap_enable_heatmap", False)),
                        )
                    except Exception as e:
                        print(f"[shap] skipped due to error: {type(e).__name__}: {e}")

                with profile_stage(stage=f"shap_casebook_{run_mode}", args=self.args, eff_csv=eff_csv):
                    try:
                        save_dir = Path(run_save_dir)
                        shap_dir = save_dir / "shap"
                        shap_values = np.load(shap_dir / "shap_values_test.npy")
                        with open(shap_dir / "shap_expected_value_test.json", "r", encoding="utf-8") as f:
                            expected_value = float(json.load(f)["expected_value"])

                        feature_names = list(meta.get("feature_names", []))
                        tsfm_feature_name = str(getattr(self.args, "casebook_tsfm_feature_name", "")).strip()
                        if not tsfm_feature_name:
                            target_col = get_tsfm_target_col(args=self.args)
                            models = [m.strip().lower() for m in str(getattr(self.args, "tsfm_models", "")).split(",") if m.strip()]
                            preferred_names = [f"pred_{m}_{target_col}" for m in models]
                            tsfm_feature_name = next(
                                (name for name in preferred_names if name in feature_names),
                                "",
                            )
                        if not tsfm_feature_name:
                            target_col = get_tsfm_target_col(args=self.args)
                            tsfm_feature_name = next(
                                (name for name in feature_names if name.startswith("pred_") and name.endswith(f"_{target_col}")),
                                "",
                            )
                        if not tsfm_feature_name:
                            tsfm_feature_name = next((name for name in feature_names if name.startswith("pred_")), "")
                        if not tsfm_feature_name:
                            raise ValueError("no TSFM feature found for casebook")

                        export_shap_casebook_low_high(
                            out_dir=shap_dir / "casebook_low_high_png",
                            shap_values=shap_values,
                            expected_value=expected_value,
                            X=X_te,
                            feature_names=feature_names,
                            y_true=y_te,
                            y_pred=y_te_pred,
                            L=int(meta["L"]),
                            tsfm_feature_name=tsfm_feature_name,
                            time_points=meta.get("te_time", None),
                            max_cases=int(getattr(self.args, "casebook_max_cases", 50)),
                            topk_waterfall=int(getattr(self.args, "casebook_topk", 10)),
                            quantile_q=float(getattr(self.args, "casebook_q", 0.10)),
                            seed=int(getattr(self.args, "seed", 0)),
                        )
                    except Exception as e:
                        print(f"[casebook] skipped due to error: {type(e).__name__}: {e}")
