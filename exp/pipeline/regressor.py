from __future__ import annotations

from typing import Dict

import lightgbm as lgb
import numpy as np
from sklearn.linear_model import ElasticNet, Lasso, Ridge


def lgbm_regression(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_va: np.ndarray,
    y_va: np.ndarray,
    args,
) -> lgb.Booster:
    dtr = lgb.Dataset(X_tr, label=y_tr)
    dva = lgb.Dataset(X_va, label=y_va, reference=dtr)

    params = dict(
        objective="regression",
        metric="rmse",
        boosting_type="gbdt",
        learning_rate=args.lgbm_learning_rate,
        num_leaves=args.num_leaves,
        max_depth=args.max_depth,
        min_data_in_leaf=args.min_data_in_leaf,
        min_sum_hessian_in_leaf=args.min_sum_hessian_in_leaf,
        min_gain_to_split=args.min_gain_to_split,
        linear_tree=args.linear_tree,
        feature_fraction=args.feature_fraction,
        bagging_fraction=args.bagging_fraction,
        bagging_freq=args.bagging_freq,
        lambda_l1=args.reg_alpha,
        lambda_l2=args.reg_lambda,
        verbose=-1,
        seed=args.seed,
        num_threads=args.n_jobs,
    )

    return lgb.train(
        params=params,
        train_set=dtr,
        valid_sets=[dtr, dva],
        valid_names=["train", "valid"],
        num_boost_round=args.num_boost_round,
        callbacks=[
            lgb.early_stopping(stopping_rounds=args.early_stopping_rounds, verbose=True),
            lgb.log_evaluation(period=args.log_period),
        ],
    )


def _make_linear_model(
    *,
    method: str,
    alpha: float,
    l1_ratio: float,
    fit_intercept: bool,
    max_iter: int,
    tol: float,
    random_state: int,
):
    name = method.strip().lower()
    if name == "ridge":
        return Ridge(alpha=float(alpha), fit_intercept=fit_intercept, random_state=random_state)
    if name == "lasso":
        return Lasso(
            alpha=float(alpha),
            fit_intercept=fit_intercept,
            max_iter=int(max_iter),
            tol=float(tol),
            random_state=random_state,
        )
    if name == "elasticnet":
        return ElasticNet(
            alpha=float(alpha),
            l1_ratio=float(l1_ratio),
            fit_intercept=fit_intercept,
            max_iter=int(max_iter),
            tol=float(tol),
            random_state=random_state,
        )
    raise ValueError(f"Unknown method={method}, expected ridge|lasso|elasticnet")


def Linear_regression(
    *,
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    method: str = "ridge",
    alpha: float = 1.0,
    l1_ratio: float = 0.5,
    fit_intercept: bool = True,
    max_iter: int = 20000,
    tol: float = 1e-4,
    random_state: int = 0,
):
    if not np.isfinite(X_tr).all() or not np.isfinite(y_tr).all():
        raise ValueError("[Linear_regression] train contains NaN/Inf.")

    model = _make_linear_model(
        method=method,
        alpha=alpha,
        l1_ratio=l1_ratio,
        fit_intercept=fit_intercept,
        max_iter=max_iter,
        tol=tol,
        random_state=random_state,
    )
    model.fit(X_tr, y_tr)
    return model