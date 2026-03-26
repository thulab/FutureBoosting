from __future__ import annotations

import json
from typing import Any, Dict, List, Tuple

import chinese_calendar as calendar
import numpy as np
import pandas as pd


POINTS_PER_DAY = 96
NOON_OFFSET = 48


def read_table(path: str) -> pd.DataFrame:
    if path.endswith(".csv"):
        return pd.read_csv(path)
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    raise ValueError(path)


def _load_single_json(path: str) -> dict:
    obj = json.load(open(path, "r", encoding="utf-8"))
    return obj[0] if isinstance(obj, list) else obj


def _to_shanghai(ts):
    t = pd.to_datetime(ts, errors="raise")
    return t.tz_localize("Asia/Shanghai") if t.tzinfo is None else t.tz_convert("Asia/Shanghai")


def _to_shanghai_series(s: pd.Series) -> pd.Series:
    t = pd.to_datetime(s, errors="raise")
    return t.dt.tz_localize("Asia/Shanghai") if t.dt.tz is None else t.dt.tz_convert("Asia/Shanghai")


def slice_df(df: pd.DataFrame, time_col: str, start, end) -> pd.DataFrame:
    t = _to_shanghai_series(df[time_col])
    s = _to_shanghai(start)
    e = _to_shanghai(end)
    return df.loc[(t >= s) & (t <= e)].copy()


def _attach_tsfm_features(
    df: pd.DataFrame,
    *,
    time_col: str,
    y_col: str,
    pred_table: pd.DataFrame,
    tsfm_models: str,
    tsfm_vars: List[str],
) -> tuple[pd.DataFrame, List[str]]:
    if pred_table is None or len(pred_table) == 0:
        return df, []

    pt = pred_table.copy()
    pt[time_col] = _to_shanghai_series(pt["time"])
    if time_col != "time":
        pt = pt.drop(columns=["time"])
    pt = pt.sort_values(time_col)

    dff = df.copy()
    dff[time_col] = _to_shanghai_series(dff[time_col])

    models = [m.strip().lower() for m in tsfm_models.split(",") if m.strip()]
    want_vars = list(tsfm_vars)
    if y_col not in want_vars:
        want_vars.append(y_col)

    pred_cols: List[str] = []
    for model_name in models:
        for var_name in want_vars:
            col = f"pred_{model_name}_{var_name}"
            if col in pt.columns:
                pred_cols.append(col)

    if not pred_cols:
        return dff, []

    dff = dff.merge(pt[[time_col] + pred_cols], on=time_col, how="left")
    return dff, pred_cols


def make_xy(
    df: pd.DataFrame,
    *,
    time_col: str,
    cov_cols: List[str],
    y_col: str,
    target_hour: int,
) -> Tuple[np.ndarray, np.ndarray, int, List[str]]:
    if len(df) == 0:
        return np.zeros((0, len(cov_cols)), np.float32), np.zeros((0,), np.float32), 0, []

    dff = df.copy()
    dff["_t"] = pd.to_datetime(dff[time_col])
    dff["_d"] = dff["_t"].dt.date

    X_days, y_days, times = [], [], []

    for _, g in dff.groupby("_d"):
        g = g.sort_values("_t")
        if len(g) != POINTS_PER_DAY:
            continue
        if g["_t"].iloc[0].hour != target_hour:
            continue
        if not calendar.is_workday(g["_t"].iloc[NOON_OFFSET].date()):
            continue

        X_days.append(g[cov_cols].to_numpy(np.float32))
        y_days.append(g[y_col].to_numpy(np.float32))
        times.extend(g["_t"].astype(str).tolist())

    if not X_days:
        return np.zeros((0, len(cov_cols)), np.float32), np.zeros((0,), np.float32), 0, []

    return (
        np.concatenate(X_days, axis=0),
        np.concatenate(y_days, axis=0),
        len(X_days),
        times,
    )


def _build_epf_benchmark_datasets(
    args,
    split_cfg: Dict[str, Any] | None = None,
    cols_cfg: List[str] | None = None,
    *,
    scale: bool | None = None,
):
    from data_provider.data_loader import CovariateDatasetBenchmark

    if split_cfg is None:
        split_cfg = _load_single_json(args.data_split_path)["data_split"]
    if cols_cfg is None:
        cols_cfg = list(_load_single_json(args.regress_cols_path)["target_columns"])

    if scale is None:
        scale_flag = bool(getattr(args, "scale", False))
    else:
        scale_flag = bool(scale)

    size = [
        int(args.seq_len),
        int(getattr(args, "input_token_len", 16)),
        int(getattr(args, "output_token_len", 720)),
        int(args.pred_len),
    ]

    common_kwargs = dict(
        size=size,
        scale=scale_flag,
        data_path=str(args.data_path),
        target_columns=list(cols_cfg),
        data_split={
            "train": float(split_cfg["train"]),
            "valid": float(split_cfg["valid"]),
            "test": float(split_cfg["test"]),
        },
        clean=bool(getattr(args, "clean", False)),
        shift=int(getattr(args, "shift", 0)),
    )

    ds_tr = CovariateDatasetBenchmark(flag="train", **common_kwargs)
    ds_va = CovariateDatasetBenchmark(flag="val", **common_kwargs)
    ds_te = CovariateDatasetBenchmark(flag="test", **common_kwargs)

    print(
        f"[epf_benchmark] scale={scale_flag}, "
        f"ds_tr={len(ds_tr)}, ds_va={len(ds_va)}, ds_te={len(ds_te)}"
    )
    return ds_tr, ds_va, ds_te, split_cfg, cols_cfg


def build_std_key_grid(ds) -> pd.DataFrame:
    cols = ["anchor_time", "time", "h"]

    if len(ds) == 0:
        return pd.DataFrame(columns=cols)
    if getattr(ds, "time_split", None) is None:
        raise ValueError("CovariateDatasetBenchmark.time_split is None (need date column).")

    pred_len = int(ds.pred_len)
    seq_len = int(ds.seq_len)
    t_split = pd.to_datetime(ds.time_split, errors="raise").reset_index(drop=True)
    rows: List[dict] = []

    for i in range(len(ds)):
        anchor_idx = i + seq_len
        anchor_time = pd.to_datetime(t_split.iloc[anchor_idx]).tz_localize(None)
        for h in range(pred_len):
            rows.append(
                {
                    "anchor_time": anchor_time,
                    "time": pd.to_datetime(t_split.iloc[anchor_idx + h]).tz_localize(None),
                    "h": np.int32(h + 1),
                }
            )

    return pd.DataFrame(rows, columns=cols)


def build_std_cov_target_table(ds, *, cov_cols: List[str]) -> pd.DataFrame:
    cols = ["anchor_time", "time", "h", "y"] + list(cov_cols)

    if len(ds) == 0:
        return pd.DataFrame(columns=cols)
    if getattr(ds, "time_split", None) is None:
        raise ValueError("CovariateDatasetBenchmark.time_split is None (need date column).")

    pred_len = int(ds.pred_len)
    need_cov_dim = len(cov_cols)
    key_grid = build_std_key_grid(ds)
    rows: List[dict] = []
    seq_len = int(ds.seq_len)

    for i in range(len(ds)):
        _, _, seq_y, seq_y_cov = ds[i]
        seq_y = np.asarray(seq_y, dtype=np.float32).reshape(-1)
        seq_y_cov = np.asarray(seq_y_cov, dtype=np.float32)

        if seq_y.shape[0] != pred_len:
            raise ValueError(
                f"[std table] seq_y len={seq_y.shape[0]} != pred_len={pred_len} for flag={getattr(ds, 'flag', 'unknown')}"
            )
        if seq_y_cov.ndim != 2 or seq_y_cov.shape[0] != pred_len:
            raise ValueError(
                f"[std table] seq_y_cov shape={seq_y_cov.shape} incompatible with pred_len={pred_len} "
                f"for flag={getattr(ds, 'flag', 'unknown')}"
            )
        if seq_y_cov.shape[1] != need_cov_dim:
            raise ValueError(
                f"[std table] seq_y_cov dim={seq_y_cov.shape[1]} != len(cov_cols)={need_cov_dim} "
                f"for flag={getattr(ds, 'flag', 'unknown')}"
            )

        anchor_idx = i + seq_len
        anchor_time = key_grid.iloc[i * pred_len]["anchor_time"]
        for h in range(pred_len):
            t = key_grid.iloc[i * pred_len + h]["time"]
            row = {
                "anchor_time": anchor_time,
                "time": t,
                "h": np.int32(h + 1),
                "y": float(seq_y[h]),
            }
            row.update({cov_cols[k]: float(seq_y_cov[h, k]) for k in range(need_cov_dim)})
            rows.append(row)

    return pd.DataFrame(rows, columns=cols)


def attach_tsfm_preds_std(
    base: pd.DataFrame,
    *,
    tsfm_pred_table: pd.DataFrame | None,
    tsfm_models: str,
    tsfm_vars: List[str],
) -> tuple[pd.DataFrame, List[str]]:
    if tsfm_pred_table is None or len(tsfm_pred_table) == 0:
        return base, []

    models = [m.strip().lower() for m in tsfm_models.split(",") if m.strip()]

    pt = tsfm_pred_table.copy()
    pt["anchor_time"] = pd.to_datetime(pt["anchor_time"]).dt.tz_localize(None)
    pt["time"] = pd.to_datetime(pt["time"]).dt.tz_localize(None)
    pt["h"] = pt["h"].astype(np.int32)

    pred_cols: List[str] = []
    for model_name in models:
        for var_name in tsfm_vars:
            col = f"pred_{model_name}_{var_name}"
            if col in pt.columns:
                pred_cols.append(col)

    if not pred_cols:
        return base, []

    merged = base.merge(
        pt[["anchor_time", "time", "h"] + pred_cols],
        on=["anchor_time", "time", "h"],
        how="left",
    )
    return merged, pred_cols


def table_to_xy_std(
    df: pd.DataFrame,
    *,
    cov_cols: List[str],
    pred_cols: List[str],
) -> tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    feature_cols = list(cov_cols) + list(pred_cols)

    if len(df) == 0:
        return (
            np.zeros((0, len(feature_cols)), np.float32),
            np.zeros((0,), np.float32),
            [],
            feature_cols,
        )

    X = df[feature_cols].to_numpy(np.float32)
    y = df["y"].to_numpy(np.float32)
    times = df["time"].astype(str).tolist()
    print("[std xy] X shape:", X.shape, "y shape:", y.shape)
    return X, y, times, feature_cols


def build_features(
    *,
    args,
    pred_table=None,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    int,
    np.ndarray,
    np.ndarray,
    int,
    np.ndarray,
    np.ndarray,
    int,
    Dict[str, Any],
]:
    split_cfg = _load_single_json(args.data_split_path)["data_split"]
    cols_cfg = list(_load_single_json(args.regress_cols_path)["target_columns"])
    tsfm_vars = list(_load_single_json(args.tsfm_cols_path)["target_columns"])

    time_col = split_cfg.get("time_col", "time")
    cov_cols, y_col = cols_cfg[:-1], cols_cfg[-1]
    if y_col not in tsfm_vars:
        tsfm_vars.append(y_col)

    if bool(getattr(args, "is_std", False)):
        ds_tr, ds_va, ds_te, _, _ = _build_epf_benchmark_datasets(
            args,
            split_cfg=split_cfg,
            cols_cfg=cols_cfg,
        )

        tr_base = build_std_cov_target_table(ds_tr, cov_cols=cov_cols)
        va_base = build_std_cov_target_table(ds_va, cov_cols=cov_cols)
        te_base = build_std_cov_target_table(ds_te, cov_cols=cov_cols)

        tr_df, pred_cols = attach_tsfm_preds_std(
            tr_base,
            tsfm_pred_table=pred_table,
            tsfm_models=args.tsfm_models,
            tsfm_vars=tsfm_vars,
        )
        va_df, _ = attach_tsfm_preds_std(
            va_base,
            tsfm_pred_table=pred_table,
            tsfm_models=args.tsfm_models,
            tsfm_vars=tsfm_vars,
        )
        te_df, _ = attach_tsfm_preds_std(
            te_base,
            tsfm_pred_table=pred_table,
            tsfm_models=args.tsfm_models,
            tsfm_vars=tsfm_vars,
        )

        X_tr, y_tr, tr_time, feature_names = table_to_xy_std(tr_df, cov_cols=cov_cols, pred_cols=pred_cols)
        X_va, y_va, va_time, _ = table_to_xy_std(va_df, cov_cols=cov_cols, pred_cols=pred_cols)
        X_te, y_te, te_time, _ = table_to_xy_std(te_df, cov_cols=cov_cols, pred_cols=pred_cols)

        N_tr = len(ds_tr)
        N_va = len(ds_va)
        N_te = len(ds_te)

        # if bool(getattr(args, "std_merge_valid_to_train", False)):
        X_tr = np.concatenate([X_tr, X_va], axis=0)
        y_tr = np.concatenate([y_tr, y_va], axis=0)
        N_tr += N_va
        tr_time = tr_time + va_time
        print(
            f"[feature_select][standard] merged train+valid for training: "
            f"N_tr={N_tr}, X_tr.shape={X_tr.shape}, y_tr.shape={y_tr.shape}"
        )

        print(f"[feature_select][standard] windows: train={N_tr}, valid={N_va}, test={N_te}")
        print(f"[feature_select][standard] #features: {len(feature_names)}")
        if pred_cols:
            print(f"[feature_select][standard] TSFM features: {len(pred_cols)}")

        meta = {
            "L": int(args.pred_len),
            "N_tr": N_tr,
            "N_va": N_va,
            "N_te": N_te,
            "feature_names": feature_names,
            "tr_time": tr_time,
            "va_time": va_time,
            "te_time": te_time,
            "target_mean": float(np.asarray(ds_tr.mean_target, dtype=float).reshape(-1)[0]) if getattr(ds_tr, "mean_target", None) is not None else None,
            "target_std": float(np.asarray(ds_tr.std_target, dtype=float).reshape(-1)[0]) if getattr(ds_tr, "std_target", None) is not None else None,
            "target_scaled": bool(getattr(args, "scale", False)),
        }
        return X_tr, y_tr, N_tr, X_va, y_va, N_va, X_te, y_te, N_te, meta

    df = read_table(args.data_path)
    df, tsfm_cols = _attach_tsfm_features(
        df,
        time_col=time_col,
        y_col=y_col,
        pred_table=pred_table,
        tsfm_models=args.tsfm_models,
        tsfm_vars=tsfm_vars,
    )

    feature_names = cov_cols + tsfm_cols
    dtr = slice_df(df, time_col, split_cfg["train_start"], split_cfg["train_end"])
    dva = slice_df(df, time_col, split_cfg["valid_start"], split_cfg["valid_end"])
    dte = slice_df(df, time_col, split_cfg["test_start"], split_cfg["test_end"])

    X_tr, y_tr, N_tr, tr_time = make_xy(
        dtr,
        time_col=time_col,
        cov_cols=feature_names,
        y_col=y_col,
        target_hour=args.target_hour,
    )
    X_va, y_va, N_va, va_time = make_xy(
        dva,
        time_col=time_col,
        cov_cols=feature_names,
        y_col=y_col,
        target_hour=args.target_hour,
    )
    X_te, y_te, N_te, te_time = make_xy(
        dte,
        time_col=time_col,
        cov_cols=feature_names,
        y_col=y_col,
        target_hour=args.target_hour,
    )

    print(f"[feature_select][shanxi] days: train={N_tr}, valid={N_va}, test={N_te}")
    print(f"[feature_select][shanxi] #features: {len(feature_names)}")
    if tsfm_cols:
        print(f"[feature_select][shanxi] TSFM features: {len(tsfm_cols)}")

    meta = {
        "L": POINTS_PER_DAY,
        "N_tr": N_tr,
        "N_va": N_va,
        "N_te": N_te,
        "feature_names": feature_names,
        "tr_time": tr_time,
        "va_time": va_time,
        "te_time": te_time,
    }
    return X_tr, y_tr, N_tr, X_va, y_va, N_va, X_te, y_te, N_te, meta
