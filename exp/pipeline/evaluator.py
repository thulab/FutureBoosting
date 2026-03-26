from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go


STEP_DEFAULT = pd.Timedelta(minutes=15)


def evaluate(
    *,
    y_point_pred: np.ndarray,
    y_point_true: np.ndarray,
    N: int,
    L: int,
    split: str,
    tag: str,
    save_dir: str | Path,
    best_iteration: Optional[int] = None,
    time_points: Optional[Sequence[str]] = None,
    summary_csv: Optional[str | Path] = None,
    hlines=(200, 600),
    model: Any = None,
    feature_names: Optional[List[str]] = None,
    topk: int = 50,
    compare_series: Optional[Dict[str, np.ndarray]] = None,
    is_std: bool = False,
    std_step: Optional[str] = None,
    regression_model: str = "lgbm",
    method: Optional[str] = None,
    summary_meta: Optional[Dict[str, Any]] = None,
    plot_train: bool = False,
    train_point_pred: Optional[np.ndarray] = None,
    train_point_true: Optional[np.ndarray] = None,
    train_time_points: Optional[Sequence[str]] = None,
    train_N: Optional[int] = None,
    plot_target_mean: Optional[float] = None,
    plot_target_std: Optional[float] = None,
    compare_series_is_raw: bool = True,
) -> Dict[str, float]:
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    regression_model = str(regression_model or "lgbm").strip().lower()
    method_norm = str(method).strip().lower() if method else None
    plot_tag = tag if method_norm is None else f"{tag}_{method_norm}"

    y_point_pred = np.asarray(y_point_pred, dtype=float).reshape(-1)
    y_point_true = np.asarray(y_point_true, dtype=float).reshape(-1)
    metrics = _compute_metrics(
        y_true=y_point_true,
        y_pred=y_point_pred,
        N=N,
        L=L,
        is_std=is_std,
    )

    payload: Dict[str, Any] = {
        "tag": tag,
        "split": split,
        "regression_model": regression_model,
        **metrics,
    }
    if method_norm is not None:
        payload["method"] = method_norm
    if best_iteration is not None:
        payload["best_iteration"] = int(best_iteration)
    if summary_meta:
        payload.update(summary_meta)

    _save_metrics_json(save_dir, split, payload)
    if summary_csv is not None:
        _update_summary_csv(Path(summary_csv), payload)

    if time_points is not None:
        plot_y_true, plot_y_pred, plot_compare_series = _prepare_plot_series(
            y_true=y_point_true,
            y_pred=y_point_pred,
            compare_series=compare_series,
            plot_target_mean=plot_target_mean,
            plot_target_std=plot_target_std,
            compare_series_is_raw=compare_series_is_raw,
        )
        plot_metrics = _compute_metrics(
            y_true=plot_y_true,
            y_pred=plot_y_pred,
            N=N,
            L=L,
            is_std=is_std,
        )
        _plot_point_series(
            save_dir=save_dir,
            split=split,
            tag=plot_tag,
            y_true=plot_y_true,
            y_pred=plot_y_pred,
            time_points=time_points,
            out=plot_metrics,
            hlines=hlines,
            compare_series=plot_compare_series,
            step=_resolve_plot_step(is_std=is_std, std_step=std_step),
            title_keys=_resolve_title_keys(is_std=is_std),
            use_time_axis=(not is_std),
            N=N,
            L=L,
        )

    if plot_train and train_point_pred is not None and train_point_true is not None and train_time_points is not None:
        plot_train_true, plot_train_pred, _ = _prepare_plot_series(
            y_true=np.asarray(train_point_true, dtype=float).reshape(-1),
            y_pred=np.asarray(train_point_pred, dtype=float).reshape(-1),
            compare_series=None,
            plot_target_mean=plot_target_mean,
            plot_target_std=plot_target_std,
            compare_series_is_raw=compare_series_is_raw,
        )
        train_metrics = _compute_metrics(
            y_true=plot_train_true,
            y_pred=plot_train_pred,
            N=train_N if train_N is not None else N,
            L=L,
            is_std=is_std,
        )
        try:
            _plot_point_series(
                save_dir=save_dir,
                split="train",
                tag=plot_tag,
                y_true=plot_train_true,
                y_pred=plot_train_pred,
                time_points=train_time_points,
                out=train_metrics,
                hlines=hlines,
                compare_series=None,
                step=_resolve_plot_step(is_std=is_std, std_step=std_step),
                title_keys=_resolve_title_keys(is_std=is_std),
                use_time_axis=(not is_std),
                N=train_N if train_N is not None else N,
                L=L,
            )
        except Exception as e:
            print(f"[evaluate][plot][warn] train: skip due to {type(e).__name__}: {e}")

    if model is not None and feature_names:
        _save_model_artifacts(
            save_dir=save_dir,
            tag=plot_tag,
            model=model,
            feature_names=feature_names,
            topk=topk,
        )

    return metrics


def _prepare_plot_series(
    *,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    compare_series: Optional[Dict[str, np.ndarray]],
    plot_target_mean: Optional[float],
    plot_target_std: Optional[float],
    compare_series_is_raw: bool,
) -> tuple[np.ndarray, np.ndarray, Optional[Dict[str, np.ndarray]]]:
    y_true_plot = np.asarray(y_true, dtype=float).reshape(-1).copy()
    y_pred_plot = np.asarray(y_pred, dtype=float).reshape(-1).copy()

    if plot_target_mean is None or plot_target_std is None:
        if compare_series is None:
            return y_true_plot, y_pred_plot, None
        return y_true_plot, y_pred_plot, {
            k: np.asarray(v, dtype=float).reshape(-1).copy() for k, v in compare_series.items()
        }

    y_true_plot = y_true_plot * float(plot_target_std) + float(plot_target_mean)
    y_pred_plot = y_pred_plot * float(plot_target_std) + float(plot_target_mean)

    if compare_series is None:
        return y_true_plot, y_pred_plot, None

    out_compare: Dict[str, np.ndarray] = {}
    for model_name, arr in compare_series.items():
        arr_plot = np.asarray(arr, dtype=float).reshape(-1).copy()
        if not compare_series_is_raw:
            arr_plot = arr_plot * float(plot_target_std) + float(plot_target_mean)
        out_compare[model_name] = arr_plot
    return y_true_plot, y_pred_plot, out_compare


def _compute_metrics(
    *,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    N: int,
    L: int,
    is_std: bool,
) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    err = y_pred - y_true

    out = {
        "MSE": float(np.mean(err * err)) if len(err) else float("nan"),
        "MAE": float(np.mean(np.abs(err))) if len(err) else float("nan"),
        "R2": _safe_r2(y_true, y_pred),
    }

    if (not is_std) and N > 0 and L > 0 and N * L == len(err):
        err_2d = err.reshape(N, L)
        out["iMSE"] = float((err_2d * err_2d).mean())
        out["iMAE"] = float(np.abs(err_2d).mean())

    return out


def _safe_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    from sklearn.metrics import r2_score

    try:
        return float(r2_score(y_true, y_pred))
    except Exception:
        return float("nan")


def _resolve_title_keys(*, is_std: bool) -> tuple[str, str]:
    return ("MSE", "MAE") if is_std else ("iMSE", "iMAE")


def _resolve_plot_step(*, is_std: bool, std_step: Optional[str]) -> Optional[pd.Timedelta]:
    if is_std:
        return None
    if std_step:
        return _freq_to_timedelta(std_step) or STEP_DEFAULT
    return STEP_DEFAULT


def _save_metrics_json(save_dir: Path, split: str, payload: Dict[str, Any]) -> None:
    with open(save_dir / f"metrics_{split}.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _update_summary_csv(summary_csv: Path, payload: Dict[str, Any]) -> None:
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    new_row = pd.DataFrame([payload])

    if summary_csv.exists():
        old = pd.read_csv(summary_csv)
        for col in new_row.columns:
            if col not in old.columns:
                old[col] = np.nan
        for col in old.columns:
            if col not in new_row.columns:
                new_row[col] = np.nan
        new_row = new_row[old.columns]

        key_cols = ["tag", "split"]
        for col in ("regression_model", "method"):
            if col in new_row.columns:
                key_cols.append(col)

        key = np.ones(len(old), dtype=bool)
        for col in key_cols:
            key &= old[col].astype(str) == str(new_row.loc[0, col])

        if key.any():
            idx = int(np.flatnonzero(key)[0])
            for col in new_row.columns:
                old.loc[idx, col] = new_row.loc[0, col]
            merged = old
        else:
            merged = pd.concat([old, new_row], axis=0, ignore_index=True)
    else:
        merged = new_row

    sort_cols = [c for c in ["tag", "split", "regression_model", "method"] if c in merged.columns]
    if sort_cols:
        merged = merged.sort_values(sort_cols).reset_index(drop=True)
    merged.to_csv(summary_csv, index=False, encoding="utf-8")


def _freq_to_timedelta(freq: Optional[str]) -> Optional[pd.Timedelta]:
    if not freq:
        return None
    try:
        return pd.to_timedelta(pd.tseries.frequencies.to_offset(freq))
    except Exception:
        return None


def _plot_point_series(
    *,
    save_dir: Path,
    split: str,
    tag: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    time_points: Sequence[str],
    out: Dict[str, float],
    hlines=(200, 600),
    compare_series: Optional[Dict[str, np.ndarray]] = None,
    step: Optional[pd.Timedelta] = None,
    title_keys: tuple[str, str] = ("MSE", "MAE"),
    use_time_axis: bool = True,
    N: Optional[int] = None,
    L: Optional[int] = None,
) -> None:
    if len(time_points) != len(y_true):
        raise ValueError(f"time_points mismatch: {len(time_points)} vs {len(y_true)}")

    y_true_plot = np.asarray(y_true, dtype=float).reshape(-1).copy()
    y_pred_plot = np.asarray(y_pred, dtype=float).reshape(-1).copy()

    if use_time_axis:
        x = pd.to_datetime(pd.Series(time_points), errors="raise")
        if step is not None:
            dt = x.diff()
            is_cont = (dt <= step * 1.01) | dt.isna()
            y_true_plot[~is_cont] = np.nan
            y_pred_plot[~is_cont] = np.nan
        else:
            is_cont = np.ones(len(y_true_plot), dtype=bool)
    else:
        x = np.arange(len(y_true_plot), dtype=int)
        is_cont = np.ones(len(y_true_plot), dtype=bool)
        if (N is not None) and (L is not None) and (N > 0) and (L > 0) and (N * L == len(y_true_plot)):
            x = np.arange(len(y_true_plot), dtype=int)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y_true_plot, mode="lines", name="y_true"))
    fig.add_trace(go.Scatter(x=x, y=y_pred_plot, mode="lines", name="FutureBoosting"))

    if compare_series is not None:
        for model_name, arr in compare_series.items():
            arr = np.asarray(arr, dtype=float).reshape(-1)
            if len(arr) != len(y_true_plot):
                print(f"[evaluate][plot][warn] compare_series[{model_name}] length mismatch: {len(arr)} vs {len(y_true_plot)}")
                continue
            arr_plot = arr.copy()
            arr_plot[~is_cont] = np.nan
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=arr_plot,
                    mode="lines",
                    name=model_name,
                    line=dict(width=1.4),
                    opacity=0.90,
                )
            )

    for hline in hlines:
        fig.add_hline(y=float(hline), line_dash="dash", line_width=1, opacity=0.6)

    k1, k2 = title_keys
    fig.update_layout(
        title=f"[{tag}][{split}] y_true vs y_pred | {k1}={out.get(k1, float('nan')):.6f} {k2}={out.get(k2, float('nan')):.6f}",
        title_x=0.5,
        width=1800,
        height=560,
        hovermode="x unified",
        legend=dict(orientation="h", xanchor="center", x=0.5, y=1.08),
        margin=dict(l=60, r=40, t=100, b=60),
    )
    fig.update_xaxes(showgrid=True)
    fig.update_yaxes(showgrid=True)

    plot_dir = Path(save_dir).parent / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(plot_dir / f"plot_{split}_{tag}.html"), include_plotlyjs="cdn")


def _save_model_artifacts(
    *,
    save_dir: Path,
    tag: str,
    model,
    feature_names: List[str],
    topk: int,
) -> None:
    if hasattr(model, "feature_importance"):
        _save_lgbm_importance(
            save_dir=save_dir,
            model=model,
            feature_names=feature_names,
            topk=topk,
        )
        return

    if hasattr(model, "coef_"):
        coef = np.asarray(getattr(model, "coef_"), dtype=float).reshape(-1)
        _save_linear_importance(
            save_dir=save_dir,
            tag=tag,
            feature_names=feature_names,
            coef=coef,
            topk=topk,
        )


def _save_linear_importance(
    *,
    save_dir: Path,
    tag: str,
    feature_names: List[str],
    coef: np.ndarray,
    topk: int,
) -> None:
    import matplotlib.pyplot as plt

    coef = np.asarray(coef, dtype=float).reshape(-1)
    abs_coef = np.abs(coef)
    order = np.argsort(abs_coef)[::-1]

    df = pd.DataFrame(
        {
            "feature": [feature_names[i] for i in order],
            "coef": coef[order],
            "abs_coef": abs_coef[order],
        }
    )

    plot_dir = Path(save_dir).parent / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(plot_dir / f"coef_{tag}.csv", index=False, encoding="utf-8")

    k = min(int(topk), len(coef))
    top_idx = order[:k]
    names_k = [feature_names[i] for i in top_idx]
    coef_k = coef[top_idx]

    fig, ax = plt.subplots(figsize=(10, max(4, 0.28 * k)))
    ax.barh(range(k), coef_k[::-1])
    ax.set_yticks(range(k))
    ax.set_yticklabels(names_k[::-1])
    ax.set_xlabel("coef")
    ax.set_title(f"[{tag}] coef top-{k} by |coef|")
    fig.tight_layout()
    fig.savefig(plot_dir / f"coef_top{k}_{tag}.png", dpi=200)
    plt.close(fig)


def _save_lgbm_importance(
    *,
    save_dir: Path,
    model,
    feature_names: List[str],
    topk: int,
) -> None:
    import matplotlib.pyplot as plt

    gain = model.feature_importance(importance_type="gain")
    split = model.feature_importance(importance_type="split")

    df = pd.DataFrame({"feature": feature_names, "gain": gain, "split": split}).sort_values("gain", ascending=False)
    df.to_csv(save_dir / "lgbm_feature_importance.csv", index=False)

    d = df.head(topk).copy()

    def _pie(values: np.ndarray, labels: List[str], out_path: Path, title: str) -> None:
        v = np.asarray(values, dtype=float)
        v = np.clip(v, 0.0, None)
        s = float(v.sum())
        if s <= 0:
            return
        v = v / s
        plt.figure(figsize=(9, 9))
        plt.pie(v, labels=labels, autopct="%1.1f%%", startangle=90)
        plt.title(title)
        plt.tight_layout()
        plt.savefig(out_path, dpi=200)
        plt.close()

    try:
        _pie(
            d["gain"].to_numpy(),
            d["feature"].tolist(),
            save_dir / f"lgbm_feature_importance_gain_top{topk}_pie.png",
            f"LGBM Feature Importance (gain) Top-{topk} Share",
        )
        _pie(
            d["split"].to_numpy(),
            d["feature"].tolist(),
            save_dir / f"lgbm_feature_importance_split_top{topk}_pie.png",
            f"LGBM Feature Importance (split) Top-{topk} Share",
        )
    except Exception:
        pass
