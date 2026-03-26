from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import chinese_calendar as calendar
import numpy as np
import pandas as pd
import torch


FREQ = pd.Timedelta(minutes=15)
POINTS_PER_DAY = 96
NOON_OFFSET = 48


@dataclass(frozen=True)
class RolloutMeta:
    tag: str
    is_std: bool
    seq_len: int
    horizon: int
    batch_size: int
    num_samples: int
    device: str
    cache_path: str
    models: List[str]
    model_paths: Dict[str, str]
    split_cfg: dict
    time_col: str
    mv_cols: List[str]
    regress_cols: List[str]
    y_col: str
    key_cols: List[str]
    required_pred_cols: List[str]
    required_cols: List[str]
    use_future_covariates: bool


def run_tsfm_rollout(*, args) -> pd.DataFrame:
    if not getattr(args, "enable_tsfm", False):
        return pd.DataFrame()

    meta = _build_rollout_meta(args=args)
    if not meta.models:
        return pd.DataFrame()

    df = _read_table(args.data_path)
    df = _prepare_time_index(df, meta.time_col)

    base, anchors, anchor_offset = _build_rollout_base(args=args, meta=meta, df=df)
    cache_df = _load_cache(meta=meta)
    force_current_rerun = bool(getattr(args, "tsfm_force_current_rerun", False))

    if meta.is_std:
        model_patch_plan = _plan_patch_for_standard(
            meta=meta,
            cache_df=cache_df,
            anchors=anchors,
            force_current_rerun=force_current_rerun,
        )
        patch_base = base if any(model_patch_plan.values()) else base.iloc[0:0].copy()
        patch_anchor_offset = anchor_offset if any(model_patch_plan.values()) else {}
    else:
        model_patch_plan = _plan_patch_for_shanxi(
            meta=meta,
            cache_df=cache_df,
            anchors=anchors,
            force_current_rerun=force_current_rerun,
        )
        patch_anchors = _unique_concat(model_patch_plan.values())
        patch_base, patch_anchor_offset = _build_shanxi_patch_grid(
            anchors=patch_anchors,
            horizon=meta.horizon,
        )

    if cache_df is not None and not any(model_patch_plan.values()):
        return _finalize_rollout(args=args, meta=meta, full_table=cache_df, current_base=base)

    patch = _infer_missing_predictions(
        meta=meta,
        df=df,
        base=patch_base,
        anchor_offset=patch_anchor_offset,
        model_patch_plan=model_patch_plan,
    )
    full_table = _merge_cache_tables(meta=meta, cache_df=cache_df, patch=patch)
    return _finalize_rollout(args=args, meta=meta, full_table=full_table, current_base=base)


# =========================
# compare-series helper
# =========================

def get_tsfm_target_col(*, args) -> str:
    regress_cols = list(_load_single_json(args.regress_cols_path)["target_columns"])
    if not regress_cols:
        raise ValueError(f"[tsfm_infer] empty regress_cols_path: {args.regress_cols_path}")
    return regress_cols[-1]


def load_compare_series_from_cache(
    *,
    args,
    time_points: Sequence[str],
) -> Dict[str, np.ndarray]:
    cache_path = _get_cache_path(args)
    if not os.path.exists(cache_path):
        return {}

    target_col = get_tsfm_target_col(args=args)
    pattern = re.compile(rf"^pred_(.+)_{re.escape(target_col)}$")
    pt = pd.read_parquet(cache_path)

    matched = []
    for col in pt.columns:
        if isinstance(col, str):
            m = pattern.match(col)
            if m is not None:
                matched.append((m.group(1), col))
    if not matched:
        return {}

    matched = sorted(matched, key=lambda x: x[0])

    if bool(getattr(args, "is_std", False)):
        from .feature_select import _build_epf_benchmark_datasets, build_std_key_grid

        split_cfg = _load_single_json(args.data_split_path)["data_split"]
        cols_cfg = list(_load_single_json(args.regress_cols_path)["target_columns"])
        _, _, ds_te, _, _ = _build_epf_benchmark_datasets(
            args,
            split_cfg=split_cfg,
            cols_cfg=cols_cfg,
        )
        base = build_std_key_grid(ds_te)
        base = _normalize_pred_table_keys(base, is_std=True)

        pt = _normalize_pred_table_keys(pt, is_std=True)
        pt = pt.set_index(["anchor_time", "time", "h"]).sort_index()
        base_index = pd.MultiIndex.from_frame(base[["anchor_time", "time", "h"]])

        out: Dict[str, np.ndarray] = {}
        for model_name, col in matched:
            out[model_name] = pd.to_numeric(pt[col], errors="coerce").reindex(base_index).to_numpy(dtype=np.float32)
        return out

    pt = _normalize_pred_table_keys(pt, is_std=False)
    pt = pt.loc[pt["time"].notna()]
    pt = pt.sort_values("time").drop_duplicates(subset=["time"], keep="last").set_index("time")

    t_index = pd.DatetimeIndex(_to_shanghai_series(pd.Series(time_points)))
    out: Dict[str, np.ndarray] = {}
    for model_name, col in matched:
        out[model_name] = pd.to_numeric(pt[col], errors="coerce").reindex(t_index).to_numpy(dtype=np.float32)
    return out


# =========================
# rollout main helpers
# =========================

def _build_rollout_meta(*, args) -> RolloutMeta:
    models = _parse_models(getattr(args, "tsfm_models", ""))
    model_paths = _parse_model_paths(args.tsfm_model_paths, models) if models else {}

    split_cfg = _load_single_json(args.data_split_path)["data_split"]
    time_col = split_cfg.get("time_col", "time")

    mv_cols = list(_load_single_json(args.tsfm_cols_path)["target_columns"])
    regress_cols = list(_load_single_json(args.regress_cols_path)["target_columns"])
    y_col = regress_cols[-1]

    if y_col not in mv_cols:
        mv_cols.append(y_col)
    if mv_cols[-1] != y_col:
        mv_cols = [c for c in mv_cols if c != y_col] + [y_col]

    is_std = bool(getattr(args, "is_std", False))
    key_cols = ["anchor_time", "time", "h"] if is_std else ["time"]
    required_pred_cols = [f"pred_{m}_{v}" for m in models for v in mv_cols]
    required_cols = key_cols + required_pred_cols

    return RolloutMeta(
        tag=str(getattr(args, "tag", "pipeline")),
        is_std=is_std,
        seq_len=int(args.seq_len),
        horizon=int(args.pred_len),
        batch_size=int(getattr(args, "batch_size", 256)),
        num_samples=int(getattr(args, "tsfm_num_samples", 20)),
        device=str(getattr(args, "device", "cuda")),
        cache_path=_get_cache_path(args),
        models=models,
        model_paths=model_paths,
        split_cfg=split_cfg,
        time_col=time_col,
        mv_cols=mv_cols,
        regress_cols=regress_cols,
        y_col=y_col,
        key_cols=key_cols,
        required_pred_cols=required_pred_cols,
        required_cols=required_cols,
        use_future_covariates=bool(getattr(args, "tsfm_use_future_covariates", True)),
    )


def _load_cache(*, meta: RolloutMeta) -> Optional[pd.DataFrame]:
    if not os.path.exists(meta.cache_path):
        return None
    return _normalize_pred_table_keys(pd.read_parquet(meta.cache_path), is_std=meta.is_std)


def _plan_patch_for_standard(
    *,
    meta: RolloutMeta,
    cache_df: Optional[pd.DataFrame],
    anchors: List[pd.Timestamp],
    force_current_rerun: bool = False,
) -> Dict[str, List[pd.Timestamp]]:
    if force_current_rerun:
        print(
            f"[tsfm_rollout][standard] force_current_rerun=1 -> "
            f"recomputing current anchors for models={meta.models} (count={len(anchors)})"
        )
        return {m: list(anchors) for m in meta.models}

    if cache_df is None:
        return {m: list(anchors) for m in meta.models}

    plan: Dict[str, List[pd.Timestamp]] = {}
    for model_name in meta.models:
        model_cols = [f"pred_{model_name}_{v}" for v in meta.mv_cols]
        missing_cols = [c for c in model_cols if c not in cache_df.columns]
        if missing_cols:
            print(f"[tsfm_rollout][standard][{model_name}] cache missing columns ({len(missing_cols)}):")
            for col in missing_cols:
                print(f"  - {col}")
            plan[model_name] = list(anchors)
        else:
            plan[model_name] = []
    return plan


def _plan_patch_for_shanxi(
    *,
    meta: RolloutMeta,
    cache_df: Optional[pd.DataFrame],
    anchors: List[pd.Timestamp],
    force_current_rerun: bool = False,
) -> Dict[str, List[pd.Timestamp]]:
    if force_current_rerun:
        print(
            f"[tsfm_rollout][shanxi] force_current_rerun=1 -> "
            f"recomputing current anchors for models={meta.models} (count={len(anchors)})"
        )
        return {m: list(anchors) for m in meta.models}

    if cache_df is None:
        return {m: list(anchors) for m in meta.models}

    cache_use = cache_df.loc[:, [c for c in cache_df.columns if c == "time" or str(c).startswith("pred_")]].copy()
    cache_use = cache_use.loc[cache_use["time"].notna()]
    cache_use = cache_use.sort_values("time").drop_duplicates(subset=["time"], keep="last").set_index("time")

    plan: Dict[str, List[pd.Timestamp]] = {}
    for model_name in meta.models:
        model_cols = [f"pred_{model_name}_{v}" for v in meta.mv_cols]
        missing_cols = [c for c in model_cols if c not in cache_use.columns]
        if missing_cols:
            print(f"[tsfm_rollout][shanxi][{model_name}] cache missing columns ({len(missing_cols)}):")
            for col in missing_cols:
                print(f"  - {col}")
            plan[model_name] = list(anchors)
            continue
        missing_anchors, examples = _find_missing_shanxi_anchors(
            cache_df=cache_use,
            anchors=anchors,
            cols=model_cols,
            horizon=meta.horizon,
        )
        plan[model_name] = missing_anchors
        if missing_anchors:
            print(f"[tsfm_rollout][shanxi][{model_name}] cache missing anchors ({len(missing_anchors)}): {_preview_timestamps(missing_anchors)}")
        if examples:
            print(f"[tsfm_rollout][shanxi][{model_name}] missing time/column preview:")
            for ts, cols_missing in examples:
                print(f"  - time={ts} | cols={cols_missing}")
    return plan


def _build_rollout_base(
    *,
    args,
    meta: RolloutMeta,
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, List[pd.Timestamp], Dict[pd.Timestamp, int]]:
    if meta.is_std:
        from .feature_select import _build_epf_benchmark_datasets

        ds_tr, ds_va, ds_te, _, _ = _build_epf_benchmark_datasets(
            args,
            split_cfg=meta.split_cfg,
            cols_cfg=meta.regress_cols,
        )
        return _build_std_rollout_grid(
            datasets=[ds_tr, ds_va, ds_te],
            seq_len=meta.seq_len,
            horizon=meta.horizon,
        )

    target_hour = int(args.target_hour)
    start = min(_to_shanghai(pd.to_datetime(meta.split_cfg[k])) for k in ["train_start", "valid_start", "test_start"])
    end = max(_to_shanghai(pd.to_datetime(meta.split_cfg[k])) for k in ["train_end", "valid_end", "test_end"])
    anchors = _build_daily_anchors(start, end, target_hour)
    anchors = _filter_valid_anchors(anchors, df.index, meta.seq_len, meta.horizon)
    times, anchor_offset = _future_time_grid(anchors, meta.horizon)
    base = pd.DataFrame({"time": times})
    return base, anchors, anchor_offset


def _infer_missing_predictions(
    *,
    meta: RolloutMeta,
    df: pd.DataFrame,
    base: pd.DataFrame,
    anchor_offset: Dict[pd.Timestamp, int],
    model_patch_plan: Dict[str, List[pd.Timestamp]],
) -> pd.DataFrame:
    patch = base.copy()
    if len(base) == 0:
        return _normalize_pred_table_keys(patch, is_std=meta.is_std)

    new_cols: Dict[str, np.ndarray] = {}
    for model_name in meta.models:
        model_anchors = model_patch_plan.get(model_name, [])
        if not model_anchors:
            continue

        model = _load_model(model_name, meta.model_paths[model_name], meta.device)
        pred_by_var = _predict_multivar(
            df=df,
            mv_cols=meta.mv_cols,
            anchors=model_anchors,
            anchor_offset=anchor_offset,
            model=model,
            seq_len=meta.seq_len,
            horizon=meta.horizon,
            batch_size=meta.batch_size,
            num_samples=meta.num_samples,
            device=meta.device,
            use_future_covariates=meta.use_future_covariates,
            output_size=len(base),
        )
        for var_name in meta.mv_cols:
            new_cols[f"pred_{model_name}_{var_name}"] = pred_by_var[var_name]

    if new_cols:
        patch = pd.concat([patch.reset_index(drop=True), pd.DataFrame(new_cols)], axis=1)
    return _normalize_pred_table_keys(patch, is_std=meta.is_std)


def _merge_cache_tables(
    *,
    meta: RolloutMeta,
    cache_df: Optional[pd.DataFrame],
    patch: pd.DataFrame,
) -> pd.DataFrame:
    patch = _normalize_pred_table_keys(patch, is_std=meta.is_std)
    if cache_df is None or len(cache_df) == 0:
        full_table = patch.copy()
    else:
        cache_idx = cache_df.set_index(meta.key_cols)
        patch_idx = patch.set_index(meta.key_cols)
        full_table = patch_idx.combine_first(cache_idx).reset_index()

    full_table = full_table.sort_values(meta.key_cols).reset_index(drop=True)
    os.makedirs(os.path.dirname(meta.cache_path) or ".", exist_ok=True)
    full_table.to_parquet(meta.cache_path, index=False)
    return full_table


def _finalize_rollout(
    *,
    args,
    meta: RolloutMeta,
    full_table: pd.DataFrame,
    current_base: pd.DataFrame,
) -> pd.DataFrame:
    current = _filter_to_current_base(
        full_table=full_table,
        base=current_base,
        key_cols=meta.key_cols,
        is_std=meta.is_std,
    )
    ret_table = _keep_required_columns(current, meta.required_cols)
    _metrics_zero_shot(args=args, pred_table=ret_table, models=meta.models)
    return ret_table


def _filter_to_current_base(
    *,
    full_table: pd.DataFrame,
    base: pd.DataFrame,
    key_cols: Sequence[str],
    is_std: bool,
) -> pd.DataFrame:
    base_keys = _normalize_pred_table_keys(base.loc[:, list(key_cols)].copy(), is_std=is_std)
    src = _normalize_pred_table_keys(full_table.copy(), is_std=is_std)
    src = src.drop_duplicates(subset=list(key_cols), keep="last")
    return base_keys.merge(src, on=list(key_cols), how="left")


# =========================
# small utilities
# =========================

def _keep_required_columns(df: pd.DataFrame, required_cols: Sequence[str]) -> pd.DataFrame:
    cols = [c for c in required_cols if c in df.columns]
    return df.loc[:, cols].copy()


def _unique_concat(items: Sequence[Sequence[pd.Timestamp]]) -> List[pd.Timestamp]:
    out: List[pd.Timestamp] = []
    seen = set()
    for xs in items:
        for x in xs:
            if x not in seen:
                seen.add(x)
                out.append(x)
    return out


def _preview_timestamps(items: Sequence[pd.Timestamp], max_items: int = 10) -> List[str]:
    vals = [str(x) for x in items[: int(max_items)]]
    if len(items) > max_items:
        vals.append("...")
    return vals


def _find_missing_shanxi_anchors(
    *,
    cache_df: pd.DataFrame,
    anchors: List[pd.Timestamp],
    cols: Sequence[str],
    horizon: int,
) -> Tuple[List[pd.Timestamp], List[Tuple[str, List[str]]]]:
    missing: List[pd.Timestamp] = []
    examples: List[Tuple[str, List[str]]] = []
    for anchor in anchors:
        fut_times = pd.DatetimeIndex([anchor + h * FREQ for h in range(horizon)])
        block = cache_df.reindex(fut_times)
        bad = block[list(cols)].isna()
        if bad.any().any():
            missing.append(anchor)
            if len(examples) < 10:
                for ts, row in bad.iterrows():
                    miss_cols = [col for col, flag in row.items() if bool(flag)]
                    if miss_cols:
                        examples.append((str(ts), miss_cols))
                        if len(examples) >= 10:
                            break
    return missing, examples


def _build_shanxi_patch_grid(
    *,
    anchors: List[pd.Timestamp],
    horizon: int,
) -> Tuple[pd.DataFrame, Dict[pd.Timestamp, int]]:
    times, anchor_offset = _future_time_grid(anchors, horizon)
    return pd.DataFrame({"time": times}), anchor_offset


def _parse_models(x: str) -> List[str]:
    vals = [s.strip().lower() for s in x.split(",") if s.strip()]
    return list(dict.fromkeys(vals))


def _parse_model_paths(x: str, models: List[str]) -> Dict[str, str]:
    parsed = json.loads(x)
    return {m: parsed[m] for m in models}


def _read_table(path: str) -> pd.DataFrame:
    if path.endswith(".csv"):
        return pd.read_csv(path)
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    raise ValueError(path)


def _load_single_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    return obj[0] if isinstance(obj, list) else obj


def _get_cache_path(args) -> str:
    cache_root = str(getattr(args, "tsfm_cache_path", "") or getattr(args, "save_dir", ".") or ".")
    os.makedirs(cache_root, exist_ok=True)

    if bool(getattr(args, "is_std", False)):
        return os.path.join(cache_root, f"{args.tag}_tsfm_pred_table.parquet")

    tsfm_name = os.path.splitext(os.path.basename(args.tsfm_cols_path))[0]
    regress_name = os.path.splitext(os.path.basename(args.regress_cols_path))[0]
    futcov = int(bool(getattr(args, "tsfm_use_future_covariates", True)))
    return os.path.join(
        cache_root,
        f"shared_tsfm_pred_table.parquet",
    )


def _to_shanghai(ts: pd.Timestamp) -> pd.Timestamp:
    ts = pd.Timestamp(ts)
    return ts.tz_localize("Asia/Shanghai") if ts.tzinfo is None else ts.tz_convert("Asia/Shanghai")


def _to_shanghai_series(s: pd.Series) -> pd.Series:
    t = pd.to_datetime(s, errors="raise")
    return t.dt.tz_localize("Asia/Shanghai") if t.dt.tz is None else t.dt.tz_convert("Asia/Shanghai")


def _prepare_time_index(df: pd.DataFrame, time_col: str) -> pd.DataFrame:
    t = pd.to_datetime(df[time_col], errors="raise").map(_to_shanghai)
    return df.assign(**{time_col: t}).sort_values(time_col).set_index(time_col)


def _normalize_pred_table_keys(df: pd.DataFrame, *, is_std: bool) -> pd.DataFrame:
    dff = df.copy()
    if is_std:
        if "anchor_time" in dff.columns:
            dff["anchor_time"] = _to_shanghai_series(dff["anchor_time"])
        if "time" in dff.columns:
            dff["time"] = _to_shanghai_series(dff["time"])
        if "h" in dff.columns:
            dff["h"] = pd.to_numeric(dff["h"], errors="coerce").astype("Int64")
            if dff["h"].notna().all():
                dff["h"] = dff["h"].astype(np.int32)
    else:
        if "time" in dff.columns:
            dff["time"] = _to_shanghai_series(dff["time"])
    return dff


def _build_daily_anchors(start: pd.Timestamp, end: pd.Timestamp, target_hour: int) -> List[pd.Timestamp]:
    s = _to_shanghai(start).normalize() + pd.Timedelta(hours=target_hour)
    if _to_shanghai(start) > s:
        s += pd.Timedelta(days=1)
    e = _to_shanghai(end).normalize() + pd.Timedelta(hours=target_hour)
    if e < s:
        return []
    return list(pd.date_range(s, e, freq="D", tz="Asia/Shanghai"))


def _filter_valid_anchors(
    anchors: List[pd.Timestamp],
    index: pd.DatetimeIndex,
    seq_len: int,
    horizon: int,
) -> List[pd.Timestamp]:
    t_ns = index.view("int64")
    out = []
    for anchor in anchors:
        pos = np.searchsorted(t_ns, int(anchor.value))
        # 注意：这里必须用 `pos + horizon > len(t_ns)`，不能用 `>=`；
        # 因为 future 切片 [pos : pos+horizon] 是左闭右开，等于 len(t_ns) 时仍然合法。
        if pos < seq_len or pos + horizon > len(t_ns) or t_ns[pos] != int(anchor.value): continue
        out.append(anchor)
    return out


def _future_time_grid(
    anchors: List[pd.Timestamp],
    horizon: int,
) -> Tuple[pd.DatetimeIndex, Dict[pd.Timestamp, int]]:
    times, offset = [], {}
    for anchor in anchors:
        offset[anchor] = len(times)
        for h in range(horizon):
            times.append(anchor + h * FREQ)
    return pd.DatetimeIndex(times), offset


def _build_std_rollout_grid(
    *,
    datasets: Sequence,
    seq_len: int,
    horizon: int,
) -> Tuple[pd.DataFrame, List[pd.Timestamp], Dict[pd.Timestamp, int]]:
    anchors: List[pd.Timestamp] = []
    anchor_offset: Dict[pd.Timestamp, int] = {}
    row_anchor: List[pd.Timestamp] = []
    row_time: List[pd.Timestamp] = []
    row_h: List[int] = []

    offset = 0
    for ds in datasets:
        if len(ds) == 0:
            continue
        t_split = pd.to_datetime(ds.time_split, errors="raise").reset_index(drop=True)
        for i in range(len(ds)):
            anchor_idx = i + seq_len
            last_idx = anchor_idx + horizon - 1
            if last_idx >= len(t_split):
                continue

            anchor_ts = _to_shanghai(t_split.iloc[anchor_idx])
            anchors.append(anchor_ts)
            anchor_offset[anchor_ts] = offset

            for h in range(horizon):
                row_anchor.append(anchor_ts)
                row_time.append(_to_shanghai(t_split.iloc[anchor_idx + h]))
                row_h.append(h + 1)
                offset += 1

    base = pd.DataFrame(
        {
            "anchor_time": row_anchor,
            "time": row_time,
            "h": np.asarray(row_h, dtype=np.int32),
        }
    )
    return base, anchors, anchor_offset


def _load_model(model_name: str, model_path: str, device: str):
    from ts_models import TimeSeriesModelFactory

    model = TimeSeriesModelFactory.create_model(model_name, model_path=model_path, model_type=model_name)
    if hasattr(model, "to"):
        model.to(device)
    return model


def _as_numpy(x) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _predict_multivar(
    *,
    df: pd.DataFrame,
    mv_cols: Sequence[str],
    anchors: List[pd.Timestamp],
    anchor_offset: Dict[pd.Timestamp, int],
    model,
    seq_len: int,
    horizon: int,
    batch_size: int,
    num_samples: int,
    device: str,
    use_future_covariates: bool,
    output_size: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    total_size = int(output_size) if output_size is not None else len(anchors) * horizon
    out_by_var = {v: np.full(total_size, np.nan, dtype=np.float32) for v in mv_cols}
    if not anchors:
        return out_by_var

    num_channels = len(mv_cols)
    t_ns = df.index.view("int64")

    ctx_list = []
    fut_list = []
    good = []

    for anchor in anchors:
        pos = np.searchsorted(t_ns, int(anchor.value))
        past = df.iloc[pos - seq_len:pos][list(mv_cols)].to_numpy(np.float32)
        if past.shape != (seq_len, num_channels) or not np.isfinite(past).all():
            continue

        future = None
        if use_future_covariates:
            future = df.iloc[pos:pos + horizon][list(mv_cols)].to_numpy(np.float32)
            if future.shape != (horizon, num_channels) or not np.isfinite(future).all():
                continue

        ctx_list.append(past.T)
        if use_future_covariates:
            future = future.T
            future[-1, :] = np.nan
            fut_list.append(future)
        good.append(anchor)

    if not ctx_list:
        return out_by_var

    ctx = np.stack(ctx_list, axis=0)
    fut = np.stack(fut_list, axis=0) if use_future_covariates else None

    for start_idx in range(0, ctx.shape[0], batch_size):
        batch_ctx = torch.from_numpy(ctx[start_idx:start_idx + batch_size]).to(device)
        batch_fut = None
        if use_future_covariates:
            batch_fut = torch.from_numpy(fut[start_idx:start_idx + batch_size]).to(device)

        with torch.no_grad():
            res = model.predict(
                context=batch_ctx,
                future_covariates=batch_fut if use_future_covariates else None,
                forecast_horizon=horizon,
                num_samples=num_samples,
            )

        mean = _as_numpy(res["mean"]).astype(np.float32)
        if mean.ndim == 2:
            mean = mean[:, None, :]

        for batch_offset in range(mean.shape[0]):
            anchor = good[start_idx + batch_offset]
            global_offset = anchor_offset[anchor]
            for channel_idx, var_name in enumerate(mv_cols):
                out_by_var[var_name][global_offset:global_offset + horizon] = mean[batch_offset, channel_idx, :]

    return out_by_var


# =========================
# metrics
# =========================

def _metrics_zero_shot(*, args, pred_table: pd.DataFrame, models: List[str]) -> None:
    if bool(getattr(args, "is_std", False)):
        _metrics_zero_shot_standard(args=args, pred_table=pred_table, models=models)
    else:
        _metrics_zero_shot_shanxi(args=args, pred_table=pred_table, models=models)


def _metrics_zero_shot_standard(*, args, pred_table: pd.DataFrame, models: List[str]) -> None:
    from pathlib import Path
    from .feature_select import _build_epf_benchmark_datasets
    from .evaluator import _plot_point_series

    cache_root = str(getattr(args, "tsfm_cache_path", "") or getattr(args, "save_dir", ".") or ".")
    out_csv = os.path.join(cache_root, "tsfm_zero_shot_metrics.csv")
    row: Dict[str, object] = {"tag": getattr(args, "tag", "pipeline"), "split": "test"}

    split_cfg = _load_single_json(args.data_split_path)["data_split"]
    cols_cfg = list(_load_single_json(args.regress_cols_path)["target_columns"])
    y_col = cols_cfg[-1]
    scale_flag = bool(getattr(args, "scale", False))

    ds_tr, _, ds_te, _, _ = _build_epf_benchmark_datasets(
        args,
        split_cfg=split_cfg,
        cols_cfg=cols_cfg,
        scale=False,
    )

    if len(pred_table) == 0 or len(ds_te) == 0:
        for model_name in models:
            row[f"{model_name}_mse"] = float("nan")
            row[f"{model_name}_mae"] = float("nan")
        _upsert_row_csv(out_csv, row)
        return

    train_std = _compute_train_y_std_from_dataset(ds_tr) if scale_flag else float("nan")

    pt = _normalize_pred_table_keys(pred_table, is_std=True)
    pt = pt.set_index(["anchor_time", "time", "h"]).sort_index()

    t_split = pd.to_datetime(ds_te.time_split, errors="raise").reset_index(drop=True)
    seq_len = int(getattr(ds_te, "seq_len", getattr(args, "seq_len")))
    horizon = int(getattr(ds_te, "pred_len", getattr(args, "pred_len")))

    se = {m: 0.0 for m in models}
    ae = {m: 0.0 for m in models}
    cnt = {m: 0 for m in models}

    total = len(ds_te) * horizon
    y_true_all = np.full((total,), np.nan, dtype=np.float32)
    time_all = [""] * total
    y_pred_all = {m: np.full((total,), np.nan, dtype=np.float32) for m in models}

    for i in range(len(ds_te)):
        _, _, seq_y, _ = ds_te[i]
        y_seq = np.asarray(seq_y, dtype=np.float32).reshape(-1)
        if y_seq.shape[0] < horizon:
            y_seq = np.concatenate([y_seq, np.full((horizon - y_seq.shape[0],), np.nan, dtype=np.float32)], axis=0)
        elif y_seq.shape[0] > horizon:
            y_seq = y_seq[:horizon]

        anchor_ts = _to_shanghai(pd.to_datetime(t_split.iloc[i + seq_len]))
        for h in range(horizon):
            fut_time = _to_shanghai(pd.to_datetime(t_split.iloc[i + seq_len + h]))
            key = (anchor_ts, fut_time, np.int32(h + 1))
            global_idx = i * horizon + h
            time_all[global_idx] = str(fut_time)

            y_true_val = float(y_seq[h])
            y_true_all[global_idx] = y_true_val if np.isfinite(y_true_val) else np.nan

            try:
                row_ser = pt.loc[key]
            except KeyError:
                continue
            if isinstance(row_ser, pd.DataFrame):
                row_ser = row_ser.iloc[0]
            if not np.isfinite(y_true_val):
                continue

            for model_name in models:
                col = f"pred_{model_name}_{y_col}"
                if col not in row_ser:
                    continue
                y_pred_val = float(row_ser[col])
                if not np.isfinite(y_pred_val):
                    continue
                y_pred_all[model_name][global_idx] = y_pred_val
                err = y_pred_val - y_true_val
                se[model_name] += err * err
                ae[model_name] += abs(err)
                cnt[model_name] += 1

    for model_name in models:
        if cnt[model_name] == 0:
            row[f"{model_name}_mse"] = float("nan")
            row[f"{model_name}_mae"] = float("nan")
            continue

        raw_mse = se[model_name] / cnt[model_name]
        raw_mae = ae[model_name] / cnt[model_name]
        row[f"{model_name}_mse_raw"] = raw_mse
        row[f"{model_name}_mae_raw"] = raw_mae
        if scale_flag and np.isfinite(train_std) and train_std != 0.0:
            row[f"{model_name}_mse"] = raw_mse / (train_std ** 2)
            row[f"{model_name}_mae"] = raw_mae / train_std
        else:
            row[f"{model_name}_mse"] = raw_mse
            row[f"{model_name}_mae"] = raw_mae

    save_dir = Path(cache_root) / "metrics"
    save_dir.mkdir(parents=True, exist_ok=True)
    tag_base = str(getattr(args, "tag", "pipeline"))
    for model_name in models:
        out = {
            "MSE": float(row.get(f"{model_name}_mse", float("nan"))),
            "MAE": float(row.get(f"{model_name}_mae", float("nan"))),
        }
        _plot_point_series(
            save_dir=save_dir,
            split="test",
            tag=f"{tag_base}_{model_name}",
            y_true=y_true_all,
            y_pred=y_pred_all[model_name],
            time_points=time_all,
            out=out,
            use_time_axis=False,
            N=len(ds_te),
            L=horizon,
        )

    _upsert_row_csv(out_csv, row)


def _metrics_zero_shot_shanxi(*, args, pred_table: pd.DataFrame, models: List[str]) -> None:
    split_cfg = _load_single_json(args.data_split_path)["data_split"]
    cols_cfg = list(_load_single_json(args.regress_cols_path)["target_columns"])
    time_col = split_cfg.get("time_col", "time")
    y_col = cols_cfg[-1]

    df = _read_table(args.data_path)
    dte = _slice_df(df, time_col, split_cfg["test_start"], split_cfg["test_end"])
    dte[time_col] = _to_shanghai_series(dte[time_col])

    pt = _normalize_pred_table_keys(pred_table, is_std=False)
    if time_col != "time":
        dte = dte.rename(columns={time_col: "time"})
        time_col = "time"

    pred_cols = [f"pred_{m}_{y_col}" for m in models if f"pred_{m}_{y_col}" in pt.columns]
    dte = dte.merge(pt[["time"] + pred_cols], on="time", how="left")

    row: Dict[str, object] = {"tag": getattr(args, "tag", "pipeline"), "split": "test_workday"}
    for model_name in models:
        pred_col = f"pred_{model_name}_{y_col}"
        row[f"{model_name}_imse"], row[f"{model_name}_imae"] = _imse_imae_workday(
            dte,
            time_col="time",
            y_true=y_col,
            y_pred=pred_col,
            target_hour=int(args.target_hour),
        )

    cache_root = str(getattr(args, "tsfm_cache_path", "") or getattr(args, "save_dir", ".") or ".")
    out_csv = os.path.join(cache_root, "tsfm_zero_shot_workday.csv")
    _upsert_row_csv(out_csv, row)


def _compute_train_y_std_from_dataset(ds) -> float:
    values = []
    for i in range(len(ds)):
        _, _, seq_y, _ = ds[i]
        y = np.asarray(seq_y, dtype=np.float64).ravel()
        if y.size:
            values.append(y)

    if not values:
        return float("nan")

    arr = np.concatenate(values, axis=0)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(arr.std())


def _slice_df(df: pd.DataFrame, time_col: str, start: str, end: str) -> pd.DataFrame:
    t = _to_shanghai_series(df[time_col])
    s = _to_shanghai(pd.to_datetime(start))
    e = _to_shanghai(pd.to_datetime(end))
    return df.loc[(t >= s) & (t <= e)].copy()


def _imse_imae_workday(
    df: pd.DataFrame,
    *,
    time_col: str,
    y_true: str,
    y_pred: str,
    target_hour: int,
) -> Tuple[float, float]:
    if y_pred not in df.columns:
        return float("nan"), float("nan")

    dff = df[[time_col, y_true, y_pred]].copy()
    dff["_d"] = dff[time_col].dt.date

    mses, maes = [], []
    for _, g in dff.groupby("_d"):
        g = g.sort_values(time_col)
        if len(g) != POINTS_PER_DAY:
            continue
        if g[time_col].iloc[0].hour != target_hour:
            continue
        if not calendar.is_workday(g[time_col].iloc[NOON_OFFSET].date()):
            continue

        yt = pd.to_numeric(g[y_true], errors="coerce").to_numpy(np.float32)
        yp = pd.to_numeric(g[y_pred], errors="coerce").to_numpy(np.float32)
        if (not np.isfinite(yt).all()) or (not np.isfinite(yp).all()):
            continue

        err = yp - yt
        mses.append(float(np.mean(err * err)))
        maes.append(float(np.mean(np.abs(err))))

    if not mses:
        return float("nan"), float("nan")
    return float(np.mean(mses)), float(np.mean(maes))


def _upsert_row_csv(path: str, row: Dict[str, object]) -> None:
    new = pd.DataFrame([row])
    if os.path.exists(path):
        old = pd.read_csv(path)
        for col in new.columns:
            if col not in old.columns:
                old[col] = np.nan
        for col in old.columns:
            if col not in new.columns:
                new[col] = np.nan
        new = new[old.columns]

        key = (old["tag"].astype(str) == str(row["tag"])) & (old["split"].astype(str) == str(row["split"]))
        if key.any():
            idx = int(np.flatnonzero(key.values)[0])
            for col in new.columns:
                val = new.loc[0, col]
                if pd.notna(val):
                    old.loc[idx, col] = val
            out = old
        else:
            out = pd.concat([old, new], ignore_index=True)
    else:
        out = new

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    out.to_csv(path, index=False)
