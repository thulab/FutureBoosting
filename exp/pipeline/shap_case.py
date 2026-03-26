from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Tuple, List
import re

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =========================
# Helpers
# =========================

_ZWSP = "\u200b"  # zero-width space (不可见，用于保证 tick label 唯一但视觉不变)


def _set_paper_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            # larger fonts (you asked: "再大一点")
            "font.size": 16,
            "axes.labelsize": 18,
            "axes.titlesize": 18,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            "legend.fontsize": 16,
        }
    )


def _reshape_by_horizon(x: np.ndarray, L: int) -> np.ndarray:
    x = np.asarray(x).reshape(-1)
    assert x.ndim == 1
    assert x.shape[0] % L == 0, f"len={x.shape[0]} not divisible by L={L}"
    return x.reshape(-1, L)


def _pick_step_by_gain(
    *,
    y_true_L: np.ndarray,  # [L]
    y_hyb_L: np.ndarray,  # [L]
    y_tsfm_L: np.ndarray,  # [L]
    mode: str,  # "low" or "high"
    q: float = 0.10,  # low: q, high: 1-q
) -> int:
    yt = np.asarray(y_true_L).reshape(-1)
    yh = np.asarray(y_hyb_L).reshape(-1)
    yz = np.asarray(y_tsfm_L).reshape(-1)
    gain = np.abs(yz - yt) - np.abs(yh - yt)  # positive => Hybrid better than TSFM

    if mode == "low":
        thr = np.quantile(yt, q)
        cand = np.where(yt <= thr)[0]
        if cand.size == 0:
            return int(np.argmin(yt))
    elif mode == "high":
        thr = np.quantile(yt, 1.0 - q)
        cand = np.where(yt >= thr)[0]
        if cand.size == 0:
            return int(np.argmax(yt))
    else:
        raise ValueError(f"mode must be 'low' or 'high', got {mode}")

    best = cand[np.argmax(gain[cand])]
    return int(best)


def _pretty_feature_names_no_tag(
    feature_names: Sequence[str],
    pred_prefix: str,
) -> Tuple[List[str], List[bool], List[str]]:
    """
    Return:
      - pretty_names: passed to SHAP (may include zero-width spaces for uniqueness)
      - is_pred:      whether original feature is pred feature
      - base_names:   human-visible base name (no ZWSP)

    Rules:
      pred_chronos2_xxx -> xxx
      others            -> original
      '_' -> ' '

    If duplicate labels occur, append k times ZWSP to make them unique *without changing appearance*.
    """
    pretty: List[str] = []
    is_pred: List[bool] = []
    base: List[str] = []
    seen: dict[str, int] = {}

    for fn in feature_names:
        if fn.startswith(pred_prefix):
            name = fn[len(pred_prefix) :]
            p = True
        else:
            name = fn
            p = False

        name = name.replace("_", " ")
        base_name = name

        cnt = seen.get(base_name, 0)
        seen[base_name] = cnt + 1

        disp = base_name
        if cnt > 0:
            # add ZWSP (visual identical, logically unique)
            disp = base_name + (_ZWSP * cnt)

        pretty.append(disp)
        is_pred.append(p)
        base.append(base_name)

    return pretty, is_pred, base


def _infer_pred_prefix(tsfm_feature_name: str) -> str:
    if not tsfm_feature_name.startswith("pred_"):
        return "pred_"
    parts = tsfm_feature_name.split("_", 2)
    if len(parts) < 3:
        return "pred_"
    return f"{parts[0]}_{parts[1]}_"


def _infer_series_label(tsfm_feature_name: str) -> str:
    if not tsfm_feature_name.startswith("pred_"):
        return "TSFM"
    parts = tsfm_feature_name.split("_", 2)
    if len(parts) < 3:
        return "TSFM"
    model_name = parts[1]
    if not model_name:
        return "TSFM"
    return model_name[0].upper() + model_name[1:]


def _extract_feat_from_waterfall_tick_keep_zwsp(tick_text: str) -> str:
    """
    Extract the feature part from either:
      - "value = feature"
      - "feature"
    Keep ZWSP so we can distinguish duplicates (pred vs non-pred).
    """
    t = tick_text
    if "=" in t:
        t = t.split("=", 1)[1]
    t = t.strip()

    # safety: remove SHAP-added [1] if it ever appears
    t = re.sub(r"\s*\[\d+\]\s*$", "", t).strip()
    return t


def _color_waterfall_yticklabels_pred_only(
    ax,
    pred_pretty_name_set: set[str],
    pred_color: str = "#9467bd",
) -> None:
    """
    Color only the ytick labels corresponding to pred variables.
    Matching is done on SHAP-rendered tick labels, using the ZWSP-unique pretty name.
    """
    for tl in ax.get_yticklabels():
        feat = _extract_feat_from_waterfall_tick_keep_zwsp(tl.get_text())
        if feat in pred_pretty_name_set:
            tl.set_color(pred_color)
            tl.set_fontweight("normal")


# =========================
# Main export
# =========================

def export_shap_casebook_low_high(
    *,
    out_dir: str | Path,  # directory to save per-figure pdf/png
    shap_values: np.ndarray,  # [N, F] (N = n_inst * L)
    expected_value: float,  # scalar expected value
    X: np.ndarray,  # [N, F]
    feature_names: Sequence[str],  # len=F
    y_true: np.ndarray,  # [N]
    y_pred: np.ndarray,  # [N] (Hybrid/LGBM)
    L: int,
    tsfm_feature_name: str,  # e.g., "pred_chronos2_day_ahead_clearing_price"
    time_points: Optional[np.ndarray] = None,  # [N]
    max_cases: Optional[int] = None,
    topk_waterfall: int = 10,
    quantile_q: float = 0.10,
    seed: int = 0,
    pred_prefix: Optional[str] = None,
    series_label: Optional[str] = None,
    pred_label_color: str = "#9467bd",  # purple
) -> None:
    """
    Save TWO figures per instance:
      - case_{inst:05d}_low.pdf/.png
      - case_{inst:05d}_high.pdf/.png

    Waveform colors:
      True: blue, Chronos2: orange, Hybrid: red (thicker).

    Waterfall:
      - pred_* feature names are displayed as suffix only (prefix removed)
      - pred feature tick-labels are colored purple (no bold)
      - duplicates are resolved via ZWSP (no visible [1])
    """
    _set_paper_style()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pred_prefix = str(pred_prefix or _infer_pred_prefix(tsfm_feature_name))
    series_label = str(series_label or _infer_series_label(tsfm_feature_name))

    feat_list_raw = list(feature_names)
    if tsfm_feature_name not in feat_list_raw:
        raise ValueError(f"tsfm_feature_name='{tsfm_feature_name}' not in feature_names")

    # pretty names passed into SHAP (ZWSP-unique)
    feat_pretty, feat_is_pred, _feat_base = _pretty_feature_names_no_tag(
        feature_names=feature_names, pred_prefix=pred_prefix
    )
    pred_pretty_name_set = {p for p, is_p in zip(feat_pretty, feat_is_pred) if is_p}

    # index for tsfm feature uses raw name
    tsfm_idx = feat_list_raw.index(tsfm_feature_name)

    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    X = np.asarray(X)
    shap_values = np.asarray(shap_values)

    N = int(y_true.shape[0])
    F = len(feat_list_raw)
    assert N == y_pred.shape[0] == X.shape[0] == shap_values.shape[0], "N mismatch"
    assert X.shape[1] == shap_values.shape[1] == F, f"F mismatch: X={X.shape[1]}, shap={shap_values.shape[1]}, feat={F}"
    assert N % L == 0, f"N={N} not divisible by L={L}"

    n_inst = N // L
    Yt = _reshape_by_horizon(y_true, L)
    Yh = _reshape_by_horizon(y_pred, L)
    Yz = _reshape_by_horizon(X[:, tsfm_idx], L)

    if time_points is not None:
        T = np.asarray(time_points)
        assert T.shape[0] == N
        T = T.reshape(-1, L)
    else:
        T = None

    inst_ids = np.arange(n_inst)
    if max_cases is not None and 0 < max_cases < n_inst:
        rng = np.random.default_rng(seed)
        inst_ids = np.sort(rng.choice(inst_ids, size=max_cases, replace=False))

    import shap  # type: ignore

    # fixed colors
    C_TRUE = "#1f77b4"  # blue
    C_TSF = "#ff7f0e"   # orange
    C_HYB = "#d62728"   # red

    def _draw_one(inst_i: int, mode: str) -> None:
        t_star = _pick_step_by_gain(
            y_true_L=Yt[inst_i],
            y_hyb_L=Yh[inst_i],
            y_tsfm_L=Yz[inst_i],
            mode=mode,
            q=quantile_q,
        )
        j = inst_i * L + t_star

        sv = shap_values[j]  # [F]
        x_row = X[j]         # [F]

        fig = plt.figure(figsize=(13.6, 4.8))
        gs = fig.add_gridspec(1, 2, width_ratios=[1.65, 1.0], wspace=0.28)
        ax_wf = fig.add_subplot(gs[0, 0])
        ax_wave = fig.add_subplot(gs[0, 1])

        # ---- Left: SHAP waterfall ----
        plt.sca(ax_wf)
        exp = shap.Explanation(
            values=sv,
            base_values=expected_value,
            data=x_row,
            feature_names=feat_pretty,  # ZWSP-unique + pred prefix removed
        )
        shap.plots.waterfall(exp, max_display=int(topk_waterfall), show=False)

        # color only pred feature tick labels (robust even with duplicates)
        _color_waterfall_yticklabels_pred_only(
            ax_wf, pred_pretty_name_set, pred_color=pred_label_color
        )

        # ---- Right: waveform ----
        x_axis = T[inst_i] if T is not None else np.arange(L)

        ax_wave.plot(x_axis, Yt[inst_i], label="True", color=C_TRUE, linewidth=1.9)
        ax_wave.plot(x_axis, Yz[inst_i], label=series_label, color=C_TSF, linewidth=1.9)
        ax_wave.plot(x_axis, Yh[inst_i], label="Hybrid", color=C_HYB, linewidth=2.8)

        ax_wave.axvline(
            x_axis[t_star], linestyle="--", linewidth=1.3, alpha=0.9, color="0.2"
        )

        ax_wave.set_xlabel("Time" if T is not None else "Horizon step")
        ax_wave.set_ylabel("Price")
        ax_wave.grid(True, alpha=0.22, linewidth=0.9)

        ax_wave.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False)

        ymin = np.nanmin([Yt[inst_i].min(), Yz[inst_i].min(), Yh[inst_i].min()])
        ymax = np.nanmax([Yt[inst_i].max(), Yz[inst_i].max(), Yh[inst_i].max()])
        span = max(float(ymax - ymin), 1e-6)
        ax_wave.set_ylim(float(ymin - 0.10 * span), float(ymax + 0.18 * span))

        fig.subplots_adjust(left=0.05, right=0.90, top=0.93, bottom=0.12, wspace=0.30)

        stem = out_dir / f"case_{inst_i:05d}_{mode}"
        fig.savefig(f"{stem}.pdf", bbox_inches="tight")
        fig.savefig(f"{stem}.png", dpi=220, bbox_inches="tight")
        plt.close(fig)

    for inst_i in inst_ids:
        _draw_one(inst_i, "low")
        _draw_one(inst_i, "high")

    print(f"[casebook] saved per-case figures to: {out_dir}")
