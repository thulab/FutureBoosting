from __future__ import annotations

import json
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple
import warnings

import numpy as np
import pandas as pd
def _plot_global_topk_bar_and_beeswarm(
    *,
    shap_values_arr: np.ndarray,     # [N, F]
    X_df: pd.DataFrame,              # [N, F]
    feature_names: Sequence[str],
    out_dir: Path,
    split: str,
    topk: int = 10,
    dpi: int = 220,
    # colors
    c_pred_label: str = "#9467bd",   # purple label for pred vars (yticks)
    c_pred_bar: str = "#a393eb",     # bar color for pred vars
    c_feat_bar: str = "#ec6d71",     # bar color for normal vars
) -> dict:
    import shap  # type: ignore
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import rcParams
    import numpy as np

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------- style ----------
    FONT_AX = 16
    FONT_TICK = 16
    FONT_LEG = 16

    rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.unicode_minus": False,
        "figure.constrained_layout.use": False,
        "figure.autolayout": False,
        "axes.labelsize": FONT_AX,
        "axes.titlesize": FONT_AX,
        "xtick.labelsize": FONT_TICK,
        "ytick.labelsize": FONT_TICK,
        "legend.fontsize": FONT_LEG,
    })

    def _bold_axes_text(ax):
        """Make axis labels + tick labels bold."""
        # axis labels
        ax.xaxis.label.set_fontweight("bold")
        ax.yaxis.label.set_fontweight("bold")

        # tick labels
        for tl in ax.get_xticklabels():
            tl.set_fontweight("bold")
        for tl in ax.get_yticklabels():
            tl.set_fontweight("bold")

        # tick marks (optional but often looks consistent)
        ax.tick_params(axis="both", width=1.6, length=6)

        # spines (optional but consistent with bold text)
        for s in ax.spines.values():
            s.set_linewidth(1.6)

    feat_list = list(feature_names)

    # ---------- compute top-k by mean(|SHAP|) ----------
    mean_abs = np.mean(np.abs(shap_values_arr), axis=0)  # [F]
    order = np.argsort(mean_abs)[::-1]
    topk = min(int(topk), len(order))
    top_idx = order[:topk]

    def _pretty_name(raw: str) -> str:
        if raw.startswith("pred_"):
            parts = raw.split("_", 2)
            if len(parts) == 3:
                return parts[2].replace("_", " ")
        return raw.replace("_", " ")

    top_raw_names = [feat_list[i] for i in top_idx]
    top_names = [_pretty_name(n) for n in top_raw_names]
    top_vals = mean_abs[top_idx]

    is_pred_top = [n.startswith("pred_") for n in top_raw_names]
    pred_name_set = set(_pretty_name(n) for n in top_raw_names if n.startswith("pred_"))

    # ---------- geometry ----------
    FIGSIZE = (10, 7)
    LEFT = 0.34
    RIGHT = 0.90
    TOP = 0.90
    BOTTOM = 0.18

    def _apply_margins(fig):
        fig.subplots_adjust(left=LEFT, right=RIGHT, top=TOP, bottom=BOTTOM)

    # ===================== 1) BAR =====================
    bar_pdf = out_dir / f"shap_global_{split}_bar_top{topk}.pdf"
    bar_png = out_dir / f"shap_global_{split}_bar_top{topk}.png"

    fig, ax = plt.subplots(figsize=FIGSIZE)
    _apply_margins(fig)

    y = np.arange(topk)
    colors = [c_pred_bar if p else c_feat_bar for p in is_pred_top]

    ax.barh(y, top_vals, height=0.55, color=colors, alpha=0.98)
    ax.set_yticks(y)
    ax.set_yticklabels(top_names)
    ax.invert_yaxis()
    ax.set_xlabel("mean(|SHAP|)")
    ax.grid(axis="x", alpha=0.25, linewidth=0.9)

    # pred labels: keep purple but bold is handled by _bold_axes_text
    for tl, p in zip(ax.get_yticklabels(), is_pred_top):
        if p:
            tl.set_color(c_pred_label)

    # remove top/right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # === NEW: bold axis text ===
    _bold_axes_text(ax)
    # (top/right spines already hidden; linewidth calls don't re-show them)

    fig.savefig(bar_pdf, dpi=dpi)
    fig.savefig(bar_png, dpi=dpi)
    plt.close(fig)

    # ===================== 2) BEESWARM =====================
    bees_pdf = out_dir / f"shap_global_{split}_beeswarm_top{topk}.pdf"
    bees_png = out_dir / f"shap_global_{split}_beeswarm_top{topk}.png"

    X_df_top = X_df.iloc[:, top_idx].copy()
    X_df_top.columns = top_names

    fig, ax = plt.subplots(figsize=FIGSIZE)
    _apply_margins(fig)
    plt.sca(ax)

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"The NumPy global RNG was seeded by calling `np\.random\.seed`.*",
            category=FutureWarning,
        )
        shap.summary_plot(
            shap_values_arr[:, top_idx],
            X_df_top,
            show=False,
            max_display=int(topk),
        )

    _apply_margins(fig)

    # recolor y tick labels for pred vars
    for tl in ax.get_yticklabels():
        if tl.get_text() in pred_name_set:
            tl.set_color(c_pred_label)

    # === NEW: bold axis text for main beeswarm axis ===
    _bold_axes_text(ax)

    # === NEW: bold colorbar axis text (SHAP creates extra axes) ===
    for a in fig.axes:
        if a is ax:
            continue
        # usually this is the colorbar axis
        for tl in a.get_xticklabels() + a.get_yticklabels():
            tl.set_fontweight("bold")
        a.xaxis.label.set_fontweight("bold")
        a.yaxis.label.set_fontweight("bold")
        a.tick_params(axis="both", width=1.6, length=6)
        for s in a.spines.values():
            s.set_linewidth(1.6)

    fig.savefig(bees_pdf, dpi=dpi)
    fig.savefig(bees_png, dpi=dpi)
    plt.close(fig)

    return {
        "global_bar_pdf": str(bar_pdf),
        "global_bar_png": str(bar_png),
        "global_beeswarm_pdf": str(bees_pdf),
        "global_beeswarm_png": str(bees_png),
    }




def run_model_shap_explain(
    *,
    model: Any,
    X: np.ndarray,
    feature_names: Sequence[str],
    save_dir: str | Path,
    split: str,
    tag: str,
    max_samples: Optional[int] = None,
    seed: int = 2021,
    topk: int = 50,
    n_dependence_plots: int = 10,
    n_waterfall_samples: int = 3,
    n_decision_samples: int = 20,
    enable_interaction: bool = True,
    enable_heatmap: bool = True,
) -> Optional[dict]:
    """
    Compute SHAP values for LightGBM tree model and save artifacts under save_dir/shap/.

    Artifacts:
      - shap_values_{split}.npy
      - shap_expected_value_{split}.json
      - shap_importance_{split}.csv  (mean(|shap|) desc)
      - shap_summary_{split}_beeswarm.pdf (if matplotlib available)
      - shap_summary_{split}_bar.pdf      (if matplotlib available)
      - shap_dependence_{feature}_{split}.pdf (for top features)
      - shap_waterfall_{split}_sample{i}.pdf (for representative samples)
      - shap_decision_{split}.pdf (multi-sample decision paths)
      - shap_heatmap_{split}.pdf (SHAP values heatmap, if enabled)
      - shap_interaction_values_{split}.npy (if interaction enabled)
      - shap_interaction_summary_{split}.pdf (if interaction enabled)
    """
    save_dir = Path(save_dir)
    out_dir = save_dir / "shap"
    out_dir.mkdir(parents=True, exist_ok=True)

    if X is None or len(X) == 0:
        _write_text(out_dir / f"shap_{split}_skipped.txt", "Skip SHAP: empty X.\n")
        return None

    if len(feature_names) != int(X.shape[1]):
        _write_text(
            out_dir / f"shap_{split}_skipped.txt",
            f"Skip SHAP: feature_names mismatch ({len(feature_names)} vs {X.shape[1]}).\n",
        )
        return None

    # ---- sampling (avoid huge cost) ----
    n = int(X.shape[0])
    if max_samples is not None and max_samples > 0:
        take = min(int(max_samples), n)
        rng = np.random.default_rng(int(seed))
        if take < n:
            idx = rng.choice(n, size=take, replace=False)
            idx.sort()
            X_use = X[idx]
        else:
            idx = None
            X_use = X
    else:
        idx = None
        X_use = X

    X_df = pd.DataFrame(X_use, columns=list(feature_names))

    # ---- SHAP ----
    try:
        import shap  # type: ignore
    except Exception as e:
        _write_text(
            out_dir / f"shap_{split}_missing_shap.txt",
            f"SHAP not installed or failed to import: {type(e).__name__}: {e}\n",
        )
        return None

    explainer, explainer_kind = _build_shap_explainer(
        model=model,
        X_background=X_use,
        seed=seed,
    )
    try:
        if explainer_kind == "generic":
            explanation = explainer(X_use)
            shap_values = explanation.values
            expected_value = explanation.base_values
        elif explainer_kind == "tree":
            shap_values = explainer.shap_values(X_df)
            expected_value = getattr(explainer, "expected_value", None)
        else:
            shap_values = explainer.shap_values(X_use)
            expected_value = getattr(explainer, "expected_value", None)
    except Exception:
        if explainer_kind == "generic":
            explanation = explainer(X_use)
            shap_values = explanation.values
            expected_value = explanation.base_values
        else:
            shap_values = explainer.shap_values(X_use)
            expected_value = getattr(explainer, "expected_value", None)

    # shap_values may be list (e.g. multiclass); for regression we want 2D array
    if isinstance(shap_values, list):
        shap_values_arr = np.asarray(shap_values[0])
    else:
        shap_values_arr = np.asarray(shap_values)

    if shap_values_arr.ndim != 2:
        _write_text(
            out_dir / f"shap_{split}_skipped.txt",
            f"Skip SHAP: unexpected shap_values shape {tuple(shap_values_arr.shape)}.\n",
        )
        return None

    base_value = _normalize_expected_value(expected_value)

    np.save(out_dir / f"shap_values_{split}.npy", shap_values_arr)

    with open(out_dir / f"shap_expected_value_{split}.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "tag": tag,
                "split": split,
                "expected_value": _jsonable(base_value),
                "explainer_kind": explainer_kind,
                "n_rows_total": n,
                "n_rows_used": int(X_use.shape[0]),
                "sampled": bool(idx is not None),
                "max_samples": int(max_samples) if max_samples is not None else None,
                "seed": int(seed),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    # ---- importance table ----
    mean_abs = np.mean(np.abs(shap_values_arr), axis=0)
    mean_signed = np.mean(shap_values_arr, axis=0)
    df_imp = pd.DataFrame(
        {
            "feature": list(feature_names),
            "mean_abs_shap": mean_abs,
            "mean_shap": mean_signed,
        }
    ).sort_values("mean_abs_shap", ascending=False)
    df_imp.to_csv(out_dir / f"shap_importance_{split}.csv", index=False, encoding="utf-8")

    # ---- plots (optional) ----
    plot_results = {}
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib import rcParams

        # ---- global font config (Times New Roman) ----
        rcParams.update({
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "pdf.fonttype": 42,   # embed TrueType fonts in PDF
            "ps.fonttype": 42,
            "axes.unicode_minus": False,
        })


        # ---- global pair (Top-10 + Others bar, Top-10 beeswarm) ----
        global_paths = _plot_global_topk_bar_and_beeswarm(
            shap_values_arr=shap_values_arr,
            X_df=X_df,
            feature_names=feature_names,
            out_dir=out_dir,
            split=split,
            topk=10,
            dpi=220,
        )
        plot_results.update(global_paths)
        print(
            "SHAP global plots saved:"
            f" bar={global_paths['global_bar_pdf']},"
            f" beeswarm={global_paths['global_beeswarm_pdf']}"
        )


        # ---- Dependence plots (top features) ----
        n_dep = min(int(n_dependence_plots), len(df_imp))
        if n_dep > 0:
            try:
                dep_features = df_imp.iloc[:n_dep]["feature"].tolist()
                dep_feature_indices = [list(feature_names).index(f) for f in dep_features]
                dep_paths = []
                for feat_name, feat_idx in zip(dep_features, dep_feature_indices):
                    try:
                        plt.figure(figsize=(10, 6))
                        shap.dependence_plot(
                            feat_idx,
                            shap_values_arr,
                            X_df,
                            interaction_index="auto",
                            show=False,
                        )
                        plt.tight_layout()
                        dep_path = out_dir / f"shap_dependence_{feat_name}_{split}.pdf"
                        plt.savefig(dep_path, dpi=220, bbox_inches="tight")
                        plt.close()
                        dep_paths.append(str(dep_path))
                    except Exception as e:
                        _write_text(
                            out_dir / f"shap_dependence_{feat_name}_{split}_failed.txt",
                            f"Failed to create dependence plot: {type(e).__name__}: {e}\n",
                        )
                if dep_paths:
                    plot_results["dependence"] = dep_paths
                    print(f"SHAP dependence plots saved ({len(dep_paths)} files)")
            except Exception as e:
                _write_text(
                    out_dir / f"shap_dependence_{split}_failed.txt",
                    f"Dependence plots failed: {type(e).__name__}: {e}\n",
                )

        # ---- Waterfall plots (representative samples) ----
        n_wf = min(int(n_waterfall_samples), X_df.shape[0])
        if n_wf > 0 and base_value is not None:
            try:
                # Select diverse samples: high, medium, low SHAP impact
                sample_indices = _select_diverse_samples(shap_values_arr, n_wf, seed)
                wf_paths = []
                for i, sample_idx in enumerate(sample_indices):
                    try:
                        base_val = base_value
                        # Try newer SHAP API (shap.plots.waterfall)
                        try:
                            explanation = shap.Explanation(
                                values=shap_values_arr[sample_idx],
                                base_values=base_val,
                                data=X_df.iloc[sample_idx].values if hasattr(X_df.iloc[sample_idx], 'values') else X_df.iloc[sample_idx],
                                feature_names=list(feature_names),
                            )
                            shap.plots.waterfall(explanation, show=False)
                            plt.tight_layout()
                            wf_path = out_dir / f"shap_waterfall_{split}_sample{i}.pdf"
                            plt.savefig(wf_path, dpi=220, bbox_inches="tight")
                            plt.close()
                            wf_paths.append(str(wf_path))
                        except (AttributeError, TypeError):
                            # Fallback to older API (shap.waterfall_plot)
                            plt.figure(figsize=(10, 8))
                            shap.waterfall_plot(
                                shap.Explanation(
                                    values=shap_values_arr[sample_idx],
                                    base_values=base_val,
                                    data=X_df.iloc[sample_idx].values if hasattr(X_df.iloc[sample_idx], 'values') else X_df.iloc[sample_idx],
                                    feature_names=list(feature_names),
                                ),
                                show=False,
                            )
                            plt.tight_layout()
                            wf_path = out_dir / f"shap_waterfall_{split}_sample{i}.pdf"
                            plt.savefig(wf_path, dpi=220, bbox_inches="tight")
                            plt.close()
                            wf_paths.append(str(wf_path))
                    except Exception as e:
                        _write_text(
                            out_dir / f"shap_waterfall_{split}_sample{i}_failed.txt",
                            f"Failed to create waterfall plot: {type(e).__name__}: {e}\n",
                        )
                if wf_paths:
                    plot_results["waterfall"] = wf_paths
                    print(f"SHAP waterfall plots saved ({len(wf_paths)} files)")
            except Exception as e:
                _write_text(
                    out_dir / f"shap_waterfall_{split}_failed.txt",
                    f"Waterfall plots failed: {type(e).__name__}: {e}\n",
                )

        # ---- Decision plot (multi-sample) ----
        n_dec = min(int(n_decision_samples), X_df.shape[0])
        if n_dec > 0 and base_value is not None:
            try:
                # Select diverse samples
                sample_indices = _select_diverse_samples(shap_values_arr, n_dec, seed)
                base_val = base_value
                
                plt.figure(figsize=(12, max(6, n_dec * 0.3)))
                shap.decision_plot(
                    base_val,
                    shap_values_arr[sample_indices],
                    X_df.iloc[sample_indices],
                    feature_names=list(feature_names),
                    feature_order="importance",
                    show=False,
                    legend_location="best",
                )
                plt.tight_layout()
                dec_path = out_dir / f"shap_decision_{split}.pdf"
                plt.savefig(dec_path, dpi=220, bbox_inches="tight")
                plt.close()
                plot_results["decision"] = str(dec_path)
                print(f"SHAP decision plot saved to {dec_path}")
            except Exception as e:
                _write_text(
                    out_dir / f"shap_decision_{split}_failed.txt",
                    f"Decision plot failed: {type(e).__name__}: {e}\n",
                )

        # ---- Heatmap ----
        if enable_heatmap and base_value is not None:
            try:
                # Limit samples for heatmap to avoid memory issues
                n_heatmap = min(100, X_df.shape[0])
                if n_heatmap > 0:
                    heatmap_indices = _select_diverse_samples(shap_values_arr, n_heatmap, seed)
                    explanation = shap.Explanation(
                        values=shap_values_arr[heatmap_indices],
                        base_values=base_value,
                        data=X_df.iloc[heatmap_indices],
                        feature_names=list(feature_names),
                    )
                    # Dynamically adjust figure size to ensure each cell has sufficient size
                    # Calculate based on desired cell size
                    n_features_display = min(int(topk), len(feature_names))
                    
                    # Define minimum cell size (in inches)
                    # Width per feature: larger value = bigger cells horizontally
                    # Height per sample: larger value = bigger cells vertically
                    cell_width = 0.6  # inches per feature (increase for larger cells)
                    cell_height = 0.25  # inches per sample (increase for larger cells)
                    
                    # Calculate figure size: base size + (cells * cell_size)
                    # Add extra space for labels and padding
                    fig_width = max(16, 4 + n_features_display * cell_width)
                    fig_height = max(12, 4 + n_heatmap * cell_height)
                    
                    # Limit maximum size to avoid memory issues (cap at reasonable values)
                    fig_width = min(fig_width, 40)  # max 40 inches width
                    fig_height = min(fig_height, 30)  # max 30 inches height
                    
                    plt.figure(figsize=(fig_width, fig_height))
                    shap.plots.heatmap(explanation, show=False, max_display=int(topk))
                    
                    # Adjust font sizes and label rotation for better readability
                    ax = plt.gca()
                    
                    # Increase font size for feature names (x-axis labels)
                    # Larger font when we have more space
                    feature_fontsize = min(10, max(7, 14 - n_features_display * 0.08))
                    if ax.get_xticklabels():
                        plt.setp(ax.get_xticklabels(), fontsize=feature_fontsize, rotation=45, ha='right')
                    
                    # Increase font size for sample indices (y-axis labels) if needed
                    sample_fontsize = min(8, max(6, 10 - n_heatmap * 0.02))
                    if ax.get_yticklabels():
                        plt.setp(ax.get_yticklabels(), fontsize=sample_fontsize)
                    
                    # Adjust layout with more padding
                    plt.tight_layout(pad=2.0)
                    
                    heatmap_path = out_dir / f"shap_heatmap_{split}.pdf"
                    plt.savefig(heatmap_path, dpi=220, bbox_inches="tight", pad_inches=0.3)
                    plt.close()
                    plot_results["heatmap"] = str(heatmap_path)
                    print(f"SHAP heatmap saved to {heatmap_path}")
            except Exception as e:
                _write_text(
                    out_dir / f"shap_heatmap_{split}_failed.txt",
                    f"Heatmap failed: {type(e).__name__}: {e}\n",
                )

        # ---- Interaction values (optional, computationally expensive) ----
        if enable_interaction and explainer_kind == "tree" and hasattr(explainer, "shap_interaction_values"):
            try:
                # Limit samples for interaction to avoid memory issues
                n_interaction = min(3, X_df.shape[0])
                if n_interaction > 0:
                    interaction_indices = _select_diverse_samples(shap_values_arr, n_interaction, seed)
                    print(f"Computing SHAP interaction values for {n_interaction} samples (this may take a while)...")
                    shap_interaction_values = explainer.shap_interaction_values(X_df.iloc[interaction_indices])
                    
                    # Save interaction values
                    if isinstance(shap_interaction_values, list):
                        shap_interaction_arr = np.asarray(shap_interaction_values[0])
                    else:
                        shap_interaction_arr = np.asarray(shap_interaction_values)
                    
                    if shap_interaction_arr.ndim == 3:
                        np.save(out_dir / f"shap_interaction_values_{split}.npy", shap_interaction_arr)
                        
                        # Summary plot for interaction (average interaction strength)
                        mean_interaction = np.mean(np.abs(shap_interaction_arr), axis=0)
                        # Sum interactions per feature pair (symmetric matrix)
                        feature_interaction_strength = np.sum(mean_interaction, axis=0) + np.sum(mean_interaction, axis=1) - np.diag(mean_interaction)
                        
                        plt.figure(figsize=(12, 10))
                        top_interaction_features = np.argsort(feature_interaction_strength)[-int(topk):][::-1]
                        
                        # Plot interaction summary for top features
                        interaction_matrix = mean_interaction[np.ix_(top_interaction_features, top_interaction_features)]
                        plt.imshow(interaction_matrix, aspect="auto", cmap="viridis")
                        plt.colorbar(label="Mean |SHAP Interaction|")
                        plt.xticks(range(len(top_interaction_features)), [feature_names[i] for i in top_interaction_features], rotation=45, ha="right")
                        plt.yticks(range(len(top_interaction_features)), [feature_names[i] for i in top_interaction_features])
                        plt.title(f"SHAP Interaction Values (Top {len(top_interaction_features)} Features)")
                        plt.tight_layout()
                        interaction_path = out_dir / f"shap_interaction_summary_{split}.pdf"
                        plt.savefig(interaction_path, dpi=220, bbox_inches="tight")
                        plt.close()
                        plot_results["interaction"] = str(interaction_path)
                        print(f"SHAP interaction values saved to {out_dir / f'shap_interaction_values_{split}.npy'}")
                        print(f"SHAP interaction summary plot saved to {interaction_path}")
            except Exception as e:
                _write_text(
                    out_dir / f"shap_interaction_{split}_failed.txt",
                    f"Interaction values failed: {type(e).__name__}: {e}\n",
                )

    except Exception as e:
        _write_text(
            out_dir / f"shap_{split}_plot_failed.txt",
            f"SHAP computed, but plotting failed: {type(e).__name__}: {e}\n",
        )

    result = {
        "tag": tag,
        "split": split,
        "n_rows_total": n,
        "n_rows_used": int(X_use.shape[0]),
        "top_feature": str(df_imp.iloc[0]["feature"]) if len(df_imp) else None,
    }
    if plot_results:
        result["plots"] = plot_results
    return result


def run_lgbm_shap_explain(**kwargs):
    return run_model_shap_explain(**kwargs)


def _select_diverse_samples(
    shap_values: np.ndarray,
    n_samples: int,
    seed: int,
) -> np.ndarray:
    """
    Select diverse samples based on SHAP value magnitudes:
    - High impact (large |SHAP| sum)
    - Medium impact
    - Low impact (small |SHAP| sum)
    """
    n = shap_values.shape[0]
    if n_samples >= n:
        return np.arange(n)
    
    # Calculate total SHAP impact per sample
    total_impact = np.sum(np.abs(shap_values), axis=1)
    
    # Select samples across the distribution
    sorted_indices = np.argsort(total_impact)
    
    # Distribute across quantiles
    selected = []
    rng = np.random.default_rng(seed)
    
    
    if n_samples <= 3:
        # Simple selection: min, median, max
        selected.append(sorted_indices[0])  # low
        if n_samples >= 2:
            selected.append(sorted_indices[len(sorted_indices) // 2])  # median
        if n_samples >= 3:
            selected.append(sorted_indices[-1])  # high
    else:
        # Select from different quantiles
        quantiles = np.linspace(0, len(sorted_indices) - 1, n_samples, dtype=int)
        selected = sorted_indices[quantiles].tolist()
        
        # Add some randomness within each quantile range
        if n_samples > 10:
            n_random = min(n_samples // 3, 5)
            for _ in range(n_random):
                idx = rng.integers(0, n)
                if idx not in selected:
                    selected.append(idx)
    
    return np.array(sorted(set(selected))[:n_samples])


def _write_text(path: Path, s: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(s, encoding="utf-8")


def _jsonable(x: Any) -> Any:
    if x is None:
        return None
    if isinstance(x, (str, int, float, bool)):
        return x
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    try:
        return float(x)
    except Exception:
        return str(x)


def _normalize_expected_value(expected_value: Any) -> Optional[float]:
    if expected_value is None:
        return None
    arr = np.asarray(expected_value, dtype=float)
    if arr.size == 0:
        return None
    return float(arr.reshape(-1)[0])


def _sample_background(X: np.ndarray, *, max_rows: int, seed: int) -> np.ndarray:
    X = np.asarray(X)
    n = int(X.shape[0])
    if n <= max_rows:
        return X
    rng = np.random.default_rng(int(seed))
    idx = np.sort(rng.choice(n, size=int(max_rows), replace=False))
    return X[idx]


def _build_shap_explainer(*, model: Any, X_background: np.ndarray, seed: int):
    import shap  # type: ignore

    if hasattr(model, "feature_importance"):
        return shap.TreeExplainer(model), "tree"

    background = _sample_background(X_background, max_rows=min(1000, len(X_background)), seed=seed)
    if hasattr(model, "coef_"):
        return shap.LinearExplainer(model, background), "linear"

    masker = shap.maskers.Independent(background)
    predict_fn = getattr(model, "predict", model)
    return shap.Explainer(predict_fn, masker), "generic"
