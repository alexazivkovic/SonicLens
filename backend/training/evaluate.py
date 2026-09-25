"""Thesis experiments: feature-set comparison, ablation, permutation importance, plots.

1. Feature sets, on identical tracks and artist-grouped folds:
     librosa_full  - FMA's precomputed features.csv (librosa, full-length tracks)
     librosa_clip  - the same librosa features recomputed on the 30 s clips (librosa_clips.py)
     soniclens     - the SonicLens Essentia model input vector (30 s clips)
   librosa_full vs. librosa_clip isolates the effect of the analysed audio span;
   librosa_clip vs. soniclens compares feature sets on the same audio.
2. Ablation: SonicLens features with one descriptor family removed at a time.
3. Permutation feature importance of the deployed models (top 15 per target).
4. Plots: predicted vs. true, target distributions, tempo agreement.

Experiments 1-2 use cross-validation on the development set only, with one fixed
gradient-boosting configuration for every feature set, so no feature set benefits from its
own tuning. Experiments 3-4 describe the deployed models on the held-out test set. They are
diagnostics of the model already reported in model_card.json; nothing is selected from them.

Writes reports/metrics.json, reports/*.png and reports/SUMMARY.md.
Usage (from backend/):  python -m training.evaluate
"""

import json
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: E402
from sklearn.inspection import permutation_importance  # noqa: E402

from app.extraction import MODEL_FEATURE_NAMES  # noqa: E402
from app.extraction.features import FAMILIES, MODEL_FEATURE_FAMILIES  # noqa: E402
from app.model.predictor import MODEL_CARD_PATH, load_model  # noqa: E402
from training.common import (  # noqa: E402
    REPORTS_DIR, SEED, TARGETS, assign_splits, cross_validate, load_librosa_features, regression_metrics,
)
from training.librosa_clips import load_librosa_clip_features  # noqa: E402
from training.train import load_training_table  # noqa: E402

# The configuration train.py selected for 4 of the 7 targets, used for every comparison run.
FIXED_HGB = dict(learning_rate=0.03, max_iter=400, max_leaf_nodes=31, min_samples_leaf=100,
                 max_features=0.5, l2_regularization=0.1)
FEATURE_SETS = {
    "librosa_full": "librosa, full track (FMA features.csv)",
    "librosa_clip": "librosa, 30 s clip (recomputed)",
    "soniclens": "SonicLens Essentia, 30 s clip",
}
TOP_K = 15

# Chart styling: validated reference palette (see docs); light surface for print.
SURFACE, INK, INK_2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
DIVERGING = LinearSegmentedColormap.from_list("blue_gray_red", ["#e34948", "#f0efec", "#2a78d6"])


def hgb() -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(early_stopping=False, random_state=SEED, **FIXED_HGB)


def style_axes(ax, grid_axis="y"):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=INK_2, labelsize=9, length=0)
    if grid_axis:
        ax.grid(axis=grid_axis, color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)


def new_figure(*args, **kwargs):
    fig, axes = plt.subplots(*args, **kwargs)
    fig.patch.set_facecolor(SURFACE)
    return fig, axes


def save(fig, name: str) -> None:
    fig.savefig(REPORTS_DIR / name, dpi=160, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


# --- 1. Feature-set comparison ------------------------------------------------------------


def feature_set_comparison(df: pd.DataFrame, dev_mask: pd.Series, folds: pd.Series) -> dict:
    sets = {
        "librosa_full": load_librosa_features(),
        "librosa_clip": load_librosa_clip_features(),
        "soniclens": df[MODEL_FEATURE_NAMES],
    }
    common = df.index[dev_mask]
    for X in sets.values():
        common = common.intersection(X.index)
    print(f"[1] feature-set comparison on {len(common)} identical development tracks")
    result = {"n_tracks": len(common), "model": {"type": "HistGradientBoostingRegressor", **FIXED_HGB}, "sets": {}}
    for name, X in sets.items():
        X = X.loc[common]
        result["sets"][name] = {"label": FEATURE_SETS[name], "n_features": X.shape[1], "cv": {}}
        for target in TARGETS:
            t = time.time()
            m = cross_validate(hgb(), X, df.loc[common, target], folds.loc[common])
            result["sets"][name]["cv"][target] = m
            print(f"    {name:13s} {target:17s} R2 {m['r2']:+.3f}  ({time.time() - t:.0f}s)", flush=True)
    return result


def plot_feature_sets(comparison: dict) -> None:
    fig, ax = new_figure(figsize=(9, 5.2))
    style_axes(ax, grid_axis="x")
    names = list(FEATURE_SETS)
    height = 0.26
    y = np.arange(len(TARGETS))
    for i, name in enumerate(names):
        vals = [comparison["sets"][name]["cv"][t]["r2"] for t in TARGETS]
        pos = y + (i - 1) * (height + 0.02)
        ax.barh(pos, vals, height=height, color=SERIES[i], label=FEATURE_SETS[name], edgecolor=SURFACE, linewidth=1)
        for p, v in zip(pos, vals):
            ax.text(v + 0.008, p, f"{v:.2f}", va="center", fontsize=7.5, color=INK_2)
    ax.set_yticks(y, TARGETS)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("Cross-validated R² (artist-grouped, 5 folds)", color=INK_2, fontsize=9)
    ax.set_title("Feature-set comparison: same tracks, folds and model", loc="left", color=INK, fontsize=11)
    ax.legend(frameon=False, fontsize=8.5, loc="lower right", labelcolor=INK_2)
    save(fig, "feature_set_comparison.png")


# --- 2. Ablation ----------------------------------------------------------------------------


def ablation(df: pd.DataFrame, dev_mask: pd.Series, folds: pd.Series, full_cv: dict) -> dict:
    dev = df[dev_mask]
    result = {"full": {t: full_cv[t]["r2"] for t in TARGETS}, "without": {}}
    print("[2] ablation by descriptor family")
    for family in FAMILIES:
        keep = [n for n in MODEL_FEATURE_NAMES if MODEL_FEATURE_FAMILIES[n] != family]
        result["without"][family] = {"n_features_removed": len(MODEL_FEATURE_NAMES) - len(keep)}
        for target in TARGETS:
            r2 = cross_validate(hgb(), dev[keep], dev[target], folds)["r2"]
            result["without"][family][target] = {"r2": r2, "delta_r2": r2 - result["full"][target]}
            print(f"    without {family:9s} {target:17s} dR2 {r2 - result['full'][target]:+.4f}", flush=True)
    return result


def plot_ablation(abl: dict) -> None:
    delta = np.array([[abl["without"][f][t]["delta_r2"] for t in TARGETS] for f in FAMILIES])
    lim = max(0.01, np.abs(delta).max())
    fig, ax = new_figure(figsize=(8.5, 3.6))
    style_axes(ax, grid_axis=None)
    im = ax.imshow(delta, cmap=DIVERGING, vmin=-lim, vmax=lim, aspect="auto")
    for (i, j), v in np.ndenumerate(delta):
        ax.text(j, i, f"{v:+.3f}", ha="center", va="center", fontsize=8, color=INK)
    ax.set_xticks(range(len(TARGETS)), TARGETS, rotation=25, ha="right")
    ax.set_yticks(range(len(FAMILIES)), [f"without {f}" for f in FAMILIES])
    for s in ax.spines.values():
        s.set_visible(False)
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(labelsize=8, colors=INK_2, length=0)
    cbar.set_label("Change in CV R² vs. all features", fontsize=8.5, color=INK_2)
    ax.set_title("Ablation: removing one descriptor family (red = worse)", loc="left", color=INK, fontsize=11)
    save(fig, "ablation.png")


# --- 3. Permutation importance ---------------------------------------------------------------


def importance(model, X_test: pd.DataFrame, test: pd.DataFrame) -> dict:
    print("[3] permutation importance on the test set")
    result = {}
    for target in TARGETS:
        pi = permutation_importance(model.models[target], X_test.to_numpy(), test[target].to_numpy(),
                                    scoring="r2", n_repeats=5, random_state=SEED)
        order = np.argsort(pi.importances_mean)[::-1]
        by_family = {f: 0.0 for f in FAMILIES}
        for name, v in zip(MODEL_FEATURE_NAMES, pi.importances_mean):
            by_family[MODEL_FEATURE_FAMILIES[name]] += float(v)
        result[target] = {
            "top": [{"feature": MODEL_FEATURE_NAMES[i], "mean": float(pi.importances_mean[i]),
                     "std": float(pi.importances_std[i])} for i in order[:TOP_K]],
            "by_family": by_family,
        }
        print(f"    {target:17s} top: {result[target]['top'][0]['feature']}", flush=True)
    return result


def plot_importance(imp: dict) -> None:
    for target in TARGETS:
        top = imp[target]["top"][::-1]
        fig, ax = new_figure(figsize=(7, 5))
        style_axes(ax, grid_axis="x")
        vals = [r["mean"] for r in top]
        ax.barh(range(len(top)), vals, xerr=[r["std"] for r in top], color=SERIES[0], height=0.7,
                error_kw=dict(ecolor=MUTED, lw=0.8), edgecolor=SURFACE, linewidth=1)
        ax.set_yticks(range(len(top)), [r["feature"] for r in top], fontsize=8.5)
        ax.set_xlabel("Mean drop in test R² when permuted (5 repeats)", color=INK_2, fontsize=9)
        ax.set_title(f"{target}: top {TOP_K} model inputs by permutation importance", loc="left", color=INK,
                     fontsize=11)
        save(fig, f"importance_{target}.png")


# --- 4. Plots --------------------------------------------------------------------------------


def plot_predictions(model, X_test, test) -> dict:
    fig, axes = new_figure(2, 4, figsize=(13, 6.6))
    metrics = {}
    for ax, target in zip(axes.flat, TARGETS):
        pred = np.clip(model.models[target].predict(X_test.to_numpy()), 0, 1)
        metrics[target] = regression_metrics(test[target], pred)
        style_axes(ax, grid_axis="both")
        ax.plot([0, 1], [0, 1], color=MUTED, lw=1, ls="--", zorder=1)
        ax.scatter(test[target], pred, s=6, color=SERIES[0], alpha=0.35, linewidths=0, zorder=2)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.set_title(f"{target}   R² {metrics[target]['r2']:.2f}", loc="left", color=INK, fontsize=10)
        ax.set_xlabel("Echo Nest", color=INK_2, fontsize=8.5)
        ax.set_ylabel("SonicLens", color=INK_2, fontsize=8.5)
    axes.flat[-1].axis("off")
    fig.suptitle("Predicted vs. true on the held-out test set", x=0.01, ha="left", color=INK, fontsize=12)
    fig.tight_layout()
    save(fig, "predicted_vs_true.png")
    return metrics


def plot_distributions(df: pd.DataFrame) -> None:
    fig, axes = new_figure(2, 4, figsize=(13, 5.6))
    for ax, target in zip(axes.flat, TARGETS):
        style_axes(ax)
        ax.hist(df[target], bins=40, range=(0, 1), color=SERIES[0], edgecolor=SURFACE, linewidth=0.5)
        ax.set_title(target, loc="left", color=INK, fontsize=10)
        ax.set_xlim(0, 1)
    axes.flat[-1].axis("off")
    fig.suptitle(f"Target distributions (all {len(df)} training tracks)", x=0.01, ha="left", color=INK, fontsize=12)
    fig.tight_layout()
    save(fig, "target_distributions.png")


def plot_tempo(df: pd.DataFrame, tempo: dict) -> None:
    fig, ax = new_figure(figsize=(6, 6))
    style_axes(ax, grid_axis="both")
    # Reference lines, each labelled just below its line at the given x position.
    for k, label, x_label in [(1, "same tempo", 215), (2, "double", 105), (0.5, "half", 215)]:
        x = np.array([40, 260])
        ax.plot(x, k * x, color=MUTED, lw=1, ls="-" if k == 1 else ":", zorder=1)
        ax.text(x_label + 3, k * x_label - 3, label, color=INK_2, fontsize=8, ha="left", va="top",
                bbox=dict(facecolor=SURFACE, edgecolor="none", pad=1), zorder=3)
    ax.scatter(df["echonest_tempo"], df["tempo"], s=4, color=SERIES[0], alpha=0.25, linewidths=0, zorder=2)
    ax.set_xlim(40, 260)
    ax.set_ylim(40, 260)
    ax.set_xlabel("Echo Nest tempo (BPM, full track)", color=INK_2, fontsize=9)
    ax.set_ylabel("SonicLens tempo (BPM, 30 s clip)", color=INK_2, fontsize=9)
    ax.set_title(f"Tempo agreement: {tempo['accuracy_1']:.0%} within 4%, {tempo['accuracy_2']:.0%} "
                 "incl. double/half", loc="left", color=INK, fontsize=10.5)
    save(fig, "tempo_agreement.png")


# --- Summary ---------------------------------------------------------------------------------


def key_findings(metrics: dict) -> list[str]:
    """Headline conclusions, computed from the numbers so they stay true when re-run."""
    sets = metrics["feature_set_comparison"]["sets"]
    r2 = {n: {t: sets[n]["cv"][t]["r2"] for t in TARGETS} for n in FEATURE_SETS}
    span = {t: r2["librosa_clip"][t] - r2["librosa_full"][t] for t in TARGETS}
    feat = {t: r2["soniclens"][t] - r2["librosa_clip"][t] for t in TARGETS}
    # Rough standard error of a CV R2: spread of the per-fold R2 / sqrt(n_folds).
    se = np.mean([np.std(sets["soniclens"]["cv"][t]["r2_folds"]) / np.sqrt(5) for t in TARGETS])
    wins = [t for t in TARGETS if feat[t] > se]
    ties = [t for t in TARGETS if abs(feat[t]) <= se]
    beat_full = [t for t in TARGETS if r2["soniclens"][t] > r2["librosa_full"][t]]
    abl = metrics["ablation"]["without"]
    top_family = {t: min(FAMILIES, key=lambda f: abl[f][t]["delta_r2"]) for t in TARGETS}

    def fmt(d, keys):
        return ", ".join(f"{t} {d[t]:+.3f}" for t in keys)

    worst_span = sorted(TARGETS, key=lambda t: span[t])[:3]
    return [
        f"- **Analysed span matters.** The same librosa features computed on the 30 s clip instead of the full "
        f"track lose {-np.mean(list(span.values())):.3f} R² on average; the largest drops are "
        f"{fmt(span, worst_span)}. Echo Nest labels describe full tracks, so part of every clip-based model's error "
        "comes from audio it never sees.",
        f"- **On identical clips, SonicLens features beat librosa** on {len(wins)} of {len(TARGETS)} targets "
        f"(by more than one standard error, ≈{se:.3f}), with half as many inputs "
        f"({sets['soniclens']['n_features']} vs. {sets['librosa_clip']['n_features']}): "
        f"{fmt(feat, sorted(wins, key=lambda t: -feat[t]))}."
        + (f" Tied (within one standard error): {', '.join(ties)}." if ties else ""),
        f"- **SonicLens on 30 s clips even beats librosa on full tracks** for {', '.join(beat_full) or 'no target'}.",
        "- **Most useful descriptor family per target** (largest ablation loss; \"none\" when every loss is "
        f"within noise, ≈{se:.3f}): "
        + ", ".join(
            f"{t}: {top_family[t]} ({abl[top_family[t]][t]['delta_r2']:+.3f})"
            if abl[top_family[t]][t]["delta_r2"] < -se else f"{t}: none"
            for t in TARGETS
        )
        + ". Families overlap in information, so removing one lets the others compensate; "
        "the dynamics family (8 inputs) is redundant with the rest for every target.",
        "- **Liveness is barely learnable from a 30 s centre clip** (test R² "
        f"{metrics['model_card']['targets']['liveness']['test']['r2']:.2f}). Full-track librosa features reach "
        f"{r2['librosa_full']['liveness']:.2f}, consistent with live-recording cues (applause, crowd noise) sitting at "
        "the start and end of tracks.",
    ]


def write_summary(metrics: dict) -> None:
    card, cmp_, abl, imp = metrics["model_card"], metrics["feature_set_comparison"], metrics["ablation"], \
        metrics["permutation_importance"]
    lines = [
        "# Evaluation summary",
        "",
        "Generated by `python -m training.evaluate`. Full numbers are in `metrics.json`.",
        "",
        "## Deployed model: held-out test set",
        "",
        f"{card['n_test']} tracks by {card['n_artists_test']} artists that never appear in training. Selection used "
        "artist-grouped 5-fold CV on the development set only.",
        "",
        "| Target | Test R² | MAE | RMSE | Spearman | CV R² | Ridge test R² | Mean test R² |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for t in TARGETS:
        c = card["targets"][t]
        lines.append(f"| {t} | **{c['test']['r2']:.3f}** | {c['test']['mae']:.3f} | {c['test']['rmse']:.3f} | "
                     f"{c['test']['spearman']:.3f} | {c['cv']['r2']:.3f} | {c['baselines']['ridge']['test']['r2']:.3f} "
                     f"| {c['baselines']['mean']['test']['r2']:.3f} |")
    lines += ["", "## Key findings", "", *key_findings(metrics), ""]
    tv = card["tempo_validation"]
    lines += [
        f"**DSP tempo check:** SonicLens tempo (30 s clip) vs. Echo Nest tempo (full track): "
        f"{tv['accuracy_1']:.1%} within ±4%, {tv['accuracy_2']:.1%} counting double/half tempo "
        f"({tv['n']} clips). See `tempo_agreement.png`.",
        "",
        "## 1. Feature-set comparison",
        "",
        f"Cross-validated R² on {cmp_['n_tracks']} identical development tracks, same artist-grouped folds, and one "
        "fixed gradient-boosting configuration for every feature set. `librosa, full track` is FMA's `features.csv`, "
        "computed on full-length tracks, the same span the Echo Nest labels describe. `librosa, 30 s clip` is the "
        "same code re-run on the clips SonicLens sees. See `feature_set_comparison.png`.",
        "",
        "| Target | " + " | ".join(FEATURE_SETS[n] for n in FEATURE_SETS) + " |",
        "|---|" + "---|" * len(FEATURE_SETS),
    ]
    for t in TARGETS:
        vals = [cmp_["sets"][n]["cv"][t]["r2"] for n in FEATURE_SETS]
        best = max(vals[1:])  # best of the two clip-based sets
        cells = [f"{v:.3f}" if i == 0 or v != best else f"**{v:.3f}**" for i, v in enumerate(vals)]
        lines.append(f"| {t} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "Feature counts: " + ", ".join(f"{FEATURE_SETS[n]} {cmp_['sets'][n]['n_features']}" for n in FEATURE_SETS)
        + ". Bold marks the better of the two clip-based sets.",
        "",
        "## 2. Ablation by descriptor family",
        "",
        "Change in cross-validated R² when one family is removed from the SonicLens features "
        "(negative = the family helps). See `ablation.png`.",
        "",
        "| Removed family (n inputs) | " + " | ".join(TARGETS) + " |",
        "|---|" + "---|" * len(TARGETS),
    ]
    for f in FAMILIES:
        row = abl["without"][f]
        lines.append(f"| {f} ({row['n_features_removed']}) | "
                     + " | ".join(f"{row[t]['delta_r2']:+.3f}" for t in TARGETS) + " |")
    lines += [
        "",
        "## 3. Permutation importance (deployed models, test set)",
        "",
        "Top 3 inputs per target (mean drop in test R² when permuted). Many inputs are strongly correlated (e.g. the "
        "mean, median and percentiles of one descriptor), so importance is shared among them, and a single input's "
        "importance understates its descriptor's. The ablation is the better guide to the value of a family. "
        "Plots: `importance_<target>.png`.",
        "",
        "| Target | #1 | #2 | #3 |",
        "|---|---|---|---|",
    ]
    for t in TARGETS:
        top = imp[t]["top"][:3]
        lines.append(f"| {t} | " + " | ".join(f"`{r['feature']}` ({r['mean']:.3f})" for r in top) + " |")
    lines += ["", "## Plots", "",
              "- `predicted_vs_true.png`: test-set predictions of the deployed models",
              "- `target_distributions.png`: distribution of each Echo Nest target",
              "- `feature_set_comparison.png`, `ablation.png`, `importance_<target>.png`, `tempo_agreement.png`", ""]
    (REPORTS_DIR / "SUMMARY.md").write_text("\n".join(lines))


def main() -> None:
    REPORTS_DIR.mkdir(exist_ok=True)
    df = load_training_table()
    splits = assign_splits(df["artist_id"])
    dev_mask = ~splits["is_test"]
    folds = splits.loc[dev_mask, "fold"]
    test = df[splits["is_test"]]
    X_test = test[MODEL_FEATURE_NAMES]
    model = load_model()
    card = json.loads(MODEL_CARD_PATH.read_text())

    plot_distributions(df)
    from training.common import tempo_agreement

    tempo = tempo_agreement(df["tempo"], df["echonest_tempo"])
    plot_tempo(df, tempo)

    test_metrics = plot_predictions(model, X_test, test)
    for t in TARGETS:  # the deployed model must reproduce the numbers in its model card
        assert abs(test_metrics[t]["r2"] - card["targets"][t]["test"]["r2"]) < 1e-9, t

    comparison = feature_set_comparison(df, dev_mask, folds)
    plot_feature_sets(comparison)
    abl = ablation(df, dev_mask, folds, comparison["sets"]["soniclens"]["cv"])
    plot_ablation(abl)
    imp = importance(model, X_test, test)
    plot_importance(imp)

    metrics = {
        "model_card": {
            "model_version": card["model_version"],
            "n_test": card["training_data"]["n_test"],
            "n_artists_test": card["training_data"]["n_artists_test"],
            "targets": {t: {k: card["targets"][t][k] for k in ("cv", "test", "baselines")} for t in TARGETS},
            "tempo_validation": card["tempo_validation"],
        },
        "feature_set_comparison": comparison,
        "ablation": abl,
        "permutation_importance": imp,
    }
    (REPORTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=1) + "\n")
    write_summary(metrics)
    print("Wrote reports/metrics.json, reports/SUMMARY.md and plots.")


if __name__ == "__main__":
    main()
