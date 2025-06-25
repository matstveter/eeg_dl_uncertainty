import os
from fractions import Fraction

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pandas._libs.parsers import CategoricalDtype
from scipy import stats

from scripts.plots.utils import format_ci, print_df


def set_paper_plot_style():
    sns.set_context("paper", font_scale=1.2)  # "paper" context + scaling
    sns.set_style("darkgrid")  # "darkgrid" style)

    label = 36

    plt.rcParams.update({
        "figure.figsize": (12, 8),
        "axes.titlesize": label+8,
        "axes.labelsize": label+8,
        "xtick.labelsize": label+8,
        "ytick.labelsize": label+8,
        "legend.fontsize": label+8,
        "lines.linewidth": 4,
        "lines.markersize": 6,
        "savefig.dpi": 300,
    })
    sns.set_palette("colorblind")


def static_plot(df, shift_name, metric, hue_order, palette,
                with_age=True, show_conf_interval=True, save_path=None, no_axis=False,
                no_legend=False):
    df_baseline = df[df["shift_name"] == "baseline"].copy()

    if shift_name in ["slow_drift", "peak_shift", "timewarp", "circular_shift"]:
        df_baseline["shift_intensity"] = df_baseline["shift_intensity"].astype(str)
    else:
        df_baseline["shift_intensity"] = df_baseline["shift_intensity"].astype(float)

    shift_key = shift_name if with_age else f"{shift_name}_without_age"
    df_shift = df[df["shift_name"] == shift_key].copy()

    if shift_name not in ["slow_drift", "peak_shift", "timewarp", "circular_shift"]:
        df_shift["shift_intensity"] = df_shift["shift_intensity"].astype(float)

    ensemble_keys = df_shift[["ensemble_type", "model_key"]].drop_duplicates()
    df_baseline = pd.merge(df_baseline, ensemble_keys, on=["ensemble_type", "model_key"])

    df_plot = pd.concat([df_baseline, df_shift], ignore_index=True)

    plt.figure()

    if shift_name in ['amplitude', 'gaussian']:
        df_plot["shift_intensity"] = df_plot["shift_intensity"].apply(lambda x: f"{x:.4g}")
    elif shift_name in ['interpolate', 'rotation']:
        # convert to percentage
        df_plot["shift_intensity"] = df_plot["shift_intensity"].apply(lambda x: f"{x * 100:.4g}")
    elif shift_name in ['phase_shift']:
        # convert to radians
        df_plot["shift_intensity"] = df_plot["shift_intensity"].apply(format_as_pi_fraction)

    if shift_name == "amplitude":
        # Define the desired order of categories
        desired_order = ['0.001', '0.01', '0.05', '0.1', '0.25',
                         '0.5', '0.75', '0', '1.5', '2', '2.5', '4', '6', '8']

        # Convert to CategoricalDtype with the desired order
        cat_dtype = CategoricalDtype(categories=desired_order, ordered=True)
        df_plot["shift_intensity"] = df_plot["shift_intensity"].astype(cat_dtype)
    elif shift_name == "slow_drift":
        desired_order = ['0.0', '0.1_1', '0.25_2', '0.5_3', '1.0_5', '2.0_7', '3.0_10', '4.0_15']
        df_plot = df_plot[df_plot["shift_intensity"].isin(desired_order)].copy()

        # Convert to CategoricalDtype with the desired order
        cat_dtype = CategoricalDtype(categories=desired_order, ordered=True)
        df_plot["shift_intensity"] = df_plot["shift_intensity"].astype(cat_dtype)
    elif shift_name == "peak_shift":
        desired_order = ['0.0', '0.1',  '0.25', '0.5',  '1.0',   '1.5',  '2.0',   '3.0',  '4.0',   '6.0',   '8.0']
        cat_dtype = CategoricalDtype(categories=desired_order, ordered=True)
        df_plot["shift_intensity"] = df_plot["shift_intensity"].astype(cat_dtype)
    elif shift_name == "circular_shift":

        desired_order = ['0.0', '200', '400', '800', '1600', '2000', '3000', '4000', '6000']

        cat_dtype = CategoricalDtype(categories=desired_order, ordered=True)
        df_plot["shift_intensity"] = df_plot["shift_intensity"].astype(cat_dtype)

    sns.lineplot(
        data=df_plot,
        x="shift_intensity",
        y=metric,
        hue="ensemble_type",
        hue_order=hue_order,
        palette=palette,
        errorbar='ci' if show_conf_interval else None,
        err_style='band',
        estimator="mean",
        marker="D"
    )

    # plt.title(f"{shift_name.title()} Shift {'' if with_age else 'w/o age'}")

    replace_xtick_zero_with_baseline(valid_values=df_plot["shift_intensity"].unique())

    if shift_name == "slow_drift":
        plt.xlabel("Shift intensity (maxDrift_numSinus)")
    elif shift_name in ["interpolate", "rotation"]:
        plt.xlabel("Shift intensity (% of channels)")
    elif shift_name == "phase_shift":
        plt.xlabel("Shift intensity (radians)")
    elif shift_name == "amplitude":
        plt.axvline(x=7, linestyle='--', color='gray', linewidth=1)
        plt.xlabel("Shift intensity (factor)")
    elif shift_name == "gaussian":
        plt.xlabel("Shift intensity (σ)")
    elif shift_name == "peak_shift":
        plt.xlabel("Shift intensity (Hz)")
    elif shift_name == "timewarp":
        plt.xlabel("Warp ratio")
    elif shift_name == "circular_shift":
        plt.xlabel("Shift intensity (time points)")
    else:
        plt.xlabel("Shift intensity")

    if metric == "auc":
        plt.ylabel("AUC")
        plt.ylim(0.45, 0.9)
    elif "auc_class" in metric:
        plt.ylabel(metric.replace("_", " ").capitalize())
        plt.ylim(0.45, 0.9)
    elif metric == "ece":
        plt.ylabel("ECE")
        plt.ylim(0.0, 0.55)
    elif metric == "accuracy":
        plt.ylabel("Accuracy")
        plt.ylim(0.25, 0.75)
    elif metric == "brier":
        plt.ylabel("Brier score")
        plt.ylim(0.4, 1.2)
    else:
        plt.ylabel(metric.replace("_", " ").capitalize())

    if no_legend:
        # Remove the legend
        plt.legend().remove()
    else:    
        plt.legend(loc='upper left' if metric == "brier" or metric == "ece" else 'lower left')

    if no_axis:
        plt.gca().tick_params(left=False, labelleft=False)
        plt.ylabel("")

    plt.grid(True)
    plt.tight_layout()
    if save_path:
        if no_axis:
            new_shift_name = f"{shift_name}_no_axis"
        else:
            new_shift_name = f"{shift_name}"
        save_name = f"{save_path}/{new_shift_name}_{metric}.pdf"
        plt.savefig(save_name, dpi=300, format="pdf", bbox_inches=None, pad_inches=0.1)
    else:
        plt.show()

    plt.close()


def bandstop_plot(df, metric, palette, with_age=True, save_path=None, no_legend=False):
    df_baseline = df[df["shift_name"] == "baseline"].copy()

    shift_key = "bandstop"

    df_shift = df[df["shift_name"].str.contains(shift_key)].copy()

    # The above might include "without_age" so we need to filter it out if shift_key is "bandstop"
    if with_age:
        df_shift = df_shift[~df_shift["shift_name"].str.contains("without_age")]
    else:
        df_shift = df_shift[df_shift["shift_name"].str.contains("without_age")]

    df_shift = df_shift[df_shift["shift_intensity"] == 1.0]

    ensemble_keys = df_shift[["ensemble_type", "model_key"]].drop_duplicates()
    df_baseline = pd.merge(df_baseline, ensemble_keys, on=["ensemble_type", "model_key"])

    df_plot = pd.concat([df_baseline, df_shift], ignore_index=True)

    # calculate_metrics(df=df_plot, metric=metric)
    # return

    plt.figure(figsize=(24, 12))

    ax = sns.boxplot(
        data=df_plot,
        x="shift_name",
        y=metric,
        hue="ensemble_type",
        dodge=True,
        showfliers=False
    )

    # get number of unique shifts
    n = len(df_plot["shift_name"].unique())

    # choose two alternating colors
    colors = ["#e0e0e0", "#b0b0b0"]

    for i in range(n):
        ax.axvspan(i - 0.5, i + 0.5,
                   facecolor=colors[i % 2],
                   alpha=0.5,
                   zorder=0)

    sns.boxplot(
        data=df_plot,
        x="shift_name",
        y=metric,
        hue="ensemble_type",
        palette=palette,
        dodge=True,
        showfliers=False
    )

    # plt.title(f"Bandstop Shift {'' if with_age else 'w/o age'}")

    ax = plt.gca()
    tick_labels_raw = [label.get_text() for label in ax.get_xticklabels()]
    tick_labels_cleaned = [label.replace("bandstop_", "").replace("_without_age", "") for label in tick_labels_raw]

    # Change the baseline tick to test
    tick_labels_cleaned = ["Test" if label == "baseline" else label for label in tick_labels_cleaned]

    ax.set_xticklabels(tick_labels_cleaned, rotation=45, ha='right', rotation_mode="anchor")
    
    if metric == "auc":
        plt.ylabel("AUC")
        plt.ylim(0.45, 0.9)
    else:
        plt.ylabel(metric.replace("_", " ").capitalize())

    plt.xlabel("Frequency band")
    if no_legend:
        # Remove the legend
        plt.legend().remove()
    else:    
        plt.legend(loc='upper left' if metric == "brier" or metric == "ece" else 'lower left')

    plt.grid(True)
    plt.tight_layout()
    shift_name = "bandstop"
    if save_path:
        save_name = f"{save_path}/{shift_name}_{metric}.pdf"
        plt.savefig(save_name, dpi=300, format="pdf", bbox_inches=None, pad_inches=0.1)
    else:
        plt.show()

    plt.close()


def calculate_metrics(df, metric):
    """Calculate means and confidence intervals for baseline and shift types."""
    ensemble_types = df["ensemble_type"].unique()
    results = []

    for ensemble_type in ensemble_types:
        df_ensemble = df[df["ensemble_type"] == ensemble_type].copy()

        # Calculate for baseline
        df_baseline = df_ensemble[df_ensemble["shift_name"] == "baseline"]
        baseline_mean, baseline_ci_lower, baseline_ci_upper = calculate_confidence_intervals(df_baseline[metric])

        # Store baseline results
        results.append({
            'ensemble_type': ensemble_type,
            'shift_name': 'baseline',
            'mean': baseline_mean,
            'ci_lower': baseline_ci_lower,
            'ci_upper': baseline_ci_upper
        })

        # Unique shift types
        shift_types = df_ensemble['shift_name'].unique()
        for shift_type in shift_types:
            if shift_type == "baseline":
                continue  # Skip baseline as we already calculated it

            df_shift = df_ensemble[df_ensemble["shift_name"] == shift_type]

            # Calculate mean and CI for shift
            shift_mean, shift_ci_lower, shift_ci_upper = calculate_confidence_intervals(df_shift[metric])

            # Perform t-test against the baseline
            p_value = perform_t_test(df_shift[metric], df_baseline[metric])

            # Store shift results
            results.append({
                'ensemble_type': ensemble_type,
                'shift_name': shift_type,
                'mean': shift_mean,
                'ci_lower': shift_ci_lower,
                'ci_upper': shift_ci_upper,
                'p_value': p_value
            })

    # Create a DataFrame with the results
    results_df = pd.DataFrame(results)
    # print(results_df)
    significant_results = results_df[results_df['p_value'] < 0.05]
    print("Significant results:", significant_results)
    return results_df


def perform_t_test(shift_data, baseline_data):
    """Perform a t-test comparing shift data to baseline data."""
    t_stat, p_value = stats.ttest_ind(shift_data, baseline_data, nan_policy='omit')
    return p_value


def calculate_confidence_intervals(data, confidence=0.95):
    """Calculate the mean and confidence intervals for the given data."""
    if len(data) <= 1:
        return np.nan, np.nan, np.nan  # Handle cases with insufficient data
    mean = np.mean(data)
    sem = stats.sem(data)  # Standard error of the mean
    h = sem * stats.t.ppf((1 + confidence) / 2.0, len(data) - 1)  # Margin of error
    return mean, mean - h, mean + h  # Return mean and lower, upper bounds of the CI


def replace_xtick_zero_with_baseline(valid_values):
    """
    Sets custom x-ticks using valid_values.
    Replaces 0 or '0' with 'baseline'. Supports float and str values.
    """
    ax = plt.gca()
    xticks = valid_values
    labels = []

    for v in valid_values:
        if isinstance(v, str):
            if v.strip() == "0" or v.strip() == "0.0":
                labels.append("Test")
            else:
                labels.append(v)
        else:
            if abs(float(v)) == 0.0:
                labels.append("Test")
            else:
                labels.append(f"{float(v):.4g}")

    ax.set_xticks(xticks)
    ax.set_xticklabels(labels, rotation=45, ha='right', rotation_mode="anchor")


def format_as_pi_fraction(x):
    multiple = Fraction(x / np.pi).limit_denominator(16)
    if multiple == 0:
        return "0"
    elif multiple == 1:
        return "π"
    elif multiple.denominator == 1:
        return f"{multiple.numerator}π"
    else:
        return f"{multiple.numerator}π/{multiple.denominator}"


def investigate_class(df, shift_name, plot_per_class=False, compare_max=True):
    df_baseline = df[df["shift_name"] == "baseline"].copy()
    df_baseline["shift_intensity"] = df_baseline["shift_intensity"].astype(float)

    df_shift = df[(df["shift_name"] == shift_name)].copy()
    df_shift["shift_intensity"] = df_shift["shift_intensity"].astype(float)

    df_shift = df_shift[df_shift["shift_intensity"] == max(df_shift["shift_intensity"].unique())]

    ensemble_keys = df_shift[["ensemble_type", "model_key"]].drop_duplicates()
    df_baseline = pd.merge(df_baseline, ensemble_keys, on=["ensemble_type", "model_key"])

    df_plot = pd.concat([df_baseline, df_shift], ignore_index=True)

    if plot_per_class:
        # Melt into long format
        df_plot = pd.melt(
            df_plot,
            id_vars=["ensemble_type", "model_key", "shift_name"],
            value_vars=["auc_class_0", "auc_class_1", "auc_class_2"],
            var_name="class",
            value_name="metric"
        )

        # Rename classes
        df_plot["class"] = df_plot["class"].replace({
            "auc_class_0": "Normal",
            "auc_class_1": "MCI",
            "auc_class_2": "Dementia"
        })

        # Plot one subplot per class
        g = sns.catplot(
            data=df_plot,
            x="ensemble_type",
            y="metric",
            hue="shift_name",
            col="class",
            kind="box",
            height=4,
            aspect=1,
            sharey=True,
            showfliers=False
        )

        g.set_axis_labels("Ensemble", "AUC")
        g.set_titles("{col_name}")
        g.fig.subplots_adjust(top=0.85)
        g.fig.suptitle(f"AUC per Class – Shift: {shift_name.title()}", fontsize=14)
        plt.show()

    else:
        df_plot = pd.melt(
            df_plot,
            id_vars=["shift_name"],
            value_vars=["auc_class_0", "auc_class_1", "auc_class_2"],
            var_name="class",
            value_name="metric"
        )

        df_plot["class"] = df_plot["class"].replace({
            "auc_class_0": "Normal",
            "auc_class_1": "MCI",
            "auc_class_2": "Dementia"
        })

        # Tegn boxplot med class som x-akse
        sns.boxplot(
            data=df_plot,
            x="class",
            y="metric",
            hue="shift_name",
            dodge=True,
            showfliers=False
        )
        plt.title(f"{shift_name.title()} – AUC per class")
        plt.ylabel("AUC")
        plt.xlabel("Class")
        plt.grid(True, axis='y', linestyle='--', linewidth=0.5)
        plt.tight_layout()
        plt.show()


def investigate_age_effect(df, metric="auc"):
    cleaned_shift = [shift for shift in df["shift_name"].unique() if "without_age" not in shift]

    results = []

    actual_values = []

    for shift in cleaned_shift:
        print(f"Investigating age effect for shift: {shift}")

        df_with = df[df["shift_name"] == shift].copy()
        df_without = df[df["shift_name"] == f"{shift}_without_age"].copy()

        if shift == "slow_drift":
            # Slow drift is a dictionary, so we need to convert it to a string
            df_with["shift_intensity"] = df_with["shift_intensity"].astype(str)
            df_without["shift_intensity"] = df_without["shift_intensity"].astype(str)

        # Merge on the common columns
        merged = pd.merge(
            df_with,
            df_without,
            on=["ensemble_type", "model_key", "shift_intensity"],
            suffixes=("_with_age", "_without_age")
        )

        merged["delta_auc"] = abs(merged[f"auc_with_age"] - merged[f"auc_without_age"])
        merged["delta_accuracy"] = abs(merged[f"accuracy_with_age"] - merged[f"accuracy_without_age"])
        merged["delta_brier"] = abs(merged[f"brier_with_age"] - merged[f"brier_without_age"])
        merged['delta_auc_class_0'] = abs(merged[f"auc_class_0_with_age"] - merged[f"auc_class_0_without_age"])
        merged['delta_auc_class_1'] = abs(merged[f"auc_class_1_with_age"] - merged[f"auc_class_1_without_age"])
        merged['delta_auc_class_2'] = abs(merged[f"auc_class_2_with_age"] - merged[f"auc_class_2_without_age"])

        merged["shift_name"] = shift

        results.append(merged[["ensemble_type", "model_key", "shift_intensity", "shift_name",
                               "delta_auc", "delta_accuracy", "delta_brier", "delta_auc_class_0",
                               "delta_auc_class_1", "delta_auc_class_2"]])
        
        actual_values.append(merged[[
            "ensemble_type", "model_key", "shift_name",
            "auc_with_age", "auc_without_age", "accuracy_with_age", "accuracy_without_age",
            "brier_with_age", "brier_without_age", "auc_class_0_with_age", "auc_class_0_without_age",
            "auc_class_1_with_age", "auc_class_1_without_age", "auc_class_2_with_age", "auc_class_2_without_age"
        ]])

    diff_df = pd.concat(results, ignore_index=True)

    actual_auc_df = pd.concat(actual_values, ignore_index=True)
    baseline_auc_df = actual_auc_df[actual_auc_df["shift_name"] == "baseline"].reset_index(drop=True)

    # list all your metric columns
    metric_cols = [
        "auc_with_age", "auc_without_age",
        "accuracy_with_age", "accuracy_without_age",
        "brier_with_age", "brier_without_age",
        "auc_class_0_with_age", "auc_class_0_without_age",
        "auc_class_1_with_age", "auc_class_1_without_age",
        "auc_class_2_with_age", "auc_class_2_without_age"
    ]

    ci_str_df = (
        baseline_auc_df
        .groupby("ensemble_type")[metric_cols]
        .agg(lambda col: format_ci(col))
        .reset_index()
    )

    print_df(ci_str_df)


def plot_baseline_drift(df, metric, show_conf_interval=True, with_age=True):
    df_baseline = df[df["shift_name"] == "baseline"].copy()
    df_baseline["shift_intensity"] = df_baseline["shift_intensity"].astype(float)

    plotting_val = ['0.1_1', '0.25_2', '0.5_3', '1.0_5', '2.0_7', '3.0_10', '4.0_15']
    # Filter out the baseline shift
    df_shift = df[df["shift_name"] == "slow_drift"].copy()
    # Select only the plotting_val
    df_shift = df_shift[df_shift["shift_intensity"].isin(plotting_val)].copy()
    df_shift["shift_intensity"] = df_shift["shift_intensity"].astype(str)
    
    ensemble_keys = df_shift[["ensemble_type", "model_key"]].drop_duplicates()
    df_baseline = pd.merge(df_baseline, ensemble_keys, on=["ensemble_type", "model_key"])
    df_plot = pd.concat([df_baseline, df_shift], ignore_index=True)

    print(df_plot)
    
    sns.lineplot(
        data=df_plot,
        x="shift_intensity",
        y=metric,
        hue="ensemble_type",
        errorbar='ci' if show_conf_interval else None,
        err_style='band',
        estimator="mean",
    )

    plt.title(f"Slow Drift Shift {'with age' if with_age else 'w/o age'}")

    replace_xtick_zero_with_baseline(valid_values=df_plot["shift_intensity"].unique())

    plt.xlabel("Shift intensity")
    plt.ylabel(metric.replace("_", " ").capitalize())
    plt.legend(loc='upper left' if metric == "brier" else 'lower left')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def make_legend_plot(ensemble_types, palette, save_path=None, ncol=2):
    from matplotlib.lines import Line2D

    marker_style = {"marker": "D", "linestyle": "-"}

    ensemble_types = ['Augmentation', 'Deep Ensemble']

    # We only need the first and last color from the palette
    palette = [palette[0], palette[-1]]

    # 3) Build the standalone legend figure
    # fig, ax = plt.subplots(figsize=(12, 8))
    fig, ax = plt.subplots(figsize=(6, 1.5))  # wider and shorter for a single‐row
    ax.axis("off")  # hide axes

    # 4) Create proxy artists for each category
    handles = [
        Line2D([0], [0], color=palette[i], **marker_style)
        for i in range(len(ensemble_types))
    ]

    ax.legend(
        handles,
        ensemble_types,
        loc="center",
        ncol=ncol,  # two columns, will wrap into two rows for 3 entries
        frameon=False,
        columnspacing=1.0,  # adjust spacing between columns
        handletextpad=0.5  # adjust spacing between marker and text
    )

    plt.tight_layout()

    if save_path:
        # 6) Save the legend as its own PDF
        out_dir = os.path.join(save_path, f"legend_only")
        os.makedirs(out_dir, exist_ok=True)
        out_file = os.path.join(out_dir, f"{ncol}_legend.pdf")
        fig.savefig(out_file, dpi=300, bbox_inches='tight', pad_inches=0.0)
        plt.close(fig)
    else:
        plt.show()
        plt.close(fig)

