import os
from fractions import Fraction

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pandas._libs.parsers import CategoricalDtype

from scripts.plots.utils import format_ci, print_df


def set_paper_plot_style():
    sns.set_context("paper", font_scale=1.2)  # "paper" context + scaling
    sns.set_style("darkgrid")  # "darkgrid" style)

    label = 16

    plt.rcParams.update({
        "figure.figsize": (12, 8),
        "axes.titlesize": label+8,
        "axes.labelsize": label+8,
        "xtick.labelsize": label+8,
        "ytick.labelsize": label+8,
        "legend.fontsize": label,
        "lines.linewidth": 4,
        "lines.markersize": 6,
        "savefig.dpi": 300,
    })
    sns.set_palette("colorblind")


def static_plot(df, shift_name, metric, with_age=True, show_conf_interval=True, save_path=None):
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
    elif shift_name in ['interpolate', 'channel_rotation']:
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
        errorbar='ci' if show_conf_interval else None,
        err_style='band',
        estimator="mean",
        marker="D"
    )

    plt.title(f"{shift_name.title()} Shift {'' if with_age else 'w/o age'}")

    replace_xtick_zero_with_baseline(valid_values=df_plot["shift_intensity"].unique())

    if shift_name == "slow_drift":
        plt.xlabel("Shift intensity (maxDrift_numSinus)")
    elif shift_name in ["interpolate", "channel_rotation"]:
        plt.xlabel("Shift intensity (% of channels)")
    elif shift_name == "phase_shift":
        plt.xlabel("Shift intensity (radians)")
    elif shift_name == "amplitude":
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
    else:
        plt.ylabel(metric.replace("_", " ").capitalize())
    plt.legend(loc='upper left' if metric == "brier" or metric == "ece" else 'lower left')
    plt.grid(True)
    plt.tight_layout()
    if save_path:
        shift_path = os.path.join(save_path, shift_name)
        # Create the directory if it doesn't exist
        os.makedirs(shift_path, exist_ok=True)
        save_name = f"{shift_path}/{shift_name}_{metric}.pdf"
        plt.savefig(save_name, dpi=300, bbox_inches='tight', format="pdf")
    else:
        plt.show()

    plt.close()


def bandstop_plot(df, metric, with_age=True, save_path=None):
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

    sns.boxplot(
        data=df_plot,
        x="shift_name",
        y=metric,
        hue="ensemble_type",
        dodge=True,
        showfliers=False
    )
    plt.title(f"Bandstop Shift {'' if with_age else 'w/o age'}")

    ax = plt.gca()
    tick_labels_raw = [label.get_text() for label in ax.get_xticklabels()]
    tick_labels_cleaned = [label.replace("bandstop_", "").replace("_without_age", "") for label in tick_labels_raw]
    ax.set_xticklabels(tick_labels_cleaned, rotation=45, ha='right', rotation_mode="anchor")
    
    if metric == "auc":
        plt.ylabel("AUC")
    else:
        plt.ylabel(metric.replace("_", " ").capitalize())

    plt.xlabel("Frequency band")

    plt.legend(loc='upper left' if metric == "brier" or metric == "ece" else 'lower left')
    plt.grid(True)
    plt.tight_layout()
    shift_name = "bandstop"
    if save_path:
        shift_path = os.path.join(save_path, shift_name)
        # Create the directory if it doesn't exist
        os.makedirs(shift_path, exist_ok=True)
        save_name = f"{shift_path}/{shift_name}_{metric}.pdf"
        plt.savefig(save_name, dpi=300, bbox_inches='tight', format="pdf")
    else:
        plt.show()

    plt.close()


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
                labels.append("baseline")
            else:
                labels.append(v)
        else:
            if abs(float(v)) == 0.0:
                labels.append("baseline")
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

    