"""
Exploratory Data Analysis (EDA): Abalone Age Prediction
Course: First-Year Exploratory Data Analysis (BCA / B.Sc Data Science)
Author: Harsh Maheshkumar Bheda
Dataset: UCI Machine Learning Repository (4,177 instances, 9 attributes)
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import seaborn as sns


def main():
    print("=" * 65)
    print("      ABALONE EXPLORATORY DATA ANALYSIS (EDA) SCRIPT")
    print("=" * 65)

    # 1. LOAD DATASET
    cols = [
        "Sex",
        "Length",
        "Diameter",
        "Height",
        "Whole_weight",
        "Shucked_weight",
        "Viscera_weight",
        "Shell_weight",
        "Rings",
    ]

    csv_file = "abalone.csv"
    if not os.path.exists(csv_file):
        print(f"Error: '{csv_file}' not found in the current directory.")
        return

    df = pd.read_csv(csv_file, header=None, names=cols)
    print(f"\n[+] Dataset loaded successfully! Shape: {df.shape}")
    print(df.head())

    # 2. DATA QUALITY ASSESSMENT
    print("\n" + "=" * 65)
    print("1. DATA QUALITY ASSESSMENT")
    print("=" * 65)

    # Check for missing values
    null_count = df.isnull().sum().sum()
    print(f"• Total missing values (NaN/Null): {null_count}")

    # Check physical anomaly (Height = 0.0)
    zero_height = df[df["Height"] == 0]
    print(f"• Specimens with Height == 0.0 mm: {len(zero_height)} rows")
    if len(zero_height) > 0:
        print(
            zero_height[
                ["Sex", "Length", "Diameter", "Height", "Shell_weight", "Rings"]
            ]
        )

    # Check mass conservation discrepancy
    weight_err = df[
        df["Whole_weight"]
        < (df["Shucked_weight"] + df["Viscera_weight"] + df["Shell_weight"])
    ]
    print(
        f"• Specimens where Whole_weight < sum of parts: {len(weight_err)} rows (moisture loss)"
    )

    # 3. CALCULATE AGE & LIFE-STAGE COHORTS
    df["True_Age"] = df["Rings"] + 1.5

    conditions = [
        (df["Rings"] <= 8),
        (df["Rings"] >= 9) & (df["Rings"] <= 11),
        (df["Rings"] >= 12),
    ]
    cohort_names = ["Young (<=8)", "Mature (9-11)", "Old (12+)"]
    df["Age_Cohort"] = pd.cut(
    df["Rings"],
    bins=[-float("inf"), 8, 11, float("inf")],
    labels=["Young (<=8)", "Mature (9-11)", "Old (12+)"],
).astype(str)

    # 4. DESCRIPTIVE STATISTICS & GROUPING
    print("\n" + "=" * 65)
    print("2. DESCRIPTIVE STATISTICS")
    print("=" * 65)
    desc = df.describe().T[["mean", "std", "min", "50%", "max"]]
    desc["skew"] = df.skew(numeric_only=True)
    print(desc.round(4))

    print("\n• Sex Distribution:")
    print(df["Sex"].value_counts(normalize=True).mul(100).round(2).astype(str) + "%")

    print("\n• Summary by Cohort:")
    cohort_summary = df.groupby("Age_Cohort").agg(
        Count=("Rings", "count"),
        Avg_Length=("Length", "mean"),
        Avg_Whole_Wt=("Whole_weight", "mean"),
        Avg_Shell_Wt=("Shell_weight", "mean"),
        Median_Shell_Wt=("Shell_weight", "median"),
    )
    print(cohort_summary.round(3))

    # 5. CORRELATION ANALYSIS
    print("\n" + "=" * 65)
    print("3. PEARSON CORRELATION WITH RINGS (RANKED)")
    print("=" * 65)
    corr_matrix = df.corr(numeric_only=True)
    rings_corr = corr_matrix["Rings"].sort_values(ascending=False)
    print(rings_corr.round(4))

    # 6. GENERATE VISUALIZATIONS
    print("\n" + "=" * 65)
    print("4. GENERATING CHARTS (DARK THEME)")
    print("=" * 65)

    plt.style.use("dark_background")
    bg_color = "#0B0C10"
    card_color = "#111318"

    # --- Chart 1: Correlation Heatmap ---
    plt.figure(figsize=(9, 7), facecolor=bg_color)
    ax1 = plt.gca()
    ax1.set_facecolor(bg_color)

    num_cols = [
        "Length",
        "Diameter",
        "Height",
        "Whole_weight",
        "Shucked_weight",
        "Viscera_weight",
        "Shell_weight",
        "Rings",
    ]
    sns.heatmap(
        df[num_cols].corr(),
        annot=True,
        fmt=".2f",
        cmap="mako",
        cbar=True,
        linewidths=1.2,
        linecolor="#1F2833",
        annot_kws={"size": 10, "weight": "bold", "color": "#FFFFFF"},
    )
    plt.title(
        "Pearson Correlation Heatmap (Abalone Metrics)",
        fontsize=14,
        weight="bold",
        color="#00E5FF",
        pad=15,
    )
    plt.xticks(rotation=45, ha="right", fontsize=10, color="#E0E0E0")
    plt.yticks(rotation=0, fontsize=10, color="#E0E0E0")
    plt.tight_layout()
    plt.savefig("correlation_heatmap.png", dpi=300, facecolor=bg_color)
    plt.close()
    print("  [✓] Saved: correlation_heatmap.png")

    # --- Chart 2: Allometric Scatter (Length vs. Whole Weight) ---
    def power_law(x, a, b):
        return a * (x**b)

    popt, _ = curve_fit(power_law, df["Length"], df["Whole_weight"])
    a_fit, b_fit = popt

    plt.figure(figsize=(9, 6.5), facecolor=bg_color)
    ax2 = plt.gca()
    ax2.set_facecolor(card_color)

    plt.scatter(
        df["Length"],
        df["Whole_weight"],
        c="#00E5FF",
        alpha=0.35,
        s=25,
        edgecolors="none",
        label="Abalone Specimens (N=4,177)",
    )

    x_vals = np.linspace(df["Length"].min(), df["Length"].max(), 300)
    y_vals = power_law(x_vals, *popt)
    plt.plot(
        x_vals,
        y_vals,
        color="#FF2A85",
        linewidth=2.8,
        label=f"Allometric Fit: W = {a_fit:.2f} * L^{b_fit:.2f}\n(Volumetric 3D Growth)",
    )

    plt.title(
        "Allometric Growth: Shell Length vs. Whole Weight",
        fontsize=14,
        weight="bold",
        color="#00E5FF",
        pad=15,
    )
    plt.xlabel(
        "Shell Length (mm)", fontsize=11, weight="bold", color="#E0E0E0"
    )
    plt.ylabel(
        "Whole Weight (grams)", fontsize=11, weight="bold", color="#E0E0E0"
    )
    plt.grid(True, color="#2B2D42", linestyle="--", alpha=0.5)

    legend = plt.legend(
        frameon=True, facecolor="#1F2833", edgecolor="#00E5FF", fontsize=10
    )
    for text in legend.get_texts():
        text.set_color("white")

    plt.tight_layout()
    plt.savefig("allometric_scatter.png", dpi=300, facecolor=bg_color)
    plt.close()
    print("  [✓] Saved: allometric_scatter.png")

    # --- Chart 3: Cohort Boxplots ---
    cohort_order = ["Young (<=8)", "Mature (9-11)", "Old (12+)"]
    palette = {
        "Young (<=8)": "#00E5FF",
        "Mature (9-11)": "#B537F2",
        "Old (12+)": "#FFD166",
    }

    plt.figure(figsize=(9, 6.5), facecolor=bg_color)
    ax3 = plt.gca()
    ax3.set_facecolor(card_color)

    sns.boxplot(
        x="Age_Cohort",
        y="Shell_weight",
        data=df,
        order=cohort_order,
        palette=palette,
        width=0.5,
        flierprops=dict(
            marker="o",
            markersize=3,
            markerfacecolor="#888888",
            markeredgecolor="none",
            alpha=0.3,
        ),
        boxprops=dict(edgecolor="#FFFFFF", linewidth=1.5),
        whiskerprops=dict(color="#FFFFFF", linewidth=1.5),
        capprops=dict(color="#FFFFFF", linewidth=1.5),
        medianprops=dict(color="#FFFFFF", linewidth=2.5),
    )

    for i, c_name in enumerate(cohort_order):
        med = df[df["Age_Cohort"] == c_name]["Shell_weight"].median()
        ax3.text(
            i,
            med + 0.03,
            f"Med: {med:.3f}g",
            horizontalalignment="center",
            size=10,
            color="#FFFFFF",
            weight="bold",
        )

    plt.title(
        "Dry Shell Weight Distribution across Age Cohorts",
        fontsize=14,
        weight="bold",
        color="#00E5FF",
        pad=15,
    )
    plt.xlabel("Age Cohort", fontsize=11, weight="bold", color="#E0E0E0")
    plt.ylabel(
        "Shell Weight (grams)", fontsize=11, weight="bold", color="#E0E0E0"
    )
    plt.grid(True, axis="y", color="#2B2D42", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig("cohort_boxplot.png", dpi=300, facecolor=bg_color)
    plt.close()
    print("  [✓] Saved: cohort_boxplot.png")

    print("\n[✓] All tasks finished! Images generated in your folder.")


if __name__ == "__main__":
    main()