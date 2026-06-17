# ============================================================
# CodeAlpha Internship — Task 3: Data Visualization
# Description: Build compelling visualizations using the
#              World Happiness Report dataset
# Libraries  : pandas, matplotlib, seaborn, plotly
# ============================================================

# Install dependencies:
# pip install pandas matplotlib seaborn plotly

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

sns.set_theme(style="whitegrid", palette="muted")

# ── 1. LOAD DATA ────────────────────────────────────────────
def load_data():
    """Load World Happiness Report 2021 from a public URL."""
    url = "https://raw.githubusercontent.com/dsrscientist/dataset1/master/happiness_score_dataset.csv"
    try:
        df = pd.read_csv(url)
    except Exception:
        # Fallback: create sample data if URL unavailable
        print("URL unavailable — generating sample data.")
        import numpy as np
        np.random.seed(42)
        regions = ["Western Europe", "North America", "Latin America",
                   "Eastern Asia", "South Asia", "Sub-Saharan Africa"]
        data = {
            "Country": [f"Country_{i}" for i in range(1, 51)],
            "Region": np.random.choice(regions, 50),
            "Happiness Score": np.random.uniform(3.0, 7.8, 50).round(3),
            "GDP per Capita": np.random.uniform(0.5, 1.8, 50).round(3),
            "Social Support": np.random.uniform(0.5, 1.5, 50).round(3),
            "Health": np.random.uniform(0.3, 1.0, 50).round(3),
            "Freedom": np.random.uniform(0.1, 0.7, 50).round(3),
            "Generosity": np.random.uniform(0.0, 0.5, 50).round(3),
            "Corruption": np.random.uniform(0.0, 0.5, 50).round(3),
        }
        df = pd.DataFrame(data)
    print(f"Dataset loaded: {df.shape[0]} rows × {df.shape[1]} columns")
    print(df.head())
    return df


# ── 2. BAR CHART — Top 15 Happiest Countries ────────────────
def plot_top_countries(df):
    score_col = [c for c in df.columns if "happiness" in c.lower() or "score" in c.lower()][0]
    country_col = [c for c in df.columns if "country" in c.lower()][0]

    top15 = df.nlargest(15, score_col)

    plt.figure(figsize=(12, 7))
    bars = plt.barh(top15[country_col], top15[score_col],
                    color=sns.color_palette("Blues_r", 15))
    plt.xlabel("Happiness Score", fontsize=12)
    plt.title("Top 15 Happiest Countries", fontsize=15, fontweight="bold")
    plt.gca().invert_yaxis()
    for bar, val in zip(bars, top15[score_col]):
        plt.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2,
                 f"{val:.2f}", va="center", fontsize=9)
    plt.tight_layout()
    plt.savefig("viz_top_countries.png", dpi=150)
    plt.show()
    print("Saved: viz_top_countries.png")


# ── 3. PIE CHART — Average Happiness by Region ──────────────
def plot_region_pie(df):
    score_col = [c for c in df.columns if "happiness" in c.lower() or "score" in c.lower()][0]
    region_col = [c for c in df.columns if "region" in c.lower()]

    if not region_col:
        print("No 'Region' column found — skipping pie chart.")
        return

    region_col = region_col[0]
    region_avg = df.groupby(region_col)[score_col].mean().sort_values(ascending=False)

    plt.figure(figsize=(10, 7))
    wedges, texts, autotexts = plt.pie(
        region_avg,
        labels=region_avg.index,
        autopct="%1.1f%%",
        startangle=140,
        colors=sns.color_palette("Set3", len(region_avg)),
        pctdistance=0.8
    )
    for text in autotexts:
        text.set_fontsize(9)
    plt.title("Average Happiness Score by Region", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("viz_region_pie.png", dpi=150)
    plt.show()
    print("Saved: viz_region_pie.png")


# ── 4. SCATTER PLOT — GDP vs Happiness ──────────────────────
def plot_gdp_vs_happiness(df):
    score_col = [c for c in df.columns if "happiness" in c.lower() or "score" in c.lower()][0]
    gdp_col = [c for c in df.columns if "gdp" in c.lower() or "economy" in c.lower()]

    if not gdp_col:
        print("No GDP column found — skipping scatter plot.")
        return

    gdp_col = gdp_col[0]
    region_col = [c for c in df.columns if "region" in c.lower()]

    plt.figure(figsize=(11, 7))
    if region_col:
        regions = df[region_col[0]].unique()
        palette = sns.color_palette("tab10", len(regions))
        for i, region in enumerate(regions):
            subset = df[df[region_col[0]] == region]
            plt.scatter(subset[gdp_col], subset[score_col],
                        label=region, color=palette[i], alpha=0.8, s=70)
        plt.legend(title="Region", bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    else:
        plt.scatter(df[gdp_col], df[score_col], color="steelblue", alpha=0.7, s=70)

    # Trend line
    import numpy as np
    valid = df[[gdp_col, score_col]].dropna()
    z = np.polyfit(valid[gdp_col], valid[score_col], 1)
    p = np.poly1d(z)
    x_line = sorted(valid[gdp_col])
    plt.plot(x_line, p(x_line), "r--", linewidth=1.5, label="Trend Line")

    plt.xlabel("GDP per Capita", fontsize=12)
    plt.ylabel("Happiness Score", fontsize=12)
    plt.title("GDP per Capita vs Happiness Score", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("viz_gdp_vs_happiness.png", dpi=150)
    plt.show()
    print("Saved: viz_gdp_vs_happiness.png")


# ── 5. HEATMAP — Factor Correlations ────────────────────────
def plot_correlation_heatmap(df):
    import numpy as np
    numeric_df = df.select_dtypes(include=[np.number])

    plt.figure(figsize=(10, 7))
    corr = numeric_df.corr()
    mask = pd.DataFrame(False, index=corr.index, columns=corr.columns)
    # Mask upper triangle for cleaner look
    import numpy as np
    mask_arr = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask_arr, annot=True, fmt=".2f",
                cmap="RdYlGn", linewidths=0.5, vmin=-1, vmax=1)
    plt.title("Correlation Between Happiness Factors", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("viz_correlation_heatmap.png", dpi=150)
    plt.show()
    print("Saved: viz_correlation_heatmap.png")


# ── 6. BOX PLOT — Score Distribution by Region ──────────────
def plot_boxplot_by_region(df):
    score_col = [c for c in df.columns if "happiness" in c.lower() or "score" in c.lower()][0]
    region_col = [c for c in df.columns if "region" in c.lower()]

    if not region_col:
        print("No 'Region' column found — skipping box plot.")
        return

    region_col = region_col[0]
    plt.figure(figsize=(13, 7))
    order = df.groupby(region_col)[score_col].median().sort_values(ascending=False).index
    sns.boxplot(data=df, x=region_col, y=score_col, order=order,
                palette="Set2")
    plt.xticks(rotation=30, ha="right", fontsize=9)
    plt.xlabel("Region", fontsize=12)
    plt.ylabel("Happiness Score", fontsize=12)
    plt.title("Happiness Score Distribution by Region", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("viz_boxplot_region.png", dpi=150)
    plt.show()
    print("Saved: viz_boxplot_region.png")


# ── MAIN ────────────────────────────────────────────────────
def main():
    print("=" * 50)
    print("  CodeAlpha — Task 3: Data Visualization")
    print("=" * 50)

    df = load_data()
    plot_top_countries(df)
    plot_region_pie(df)
    plot_gdp_vs_happiness(df)
    plot_correlation_heatmap(df)
    plot_boxplot_by_region(df)

    print("\nAll visualizations saved as PNG files!")


if __name__ == "__main__":
    main()
