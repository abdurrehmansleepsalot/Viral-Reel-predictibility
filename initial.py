import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

INPUT_CSV = "viral_shorts_reels_performance_dataset.csv"
DATE_COL = "upload_time"

# Toggle options
PLOT_WEEKLY_NICHE = True      # Set False if you don't want niche weekly multi-line
ROLLING_WINDOW = 7            # Set None to disable rolling average plot
PLOT_DURATION_BUCKETS = True  # Toggle duration bucket bar chart

sns.set_style("whitegrid")

def load_data(path):
    df = pd.read_csv(path)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    df = df.dropna(subset=[DATE_COL]).sort_values(DATE_COL)
    # Time features
    df["dow"] = df[DATE_COL].dt.dayofweek  # 0=Mon
    df["month"] = df[DATE_COL].dt.month
    # Duration buckets for bar chart
    bins = [0, 10, 20, 30, 40, 100]
    labels = ["<=10s","11-20s","21-30s","31-40s",">40s"]
    df["duration_bucket"] = pd.cut(df["duration_sec"], bins=bins, labels=labels, include_lowest=True)
    return df

def basic_summary(df):
    print("=== BASIC SUMMARY ===")
    print("Rows:", len(df))
    print("Columns:", list(df.columns))
    print("\nData Types:\n", df.dtypes)
    print("\nNumeric Summary:\n", df.select_dtypes(include=[np.number]).describe().T)
    print("\nMissing Values:\n", df.isna().sum())

def correlation_matrix(df):
    num = df.select_dtypes(include=[np.number])
    corr = num.corr(method="spearman")
    print("\n=== Spearman Correlation Matrix (rounded) ===")
    print(corr.round(3))

def make_daily_trends(df):
    df["date"] = df[DATE_COL].dt.date
    daily = df.groupby("date").agg({
        "views_first_hour": "mean",
        "views_total": "mean",
        "retention_rate": "mean"
    }).reset_index()
    daily["date"] = pd.to_datetime(daily["date"])

    plt.figure(figsize=(10,4))
    plt.plot(daily["date"], daily["views_first_hour"], label="Mean First-hour Views", color="#1f77b4")
    plt.title("Daily Trend: Mean First-hour Views")
    plt.xlabel("Date"); plt.ylabel("Views")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10,4))
    plt.plot(daily["date"], daily["views_total"], label="Mean Total Views", color="#ff7f0e")
    plt.title("Daily Trend: Mean Total Views")
    plt.xlabel("Date"); plt.ylabel("Views")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10,4))
    plt.plot(daily["date"], daily["retention_rate"], label="Mean Retention Rate", color="#2ca02c")
    plt.title("Daily Trend: Mean Retention Rate")
    plt.xlabel("Date"); plt.ylabel("Retention Rate")
    plt.tight_layout()
    plt.show()

    if ROLLING_WINDOW:
        daily["first_hour_roll"] = daily["views_first_hour"].rolling(ROLLING_WINDOW).mean()
        plt.figure(figsize=(10,4))
        plt.plot(daily["date"], daily["views_first_hour"], alpha=0.4, label="Daily Mean")
        plt.plot(daily["date"], daily["first_hour_roll"], color="red", label=f"{ROLLING_WINDOW}-day Rolling Mean")
        plt.title(f"First-hour Views (Raw vs {ROLLING_WINDOW}-day Smoothed)")
        plt.xlabel("Date"); plt.ylabel("Views")
        plt.legend()
        plt.tight_layout()
        plt.show()

def weekly_niche_trend(df):
    if not PLOT_WEEKLY_NICHE:
        return
    df["week_start"] = df[DATE_COL] - pd.to_timedelta(df[DATE_COL].dt.dayofweek, unit="D")
    weekly = df.groupby(["week_start","niche"]).agg({"views_first_hour":"mean"}).reset_index()

    plt.figure(figsize=(11,5))
    for niche in weekly["niche"].unique():
        sub = weekly[weekly["niche"] == niche]
        plt.plot(sub["week_start"], sub["views_first_hour"], label=niche)
    plt.title("Weekly Mean First-hour Views by Niche")
    plt.xlabel("Week Start"); plt.ylabel("Views")
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.show()

# ====== NEW: Readable bar charts (no prediction) ======
def bar_chart_mean(df, x_col, y_col="views_first_hour", title=None, order_desc=True, palette="Blues_d"):
    agg = df.groupby(x_col)[y_col].mean().reset_index()
    if order_desc:
        agg = agg.sort_values(y_col, ascending=False)
    plt.figure(figsize=(9,4))
    sns.barplot(data=agg, x=x_col, y=y_col, palette=palette)
    plt.title(title or f"Mean {y_col} by {x_col}")
    if x_col == "dow":
        plt.xlabel("Day of Week (0=Mon)")
    else:
        plt.xlabel(x_col)
    plt.ylabel(f"Mean {y_col}")
    plt.tight_layout()
    plt.show()
    return agg

def top_k_bar(df, x_col, y_col="views_first_hour", k=10, title=None, palette="Greens_d"):
    agg = df.groupby(x_col)[y_col].mean().reset_index().sort_values(y_col, ascending=False).head(k)
    plt.figure(figsize=(9,4))
    sns.barplot(data=agg, x=x_col, y=y_col, palette=palette)
    plt.title(title or f"Top {k}: Mean {y_col} by {x_col}")
    plt.ylabel(f"Mean {y_col}")
    plt.tight_layout()
    plt.show()
    return agg

def main():
    if not os.path.exists(INPUT_CSV):
        print(f"ERROR: {INPUT_CSV} not found. Current dir: {os.getcwd()}")
        return

    df = load_data(INPUT_CSV)
    basic_summary(df)
    correlation_matrix(df)

    # Line charts you already had
    make_daily_trends(df)
    weekly_niche_trend(df)

    # ===== Bar charts for readability =====
    print("\n[Bar] Day-of-week performance (mean first-hour views)")
    bar_chart_mean(df, "dow", "views_first_hour", title="Mean First-hour Views by Day-of-Week (0=Mon)", palette="Blues_d")

    print("\n[Bar] Month performance (mean first-hour views)")
    bar_chart_mean(df, "month", "views_first_hour", title="Mean First-hour Views by Month", palette="Oranges_d")

    print("\n[Bar] Mean First-hour Views by Niche")
    top_k_bar(df, "niche", "views_first_hour", k=len(df['niche'].unique()), title="Mean First-hour Views by Niche", palette="Purples_d")

    print("\n[Bar] Mean First-hour Views by Music Type")
    bar_chart_mean(df, "music_type", "views_first_hour", title="Mean First-hour Views by Music Type", palette="Greens_d")

    if PLOT_DURATION_BUCKETS:
        print("\n[Bar] Mean First-hour Views by Duration Bucket")
        bar_chart_mean(df, "duration_bucket", "views_first_hour", title="Mean First-hour Views by Duration Bucket", palette="Greys")

    print("\nMinimal EDA with bar charts complete.")

if __name__ == "__main__":
    main()