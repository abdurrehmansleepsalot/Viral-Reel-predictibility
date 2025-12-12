import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set style
sns.set_style("whitegrid")

def perform_eda():
    print("="*50)
    print("STARTING EXPLORATORY DATA ANALYSIS (EDA)")
    print("="*50)

    # Load Data
    df = pd.read_csv('viral_shorts_reels_performance_dataset.csv')
    
    # 1. Summary Statistics
    print("\n--- Summary Statistics ---")
    print(df.describe())
    
    print("\n--- Data Types & Missing Values ---")
    print(df.info())
    
    # 2. Visualizations
    
    # Distribution of Target Variable (Views)
    plt.figure(figsize=(10, 6))
    sns.histplot(df['views_total'], bins=30, kde=True, color='blue')
    plt.title('Distribution of Total Views')
    plt.savefig('distributions.png')
    plt.close()
    
    # Correlation Matrix
    plt.figure(figsize=(12, 8))
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Feature Correlation Matrix')
    plt.savefig('correlation_heatmap.png')
    plt.close()
    
    # Views by Niche (Boxplot)
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='niche', y='views_total', data=df)
    plt.xticks(rotation=45)
    plt.title('Total Views by Niche')
    plt.savefig('niche_analysis.png')
    plt.close()

    # Views by Music Type (Bar Plot)
    plt.figure(figsize=(10, 6))
    sns.barplot(x='music_type', y='views_total', data=df, estimator='mean', errorbar=None)
    plt.title('Average Views by Music Type')
    plt.savefig('music_type_analysis.png')
    plt.close()

    # Trend Analysis (if dates exist)
    if 'upload_time' in df.columns:
        df['upload_time'] = pd.to_datetime(df['upload_time'])
        daily_views = df.groupby('upload_time')['views_total'].mean()
        plt.figure(figsize=(12, 6))
        daily_views.plot()
        plt.title('Average Views Trend Over Time')
        plt.ylabel('Average Views')
        plt.savefig('trend_analysis.png')
        plt.close()

    print("\n✓ EDA Completed! Images saved:")
    print("- distributions.png")
    print("- correlation_heatmap.png")
    print("- niche_analysis.png")
    print("- music_type_analysis.png")
    print("- trend_analysis.png")

if __name__ == "__main__":
    perform_eda()