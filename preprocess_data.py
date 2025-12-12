import os
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# ==================== CONFIGURATION ====================
INPUT_CSV = "viral_shorts_reels_performance_dataset.csv"
DATE_COL = "upload_time"
TARGET_COLUMN = "views_first_hour"
TEST_SIZE = 0.2
RANDOM_STATE = 42

def load_data(path):
    """Load and parse dates"""
    print("=" * 60)
    print("LOADING DATA")
    print("=" * 60)
    
    df = pd.read_csv(path)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    df = df.dropna(subset=[DATE_COL]). sort_values(DATE_COL)
    
    print(f"\n✅ Loaded {len(df)} rows, {len(df.columns)} columns")
    return df

def handle_missing_values(df):
    """Handle missing values"""
    print("\n" + "=" * 60)
    print("HANDLING MISSING VALUES")
    print("=" * 60)
    
    print("\nMissing values before:")
    print(df. isna().sum()[df.isna().sum() > 0])
    
    # Drop rows with missing target
    if TARGET_COLUMN in df.columns:
        df = df.dropna(subset=[TARGET_COLUMN])
    
    # Fill numeric with median
    numeric_cols = df.select_dtypes(include=[np.number]). columns
    for col in numeric_cols:
        if df[col].isna().sum() > 0:
            df[col].fillna(df[col].median(), inplace=True)
    
    # Fill categorical with mode
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if df[col].isna().sum() > 0:
            df[col].fillna(df[col].mode()[0], inplace=True)
    
    print("\n✅ Missing values handled")
    return df

def create_time_features(df):
    """Extract time-based features"""
    print("\n" + "=" * 60)
    print("CREATING TIME FEATURES")
    print("=" * 60)
    
    df["year"] = df[DATE_COL].dt.year
    df["month"] = df[DATE_COL].dt.month
    df["day"] = df[DATE_COL].dt.day
    df["dow"] = df[DATE_COL].dt.dayofweek  # 0=Monday
    df["hour"] = df[DATE_COL].dt.hour
    df["is_weekend"] = df["dow"].isin([5, 6]).astype(int)
    
    # Cyclical encoding
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["dow"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["dow"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    
    print("\n✅ Created time features:")
    print("   - year, month, day, dow, hour, is_weekend")
    print("   - Cyclical: hour_sin/cos, dow_sin/cos, month_sin/cos")
    
    return df

def create_duration_features(df):
    """Create duration-based features"""
    print("\n" + "=" * 60)
    print("CREATING DURATION FEATURES")
    print("=" * 60)
    
    if "duration_sec" in df.columns:
        df["duration_squared"] = df["duration_sec"] ** 2
        df["duration_log"] = np.log1p(df["duration_sec"])
        
        print("\n✅ Created duration features:")
        print("   - duration_squared, duration_log")
    
    return df

def create_interaction_features(df):
    """Create interaction features"""
    print("\n" + "=" * 60)
    print("CREATING INTERACTION FEATURES")
    print("=" * 60)
    
    created = []
    
    if "duration_sec" in df.columns and "retention_rate" in df.columns:
        df["duration_retention"] = df["duration_sec"] * df["retention_rate"]
        created.append("duration_retention")
    
    if "likes" in df.columns and "comments" in df.columns:
        df["engagement_ratio"] = df["likes"] / (df["comments"] + 1)
        created.append("engagement_ratio")
    
    if "shares" in df.columns and "views_total" in df.columns:
        df["share_rate"] = df["shares"] / (df["views_total"] + 1)
        created.append("share_rate")
    
    print(f"\n✅ Created {len(created)} interaction features:")
    for feat in created:
        print(f"   - {feat}")
    
    return df

def encode_categorical_features(df):
    """Encode categorical variables"""
    print("\n" + "=" * 60)
    print("ENCODING CATEGORICAL FEATURES")
    print("=" * 60)
    
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns. tolist()
    
    if DATE_COL in categorical_cols:
        categorical_cols.remove(DATE_COL)
    
    print(f"\nCategorical columns found: {categorical_cols}")
    
    label_encoders = {}
    
    for col in categorical_cols:
        le = LabelEncoder()
        df[f"{col}_encoded"] = le. fit_transform(df[col]. astype(str))
        label_encoders[col] = le
        print(f"   ✅ Encoded '{col}' -> '{col}_encoded' ({len(le.classes_)} categories)")
    
    # Save encoders
    with open('label_encoders.pkl', 'wb') as f:
        pickle.dump(label_encoders, f)
    print("\n✅ Saved: 'label_encoders.pkl'")
    
    # Drop original categorical columns
    df = df.drop(columns=categorical_cols + [DATE_COL])
    
    return df, label_encoders

def split_and_scale(df):
    """Split data and scale features"""
    print("\n" + "=" * 60)
    print("SPLITTING & SCALING DATA")
    print("=" * 60)
    
    # Separate features and target
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    
    print(f"\n✅ Split completed:")
    print(f"   Train: {len(X_train)} samples")
    print(f"   Test:  {len(X_test)} samples")
    print(f"   Features: {X_train.shape[1]}")
    
    # Scale features
    print("\nScaling features...")
    exclude_cols = ["year", "month", "day", "dow", "hour", "is_weekend"]
    cols_to_scale = [col for col in X_train.columns if col not in exclude_cols]
    
    scaler = StandardScaler()
    X_train[cols_to_scale] = scaler.fit_transform(X_train[cols_to_scale])
    X_test[cols_to_scale] = scaler. transform(X_test[cols_to_scale])
    
    print(f"   Scaled {len(cols_to_scale)} features")
    
    # Save everything
    X_train.to_csv("X_train.csv", index=False)
    X_test.to_csv("X_test.csv", index=False)
    y_train.to_csv("y_train.csv", index=False)
    y_test.to_csv("y_test.csv", index=False)
    
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    print("\n✅ Saved files:")
    print("   - X_train.csv")
    print("   - X_test.csv")
    print("   - y_train.csv")
    print("   - y_test.csv")
    print("   - scaler.pkl")
    
    return X_train, X_test, y_train, y_test

def main():
    if not os.path.exists(INPUT_CSV):
        print(f"❌ ERROR: '{INPUT_CSV}' not found!")
        print(f"Current directory: {os.getcwd()}")
        return
    
    print("\n" + "=" * 60)
    print("🚀 DATA PREPROCESSING PIPELINE")
    print("=" * 60)
    
    # Step 1: Load
    df = load_data(INPUT_CSV)
    
    # Step 2: Handle missing values
    df = handle_missing_values(df)
    
    # Step 3: Create time features
    df = create_time_features(df)
    
    # Step 4: Create duration features
    df = create_duration_features(df)
    
    # Step 5: Create interaction features
    df = create_interaction_features(df)
    
    # Step 6: Encode categorical
    df, encoders = encode_categorical_features(df)
    
    # Step 7: Split and scale
    X_train, X_test, y_train, y_test = split_and_scale(df)
    
    # Summary
    print("\n" + "=" * 60)
    print("✅ PREPROCESSING COMPLETE!")
    print("=" * 60)
    print("\n📁 Generated Files:")
    print("   1. X_train.csv - Training features")
    print("   2.  X_test.csv - Test features")
    print("   3. y_train.csv - Training target")
    print("   4.  y_test.csv - Test target")
    print("   5.  scaler.pkl - Feature scaler")
    print("   6. label_encoders.pkl - Categorical encoders")
    print("\n🎯 Next step: Run 'train_models.py' to train ML models!")

if __name__ == "__main__":
    main()