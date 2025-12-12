import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle

def preprocess_data():
    print("\n" + "="*50)
    print("STARTING DATA PREPROCESSING")
    print("="*50)

    # Load Data
    df = pd.read_csv('viral_shorts_reels_performance_dataset.csv')
    
    # 1. Handle Missing Values (None expected, but good practice)
    df = df.dropna()
    
    # 2. Feature Selection
    # Drop IDs and Dates for modeling (Dates are hard to predict with in simple apps)
    X = df.drop(['video_id', 'upload_time', 'views_total'], axis=1)
    y = df['views_total']
    
    # 3. Encode Categorical Variables (One-Hot Encoding)
    X = pd.get_dummies(X, columns=['niche', 'music_type'], drop_first=True)
    
    # Save the feature column names to ensure the App uses the same structure
    feature_columns = X.columns.tolist()
    with open('feature_columns.pkl', 'wb') as f:
        pickle.dump(feature_columns, f)
        
    # 4. Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 5. Save Processed Data
    np.save('X_train.npy', X_train)
    np.save('X_test.npy', X_test)
    np.save('y_train.npy', y_train)
    np.save('y_test.npy', y_test)
    
    print("✓ Data processed and saved successfully!")
    print(f"Training features: {X.shape[1]}")
    print("Files created: X_train.npy, X_test.npy, y_train.npy, y_test.npy, feature_columns.pkl")

if __name__ == "__main__":
    preprocess_data()