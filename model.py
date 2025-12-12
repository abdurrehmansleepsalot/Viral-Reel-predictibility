import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import pickle

def train_model():
    print("\n" + "="*50)
    print("TRAINING MACHINE LEARNING MODEL")
    print("="*50)
    
    # 1. Load Data
    # We add allow_pickle=True to fix the ValueError
    # We add .astype(float) to ensure data is numeric for the Random Forest
    print("Loading data...")
    X_train = np.load('X_train.npy', allow_pickle=True).astype(float)
    X_test = np.load('X_test.npy', allow_pickle=True).astype(float)
    y_train = np.load('y_train.npy', allow_pickle=True).astype(float)
    y_test = np.load('y_test.npy', allow_pickle=True).astype(float)
    
    # 2. Initialize Model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    
    # 3. Train
    print("Training Random Forest Regressor...")
    model.fit(X_train, y_train)
    
    # 4. Evaluate
    predictions = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)
    
    print(f"\nModel Performance:")
    print(f"RMSE: {rmse:,.2f}")
    print(f"R² Score: {r2:.4f}")
    
    # 5. Save Model
    with open('best_model.pkl', 'wb') as f:
        pickle.dump(model, f)
        
    print("\n✓ Model saved as 'best_model.pkl'")

if __name__ == "__main__":
    train_model()