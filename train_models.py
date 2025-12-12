import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')

# Configuration
RANDOM_STATE = 42
sns.set_style("whitegrid")

def load_data():
    """Load preprocessed data"""
    print("=" * 60)
    print("LOADING PREPROCESSED DATA")
    print("=" * 60)
    
    X_train = pd.read_csv("X_train.csv")
    X_test = pd.read_csv("X_test.csv")
    y_train = pd.read_csv("y_train.csv"). squeeze()
    y_test = pd.read_csv("y_test.csv").squeeze()
    
    print(f"\n✅ Data loaded:")
    print(f"   Train: {len(X_train)} samples")
    print(f"   Test:  {len(X_test)} samples")
    print(f"   Features: {X_train.shape[1]}")
    
    return X_train, X_test, y_train, y_test

def initialize_models():
    """Initialize ML models"""
    print("\n" + "=" * 60)
    print("INITIALIZING MODELS")
    print("=" * 60)
    
    models = {
        "Random Forest": RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            random_state=RANDOM_STATE
        ),
        "Ridge Regression": Ridge(
            alpha=10.0,
            random_state=RANDOM_STATE
        )
    }
    
    print("\n✅ Initialized 3 models:")
    print("   1. Random Forest")
    print("   2. Gradient Boosting")
    print("   3.  Ridge Regression")
    
    return models

def train_and_evaluate(models, X_train, X_test, y_train, y_test):
    """Train and evaluate all models"""
    print("\n" + "=" * 60)
    print("TRAINING MODELS")
    print("=" * 60)
    
    results = []
    predictions = {}
    
    for model_name, model in models.items():
        print(f"\n{'─'*60}")
        print(f"Training: {model_name}")
        print(f"{'─'*60}")
        
        # Train
        model.fit(X_train, y_train)
        
        # Predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        # Metrics
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        test_mae = mean_absolute_error(y_test, y_test_pred)
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2', n_jobs=-1)
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()
        
        # Store results
        results.append({
            "Model": model_name,
            "Train R²": train_r2,
            "Test R²": test_r2,
            "Test RMSE": test_rmse,
            "Test MAE": test_mae,
            "CV R² Mean": cv_mean,
            "CV R² Std": cv_std
        })
        
        predictions[model_name] = y_test_pred
        
        # Print results
        print(f"\n📊 Results:")
        print(f"   Train R²:  {train_r2:.4f}")
        print(f"   Test R²:   {test_r2:.4f}")
        print(f"   Test RMSE: {test_rmse:.2f}")
        print(f"   Test MAE:  {test_mae:.2f}")
        print(f"   CV R²:     {cv_mean:.4f} ± {cv_std:.4f}")
        
        # Overfitting check
        r2_diff = train_r2 - test_r2
        if r2_diff > 0.1:
            print(f"   ⚠️  Overfitting detected (diff = {r2_diff:.4f})")
        else:
            print(f"   ✅ Good generalization (diff = {r2_diff:. 4f})")
    
    return pd.DataFrame(results), predictions

def plot_comparison(results_df):
    """Plot model comparison"""
    print("\n" + "=" * 60)
    print("CREATING VISUALIZATIONS")
    print("=" * 60)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    models = results_df['Model']
    x = np.arange(len(models))
    
    # 1. R² Comparison
    ax = axes[0, 0]
    width = 0.35
    ax.bar(x - width/2, results_df['Train R²'], width, label='Train', alpha=0.8)
    ax.bar(x + width/2, results_df['Test R²'], width, label='Test', alpha=0.8)
    ax.set_ylabel('R² Score')
    ax.set_title('R² Score Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # 2. RMSE
    ax = axes[0, 1]
    ax.bar(x, results_df['Test RMSE'], color='coral', alpha=0.8)
    ax.set_ylabel('RMSE')
    ax.set_title('Test RMSE (Lower is Better)')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha='right')
    ax.grid(axis='y', alpha=0.3)
    
    # 3. MAE
    ax = axes[1, 0]
    ax.bar(x, results_df['Test MAE'], color='green', alpha=0.8)
    ax.set_ylabel('MAE')
    ax.set_title('Test MAE (Lower is Better)')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha='right')
    ax. grid(axis='y', alpha=0.3)
    
    # 4. Cross-Validation
    ax = axes[1, 1]
    ax.bar(x, results_df['CV R² Mean'], color='purple', alpha=0.8)
    ax.errorbar(x, results_df['CV R² Mean'], yerr=results_df['CV R² Std'],
                fmt='none', ecolor='black', capsize=5)
    ax.set_ylabel('CV R² Score')
    ax. set_title('Cross-Validation R²')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha='right')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
    print("\n✅ Saved: 'model_comparison.png'")
    plt.show()

def plot_predictions(y_test, predictions):
    """Plot actual vs predicted"""
    n = len(predictions)
    fig, axes = plt.subplots(1, n, figsize=(6*n, 5))
    
    if n == 1:
        axes = [axes]
    
    for ax, (name, y_pred) in zip(axes, predictions.items()):
        ax.scatter(y_test, y_pred, alpha=0.5, s=20)
        
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect')
        
        r2 = r2_score(y_test, y_pred)
        ax.set_xlabel('Actual')
        ax.set_ylabel('Predicted')
        ax.set_title(f'{name}\nR² = {r2:.4f}')
        ax. legend()
        ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('predictions.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: 'predictions.png'")
    plt.show()

def plot_feature_importance(models, feature_names):
    """Plot feature importance for tree models"""
    for name, model in models.items():
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1]
            top_n = min(20, len(feature_names))
            
            plt.figure(figsize=(10, 8))
            plt.barh(range(top_n), importances[indices[:top_n]])
            plt.yticks(range(top_n), [feature_names[i] for i in indices[:top_n]])
            plt.xlabel('Importance')
            plt.title(f'Top {top_n} Features - {name}')
            plt. gca().invert_yaxis()
            plt.tight_layout()
            
            filename = f"feature_importance_{name. replace(' ', '_')}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"✅ Saved: '{filename}'")
            plt.show()

def save_best_model(models, results_df):
    """Save the best model"""
    best_idx = results_df['Test R²'].idxmax()
    best_name = results_df.loc[best_idx, 'Model']
    best_r2 = results_df.loc[best_idx, 'Test R²']
    
    best_model = models[best_name]
    
    with open('best_model.pkl', 'wb') as f:
        pickle.dump(best_model, f)
    
    print(f"\n✅ Best model saved: '{best_name}'")
    print(f"   File: 'best_model.pkl'")
    print(f"   Test R²: {best_r2:.4f}")

def main():
    print("\n" + "=" * 60)
    print("🚀 MACHINE LEARNING TRAINING PIPELINE")
    print("=" * 60)
    
    # Load data
    X_train, X_test, y_train, y_test = load_data()
    
    # Initialize models
    models = initialize_models()
    
    # Train and evaluate
    results_df, predictions = train_and_evaluate(models, X_train, X_test, y_train, y_test)
    
    # Display results table
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print("\n" + results_df.to_string(index=False))
    
    # Save results
    results_df.to_csv('model_results.csv', index=False)
    print("\n✅ Saved: 'model_results. csv'")
    
    # Visualizations
    plot_comparison(results_df)
    plot_predictions(y_test, predictions)
    plot_feature_importance(models, X_train.columns. tolist())
    
    # Save best model
    save_best_model(models, results_df)
    
    print("\n" + "=" * 60)
    print("🎉 TRAINING COMPLETE!")
    print("=" * 60)
    print("\n📁 Generated Files:")
    print("   1. model_results.csv")
    print("   2. model_comparison.png")
    print("   3. predictions.png")
    print("   4. feature_importance_*.png")
    print("   5. best_model.pkl")

if __name__ == "__main__":
    main()