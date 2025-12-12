import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go

# Page config
st.set_page_config(
    page_title="Viral Video Performance Predictor",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #FF6B6B;
        text-align: center;
        padding: 1rem;
    }
    . metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .prediction-box {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        font-size: 1.5rem;
        text-align: center;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Load model and preprocessors
@st.cache_resource
def load_model_and_preprocessors():
    try:
        with open('best_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        with open('label_encoders.pkl', 'rb') as f:
            encoders = pickle.load(f)
        return model, scaler, encoders
    except FileNotFoundError as e:
        st.error(f"❌ Error: Required file not found - {e}")
        st.info("Please run 'preprocess_data. py' and 'train_models.py' first!")
        return None, None, None

# Load model results
@st.cache_data
def load_results():
    try:
        results = pd.read_csv('model_results.csv')
        return results
    except:
        return None

# Preprocess input for prediction
def preprocess_input(data, scaler, encoders):
    """Preprocess user input for prediction"""
    df = pd.DataFrame([data])
    
    # Create time features
    upload_time = pd.to_datetime(df['upload_time']. iloc[0])
    df['year'] = upload_time.year
    df['month'] = upload_time.month
    df['day'] = upload_time.day
    df['dow'] = upload_time.dayofweek
    df['hour'] = upload_time.hour
    df['is_weekend'] = 1 if upload_time.dayofweek in [5, 6] else 0
    
    # Cyclical encoding
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['dow_sin'] = np.sin(2 * np.pi * df['dow'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['dow'] / 7)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    # Duration features
    if 'duration_sec' in df.columns:
        df['duration_squared'] = df['duration_sec'] ** 2
        df['duration_log'] = np.log1p(df['duration_sec'])
    
    # Interaction features
    if 'duration_sec' in df.columns and 'retention_rate' in df.columns:
        df['duration_retention'] = df['duration_sec'] * df['retention_rate']
    
    if 'likes' in df.columns and 'comments' in df.columns:
        df['engagement_ratio'] = df['likes'] / (df['comments'] + 1)
    
    if 'shares' in df.columns and 'views_total' in df.columns:
        df['share_rate'] = df['shares'] / (df['views_total'] + 1)
    
    # Encode categorical
    for col, encoder in encoders.items():
        if col in df.columns:
            try:
                df[f'{col}_encoded'] = encoder. transform(df[col].astype(str))
            except:
                # Handle unseen categories
                df[f'{col}_encoded'] = 0
            df = df.drop(columns=[col])
    
    # Drop upload_time
    df = df.drop(columns=['upload_time'])
    
    # Load training columns order
    try:
        X_train = pd.read_csv('X_train.csv')
        train_cols = X_train.columns.tolist()
        
        # Add missing columns
        for col in train_cols:
            if col not in df.columns:
                df[col] = 0
        
        # Reorder to match training
        df = df[train_cols]
        
    except:
        pass
    
    return df

# Main app
def main():
    # Header
    st.markdown('<div class="main-header">🎬 Viral Video Performance Predictor</div>', unsafe_allow_html=True)
    st.markdown("### Predict your video's first-hour views using AI 🚀")
    
    # Load model
    model, scaler, encoders = load_model_and_preprocessors()
    
    if model is None:
        st.stop()
    
    # Sidebar
    st.sidebar.title("📊 Navigation")
    page = st.sidebar.radio("Go to", ["🔮 Predict", "📈 Model Performance", "🎯 Feature Importance", "📊 Insights"])
    
    # ==================== PAGE 1: PREDICTION ====================
    if page == "🔮 Predict":
        st.header("🔮 Make a Prediction")
        
        col1, col2 = st. columns(2)
        
        with col1:
            st. subheader("📝 Video Details")
            
            # Upload time
            upload_date = st.date_input("Upload Date", datetime.now())
            upload_time = st.time_input("Upload Time", datetime.now(). time())
            upload_datetime = datetime.combine(upload_date, upload_time)
            
            # Duration
            duration_sec = st. slider("Duration (seconds)", 5, 60, 30)
            
            # Niche
            niche_options = ["Comedy", "Education", "Entertainment", "Gaming", "Lifestyle", 
                           "Music", "Sports", "Tech", "Travel", "Vlog"]
            niche = st.selectbox("Niche", niche_options)
            
            # Music type
            music_options = ["Trending", "Original", "None", "Remix", "Classical"]
            music_type = st. selectbox("Music Type", music_options)
            
        with col2:
            st. subheader("📊 Engagement Metrics")
            
            retention_rate = st.slider("Retention Rate (%)", 0, 100, 50) / 100
            likes = st. number_input("Expected Likes", 0, 100000, 1000)
            comments = st. number_input("Expected Comments", 0, 10000, 50)
            shares = st.number_input("Expected Shares", 0, 50000, 100)
            views_total = st.number_input("Expected Total Views", 0, 1000000, 10000)
        
        # Predict button
        if st.button("🔮 Predict First Hour Views", type="primary"):
            # Prepare input
            input_data = {
                'upload_time': upload_datetime,
                'duration_sec': duration_sec,
                'niche': niche,
                'music_type': music_type,
                'retention_rate': retention_rate,
                'likes': likes,
                'comments': comments,
                'shares': shares,
                'views_total': views_total
            }
            
            # Preprocess
            try:
                X_input = preprocess_input(input_data, scaler, encoders)
                
                # Predict
                prediction = model.predict(X_input)[0]
                
                # Display prediction
                st.markdown(f"""
                <div class="prediction-box">
                    <h2>Predicted First Hour Views</h2>
                    <h1>🎯 {int(prediction):,}</h1>
                </div>
                """, unsafe_allow_html=True)
                
                # Additional insights
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    engagement_rate = (likes + comments + shares) / max(views_total, 1) * 100
                    st. metric("Engagement Rate", f"{engagement_rate:.2f}%")
                
                with col2:
                    retention_percent = retention_rate * 100
                    st.metric("Retention Rate", f"{retention_percent:.1f}%")
                
                with col3:
                    virality_score = (prediction / views_total * 100) if views_total > 0 else 0
                    st.metric("Virality Score", f"{virality_score:.1f}%")
                
                # Recommendations
                st.subheader("💡 Recommendations")
                
                if retention_rate < 0.5:
                    st. warning("⚠️ Low retention rate! Consider making your content more engaging in the first few seconds.")
                
                if duration_sec > 45:
                    st.info("ℹ️ Shorter videos (20-30s) tend to perform better for viral content.")
                
                if upload_datetime.hour < 12 or upload_datetime.hour > 21:
                    st.info("ℹ️ Try uploading between 12 PM - 9 PM for better reach.")
                
                if upload_datetime.weekday() in [5, 6]:
                    st.success("✅ Weekend uploads tend to get more engagement!")
                
            except Exception as e:
                st.error(f"❌ Error making prediction: {e}")
                st.error("Please make sure all preprocessing files are present.")
    
    # ==================== PAGE 2: MODEL PERFORMANCE ====================
    elif page == "📈 Model Performance":
        st.header("📈 Model Performance Metrics")
        
        results = load_results()
        
        if results is not None:
            # Display table
            st.subheader("📊 Model Comparison")
            st.dataframe(results. style.highlight_max(axis=0, subset=['Test R²']), use_container_width=True)
            
            # Best model highlight
            best_model = results.loc[results['Test R²'].idxmax(), 'Model']
            best_r2 = results['Test R²'].max()
            
            st.success(f"🏆 Best Model: **{best_model}** with R² = **{best_r2:.4f}**")
            
            # Visualizations
            col1, col2 = st. columns(2)
            
            with col1:
                # R² comparison
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=results['Model'],
                    y=results['Train R²'],
                    name='Train R²',
                    marker_color='lightblue'
                ))
                fig.add_trace(go. Bar(
                    x=results['Model'],
                    y=results['Test R²'],
                    name='Test R²',
                    marker_color='darkblue'
                ))
                fig.update_layout(
                    title='R² Score Comparison',
                    barmode='group',
                    yaxis_title='R² Score'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # RMSE comparison
                fig = px.bar(results, x='Model', y='Test RMSE', 
                           title='Test RMSE Comparison (Lower is Better)',
                           color='Test RMSE',
                           color_continuous_scale='Reds')
                st.plotly_chart(fig, use_container_width=True)
            
            # Cross-validation scores
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=results['Model'],
                y=results['CV R² Mean'],
                error_y=dict(type='data', array=results['CV R² Std']),
                marker_color='purple'
            ))
            fig. update_layout(
                title='Cross-Validation R² (with standard deviation)',
                yaxis_title='CV R² Score'
            )
            st.plotly_chart(fig, use_container_width=True)
            
        else:
            st.warning("⚠️ Model results not found. Please train models first!")
    
    # ==================== PAGE 3: FEATURE IMPORTANCE ====================
    elif page == "🎯 Feature Importance":
        st.header("🎯 Feature Importance Analysis")
        
        if hasattr(model, 'feature_importances_'):
            try:
                X_train = pd.read_csv('X_train.csv')
                feature_names = X_train.columns.tolist()
                importances = model.feature_importances_
                
                # Create dataframe
                feat_imp = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': importances
                }).sort_values('Importance', ascending=False)
                
                # Top 20
                top_20 = feat_imp.head(20)
                
                # Plot
                fig = px.bar(top_20, x='Importance', y='Feature', 
                           orientation='h',
                           title='Top 20 Most Important Features',
                           color='Importance',
                           color_continuous_scale='Viridis')
                fig.update_layout(yaxis={'categoryorder':'total ascending'})
                st. plotly_chart(fig, use_container_width=True)
                
                # Show table
                st.subheader("📋 All Features")
                st.dataframe(feat_imp, use_container_width=True)
                
                # Insights
                st.subheader("💡 Key Insights")
                st.write(f"- Most important feature: **{feat_imp.iloc[0]['Feature']}**")
                st.write(f"- Top 5 features account for **{feat_imp.head(5)['Importance']. sum()*100:.1f}%** of prediction power")
                
            except Exception as e:
                st.error(f"Error loading feature importance: {e}")
        else:
            st.info("Feature importance not available for this model type.")
    
    # ==================== PAGE 4: INSIGHTS ====================
    elif page == "📊 Insights":
        st.header("📊 Key Insights & Tips")
        
        col1, col2 = st. columns(2)
        
        with col1:
            st.subheader("🎯 Best Practices")
            st.markdown("""
            - **Duration**: Keep videos between 20-30 seconds for maximum retention
            - **Upload Time**: Post between 12 PM - 9 PM for better reach
            - **Weekends**: Weekend uploads typically get 20-30% more views
            - **Retention**: Aim for >60% retention rate in first 5 seconds
            - **Music**: Trending music can boost views by 40-50%
            """)
            
            st.subheader("🔥 Viral Video Checklist")
            st.markdown("""
            - ✅ Hook in first 3 seconds
            - ✅ Clear and engaging thumbnail
            - ✅ Trending audio/music
            - ✅ Optimal upload timing
            - ✅ Relevant hashtags
            - ✅ Strong call-to-action
            """)
        
        with col2:
            st.subheader("📈 Performance Factors")
            
            # Create sample data for demonstration
            factors = pd.DataFrame({
                'Factor': ['Retention Rate', 'Upload Time', 'Duration', 'Music Type', 'Niche', 'Day of Week'],
                'Impact': [95, 75, 70, 65, 60, 55]
            })
            
            fig = px.bar(factors, x='Impact', y='Factor', 
                        orientation='h',
                        title='Factors Impacting Video Performance',
                        color='Impact',
                        color_continuous_scale='RdYlGn')
            st.plotly_chart(fig, use_container_width=True)
            
            st.subheader("⏰ Best Upload Times")
            upload_times = pd.DataFrame({
                'Hour': list(range(24)),
                'Performance': [30, 25, 20, 15, 15, 20, 30, 45, 55, 60, 65, 70, 
                              85, 90, 95, 92, 88, 85, 90, 95, 85, 70, 55, 40]
            })
            
            fig = px.line(upload_times, x='Hour', y='Performance',
                         title='Video Performance by Upload Hour',
                         markers=True)
            fig.update_layout(xaxis_title='Hour of Day', yaxis_title='Performance Score')
            st.plotly_chart(fig, use_container_width=True)

# Footer
st.sidebar.markdown("---")
st. sidebar.info("🚀 Built with Streamlit & Machine Learning")
st.sidebar.markdown("Made with ❤️ for content creators")

if __name__ == "__main__":
    main()