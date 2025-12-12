import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Viral Reels Predictor",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# 2. CUSTOM CSS (THEME & STYLING)
# -----------------------------------------------------------------------------
st.markdown("""
<style>
    /* --- GLOBAL THEME --- */
    /* Main Background */
    .stApp {
        background-color: #0F0A1E; /* Very dark purple/black */
        color: #E6E6FA; /* Lavender text */
    }
    
    /* Sidebar Background */
    section[data-testid="stSidebar"] {
        background-color: #1A1025;
        border-right: 1px solid #2E1A47;
    }

    /* --- TYPOGRAPHY --- */
    h1, h2, h3 {
        color: #D4BBFF !important; /* Bright Neon Lavender */
        font-family: 'Helvetica Neue', sans-serif;
        font-weight: 700;
    }
    p, label, .stMarkdown {
        font-size: 1.1rem !important; /* Increase general text size */
        color: #CEC0E0 !important;
    }

    /* --- VERTICAL TABS (SIDEBAR RADIO) --- */
    /* Hide the default radio circle */
    .stRadio > div[role="radiogroup"] > label > div:first-child {
        display: None;
    }
    
    /* Style the buttons */
    .stRadio > div[role="radiogroup"] > label {
        background-color: transparent;
        padding: 15px 20px;
        margin-bottom: 5px;
        border-radius: 10px;
        border: 1px solid transparent;
        transition: all 0.3s;
        cursor: pointer;
        display: block; /* Make it take full width */
    }
    
    /* Hover Effect */
    .stRadio > div[role="radiogroup"] > label:hover {
        background-color: #2E1A47;
        border-color: #4B3B60;
    }

    /* Selected Tab Styling (The "Active" state) */
    .stRadio > div[role="radiogroup"] > label[data-testid="stMarkdownContainer"] > p {
        font-size: 1.3rem !important; /* BIGGER FONT FOR TABS */
        font-weight: 600;
    }
    
    /* --- METRICS & CARDS --- */
    div[data-testid="stMetric"] {
        background-color: #1F1430;
        border: 1px solid #4B3B60;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    div[data-testid="stMetricLabel"] {
        color: #9E86C4 !important;
        font-size: 1rem !important;
    }
    div[data-testid="stMetricValue"] {
        color: #D4BBFF !important;
        font-size: 2.2rem !important;
    }

    /* --- BUTTONS --- */
    .stButton > button {
        background-color: #D4BBFF;
        color: #0F0A1E;
        font-size: 1.2rem;
        font-weight: bold;
        border-radius: 10px;
        padding: 0.75rem 1rem;
        border: none;
        transition: 0.2s;
    }
    .stButton > button:hover {
        background-color: #FFFFFF;
        box-shadow: 0 0 15px #D4BBFF;
    }
    
    /* --- INPUT FIELDS --- */
    .stSelectbox div[data-baseweb="select"] > div, 
    .stNumberInput input {
        background-color: #261938;
        color: #FFFFFF;
        border: 1px solid #4B3B60;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 3. HELPER FUNCTIONS
# -----------------------------------------------------------------------------
@st.cache_resource
def load_resources():
    try:
        base_path = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(base_path, 'best_model.pkl')
        cols_path = os.path.join(base_path, 'feature_columns.pkl')

        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        with open(cols_path, 'rb') as f:
            feature_columns = pickle.load(f)
        return model, feature_columns
    except FileNotFoundError:
        return None, None

def display_image(filename, caption):
    if os.path.exists(filename):
        st.image(filename, caption=caption, use_container_width=True)
    else:
        st.warning(f"Image '{filename}' not found. Run eda.py first.")

# -----------------------------------------------------------------------------
# 4. MAIN NAVIGATION & CONTENT
# -----------------------------------------------------------------------------
def main():
    
    # --- SIDEBAR NAVIGATION ---
    with st.sidebar:
        st.title("🎬 ViralPredict")
        st.markdown("Data-Driven Content Strategy")
        st.markdown("---")
        
        # This radio button is styled by CSS to look like vertical tabs
        page_selection = st.radio(
            "Go to",
            ["🏠  Project Home", "📊  EDA Analytics", "🤖  Model Details", "🔮  Live Predictor"],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        st.info("**Student Project**\n\nIDS Fall 2024")

    # --- PAGE: PROJECT HOME ---
    if "Project Home" in page_selection:
        st.title("🚀 Viral Shorts & Reels Predictor")
        st.markdown("### Decode the Algorithm. Predict the Future.")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown("""
            <div style="background-color: #1F1430; padding: 25px; border-radius: 15px; border-left: 5px solid #D4BBFF;">
                <h3 style="margin-top:0">The Problem</h3>
                <p>Creating content is hard. Knowing if it will go viral is harder. 
                Creators spend hours filming, only to get 100 views.</p>
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            st.markdown("""
            <div style="background-color: #1F1430; padding: 25px; border-radius: 15px; border-left: 5px solid #9E86C4;">
                <h3 style="margin-top:0">The Solution</h3>
                <p>We analyzed 400+ videos to build an AI model that predicts 
                <strong>Total Views</strong> based on Hook Strength, Music, and Duration.</p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("### 🔑 Key Metrics Analyzed")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Hook Strength", "First 3s", "Crucial")
        m2.metric("Music Choice", "Trending", "+40% Boost")
        m3.metric("Retention", "Watch Time", "Algorithm Key")
        m4.metric("Niche", "Topic", "Audience Size")

    # --- PAGE: EDA ANALYTICS ---
    elif "EDA Analytics" in page_selection:
        st.title("📊 Exploratory Data Analysis")
        st.markdown("Visualizing the DNA of viral content.")
        
        tab_eda1, tab_eda2 = st.tabs(["🔥 Correlations & Distributions", "🎭 Niche & Music"])
        
        with tab_eda1:
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Correlation Heatmap")
                display_image("correlation_heatmap.png", "Correlation Matrix")
            with col2:
                st.subheader("Views Distribution")
                display_image("distributions.png", "Target Variable Distribution")
                
        with tab_eda2:
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Niche Performance")
                display_image("niche_analysis.png", "Which topics perform best?")
            with col2:
                st.subheader("Music Impact")
                display_image("music_type_analysis.png", "Impact of Audio Choice")

    # --- PAGE: MODEL DETAILS ---
    elif "Model Details" in page_selection:
        st.title("🤖 Machine Learning Model")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Why Random Forest?")
            st.markdown("""
            We selected the **Random Forest Regressor** for its ability to handle complex, non-linear relationships in social media data.
            
            * **Ensemble Method:** Combines 100+ decision trees to reduce overfitting.
            * **Robustness:** Handles outliers (viral hits) better than Linear Regression.
            * **Transparency:** Provides clear feature importance scores.
            """)
            
            st.markdown("### Feature Importance")
            display_image("feature_importance.png", "What drives the algorithm?")
            
        with col2:
            st.markdown("""
            <div style="background-color: #261938; padding: 20px; border-radius: 15px;">
                <h4>⚙️ Model Specs</h4>
                <p><strong>Type:</strong> Regressor</p>
                <p><strong>Trees:</strong> 100</p>
                <p><strong>Split:</strong> 80/20 Train/Test</p>
            </div>
            <br>
            """, unsafe_allow_html=True)
            
            st.subheader("Performance")
            st.warning("See terminal for latest RMSE & R² scores")

    # --- PAGE: LIVE PREDICTOR ---
    elif "Live Predictor" in page_selection:
        st.title("🔮 Viral Prediction Engine")
        st.markdown("Test your content strategy before you post.")
        
        model, feature_columns = load_resources()
        
        if model:
            st.markdown("<br>", unsafe_allow_html=True)
            
            # INPUT CARD
            with st.container():
                st.markdown("""<div style="background-color: #1A1025; padding: 20px; border-radius: 15px; border: 1px solid #4B3B60;">""", unsafe_allow_html=True)
                
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.markdown("##### 📝 Content Basics")
                    niche = st.selectbox("Niche", ['Tech', 'Motivation', 'Travel', 'Food', 'Gaming', 'Comedy', 'Education', 'Fitness', 'Beauty', 'Music'])
                    duration = st.slider("Duration (Seconds)", 5, 60, 15)
                    
                with c2:
                    st.markdown("##### 🎵 Audio & Hook")
                    music_type = st.selectbox("Music Type", ['Viral Track', 'Trending', 'Remix', 'Original', 'No Music'])
                    hook_strength = st.slider("Hook Strength (0.0 - 1.0)", 0.0, 1.0, 0.6)
                    
                with c3:
                    st.markdown("##### 📈 Early Signals")
                    views_first_hour = st.number_input("Est. First Hour Views", value=1500, step=100)
                    retention = st.slider("Est. Retention Rate", 0.0, 1.0, 0.45)
                    
                # Hidden default for simpler UI
                first_3_sec = 0.5 
                st.markdown("</div>", unsafe_allow_html=True)

            # PREDICTION LOGIC
            input_data = pd.DataFrame({
                'duration_sec': [duration],
                'hook_strength_score': [hook_strength],
                'views_first_hour': [views_first_hour],
                'retention_rate': [retention],
                'first_3_sec_engagement': [first_3_sec],
                'niche': [niche],
                'music_type': [music_type]
            })
            
            # Preprocessing
            input_data = pd.get_dummies(input_data, columns=['niche', 'music_type'], drop_first=True)
            for col in feature_columns:
                if col not in input_data.columns:
                    input_data[col] = 0
            input_data = input_data[feature_columns]

            st.markdown("<br>", unsafe_allow_html=True)
            
            if st.button("🚀 PREDICT VIRALITY NOW", use_container_width=True):
                prediction = model.predict(input_data)[0]
                
                st.markdown("---")
                
                res_col1, res_col2 = st.columns([1, 2])
                
                with res_col1:
                    st.metric("Predicted Total Views", f"{int(prediction):,}", delta="Algorithm Estimate")
                
                with res_col2:
                    if prediction > 500000:
                        st.balloons()
                        st.markdown("""
                        <div style="background-color: rgba(0, 255, 0, 0.1); padding: 20px; border-radius: 10px; border: 1px solid green;">
                            <h2 style="color: #00FF00 !important; margin:0;">🔥 MEGA VIRAL HIT!</h2>
                            <p style="color: #DDFFDD !important;">This content has massive potential. Post it immediately!</p>
                        </div>
                        """, unsafe_allow_html=True)
                    elif prediction > 200000:
                        st.markdown("""
                        <div style="background-color: rgba(255, 255, 0, 0.1); padding: 20px; border-radius: 10px; border: 1px solid yellow;">
                            <h2 style="color: #FFFF00 !important; margin:0;">📈 STRONG PERFORMER</h2>
                            <p style="color: #FFFFDD !important;">Good potential. Improve the hook to push it to viral status.</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown("""
                        <div style="background-color: rgba(255, 0, 0, 0.1); padding: 20px; border-radius: 10px; border: 1px solid red;">
                            <h2 style="color: #FF4444 !important; margin:0;">📉 AVERAGE PERFORMANCE</h2>
                            <p style="color: #FFDDDD !important;">Try using a Trending Audio or cutting the duration to improve results.</p>
                        </div>
                        """, unsafe_allow_html=True)

        else:
            st.error("⚠️ Model files missing. Please run 'python model.py' first.")

if __name__ == '__main__':
    main()