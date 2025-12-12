import streamlit as st
import numpy as np
import pandas as pd
import pickle

# SESSION STATE INITIALIZATION 
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'username' not in st.session_state:
    st.session_state.username = ""

# LOAD ARTIFACTS 
@st.cache_resource
def load_artifacts():
    model = pickle.load(open("model.pkl", "rb"))
    scaler = pickle.load(open("scaler.pkl", "rb"))
    columns = pickle.load(open("columns.pkl", "rb"))
    return model, scaler, columns

model, scaler, columns = load_artifacts()

# PAGE CONFIG 

st.set_page_config(
    page_title="Real Estate Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="collapsed" if not st.session_state.logged_in else "expanded"
)

# GLOBAL CUSTOM CSS

st.markdown(
    """
    <style>
    /* Remove default padding and make background full-screen gradient */
    .stApp {
        background: radial-gradient(circle at 0% 0%, #1f2937 0, #020617 50%, #000000 100%);
        color: #f9fafb;
        font-family: "Inter", system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }

    /* Center content and give it a 3D card look */
    .main > div {
        padding-top: 1rem;
    }

    .block-container {
        max-width: 1100px;
        padding-top: 1rem;
        padding-bottom: 2rem;
    }

    /* Glassmorphism cards */
    .glass-card {
        background: linear-gradient(135deg, rgba(15,23,42,0.85), rgba(30,64,175,0.55));
        border-radius: 24px;
        padding: 1.8rem 2.2rem;
        border: 1px solid rgba(148,163,184,0.35);
        box-shadow:
            0 18px 45px rgba(0,0,0,0.65),
            0 0 0 1px rgba(148,163,184,0.15);
        backdrop-filter: blur(18px);
        -webkit-backdrop-filter: blur(18px);
        transform: translateY(0px);
        transition: transform 200ms ease, box-shadow 200ms ease, border-color 200ms ease;
    }

    .glass-card:hover {
        transform: translateY(-4px);
        box-shadow:
            0 28px 60px rgba(0,0,0,0.85),
            0 0 0 1px rgba(191,219,254,0.25);
        border-color: rgba(191,219,254,0.45);
    }

    /* Login card specific styling */
    .login-card {
        background: linear-gradient(135deg, rgba(15,23,42,0.92), rgba(30,64,175,0.65));
        border-radius: 28px;
        padding: 1.6rem 1.6rem;
        border: 1px solid rgba(148,163,184,0.4);
        box-shadow:
            0 24px 55px rgba(0,0,0,0.75),
            0 0 0 1px rgba(148,163,184,0.2);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        max-width: 300px;
        margin: 0 auto;
    }

    /* House logo styling */
    .house-logo {
        font-size: 3rem;
        text-align: center;
        margin-bottom: 0.2rem;
        filter: drop-shadow(0 0 20px rgba(59,130,246,0.5));
        animation: float 3s ease-in-out infinite;
    }

    @keyframes float {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-10px); }
    }

    /* Login title */
    .login-title {
        font-size: 1.3rem;
        font-weight: 800;
        text-align: center;
        letter-spacing: 0.15em;
        background: linear-gradient(120deg, #e5e7eb, #93c5fd, #f97316);
        -webkit-background-clip: text;
        color: transparent;
        margin-bottom: 2rem;
        text-shadow:
            0 0 18px rgba(59,130,246,0.35);
    }

    /* Main title styling with subtle neon glow */
    .main-title {
        font-size: 2.6rem;
        font-weight: 800;
        letter-spacing: 0.03em;
        background: linear-gradient(120deg, #e5e7eb, #93c5fd, #f97316);
        -webkit-background-clip: text;
        color: transparent;
        text-shadow:
            0 0 18px rgba(59,130,246,0.35),
            0 0 32px rgba(249,115,22,0.25);
        margin-bottom: 0.2rem;
    }

    .subtitle {
        color: #cbd5f5;
        font-size: 0.98rem;
    }

    /* Welcome message */
    .welcome-msg {
        background: linear-gradient(135deg, rgba(34,197,94,0.15), rgba(59,130,246,0.15));
        border-left: 4px solid #22c55e;
        padding: 1rem 1.5rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        color: #e5e7eb;
    }

    /* Logout button styling */
    .logout-container {
        text-align: right;
        margin-bottom: 1rem;
    }

    /* Sidebar glass panel */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(15,23,42,0.92), rgba(15,23,42,0.75));
        border-right: 1px solid rgba(148,163,184,0.45);
        box-shadow: 16px 0 40px rgba(0,0,0,0.6);
        backdrop-filter: blur(18px);
        -webkit-backdrop-filter: blur(18px);
    }

    [data-testid="stSidebar"] * {
        color: #e5e7eb !important;
    }

    /* Input fields styling */
    .stTextInput > div > div > input {
        background: rgba(15,23,42,0.6);
        border: 1px solid rgba(148,163,184,0.3);
        border-radius: 12px;
        color: #e5e7eb;
        padding: 0.75rem 1rem;
        font-size: 1rem;
    }

    .stTextInput > div > div > input:focus {
        border-color: #3b82f6;
        box-shadow: 0 0 0 2px rgba(59,130,246,0.3);
    }

    /* Sliders and number inputs */
    .stSlider > label, .stNumberInput > label {
        font-weight: 600;
        color: #e5e7eb;
    }

    /* 3D Predict button */
    .stButton > button {
        width: 100%;
        border-radius: 999px;
        padding: 0.9rem 1.4rem;
        border: 1px solid rgba(191,219,254,0.5);
        background: radial-gradient(circle at 0% 0%, #38bdf8, #2563eb, #4f46e5);
        color: #f9fafb;
        font-weight: 700;
        font-size: 1rem;
        letter-spacing: 0.03em;
        box-shadow:
            0 18px 30px rgba(37,99,235,0.45),
            0 0 0 1px rgba(191,219,254,0.3);
        text-shadow: 0 1px 3px rgba(15,23,42,0.75);
        transform: translateY(0px);
        transition: transform 120ms ease, box-shadow 120ms ease, filter 120ms ease;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow:
            0 26px 40px rgba(37,99,235,0.7),
            0 0 0 1px rgba(191,219,254,0.6);
        filter: brightness(1.05);
    }

    .stButton > button:active {
        transform: translateY(1px) scale(0.99);
        box-shadow:
            0 12px 24px rgba(15,23,42,0.9),
            0 0 0 1px rgba(148,163,184,0.5);
    }

    /* Metrics glow */
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, rgba(15,23,42,0.95), rgba(30,64,175,0.8));
        border-radius: 18px;
        padding: 0.85rem 0.9rem;
        box-shadow:
            0 14px 26px rgba(15,23,42,0.85),
            0 0 0 1px rgba(148,163,184,0.35);
    }

    [data-testid="stMetric"] label {
        color: #e5e7eb !important;
        font-weight: 600;
        font-size: 0.8rem;
    }

    [data-testid="stMetricValue"] {
        color: #bfdbfe !important;
        font-weight: 800 !important;
        font-size: 1.1rem !important;
    }

    /* Tables glass style */
    .stDataFrame, .stTable {
        border-radius: 18px;
        overflow: hidden;
        box-shadow: 0 14px 30px rgba(15,23,42,0.9);
    }

    /* Divider line */
    .soft-divider {
        height: 1px;
        width: 100%;
        margin: 1.6rem 0;
        background: linear-gradient(90deg, rgba(15,23,42,0), rgba(148,163,184,0.8), rgba(15,23,42,0));
    }
    </style>
    """,
    unsafe_allow_html=True,
)

#  LOGIN PAGE 
def login_page():
    # Center the login card vertically
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    # Create centered column layout
    col1, col2, col3 = st.columns([1, 1.2, 1])
    
    with col2:
        st.markdown(
            """
            <div class="login-card">
                <div class="house-logo">🏠</div>
                <div class="login-title">LOGIN</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Login form inside the glass card
        with st.form("login_form"):
            username = st.text_input("Username", placeholder="Enter your username")
            password = st.text_input("Password", type="password", placeholder="Enter your password")
            
            st.markdown("<br>", unsafe_allow_html=True)
            submit = st.form_submit_button("🔓 Login")
            
            if submit:
                # Simple authentication
                if username and password:
                    if username == "Ayush" and password == "lolipop123":
                        st.session_state.logged_in = True
                        st.session_state.username = username
                        st.rerun()
                    else:
                        st.error("❌ Invalid credentials. Please try again.")
                else:
                    st.warning("⚠️ Please enter both username and password")
        
        st.markdown(
            """
            <div style='text-align: center; margin-top: 2rem; color: #94a3b8; font-size: 0.85rem;'>
                <p>Droid-DevX: Real Estate Price Predictor</p>
            </div>
            """,
            unsafe_allow_html=True
        )

# MAIN APP 

def main_app():
    # Logout button
    col1, col2 = st.columns([4, 1])
    with col2:
        if st.button("🚪 Logout"):
            st.session_state.logged_in = False
            st.session_state.username = ""
            st.rerun()
    
    # Welcome message
    st.markdown(
        f"""
        <div class="welcome-msg">
            👋 <b>Welcome back, {st.session_state.username}!</b> Ready to predict house prices?
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # SIDEBAR INPUTS 
    
    st.sidebar.markdown("### House Configuration")
    st.sidebar.caption("Tune the parameters and generate a 3D-styled price estimate.")
    
    overall_qual = st.sidebar.slider("Overall Quality (1–10)", 1, 10, 5)
    grlivarea = st.sidebar.number_input("Ground Living Area (sq ft)", min_value=300, max_value=10000, value=1500)
    garagecars = st.sidebar.slider("Garage Capacity (cars)", 0, 5, 2)
    fullbath = st.sidebar.slider("Full Bathrooms (above grade)", 0, 4, 2)
    yearbuilt = st.sidebar.number_input("Year Built", min_value=1800, max_value=2025, value=2005)
    
    st.sidebar.markdown("---")
    st.sidebar.write(" After setting values, click **Predict House Price** on the main panel.")
    
    # BUILD FULL FEATURE ROW 
    
    input_df = pd.DataFrame(columns=columns)
    input_df.loc[0] = 0  # initialize all features to 0
    
    if "OverallQual" in input_df.columns:
        input_df.loc[0, "OverallQual"] = overall_qual
    
    if "GrLivArea" in input_df.columns:
        input_df.loc[0, "GrLivArea"] = grlivarea
    
    if "GarageCars" in input_df.columns:
        input_df.loc[0, "GarageCars"] = garagecars
    
    if "FullBath" in input_df.columns:
        input_df.loc[0, "FullBath"] = fullbath
    
    if "YearBuilt" in input_df.columns:
        input_df.loc[0, "YearBuilt"] = yearbuilt
    
    #  MAIN HEADER 
    
    st.markdown(
        """
        <div class="glass-card">
            <div class="main-title">🏠 Real Estate Value Predictor</div>
            <p class="subtitle">
                A 3D-styled ML web app powered by <b>Linear Regression</b>.  
                Estimate the sale price of a house using key structural features.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    st.markdown('<div class="soft-divider"></div>', unsafe_allow_html=True)
    
    # LAYOUT: LEFT (inputs info + chart) / RIGHT (prediction + metrics) 
    
    left_col, right_col = st.columns([1.4, 1])
    
       # LEFT SIDE
          
    with left_col:
        from datetime import datetime

        st.subheader("📋 Selected Features")

        st.write(
            pd.DataFrame(
                {
                    "Feature": ["Overall Quality", "Living Area (sq ft)", "Garage Cars", "Full Bathrooms", "Year Built"],
                    "Value": [overall_qual, grlivarea, garagecars, fullbath, yearbuilt],
                }
            )
        )

        st.markdown("### 📊 Feature Overview")

        # compute house age from YearBuilt (use current year)
        current_year = datetime.now().year
        house_age = current_year - yearbuilt
        if house_age < 0:
            house_age = 0
        house_age_display = int(house_age)

        chart_data = pd.DataFrame(
            {
                "Feature": ["OverallQual", "House Age (yrs)", "GarageCars", "FullBath"],
                "Value": [overall_qual, house_age_display, garagecars, fullbath],
            }
        ).set_index("Feature")

        st.bar_chart(chart_data, use_container_width=True)

        st.markdown("</div>", unsafe_allow_html=True)


    # RIGHT SIDE
    with right_col:
    
        st.subheader(" Price Estimation")
    
        predicted_price = None
    
        if st.button("Predict House Price"):
            # Scale features using the same scaler as training
            input_scaled = scaler.transform(input_df)
            pred_log = model.predict(input_scaled)
            price = np.expm1(pred_log)  
    
            predicted_price = float(price[0])
    
            st.success(f"Estimated House Price:\n\n### $ {predicted_price:,.0f}")
    
            st.info(
                "This estimate is generated using a Linear Regression model trained on the **Ames Housing** dataset "
                "with feature engineering, encoding, and scaling."
            )
        else:
            st.caption("Click the button above to generate an estimate based on your sidebar inputs.")
    
        st.markdown("---")
    
        st.subheader("Quick Metrics")
        m1, m2 = st.columns(2)
        m3, m4 = st.columns(2)
    
        m1.metric(label="Overall Quality", value=overall_qual)
        m2.metric(label="Living Area (sq ft)", value=grlivarea)
        m3.metric(label="Garage (cars)", value=garagecars)
        m4.metric(label="Year Built", value=yearbuilt)
    
        st.markdown("</div>", unsafe_allow_html=True)

# MAIN ROUTING 

if not st.session_state.logged_in:
    login_page()
else:
    main_app()

