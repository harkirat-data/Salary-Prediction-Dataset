import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings('ignore')

# ── PAGE CONFIG ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Salary Predictor", layout="centered")

st.markdown("<h1 style='text-align: center;'>Salary Predictor 💼</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: gray;'>Enter your details to predict your salary</p>", unsafe_allow_html=True)

# ── LOAD DATA ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    return pd.read_csv("job_salary_prediction_dataset.csv")

# ── TRAIN MODEL ───────────────────────────────────────────────────────────────
@st.cache_resource
def train_model(df):
    cat_cols   = df.select_dtypes(include='object').columns
    df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    X = df_encoded.drop('salary', axis=1)
    y = df_encoded['salary']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # All 3 models
    lr  = LinearRegression()
    rf  = RandomForestRegressor(n_estimators=100, max_depth=30, random_state=42)
    xgb = XGBRegressor(n_estimators=100, max_depth=10, random_state=42, verbosity=0)

    scaler     = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    lr.fit(X_train_sc, y_train)
    rf.fit(X_train,    y_train)
    xgb.fit(X_train,   y_train)

    results = []
    for name, model, Xte in [
        ('Linear Regression', lr,  X_test_sc),
        ('Random Forest',     rf,  X_test),
        ('XGBoost',           xgb, X_test),
    ]:
        y_pred = model.predict(Xte)
        results.append({
            'Model':     name,
            'MAE':       round(mean_absolute_error(y_test, y_pred), 2),
            'RMSE':      round(np.sqrt(mean_squared_error(y_test, y_pred)), 2),
            'R-squared': round(r2_score(y_test, y_pred), 4),
        })

    return xgb, scaler, X.columns.tolist(), pd.DataFrame(results)

# ── LOAD ──────────────────────────────────────────────────────────────────────
try:
    df = load_data()
except FileNotFoundError:
    st.error("❌ `job_salary_prediction_dataset.csv` not found! Put the CSV in the same folder as app.py.")
    st.stop()

xgb_model, scaler, feature_cols, results_df = train_model(df)

# ── SIDEBAR INPUTS ────────────────────────────────────────────────────────────
st.sidebar.header("Your Profile")

Experience = st.sidebar.slider("Experience (years):", 0, 20, 8)
Education  = st.sidebar.selectbox("Education Level:", ["High School", "Diploma", "Bachelor", "Master", "PhD"])
Industry   = st.sidebar.selectbox("Industry:", ["Healthcare", "Education", "Telecom", "Government",
                                                  "Consulting", "Manufacturing", "Media", "Finance",
                                                  "Technology", "Retail"])
Job_Title  = st.sidebar.selectbox("Job Title:", ["AI Engineer", "Data Analyst", "Frontend Developer",
                                                   "Business Analyst", "Product Manager", "Backend Developer",
                                                   "Machine Learning Engineer", "DevOps Engineer",
                                                   "Software Engineer", "Cybersecurity Analyst",
                                                   "Data Scientist", "Cloud Engineer"])
Skills     = st.sidebar.slider("Skills Count:", 1, 19, 8)
Company    = st.sidebar.selectbox("Company Size:", ["Small", "Medium", "Large", "Enterprise", "Startup"])
Location   = st.sidebar.selectbox("Location:", ["India", "USA", "UK", "Australia", "Singapore",
                                                  "Canada", "Germany", "Netherlands", "Sweden", "Remote"])
Remote     = st.sidebar.selectbox("Remote Work:", ["Yes", "No", "Hybrid"])
Certs      = st.sidebar.slider("Certifications:", 0, 5, 1)

predicted_btn = st.sidebar.button("Predict Salary")

# ── MAIN LAYOUT ───────────────────────────────────────────────────────────────
col1, col2 = st.columns([1, 2])

with col1:
    st.markdown("<h3 style='text-align: left;'>Profile Summary</h3>", unsafe_allow_html=True)
    st.markdown(f"**Experience:** {Experience} years")
    st.markdown(f"**Education Level:** {Education}")
    st.markdown(f"**Industry:** {Industry}")
    st.markdown(f"**Job Title:** {Job_Title}")
    st.markdown(f"**Skills:** {Skills}")
    st.markdown(f"**Company Size:** {Company}")
    st.markdown(f"**Location:** {Location}")
    st.markdown(f"**Remote Work:** {Remote}")
    st.markdown(f"**Certifications:** {Certs}")

with col2:
    st.markdown("<h3 style='text-align: right;'>Prediction Result</h3>", unsafe_allow_html=True)

    if predicted_btn:
        # Build input for model
        input_raw = {
            'experience_years': Experience,
            'skills_count':     Skills,
            'certifications':   Certs,
            'job_title':        Job_Title,
            'education_level':  Education,
            'industry':         Industry,
            'company_size':     Company,
            'location':         Location,
            'remote_work':      Remote,
        }

        input_df      = pd.DataFrame([input_raw])
        cat_in        = input_df.select_dtypes(include='object').columns
        input_encoded = pd.get_dummies(input_df, columns=cat_in, drop_first=True)

        # Align with training columns
        for col in feature_cols:
            if col not in input_encoded.columns:
                input_encoded[col] = 0
        input_encoded = input_encoded[feature_cols]

        predicted_salary = xgb_model.predict(input_encoded)[0]

        # Show result
        st.markdown(
            f"""
            <div style="background-color:#111827; padding:40px;
                        border-radius:12px; text-align:center;">
                <h1 style="color:#22c55e;">${predicted_salary:,.0f}</h1>
                <p style="color:gray;">Estimated Annual Salary (XGBoost)</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("")

        # Real metrics
        xgb_row = results_df[results_df['Model'] == 'XGBoost'].iloc[0]
        colA, colB, colC = st.columns(3)
        colA.metric("R² Score", f"{xgb_row['R-squared']:.4f}")
        colB.metric("RMSE",     f"{xgb_row['RMSE']:,.0f}")
        colC.metric("MAE",      f"{xgb_row['MAE']:,.0f}")

    else:
        st.info("Click the 'Predict Salary' button in the sidebar to see your estimated salary based on the provided details.")

# ── MODEL COMPARISON ──────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### 🤖 Model Comparison")
st.dataframe(results_df, use_container_width=True, hide_index=True)

col_p1, col_p2 = st.columns(2)

with col_p1:
    fig_r2 = px.bar(results_df, x='Model', y='R-squared',
                    title='R² Score Comparison',
                    color='Model', text='R-squared',
                    color_discrete_sequence=['#22c55e', '#3b82f6', '#f59e0b'])
    fig_r2.update_traces(texttemplate='%{text:.4f}', textposition='outside')
    fig_r2.update_layout(showlegend=False, yaxis_range=[0.95, 0.98])
    st.plotly_chart(fig_r2, use_container_width=True)

with col_p2:
    fig_rmse = px.bar(results_df, x='Model', y='RMSE',
                      title='RMSE Comparison',
                      color='Model', text='RMSE',
                      color_discrete_sequence=['#22c55e', '#3b82f6', '#f59e0b'])
    fig_rmse.update_traces(texttemplate='%{text:,.0f}', textposition='outside')
    fig_rmse.update_layout(showlegend=False)
    st.plotly_chart(fig_rmse, use_container_width=True)

# ── EDA ───────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### 📊 Data Insights")

col_e1, col_e2 = st.columns(2)

with col_e1:
    fig_edu = px.box(df, x='education_level', y='salary',
                     title='Salary by Education Level',
                     color='education_level')
    st.plotly_chart(fig_edu, use_container_width=True)

with col_e2:
    fig_exp = px.scatter(df.sample(3000, random_state=42),
                         x='experience_years', y='salary',
                         title='Salary vs Experience',
                         opacity=0.4,
                         color_discrete_sequence=['#22c55e'])
    st.plotly_chart(fig_exp, use_container_width=True)

col_e3, col_e4 = st.columns(2)

with col_e3:
    fig_loc = px.violin(df, x='location', y='salary',
                        title='Salary by Location',
                        color='location')
    st.plotly_chart(fig_loc, use_container_width=True)

with col_e4:
    fig_cs = px.violin(df, x='company_size', y='salary',
                       title='Salary by Company Size',
                       color='company_size')
    st.plotly_chart(fig_cs, use_container_width=True)

# ── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:gray; font-size:0.8rem;'>"
    "Built by Harkirat · B.Tech CSE (Data Science) · NIT · GDG Lead Task #2"
    "</p>",
    unsafe_allow_html=True,
)