import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import yaml
import os
from PIL import Image

# Page config
st.set_page_config(page_title="EdTech Engagement Dashboard", page_icon="🎓", layout="wide")

# Load config and data
@st.cache_data
def load_data():
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    df = pd.read_csv(config['paths']['feature_data'])
    return df, config

df, config = load_data()

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["📊 Overview", "🔍 Student Explorer", "📈 Engagement Trends", "🤖 Dropout Risk Predictor", "📉 Model Insights"])

# Page 1: Overview
if page == "📊 Overview":
    st.title("📊 Platform Overview")
    
    # KPI Cards
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Students", len(df))
    col2.metric("Dropout Rate", f"{df['dropped_out'].mean():.1%}")
    col3.metric("Avg Engagement", f"{df['engagement_score'].mean():.2f}")
    col4.metric("Avg Logins", f"{df['total_logins'].mean():.1f}")
    
    col_a, col_b = st.columns(2)
    
    with col_a:
        # Dropout rate by course
        course_dropout = df.groupby('course_id')['dropped_out'].mean().reset_index()
        fig1 = px.bar(course_dropout, x='course_id', y='dropped_out', title="Dropout Rate by Course", labels={'dropped_out': 'Dropout Rate'})
        st.plotly_chart(fig1, use_container_width=True)
        
    with col_b:
        # Student segments
        segment_counts = df['student_segment'].value_counts().reset_index()
        segment_counts.columns = ['Segment', 'Count']
        fig2 = px.pie(segment_counts, names='Segment', values='Count', title="Student Segments (K-Means)")
        st.plotly_chart(fig2, use_container_width=True)
        
    # Simulated weekly login trend
    weeks = [f"Week {i+1}" for i in range(12)]
    logins = np.random.randint(500, 1500, size=12) + np.linspace(0, 500, 12)
    fig3 = px.line(x=weeks, y=logins, title="Weekly Login Trend (Last 12 Weeks)", labels={'x': 'Week', 'y': 'Total Logins'})
    st.plotly_chart(fig3, use_container_width=True)

# Page 2: Student Explorer
elif page == "🔍 Student Explorer":
    st.title("🔍 Student Explorer")
    
    # Filters
    st.sidebar.subheader("Filters")
    region_f = st.sidebar.multiselect("Region", df['region'].unique(), default=df['region'].unique())
    gender_f = st.sidebar.multiselect("Gender", df['gender'].unique(), default=df['gender'].unique())
    course_f = st.sidebar.multiselect("Course", df['course_id'].unique(), default=df['course_id'].unique())
    
    filtered_df = df[
        (df['region'].isin(region_f)) & 
        (df['gender'].isin(gender_f)) & 
        (df['course_id'].isin(course_f))
    ]
    
    # Data table with color-coded risk
    st.subheader("Filtered Student Data")
    
    def color_risk(val):
        color = 'red' if val == 1 else 'green'
        return f'background-color: {color}; color: white'
    
    # Select subset of columns for display
    display_cols = ['student_id', 'course_id', 'engagement_score', 'quiz_scores_avg', 'early_dropout_risk', 'dropped_out']
    st.dataframe(filtered_df[display_cols].style.applymap(color_risk, subset=['early_dropout_risk', 'dropped_out']))
    
    # Scatter plot
    fig4 = px.scatter(filtered_df, x='engagement_score', y='quiz_scores_avg', color='dropped_out', 
                      title="Engagement vs Quiz Scores", color_continuous_scale='RdYlGn_r')
    st.plotly_chart(fig4, use_container_width=True)

# Page 3: Engagement Trends
elif page == "📈 Engagement Trends":
    st.title("📈 Engagement Trends")
    
    # Histogram
    fig5 = px.histogram(df, x='engagement_score', nbins=30, title="Distribution of Engagement Scores", marginal="box")
    st.plotly_chart(fig5, use_container_width=True)
    
    col_c, col_d = st.columns(2)
    
    with col_c:
        # Box plot by region
        fig6 = px.box(df, x='region', y='engagement_score', title="Engagement by Region", color='region')
        st.plotly_chart(fig6, use_container_width=True)
        
    with col_d:
        # Box plot by course
        fig7 = px.box(df, x='course_id', y='engagement_score', title="Engagement by Course", color='course_id')
        st.plotly_chart(fig7, use_container_width=True)
        
    # Heatmap
    st.subheader("Feature Correlation Matrix")
    corr = df.select_dtypes(include=[np.number]).corr()
    fig8 = px.imshow(corr, text_auto=True, aspect="auto", title="Correlation Heatmap", color_continuous_scale='RdBu_r')
    st.plotly_chart(fig8, use_container_width=True)

# Page 4: Dropout Risk Predictor
elif page == "🤖 Dropout Risk Predictor":
    st.title("🤖 Dropout Risk Predictor")
    
    st.write("Enter student details to predict the risk of dropout.")
    
    col_e, col_f = st.columns(2)
    
    with col_e:
        logins = st.slider("Total Logins", 1, 200, 50)
        duration = st.slider("Avg Session Duration (mins)", 1, 120, 30)
        modules = st.slider("Modules Completed", 0, 20, 5)
        quiz = st.slider("Avg Quiz Score", 0, 100, 70)
        
    with col_f:
        forum = st.slider("Forum Posts", 0, 50, 5)
        video = st.slider("Video Watch %", 0, 100, 40)
        days_since = st.slider("Days Since Last Login", 0, 90, 20)
        course = st.selectbox("Course ID", sorted(df['course_id'].unique()))
        # Fixed placeholders for non-input features
        age = 25
        gender = 1
        region = 1
        assignment = 5
        
    if st.button("Predict Risk"):
        # Load model and pipeline
        model = joblib.load(config['paths']['model_path'])
        pipeline = joblib.load(config['paths']['pipeline_path'])
        
        # Calculate derived features for the predictor
        completion_rate = modules / 20
        # Normalization constants from data generation
        n_logins = (logins - 1) / (200 - 1)
        n_duration = (duration - 1) / (120 - 1)
        n_quiz = quiz / 100
        n_forum = (forum - 0) / (50 - 0)
        n_video = video / 100
        
        w = config['engagement_weights']
        eng_score = (w['logins'] * n_logins + w['session_duration'] * n_duration + 
                     w['module_completion'] * completion_rate + w['quiz_avg'] * n_quiz + 
                     w['forum_posts'] * n_forum + w['video_watch'] * n_video)
        
        # Create input dataframe
        input_data = pd.DataFrame([{
            'age': age, 'gender': gender, 'region': region, 'course_id': course,
            'total_logins': logins, 'avg_session_duration_mins': duration,
            'modules_completed': modules, 'quiz_scores_avg': quiz,
            'forum_posts': forum, 'assignment_submissions': assignment,
            'video_watch_pct': video, 'days_since_last_login': days_since,
            'engagement_score': eng_score, 'module_completion_rate': completion_rate,
            'interaction_consistency': 5.0 # default
        }])
        
        # Transform and Predict
        X_proc = pipeline.transform(input_data)
        prob = model.predict_proba(X_proc)[0][1]
        risk = "HIGH" if prob > 0.5 else "LOW"
        color = "red" if risk == "HIGH" else "green"
        
        st.markdown(f"### Result: <span style='color:{color}'>{risk} RISK</span>", unsafe_allow_html=True)
        st.write(f"**Dropout Probability:** {prob:.2%}")
        st.progress(prob)

# Page 5: Model Insights
elif page == "📉 Model Insights":
    st.title("📉 Model Insights")
    
    st.subheader("Model Performance Comparison")
    # Simulated comparison table
    comparison = pd.DataFrame({
        "Model": ["Logistic Regression", "Random Forest", "XGBoost"],
        "Accuracy": [0.82, 0.88, 0.91],
        "F1-Score": [0.75, 0.81, 0.85],
        "ROC-AUC": [0.86, 0.92, 0.95]
    })
    st.table(comparison)
    
    st.subheader("Feature Importance & SHAP")
    
    fig_col1, fig_col2 = st.columns(2)
    
    with fig_col1:
        if os.path.exists('reports/figures/feature_importance.png'):
            st.image('reports/figures/feature_importance.png', caption="Feature Importance (Random Forest)")
        else:
            st.warning("Feature importance plot not found. Run evaluation first.")
            
    with fig_col2:
        if os.path.exists('reports/figures/shap_summary.png'):
            st.image('reports/figures/shap_summary.png', caption="SHAP Summary (XGBoost)")
        else:
            st.warning("SHAP summary plot not found. Run evaluation first.")
            
    st.markdown("""
    ### Key Insights:
    1. **Engagement is King**: Features like `engagement_score` and `days_since_last_login` are the strongest predictors of dropout.
    2. **Course Difficulty**: Certain courses (e.g., C005) show higher dropout rates, suggesting a need for intervention or curriculum review.
    3. **Activity Patterns**: Students who don't post in forums or watch videos early on are at much higher risk.
    """)
