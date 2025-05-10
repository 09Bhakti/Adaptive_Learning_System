
# import streamlit as st
# import pandas as pd
# from clustering import cluster_students_advanced
# from recommender import predict_learning_strategy

# st.title("📊 Adaptive Learning System")

# # Simulated Data Upload
# uploaded_file = st.file_uploader("Upload student performance CSV")

# if uploaded_file:
#     df = pd.read_csv(uploaded_file)
#     clusters_df = cluster_students_advanced(df)

#     student_id = st.selectbox("Select Student ID", clusters_df["student_id"])
#     student_row = clusters_df[clusters_df["student_id"] == student_id].iloc[0]

#     strategy = predict_learning_strategy(student_row)

#     st.subheader("📌 Personalized Recommendation")
#     st.write(f"Difficulty Level: **{strategy['difficulty']}**")
#     st.write(f"Engagement Risk: **{strategy['risk']}**")
#     st.write(f"Recommendation: {strategy['recommendation']}")


# app.py
import streamlit as st
import pandas as pd
from clustering import cluster_students_advanced
from recommender import predict_learning_strategy

st.set_page_config(
    page_title="AI-Powered Adaptive Learning System",
    page_icon="📚",
    layout="wide",
)

st.markdown("""
    <style>
    .main-title {
        font-size:40px !important;
        color:#4A90E2;
        font-weight:700;
    }
    .subheader {
        font-size:20px !important;
        margin-top:10px;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-title">📚 AI-Based Adaptive Learning Recommendation System</p>', unsafe_allow_html=True)

st.sidebar.title("Upload Student Data")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    clusters_df = cluster_students_advanced(df)

    with st.sidebar:
        student_id = st.selectbox("Select Student ID", clusters_df["student_id"])

    student_row = clusters_df[clusters_df["student_id"] == student_id].iloc[0]
    strategy = predict_learning_strategy(student_row)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🎯 Personalized Strategy")
        st.markdown(f"**Recommended Difficulty:** {strategy['difficulty']}")
        st.markdown(f"**Engagement Risk:** {strategy['risk']}")

    with col2:
        st.subheader("🧠 Actionable Guidance")
        st.markdown(f"**What to do:** {strategy['recommendation']}")

    st.markdown("""---
    <center><small>Developed for <strong>@Prasunet</strong> by Bhakti using Streamlit, AI, and ML models</small></center>
    """, unsafe_allow_html=True)
else:
    st.info("Please upload a student performance CSV to get started.")

