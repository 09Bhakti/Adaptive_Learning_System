
import streamlit as st
import pandas as pd
from clustering import cluster_students_advanced
from recommender import predict_learning_strategy

st.title("📊 Student Learning Profile Analyzer")

# Simulated Data Upload
uploaded_file = st.file_uploader("Upload student performance CSV")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    clusters_df = cluster_students_advanced(df)

    student_id = st.selectbox("Select Student ID", clusters_df["student_id"])
    student_row = clusters_df[clusters_df["student_id"] == student_id].iloc[0]

    strategy = predict_learning_strategy(student_row)

    st.subheader("📌 Personalized Recommendation")
    st.write(f"Difficulty Level: **{strategy['difficulty']}**")
    st.write(f"Engagement Risk: **{strategy['risk']}**")
    st.write(f"Recommendation: {strategy['recommendation']}")
