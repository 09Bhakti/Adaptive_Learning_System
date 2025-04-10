
import streamlit as st

st.title("Adaptive Learning System")

score = st.slider("Enter your last test score:", 0, 100)
if st.button("Get Recommendation"):
    if score < 50:
        st.write("📘 Beginner Level Content Recommended")
    elif 50 <= score < 80:
        st.write("📗 Intermediate Level Content Recommended")
    else:
        st.write("📕 Advanced Level Content Recommended")
