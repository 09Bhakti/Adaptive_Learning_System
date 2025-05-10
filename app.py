
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

import streamlit as st

st.set_page_config(page_title="AI Learning Pattern Analyzer & Predictor", layout="centered")
st.title("AI Learning Pattern Analyzer & Predictor")
st.markdown("""
This tool analyzes your learning patterns and predicts your skill level based on the data you provide. It also generates personalized recommendations and a study plan.
""")

with st.form("learning_pattern_form"):
    st.subheader("Learning Metrics")
    col1, col2 = st.columns(2)
    with col1:
        study_hours = st.slider("Weekly Study Hours", 1, 40, 10, help="Estimated number of hours spent studying per week")
        avg_score = st.slider("Average Quiz/Test Score (%)", 0, 100, 75, help="Your average score across assessments")
    with col2:
        course_completion = st.slider("Course Completion Rate (%)", 0, 100, 50, help="Percentage of courses completed")
        consistency = st.slider("Learning Consistency (1-10)", 1, 10, 5, help="How consistent your study sessions are")

    st.subheader("Additional Information")
    col3, col4 = st.columns(2)
    with col3:
        learning_style = st.selectbox("Preferred Learning Style", ["visual", "auditory", "reading/writing", "kinesthetic"], help="Choose your preferred learning style")
        difficulty = st.slider("Preferred Topic Difficulty (1-10)", 1, 10, 5, help="Level of challenge you prefer in learning material")
    with col4:
        interests = st.text_area("Learning Interests", placeholder="E.g., data science, web development, AI")
        prior_knowledge = st.slider("Prior Knowledge (1-10)", 1, 10, 3, help="Your self-assessed prior knowledge")

    submitted = st.form_submit_button("Analyze My Learning Pattern")

    if submitted:
        st.success("Form submitted successfully!")
        st.write("### Summary of Inputs")
        st.write(f"**Study Hours**: {study_hours} hours/week")
        st.write(f"**Average Score**: {avg_score}%")
        st.write(f"**Course Completion**: {course_completion}%")
        st.write(f"**Consistency**: {consistency}/10")
        st.write(f"**Learning Style**: {learning_style}")
        st.write(f"**Preferred Difficulty**: {difficulty}/10")
        st.write(f"**Prior Knowledge**: {prior_knowledge}/10")
        st.write(f"**Learning Interests**: {interests}")

        # This is where you would pass data to ML model or recommendation engine
        st.info("This is a placeholder for recommendation output. Integrate model prediction here.")
