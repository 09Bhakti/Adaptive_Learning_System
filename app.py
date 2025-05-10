
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
import pandas as pd
from clustering import cluster_student
from recommender import get_recommendations

st.set_page_config(page_title="AI Learning Pattern Analyzer", layout="wide")

st.title("AI Learning Pattern Analyzer & Predictor")
st.markdown("This tool analyzes your learning patterns and predicts your skill level based on the data you provide. It also generates personalized recommendations and a study plan.")

with st.form("learning_form"):
    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            study_hours = st.slider("Weekly Study Hours", 1, 40, 10, help="How many hours do you study per week?")
            quiz_score = st.slider("Average Quiz/Test Score (%)", 0, 100, 75, help="Your average score in quizzes or tests.")

        with col2:
            completion_rate = st.slider("Course Completion Rate (%)", 0, 100, 50, help="Percentage of completed courses.")
            consistency = st.slider("Learning Consistency (1-10)", 1, 10, 5, help="How consistent is your study routine?")

    st.markdown("### Additional Information")
    col3, col4 = st.columns(2)
    with col3:
        learning_style = st.selectbox("Preferred Learning Style", ["visual", "auditory", "reading/writing", "kinesthetic"], help="Choose your preferred learning style.")
        topic_difficulty = st.slider("Preferred Topic Difficulty (1-10)", 1, 10, 5, help="Rate the difficulty level of topics you prefer.")
    with col4:
        learning_interests = st.text_area("Learning Interests", placeholder="E.g., data science, web development, AI", help="Mention areas you are interested in.")
        prior_knowledge = st.slider("Prior Knowledge (1-10)", 1, 10, 3, help="Your prior knowledge in the domain.")

    submitted = st.form_submit_button("Analyze My Learning Pattern")

if submitted:
    input_df = pd.DataFrame([{
        "study_hours": study_hours,
        "quiz_score": quiz_score,
        "completion_rate": completion_rate,
        "consistency": consistency,
        "learning_style": learning_style,
        "topic_difficulty": topic_difficulty,
        "learning_interests": learning_interests,
        "prior_knowledge": prior_knowledge
    }])

    st.markdown("## 📊 Your Learning Pattern Analysis")
    
    cluster_label = cluster_student(input_df)
    st.write(f"🔍 **Predicted Cluster/Group:** {cluster_label}")

    st.markdown("## 📌 Personalized Recommendations")
    recommendations = get_recommendations(input_df)
    st.success(recommendations)
