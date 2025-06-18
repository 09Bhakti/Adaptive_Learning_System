import streamlit as st
import pandas as pd
from clustering import cluster_student
from recommender import get_recommendations  # This must be defined in recommender.py

# Set page layout
st.set_page_config(page_title="AI Learning Pattern Analyzer", layout="wide")

# Title and description
st.title("AI Learning Pattern Analyzer & Predictor")
st.markdown("This tool analyzes your learning patterns and predicts your skill level based on the data you provide. It also generates personalized recommendations and a study plan.")

# Input form
with st.form("learning_form"):
    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            study_hours = st.slider("Weekly Study Hours", 1, 40, 10)
            quiz_score = st.slider("Average Quiz/Test Score (%)", 0, 100, 75)

        with col2:
            completion_rate = st.slider("Course Completion Rate (%)", 0, 100, 50)
            consistency = st.slider("Learning Consistency (1-10)", 1, 10, 5)

    st.markdown("### Additional Information")
    col3, col4 = st.columns(2)
    with col3:
        learning_style = st.selectbox("Preferred Learning Style", ["visual", "auditory", "reading/writing", "kinesthetic"])
        topic_difficulty = st.slider("Preferred Topic Difficulty (1-10)", 1, 10, 5)
    with col4:
        learning_interests = st.text_area("Learning Interests", placeholder="E.g., data science, web development, AI")
        prior_knowledge = st.slider("Prior Knowledge (1-10)", 1, 10, 3)

    submitted = st.form_submit_button("Analyze My Learning Pattern")

# If form submitted
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

    st.markdown("## 💡 Personalized Recommendations")
    recommendations = get_recommendations(input_df.iloc[0].to_dict(), cluster_label)
    for rec in recommendations:
        st.write(f"✅ {rec}")




# import streamlit as st
# import pandas as pd
# from clustering import cluster_student
# from recommender import get_recommendations
# #from recommender import recommend_ncf, recommend_difficulty, predict_learning_strategy


# st.set_page_config(page_title="AI Learning Pattern Analyzer", layout="wide")

# st.title("AI Learning Pattern Analyzer & Predictor")
# st.markdown("This tool analyzes your learning patterns and predicts your skill level based on the data you provide. It also generates personalized recommendations and a study plan.")

# with st.form("learning_form"):
#     with st.container():
#         col1, col2 = st.columns(2)
#         with col1:
#             study_hours = st.slider("Weekly Study Hours", 1, 40, 10, help="How many hours do you study per week?")
#             quiz_score = st.slider("Average Quiz/Test Score (%)", 0, 100, 75, help="Your average score in quizzes or tests.")

#         with col2:
#             completion_rate = st.slider("Course Completion Rate (%)", 0, 100, 50, help="Percentage of completed courses.")
#             consistency = st.slider("Learning Consistency (1-10)", 1, 10, 5, help="How consistent is your study routine?")

#     st.markdown("### Additional Information")
#     col3, col4 = st.columns(2)
#     with col3:
#         learning_style = st.selectbox("Preferred Learning Style", ["visual", "auditory", "reading/writing", "kinesthetic"], help="Choose your preferred learning style.")
#         topic_difficulty = st.slider("Preferred Topic Difficulty (1-10)", 1, 10, 5, help="Rate the difficulty level of topics you prefer.")
#     with col4:
#         learning_interests = st.text_area("Learning Interests", placeholder="E.g., data science, web development, AI", help="Mention areas you are interested in.")
#         prior_knowledge = st.slider("Prior Knowledge (1-10)", 1, 10, 3, help="Your prior knowledge in the domain.")

#     submitted = st.form_submit_button("Analyze My Learning Pattern")

# if submitted:
#     input_df = pd.DataFrame([{
#         "study_hours": study_hours,
#         "quiz_score": quiz_score,
#         "completion_rate": completion_rate,
#         "consistency": consistency,
#         "learning_style": learning_style,
#         "topic_difficulty": topic_difficulty,
#         "learning_interests": learning_interests,
#         "prior_knowledge": prior_knowledge
#     }])

#     st.markdown("## 📊 Your Learning Pattern Analysis")
    
#     cluster_label = cluster_student(input_df)
#     st.write(f"🔍 **Predicted Cluster/Group:** {cluster_label}")

