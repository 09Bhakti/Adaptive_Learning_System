# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import StandardScaler
# from sklearn.cluster import KMeans

# def extract_advanced_features(df):
#     recent_group = df.sort_values("timestamp").groupby("student_id").tail(3)

#     features = df.groupby("student_id").agg({
#         "score": ["mean", "std"],
#         "time_spent": "mean",
#         "activity": "count",
#         "feedback_sentiment": "mean"
#     }).reset_index()

#     recent_delta = recent_group.groupby("student_id")["score"].mean() - features[("score", "mean")].values
#     features["recent_score_delta"] = recent_delta.values

#     features.columns = [
#         "student_id", "avg_score", "score_std", "avg_time_spent", 
#         "attempt_count", "sentiment_avg", "recent_score_delta"
#     ]
#     features = features.fillna(0)
#     return features

# def cluster_students_advanced(df, n_clusters=4):
#     features = extract_advanced_features(df)
#     X = features.drop(columns=["student_id"])
    
#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(X)

#     kmeans = KMeans(n_clusters=n_clusters, random_state=42)
#     clusters = kmeans.fit_predict(X_scaled)

#     features["cluster"] = clusters
#     return features

from sklearn.cluster import KMeans
import numpy as np

# This function performs clustering directly on the user's single input.
# In production, you'd usually load a pre-trained model or train on a dataset.
def cluster_student(input_df):
    # Select only numerical features relevant for clustering
    features = input_df[["study_hours", "quiz_score", "completion_rate", "consistency", "topic_difficulty", "prior_knowledge"]]

    # For simplicity, simulate model training here (should be replaced with actual model load in production)
    # Use some synthetic data along with the input to train a basic model
    synthetic_data = np.random.rand(20, 6) * [40, 100, 100, 10, 10, 10]  # simulate 20 students
    combined_data = np.vstack([synthetic_data, features.values])

    model = KMeans(n_clusters=3, random_state=42)
    model.fit(combined_data)

    # Predict for the last row (the actual user input)
    cluster = model.predict([features.values[0]])
    return int(cluster[0])
