import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

def extract_advanced_features(df):
    recent_group = df.sort_values("timestamp").groupby("student_id").tail(3)

    features = df.groupby("student_id").agg({
        "score": ["mean", "std"],
        "time_spent": "mean",
        "activity": "count",
        "feedback_sentiment": "mean"
    }).reset_index()

    recent_delta = recent_group.groupby("student_id")["score"].mean() - features[("score", "mean")].values
    features["recent_score_delta"] = recent_delta.values

    features.columns = [
        "student_id", "avg_score", "score_std", "avg_time_spent", 
        "attempt_count", "sentiment_avg", "recent_score_delta"
    ]
    features = features.fillna(0)
    return features

def cluster_students_advanced(df, n_clusters=4):
    features = extract_advanced_features(df)
    X = features.drop(columns=["student_id"])
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)

    features["cluster"] = clusters
    return features


