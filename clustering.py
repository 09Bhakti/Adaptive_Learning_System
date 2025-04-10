from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

# Select features for clustering
X = df[['score', 'time_spent']]

# Apply K-Means clustering
kmeans = KMeans(n_clusters=3, random_state=42)
df['cluster'] = kmeans.fit_predict(X)

# Plot the clusters
plt.figure(figsize=(8, 6))
sns.scatterplot(x=df['score'], y=df['time_spent'], hue=df['cluster'], palette='viridis')
plt.xlabel('Score')
plt.ylabel('Time Spent (seconds)')
plt.title('Student Clusters Based on Performance')
plt.show()


import pandas as pd
from sklearn.neighbors import NearestNeighbors
import numpy as np

# Load the generated dataset
df = pd.read_csv("synthetic_student_data.csv")

# Create a pivot table (students as rows, content as columns, scores as values)
pivot_table = df.pivot_table(index='student_id', columns='content_id', values='score').fillna(0)

# Fit the KNN model
knn = NearestNeighbors(n_neighbors=5, metric='cosine')
knn.fit(pivot_table)

# Function to recommend content for a student
