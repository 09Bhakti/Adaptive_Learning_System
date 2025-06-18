

import pandas as pd
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import scipy.stats as stats
import nltk

# # ---------------------------------------------
# # Collaborative Filtering Placeholder Function
# # ---------------------------------------------
# def recommend_content(student_id, pivot_table, knn, n_recommendations=3):
#     if student_id not in pivot_table.index:
#         return []
#     student_vector = pivot_table.loc[student_id].values.reshape(1, -1)
#     distances, indices = knn.kneighbors(student_vector, n_neighbors=5)
#     similar_students = pivot_table.iloc[indices.flatten()].drop(student_id, errors='ignore')
#     recommended_content = similar_students.mean().sort_values(ascending=False).index[:n_recommendations]
#     return recommended_content.tolist()

# ---------------------------------------------
# #Neural Collaborative Filtering (NCF)
# ---------------------------------------------
# class NCF(nn.Module):
#     def __init__(self, num_students, num_contents, embedding_dim=10):
#         super(NCF, self).__init__()
#         self.student_embedding = nn.Embedding(num_students, embedding_dim)
#         self.content_embedding = nn.Embedding(num_contents, embedding_dim)
#         self.fc = nn.Sequential(
#             nn.Linear(embedding_dim * 2, 64),
#             nn.ReLU(),
#             nn.Linear(64, 32),
#             nn.ReLU(),
#             nn.Linear(32, 1)
#         )

#     def forward(self, student, content):
#         student_embedded = self.student_embedding(student)
#         content_embedded = self.content_embedding(content)
#         x = torch.cat([student_embedded, content_embedded], dim=1)
#         return self.fc(x).squeeze()

# def recommend_ncf(model, student_id, student_ids, content_ids, n_recommendations=3):
#     student_idx = student_ids.get(student_id, None)
#     if student_idx is None:
#         return []
#     content_indices = list(content_ids.values())
#     student_tensor = torch.tensor([student_idx] * len(content_indices))
#     content_tensor = torch.tensor(content_indices)
#     scores = model(student_tensor, content_tensor).detach().numpy()
#     ranked = sorted(zip(content_ids.keys(), scores), key=lambda x: x[1], reverse=True)
#     return [content[0] for content in ranked[:n_recommendations]]

# # ---------------------------------------------
# # Deep Q-Learning (Reinforcement Learning)
# # ---------------------------------------------
# class DQN(nn.Module):
#     def __init__(self, state_size, action_size):
#         super(DQN, self).__init__()
#         self.fc1 = nn.Linear(state_size, 64)
#         self.fc2 = nn.Linear(64, 32)
#         self.fc3 = nn.Linear(32, action_size)

#     def forward(self, x):
#         x = torch.relu(self.fc1(x))
#         x = torch.relu(self.fc2(x))
#         return self.fc3(x)

# def get_student_state():
#     return np.array([random.uniform(0, 100), random.uniform(10, 300)])

# def get_reward(previous_score, new_score):
#     return (new_score - previous_score) / 10.0

# def recommend_difficulty(model, student_state, difficulty_levels):
#     state_tensor = torch.FloatTensor(student_state).unsqueeze(0)
#     action = torch.argmax(model(state_tensor)).item()
#     return difficulty_levels[action]

# # ---------------------------------------------
# # Feedback-Based Difficulty Adjustment
# # ---------------------------------------------
# def adjust_learning_path(score, time_spent, feedback):
#     if feedback == 'too easy' and score > 80:
#         return 'Increase Difficulty'
#     elif feedback == 'too hard' or score < 50:
#         return 'Reduce Difficulty'
#     else:
#         return 'Maintain Difficulty'

# # ---------------------------------------------
# # Strategy Prediction
# # ---------------------------------------------
# def predict_learning_strategy(row):
#     if row['sentiment_avg'] < -0.3 or row['recent_score_delta'] < -10:
#         return {
#             "difficulty": "Easy",
#             "risk": "High",
#             "recommendation": "Provide review material & simpler exercises"
#         }
#     elif row['recent_score_delta'] > 5 and row['avg_score'] > 75:
#         return {
#             "difficulty": "Hard",
#             "risk": "Low",
#             "recommendation": "Introduce advanced concepts"
#         }
#     elif row['avg_time_spent'] > 300:
#         return {
#             "difficulty": "Medium",
#             "risk": "Medium",
#             "recommendation": "Add interactive/video-based content"
#         }
#     else:
#         return {
#             "difficulty": "Medium",
#             "risk": "Low",
#             "recommendation": "Continue current learning path"
#         }

# # ---------------------------------------------
# # Testing Block
# # ---------------------------------------------
# if __name__ == "__main__":
#     # Load dataset
#     df = pd.read_csv("synthetic_student_data.csv")

#     # NCF Setup
#     student_ids = {id: i for i, id in enumerate(df['student_id'].unique())}
#     content_ids = {id: i for i, id in enumerate(df['content_id'].unique())}
#     df['student_id'] = df['student_id'].map(student_ids)
#     df['content_id'] = df['content_id'].map(content_ids)

#     train_data, _ = train_test_split(df, test_size=0.2, random_state=42)
#     train_tensor = torch.tensor(train_data[['student_id', 'content_id']].values, dtype=torch.long)
#     train_scores = torch.tensor(train_data['score'].values, dtype=torch.float32)

#     ncf_model = NCF(len(student_ids), len(content_ids))
#     optimizer = optim.Adam(ncf_model.parameters(), lr=0.001)
#     criterion = nn.MSELoss()

#     for epoch in range(5):
#         optimizer.zero_grad()
#         predictions = ncf_model(train_tensor[:, 0], train_tensor[:, 1])
#         loss = criterion(predictions, train_scores)
#         loss.backward()
#         optimizer.step()
#         print(f"NCF Epoch {epoch+1}, Loss: {loss.item():.4f}")

#     print("Recommended (NCF):", recommend_ncf(ncf_model, 10, student_ids, content_ids))

#     # RL Setup
#     difficulty_levels = ['easy', 'medium', 'hard']
#     rl_model = DQN(2, len(difficulty_levels))
#     rl_optimizer = optim.Adam(rl_model.parameters(), lr=0.01)
#     memory = deque(maxlen=2000)
#     gamma = 0.95
#     epsilon = 1.0
#     epsilon_min = 0.01
#     epsilon_decay = 0.995

#     for episode in range(100):
#         state = get_student_state()
#         prev_score = state[0]
#         for t in range(10):
#             state_tensor = torch.FloatTensor(state).unsqueeze(0)
#             action = random.randint(0, 2) if random.random() < epsilon else torch.argmax(rl_model(state_tensor)).item()
#             next_state = get_student_state()
#             reward = get_reward(prev_score, next_state[0])
#             memory.append((state, action, reward, next_state))
#             prev_score = next_state[0]
#             if len(memory) >= 32:
#                 batch = random.sample(memory, 32)
#                 s, a, r, ns = zip(*batch)
#                 s = torch.FloatTensor(s)
#                 a = torch.LongTensor(a)
#                 r = torch.FloatTensor(r)
#                 ns = torch.FloatTensor(ns)
#                 q_vals = rl_model(s).gather(1, a.unsqueeze(1)).squeeze()
#                 max_next = rl_model(ns).max(1)[0].detach()
#                 expected = r + gamma * max_next
#                 loss = nn.MSELoss()(q_vals, expected)
#                 rl_optimizer.zero_grad()
#                 loss.backward()
#                 rl_optimizer.step()
#             state = next_state
#         epsilon = max(epsilon_min, epsilon * epsilon_decay)
#         if episode % 20 == 0:



def get_recommendations(user_input, cluster_label):
    recommendations = []

    # Rule-based logic for learning recommendations
    if user_input["study_hours"] < 10:
        recommendations.append("Increase your weekly study hours gradually to at least 10 hours.")
    else:
        recommendations.append("Your study hours are on track.")

    if user_input["quiz_score"] < 60:
        recommendations.append("Focus more on practice tests and quizzes to improve retention.")
    elif user_input["quiz_score"] < 80:
        recommendations.append("You're doing well on tests, but there's still room to improve.")
    else:
        recommendations.append("Excellent test performance! Keep up the good work.")

    if user_input["completion_rate"] < 50:
        recommendations.append("Try to complete more of your enrolled courses to build consistency.")
    else:
        recommendations.append("You're doing well in completing your courses.")

    if user_input["consistency"] < 5:
        recommendations.append("Establish a regular study routine to improve consistency.")
    else:
        recommendations.append("Great consistency in learning!")

    if user_input["prior_knowledge"] < 5:
        recommendations.append("Start with foundational content and gradually move to advanced topics.")
    else:
        recommendations.append("You can explore more advanced and challenging material.")

    if user_input["topic_difficulty"] >= 8:
        recommendations.append("You prefer difficult topics. Consider joining peer groups or mentorships for better understanding.")
    elif user_input["topic_difficulty"] <= 3:
        recommendations.append("You prefer easier topics. Slowly increase the difficulty for growth.")
    else:
        recommendations.append("Balanced topic difficulty preference. Keep exploring at your pace.")

    # Additional suggestion based on learning style
    learning_style = user_input.get("learning_style", "").lower()
    if learning_style == "visual":
        recommendations.append("Use videos, infographics, and diagrams to enhance your learning.")
    elif learning_style == "auditory":
        recommendations.append("Try podcasts, recorded lectures, or discussion-based learning.")
    elif learning_style == "reading/writing":
        recommendations.append("Use notes, textbooks, and written summaries for better retention.")
    elif learning_style == "kinesthetic":
        recommendations.append("Engage in hands-on projects, experiments, or coding exercises.")

    # Interests-based suggestion
    interests = user_input.get("interests", "").strip()
    if interests:
        recommendations.append(f"Explore courses, blogs, and projects in: {interests}.")
    else:
        recommendations.append("Specify your learning interests to get targeted resources.")

    # Add a generic line based on cluster (optional)
    if cluster_label == 0:
        recommendations.append("You're an emerging learner. Focus on consistent effort and structured progress.")
    elif cluster_label == 1:
        recommendations.append("You’re a steady performer. Keep refining your strategy and balance.")
    else:
        recommendations.append("You are a high performer. Aim for depth and specialization.")

    return recommendations

            print(f"RL Episode {episode}, Epsilon: {epsilon:.3f}")

    print("Recommended Difficulty:", recommend_difficulty(rl_model, get_student_state(), difficulty_levels))

