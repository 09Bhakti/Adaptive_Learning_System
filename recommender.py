def recommend_content(student_id, n_recommendations=3):
    if student_id not in pivot_table.index:
        print("Student not found!")
        return []

    student_vector = pivot_table.loc[student_id].values.reshape(1, -1)
    distances, indices = knn.kneighbors(student_vector, n_neighbors=5)

    # Find similar students and get the content they engaged with
    similar_students = pivot_table.iloc[indices.flatten()].drop(student_id, errors='ignore')
    recommended_content = similar_students.mean().sort_values(ascending=False).index[:n_recommendations]

    return recommended_content.tolist()

# Example: Recommend content for student_id = 10
recommended_content = recommend_content(10)
print(f"Recommended Content for Student 10: {recommended_content}")


import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("synthetic_student_data.csv")

# Encode student_id and content_id as indices
student_ids = {id: i for i, id in enumerate(df['student_id'].unique())}
content_ids = {id: i for i, id in enumerate(df['content_id'].unique())}

df['student_id'] = df['student_id'].map(student_ids)
df['content_id'] = df['content_id'].map(content_ids)

# Train-test split
train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)

# Convert to tensors
train_tensor = torch.tensor(train_data[['student_id', 'content_id']].values, dtype=torch.long)
train_scores = torch.tensor(train_data['score'].values, dtype=torch.float32)
test_tensor = torch.tensor(test_data[['student_id', 'content_id']].values, dtype=torch.long)
test_scores = torch.tensor(test_data['score'].values, dtype=torch.float32)

# Define Neural Collaborative Filtering Model
class NCF(nn.Module):
    def __init__(self, num_students, num_contents, embedding_dim=10):
        super(NCF, self).__init__()
        self.student_embedding = nn.Embedding(num_students, embedding_dim)
        self.content_embedding = nn.Embedding(num_contents, embedding_dim)
        self.fc = nn.Sequential(
            nn.Linear(embedding_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, student, content):
        student_embedded = self.student_embedding(student)
        content_embedded = self.content_embedding(content)
        x = torch.cat([student_embedded, content_embedded], dim=1)
        return self.fc(x).squeeze()

# Initialize model
num_students = len(student_ids)
num_contents = len(content_ids)
model = NCF(num_students, num_contents)

# Define loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
epochs = 10
for epoch in range(epochs):
    optimizer.zero_grad()
    predictions = model(train_tensor[:, 0], train_tensor[:, 1])
    loss = criterion(predictions, train_scores)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item()}")

# Function to recommend content for a student
def recommend_ncf(student_id, n_recommendations=3):
    student_idx = student_ids.get(student_id, None)
    if student_idx is None:
        return []

    content_indices = list(content_ids.values())
    student_tensor = torch.tensor([student_idx] * len(content_indices))
    content_tensor = torch.tensor(content_indices)

    scores = model(student_tensor, content_tensor).detach().numpy()
    recommended_content = sorted(zip(content_ids.keys(), scores), key=lambda x: x[1], reverse=True)[:n_recommendations]

    return [content[0] for content in recommended_content]

# Example: Recommend content for student_id = 10
recommended_content = recommend_ncf(10)
print(f"Recommended Content for Student 10: {recommended_content}")


import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

# Define difficulty levels
difficulty_levels = ['easy', 'medium', 'hard']
difficulty_to_index = {level: i for i, level in enumerate(difficulty_levels)}

# Simulated student states (features: past score, time spent)
def get_student_state():
    return np.array([random.uniform(0, 100), random.uniform(10, 300)])  # [score, time_spent]

# Reward function: Encourage improvement, penalize struggling
def get_reward(previous_score, new_score):
    return (new_score - previous_score) / 10.0  # Reward if score improves

# Neural Network for Q-Learning
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Q-learning parameters
state_size = 2  # [score, time_spent]
action_size = len(difficulty_levels)
gamma = 0.95
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.995
learning_rate = 0.01
batch_size = 32
memory = deque(maxlen=2000)

# Initialize model & optimizer
model = DQN(state_size, action_size)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.MSELoss()

# Training loop
# Training loop
def train_dqn(episodes=1000):
    global epsilon
    loss = 0  # Initialize loss to prevent UnboundLocalError

    for episode in range(episodes):
        state = get_student_state()
        previous_score = state[0]

        for t in range(10):  # Simulating 10 interactions per student
            state_tensor = torch.FloatTensor(state).unsqueeze(0)

            # Choose action (difficulty level)
            if random.uniform(0, 1) < epsilon:
                action = random.randint(0, action_size - 1)  # Explore
            else:
                with torch.no_grad():
                    action = torch.argmax(model(state_tensor)).item()  # Exploit

            new_state = get_student_state()  # Get next state
            new_score = new_state[0]
            reward = get_reward(previous_score, new_score)
            previous_score = new_score

            # Store in memory
            memory.append((state, action, reward, new_state))

            # Train model with batch from memory
            if len(memory) > batch_size:
                batch = random.sample(memory, batch_size)
                states, actions, rewards, next_states = zip(*batch)

                states = torch.FloatTensor(states)
                actions = torch.LongTensor(actions)
                rewards = torch.FloatTensor(rewards)
                next_states = torch.FloatTensor(next_states)

                # Compute Q-values
                q_values = model(states).gather(1, actions.unsqueeze(1)).squeeze()
                next_q_values = model(next_states).max(1)[0].detach()
                expected_q_values = rewards + gamma * next_q_values

                loss = criterion(q_values, expected_q_values)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            state = new_state

        # Decay exploration rate
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        # ✅ Fix: Check if 'loss' exists before printing
        if episode % 100 == 0 and loss != 0:
            print(f"Episode {episode}, Loss: {loss.item()}")

# Train the agent
train_dqn(episodes=1000)

# Function to recommend difficulty level based on student state
def recommend_difficulty(student_state):
    state_tensor = torch.FloatTensor(student_state).unsqueeze(0)
    action = torch.argmax(model(state_tensor)).item()
    return difficulty_levels[action]

# Example: Recommend difficulty for a new student
student_state = get_student_state()
recommended_difficulty = recommend_difficulty(student_state)
print(f"Recommended Difficulty: {recommended_difficulty}")


import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Generate synthetic student learning data
np.random.seed(42)
num_students = 1000

data = {
    'past_score': np.random.randint(0, 100, num_students),  # Previous test scores
    'time_spent': np.random.randint(5, 300, num_students),  # Time spent on learning
    'difficulty_level': np.random.choice([1, 2, 3], num_students),  # 1: Easy, 2: Medium, 3: Hard
    'next_difficulty': np.random.choice([1, 2, 3], num_students)  # Recommended difficulty
}

df = pd.DataFrame(data)

# Split into training & test sets
X = df[['past_score', 'time_spent', 'difficulty_level']]
y = df['next_difficulty']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Decision Tree Model
model = DecisionTreeClassifier(max_depth=4)
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Function to recommend next difficulty level
def recommend_next_activity(past_score, time_spent, current_difficulty):
    input_data = np.array([[past_score, time_spent, current_difficulty]])
    next_difficulty = model.predict(input_data)[0]
    return next_difficulty

# Example: Predict next activity for a student
past_score = 75
time_spent = 120
current_difficulty = 2  # Medium

recommended_difficulty = recommend_next_activity(past_score, time_spent, current_difficulty)
print(f"Recommended Next Difficulty: {recommended_difficulty}")


import pandas as pd
import numpy as np

# Simulated student interaction dataset
feedback_data = {
    'student_id': np.random.randint(1, 100, 500),  # 500 interactions from 100 students
    'activity': np.random.choice(['lecture', 'quiz', 'assignment'], 500),
    'content_id': np.random.randint(1, 50, 500),  # 50 learning materials
    'score': np.random.uniform(0, 100, 500),  # Student scores
    'time_spent': np.random.randint(5, 300, 500),  # Time spent in seconds
    'feedback': np.random.choice(['too easy', 'just right', 'too hard'], 500)
}

df = pd.DataFrame(feedback_data)

# Preview feedback data
print(df.head())


# Calculate average score and time spent per difficulty level
feedback_summary = df.groupby('feedback').agg({'score': 'mean', 'time_spent': 'mean'}).reset_index()
print(feedback_summary)


def adjust_learning_path(score, time_spent, feedback):
    """Modify difficulty based on feedback"""
    if feedback == 'too easy' and score > 80:
        return 'Increase Difficulty'
    elif feedback == 'too hard' or score < 50:
        return 'Reduce Difficulty'
    else:
        return 'Maintain Difficulty'

# Apply to dataset
df['adjustment'] = df.apply(lambda row: adjust_learning_path(row['score'], row['time_spent'], row['feedback']), axis=1)

# View adjustments
print(df[['score', 'time_spent', 'feedback', 'adjustment']].head())


from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

# Convert feedback into numerical labels
df['feedback_encoded'] = df['feedback'].map({'too easy': 0, 'just right': 1, 'too hard': 2})
df['adjustment_encoded'] = df['adjustment'].map({'Increase Difficulty': 1, 'Maintain Difficulty': 0, 'Reduce Difficulty': -1})

X = df[['score', 'time_spent', 'feedback_encoded']]
y = df['adjustment_encoded']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Decision Tree
dt_model = DecisionTreeClassifier(max_depth=3)
dt_model.fit(X_train, y_train)

# Predict next learning path
sample_student = [[85, 120, 0]]  # Score: 85, Time Spent: 120 sec, Feedback: Too Easy
adjustment = dt_model.predict(sample_student)[0]

print(f"Recommended Adjustment: {'Increase Difficulty' if adjustment == 1 else 'Reduce Difficulty' if adjustment == -1 else 'Maintain Difficulty'}")


import numpy as np
import pandas as pd
import scipy.stats as stats

# Simulate 500 students randomly assigned to two groups
np.random.seed(42)
students = 500
group = np.random.choice(['A', 'B'], students)

# Generate quiz scores for A (Basic Model) & B (ML-Based Model)
scores_A = np.random.normal(loc=75, scale=10, size=students//2)  # Mean = 75, Std = 10
scores_B = np.random.normal(loc=80, scale=10, size=students//2)  # Mean = 80, Std = 10

# Combine into a DataFrame
df = pd.DataFrame({
    'student_id': range(1, students+1),
    'group': group,
    'score': np.concatenate([scores_A, scores_B])
})

# Analyze mean performance
mean_A = df[df['group'] == 'A']['score'].mean()
mean_B = df[df['group'] == 'B']['score'].mean()

print(f"Mean Score for Group A: {mean_A:.2f}")
print(f"Mean Score for Group B: {mean_B:.2f}")

# Perform Statistical A/B Test (t-test)
t_stat, p_value = stats.ttest_ind(scores_A, scores_B)
print(f"T-Test Statistic: {t_stat:.4f}, P-Value: {p_value:.4f}")

# Check if B is significantly better
if p_value < 0.05:
    print("Version B (ML-Based) performs significantly better! ✅")
else:
    print("No significant difference between A and B. ❌")


import pandas as pd
import nltk
