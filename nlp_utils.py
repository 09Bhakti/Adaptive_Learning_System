from nltk.sentiment import SentimentIntensityAnalyzer

nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

# Simulated student feedback data
data = {
    'student_id': [1, 2, 3, 4, 5],
    'feedback': [
        "The lecture was really helpful and easy to understand!",
        "This topic is too confusing. I don't get it.",
        "The quiz was okay, but some questions were tricky.",
        "I found the assignment really challenging.",
        "Loved the way the content was explained!"
    ]
}

df = pd.DataFrame(data)

# Apply sentiment analysis
df['sentiment_score'] = df['feedback'].apply(lambda x: sia.polarity_scores(x)['compound'])

# Classify sentiment
df['sentiment_label'] = df['sentiment_score'].apply(lambda x: 'positive' if x > 0.2 else 'negative' if x < -0.2 else 'neutral')

print(df[['feedback', 'sentiment_score', 'sentiment_label']])


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# Sample student queries
queries = [
    "How does backpropagation work in neural networks?",
    "What is gradient descent in machine learning?",
    "Can you explain decision trees?",
    "How do I improve my accuracy in deep learning?",
    "What are activation functions used for?"
]

# Convert text to numerical format (TF-IDF)
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(queries)

# Perform Topic Modeling (LDA)
lda = LatentDirichletAllocation(n_components=2, random_state=42)
lda.fit(X)

# Print top words in each topic
words = vectorizer.get_feature_names_out()
for topic_idx, topic in enumerate(lda.components_):
    top_words = [words[i] for i in topic.argsort()[:-6:-1]]
    print(f"Topic {topic_idx + 1}: {', '.join(top_words)}")

