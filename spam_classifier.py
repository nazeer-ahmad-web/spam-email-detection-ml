import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# 1. Load Dataset
data = pd.read_csv("dataset/spam.csv", encoding="latin-1")

# 2. Keep required columns
data = data[['v1', 'v2']]
data.columns = ['label', 'message']

# 3. Convert labels to numbers
data['label'] = data['label'].map({'spam': 1, 'ham': 0})

# 4. Basic text cleaning
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\W+', ' ', text)
    return text

data['message'] = data['message'].apply(clean_text)

# 5. Convert text to numbers (IMPROVED TF-IDF)
vectorizer = TfidfVectorizer(
    stop_words='english',
    ngram_range=(1, 2),   # unigrams + bigrams
    max_df=0.9,
    min_df=2
)

X = vectorizer.fit_transform(data['message'])
y = data['label']

# 6. Split data (important fix)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 7. Train model (BETTER THAN NAIVE BAYES)
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# 8. Evaluate model
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# 9. Test with custom input
email = ["ALERT: Your bank account has been temporarily suspended. "]
email_cleaned = [clean_text(email[0])]
email_vector = vectorizer.transform(email_cleaned)
prediction = model.predict(email_vector)

print("\nTest Email Result:")
print("Spam ❌" if prediction[0] == 1 else "Not Spam ✅")
