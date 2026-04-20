import pandas as pd
import re
import string
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# =========================
# 1. Load Dataset
# =========================
data = pd.read_csv("dataset/spam.csv", encoding="latin-1")

# Keep required columns
data = data[['v1', 'v2']]
data.columns = ['label', 'message']

# Convert labels
data['label'] = data['label'].map({'spam': 1, 'ham': 0})

# =========================
# 2. Clean Text
# =========================
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)  # remove numbers
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

data['message'] = data['message'].apply(clean_text)

# =========================
# 3. Vectorization
# =========================
vectorizer = TfidfVectorizer(
    stop_words='english',
    ngram_range=(1, 2),
    max_df=0.9,
    min_df=2
)

X = vectorizer.fit_transform(data['message'])
y = data['label']

# =========================
# 4. Train-Test Split
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =========================
# 5. Train Models
# =========================
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Naive Bayes": MultinomialNB()
}

best_model = None
best_accuracy = 0

print("Model Performance:\n")

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"{name} Accuracy: {acc:.4f}")

    if acc > best_accuracy:
        best_accuracy = acc
        best_model = model

# =========================
# 6. Detailed Evaluation
# =========================
y_pred = best_model.predict(X_test)

print("\nBest Model Selected")
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# =========================
# 7. Save Model + Vectorizer
# =========================
pickle.dump(best_model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("\nModel and vectorizer saved!")

# =========================
# 8. Test Custom Input
# =========================
def predict_email(text):
    cleaned = clean_text(text)
    vector = vectorizer.transform([cleaned])
    prediction = best_model.predict(vector)[0]
    prob = best_model.predict_proba(vector)[0]

    return prediction, prob

# Example Tests
test_emails = [
    "Win a FREE iPhone now!!!",
    "Hey bro, are we meeting today?",
    "URGENT! Your account is blocked. Click here",
    "Your OTP is 4582"
]

print("\nTest Predictions:\n")

for email in test_emails:
    pred, prob = predict_email(email)
    result = "Spam ❌" if pred == 1 else "Not Spam ✅"
    confidence = max(prob) * 100

    print(f"Email: {email}")
    print(f"Result: {result} ({confidence:.2f}% confidence)\n")
