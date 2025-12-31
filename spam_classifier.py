import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# 1. Load Dataset
data = pd.read_csv("dataset/spam.csv", encoding="latin-1")

# 2. Keep required columns
data = data[['v1', 'v2']]
data.columns = ['label', 'message']

# 3. Convert labels to numbers
data['label'] = data['label'].map({'spam': 1, 'ham': 0})

# 4. Convert text to numbers
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(data['message'])
y = data['label']

# 5. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 6. Train model
model = MultinomialNB()
model.fit(X_train, y_train)

# 7. Evaluate model
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# 8. Test with custom input
email = ["tommorrow there is a meeting at 11 am"]
email_vector = vectorizer.transform(email)
prediction = model.predict(email_vector)

print("\nTest Email Result:")
print("Spam ❌" if prediction[0] == 1 else "Not Spam ✅")
