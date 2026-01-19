# 📧 Spam Email Detection using Logistic Regression and TF-IDF

A machine learning project that classifies emails as **Spam** or **Not Spam (Ham)** using **Natural Language Processing (NLP)** techniques and **Logistic Regression**. This project demonstrates text preprocessing, feature extraction using TF-IDF, model training, and evaluation.

---

## 🚀 Project Overview

Spam emails are a common problem in digital communication. This project aims to build a reliable spam detection system using classical machine learning techniques. The model is trained on a labeled dataset and can predict whether a given email message is spam or not.

---

## 🧠 Machine Learning Approach

* Text Preprocessing (lowercasing, removing special characters)
* Feature Extraction using **TF-IDF Vectorization**
* Classification using **Logistic Regression**
* Model Evaluation using accuracy, precision, recall, and F1-score

---

## 🛠️ Technologies Used

* **Python**
* **Pandas**
* **Scikit-learn**
* **Regular Expressions (re)**

---

## 📂 Project Structure

```
SPAM_EMAIL_DETECTION
│── dataset
│   └── spam.csv
│── spam_classifier.py
│── requirements.txt
│── README.md
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the repository

```bash
git clone https://github.com/your-username/spam-email-detection.git
cd spam-email-detection
```

### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Run the project

```bash
python spam_classifier.py
```

(or on Windows)

```bash
py spam_classifier.py
```

---

## 📊 Model Performance

* Achieves strong accuracy on the test dataset
* Uses TF-IDF with n-grams for improved feature representation
* Handles common spam patterns such as promotional offers, fake alerts, and scam messages

⚠️ *Note:* As a classical ML model, Logistic Regression may misclassify ambiguous messages. Further improvements can be made using SVM or transformer-based models.

---

## 🧪 Example Test Input

```python
email = ["Congratulations! You have won ₹5,00,000. Click now"]
```

**Output:**

```
Spam ❌
```

---

## 🎯 Learning Outcomes

* Practical understanding of NLP pipelines
* Experience with text classification problems
* Hands-on application of Logistic Regression
* Model evaluation and error analysis

---

## 🔮 Future Improvements

* Add Support Vector Machine (SVM)
* Integrate Flask API for web usage
* Save and load trained models
* Improve accuracy using deep learning models (LSTM / BERT)

---

## 👨‍💻 Author

**Shaik Nazeer Ahmad**

* GitHub: [https://github.com/nazeer-ahmad-web](https://github.com/nazeer-ahmad-web)
* LinkedIn: [https://linkedin.com/in/shaik-nazeer-ahmad-6a94b2345](https://linkedin.com/in/shaik-nazeer-ahmad-6a94b2345)
