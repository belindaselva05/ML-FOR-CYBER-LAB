import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# --------------------------------------------------
# Step 1: Create Dataset
# --------------------------------------------------

data = {
    "url": [
        # Legitimate URLs
        "https://www.google.com",
        "https://www.amazon.com/login",
        "https://github.com/openai",
        "https://www.microsoft.com/en-us/",
        "https://www.paypal.com/home",
        "https://www.bankofamerica.com/security",
        "https://www.linkedin.com/in/johndoe",
        "https://www.apple.com/store",

        # Phishing URLs
        "http://paypal-login-security-alert.com",
        "http://verify-account-amazon.biz/login",
        "https://secure-facebook-login-update.ru",
        "http://bankofamerica-login-confirm.net",
        "https://google-account-reset-now.xyz",
        "http://microsoft-support-verification.top",
        "http://appleid-reset-alert.org",
        "https://login-update-github.io"
    ],

    "label": [
        0, 0, 0, 0, 0, 0, 0, 0,
        1, 1, 1, 1, 1, 1, 1, 1
    ]
}

df = pd.DataFrame(data)

print("SVM URL Classification Dataset")
print("--------------------------------")
print(df.head())

# --------------------------------------------------
# Step 2: Separate Features and Target
# --------------------------------------------------

X = df["url"]
y = df["label"]

# --------------------------------------------------
# Step 3: Convert URLs into TF-IDF Features
# --------------------------------------------------

vectorizer = TfidfVectorizer(
    analyzer="char",
    ngram_range=(3, 5)
)

X_vectorized = vectorizer.fit_transform(X)

# --------------------------------------------------
# Step 4: Split Dataset
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

# --------------------------------------------------
# Step 5: Train SVM Model
# --------------------------------------------------

model = SVC(
    kernel="linear",
    random_state=42
)

model.fit(X_train, y_train)

# --------------------------------------------------
# Step 6: Make Predictions
# --------------------------------------------------

y_pred = model.predict(X_test)

# --------------------------------------------------
# Step 7: Evaluate Model
# --------------------------------------------------

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)

print("\nSVM Performance")
print("----------------")
print("Accuracy :", round(accuracy * 100, 2), "%")
print("Precision:", round(precision * 100, 2), "%")
print("Recall   :", round(recall * 100, 2), "%")
print("F1 Score :", round(f1 * 100, 2), "%")

# --------------------------------------------------
# Step 8: Confusion Matrix
# --------------------------------------------------

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

plt.figure(figsize=(6, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Legitimate", "Phishing"],
    yticklabels=["Legitimate", "Phishing"]
)

plt.title("SVM - Phishing URL Detection")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()

plt.savefig(
    "svm_confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# --------------------------------------------------
# Step 9: Classification Report
# --------------------------------------------------

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Legitimate", "Phishing"],
        zero_division=0
    )
)

# --------------------------------------------------
# Step 10: Test New URLs
# --------------------------------------------------

new_urls = [
    "https://secure-login-paypal-alert.info",
    "https://www.amazon.in/gp/cart/view.html",
    "http://update-facebook-verification.com",
    "https://accounts.google.com"
]

new_features = vectorizer.transform(new_urls)

predictions = model.predict(new_features)

print("\nNew URL Prediction Results:")
print("----------------------------")

for url, prediction in zip(new_urls, predictions):

    if prediction == 1:
        result = "Phishing"
    else:
        result = "Legitimate"

    print(url, "-->", result)

print("\nExperiment 5 completed successfully.")