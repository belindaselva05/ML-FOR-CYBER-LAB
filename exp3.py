import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Dataset
df = pd.DataFrame({
    "Packet_Size":[120,150,130,900,850,1000,950,140,160,110,880,920,125,145,980],
    "Connections":[2,3,2,50,45,60,55,3,2,1,48,52,2,3,58],
    "Failed_Logins":[0,1,0,2,1,8,7,0,1,0,3,4,0,1,6],
    "Port":[80,443,80,22,21,3389,3389,443,80,53,21,22,80,443,3389],
    "Attack":["Normal","Normal","Normal","Brute Force","Brute Force",
              "DDoS","DDoS","Normal","Normal","Normal",
              "Brute Force","Brute Force","Normal","Normal","DDoS"]
})

# Features and target
X = df.drop("Attack", axis=1)
y = LabelEncoder().fit_transform(df["Attack"])

# Train and test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Random Forest Ensemble
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Performance
print("Accuracy :", round(accuracy_score(y_test,y_pred)*100,2), "%")
print("Precision:", round(precision_score(y_test,y_pred,average="weighted")*100,2), "%")
print("Recall   :", round(recall_score(y_test,y_pred,average="weighted")*100,2), "%")
print("F1 Score :", round(f1_score(y_test,y_pred,average="weighted")*100,2), "%")

# Confusion Matrix
sns.heatmap(confusion_matrix(y_test,y_pred), annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Random Forest - Cyber Attack Classification")
plt.show()