import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# Dataset
digits = load_digits()
X = digits.data
y = digits.target

# Train model
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

print("Model Accuracy:",
      round(model.score(X_test, y_test) * 100, 2), "%")

# CAPTCHA-like sequence
sample = X_test[:4]
prediction = model.predict(sample)

print("Predicted CAPTCHA:", "".join(map(str, prediction)))

# Display CAPTCHA characters
fig, ax = plt.subplots(1, 4, figsize=(6, 2))

for i in range(4):
    ax[i].imshow(sample[i].reshape(8, 8), cmap="gray")
    ax[i].axis("off")
    ax[i].set_title(prediction[i])

plt.suptitle("ML-Based CAPTCHA Recognition")
plt.tight_layout()
plt.savefig("captcha_output.png", dpi=300)
plt.show()