import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from urllib.parse import urlparse
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


# --------------------------------------------------
# Step 1: Create URL Dataset
# --------------------------------------------------

urls = [

    # Normal URLs
    "https://www.google.com",
    "https://www.amazon.com",
    "https://github.com/openai",
    "https://www.microsoft.com",
    "https://www.apple.com",
    "https://www.linkedin.com",
    "https://www.wikipedia.org",
    "https://www.python.org",
    "https://www.ibm.com",
    "https://www.oracle.com",

    # Abnormal / suspicious URLs
    "http://192.168.1.100/login",
    "http://paypal-login-security-alert.com",
    "http://verify-account-amazon.biz/login",
    "http://secure-login-update-account.xyz",
    "http://free-gift-card-win-now.com",
    "http://login@secure-account-update.com",
    "http://facebook-security-verification-alert.ru",
    "http://update-bank-account-confirm.net",
    "http://microsoft-support-verification.top",
    "http://google-account-reset-now.xyz"
]


# --------------------------------------------------
# Step 2: Feature Engineering Function
# --------------------------------------------------

def extract_features(url):

    parsed = urlparse(url)

    hostname = parsed.netloc
    path = parsed.path

    # Remove username if present
    hostname_clean = hostname.split("@")[-1]

    # URL length
    url_length = len(url)

    # Number of dots
    dot_count = url.count(".")

    # Number of hyphens
    hyphen_count = url.count("-")

    # Number of digits
    digit_count = sum(char.isdigit() for char in url)

    # Number of special characters
    special_count = sum(
        not char.isalnum()
        for char in url
    )

    # Number of path directories
    directory_count = path.count("/")

    # Presence of @ symbol
    at_symbol = 1 if "@" in url else 0

    # Presence of IP address
    ip_present = 1 if any(
        part.isdigit()
        for part in hostname_clean.split(".")
    ) and hostname_clean.replace(".", "").isdigit() else 0

    # Suspicious keywords
    suspicious_words = [
        "login",
        "verify",
        "account",
        "security",
        "update",
        "confirm",
        "free",
        "gift",
        "reset",
        "alert"
    ]

    suspicious_keyword_count = sum(
        word in url.lower()
        for word in suspicious_words
    )

    return [
        url_length,
        dot_count,
        hyphen_count,
        digit_count,
        special_count,
        directory_count,
        at_symbol,
        ip_present,
        suspicious_keyword_count
    ]


# --------------------------------------------------
# Step 3: Extract Features
# --------------------------------------------------

features = []

for url in urls:
    features.append(extract_features(url))


feature_names = [
    "URL_Length",
    "Dot_Count",
    "Hyphen_Count",
    "Digit_Count",
    "Special_Character_Count",
    "Directory_Count",
    "At_Symbol",
    "IP_Address",
    "Suspicious_Keyword_Count"
]

df = pd.DataFrame(
    features,
    columns=feature_names
)

df["URL"] = urls


print("URL Feature Dataset")
print("-------------------")
print(df)


# --------------------------------------------------
# Step 4: Prepare Features
# --------------------------------------------------

X = df[feature_names]


# --------------------------------------------------
# Step 5: Standardize Features
# --------------------------------------------------

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)


# --------------------------------------------------
# Step 6: Isolation Forest
# --------------------------------------------------

model = IsolationForest(
    contamination=0.3,
    random_state=42
)

model.fit(X_scaled)


# --------------------------------------------------
# Step 7: Detect Anomalies
# --------------------------------------------------

predictions = model.predict(X_scaled)

df["Anomaly"] = predictions


# Isolation Forest:
#  1  = Normal
# -1  = Anomaly

df["Status"] = df["Anomaly"].map({
    1: "Normal",
    -1: "Anomalous"
})


# --------------------------------------------------
# Step 8: Display Results
# --------------------------------------------------

print("\nURL Anomaly Detection Results")
print("-----------------------------")

for index, row in df.iterrows():

    print(
        row["URL"],
        "-->",
        row["Status"]
    )


# --------------------------------------------------
# Step 9: Count Results
# --------------------------------------------------

normal_count = sum(df["Anomaly"] == 1)
anomaly_count = sum(df["Anomaly"] == -1)

print("\nSummary")
print("-------")
print("Normal URLs   :", normal_count)
print("Anomalous URLs:", anomaly_count)


# --------------------------------------------------
# Step 10: Visualization
# --------------------------------------------------

plt.figure(figsize=(12, 6))

plt.scatter(
    range(len(df)),
    df["URL_Length"],
    c=df["Anomaly"],
    s=100
)

plt.axhline(
    df["URL_Length"].mean(),
    linestyle="--",
    label="Average URL Length"
)

plt.xlabel("URL Index")
plt.ylabel("URL Length")

plt.title("URL Anomaly Detection Using Feature Engineering")

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.savefig(
    "url_anomaly_detection.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


print("\nExperiment 6 completed successfully.")