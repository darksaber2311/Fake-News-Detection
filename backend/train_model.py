import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

# ----------------------------
# 1. Load dataset
# ----------------------------
# Change filename if your dataset has a different name
df = pd.read_csv("news.csv")

# Make sure your dataset has "text" and "label"
X = df["text"]
y = df["label"].map({"FAKE": 1, "REAL": 0})  # convert labels to 1/0

# ----------------------------
# 2. Split dataset
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ----------------------------
# 3. Convert text to features
# ----------------------------
vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
X_vec = vectorizer.fit_transform(X)

model = LogisticRegression(max_iter=1000)
model.fit(X_vec, y)

joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")
print("✅ Model trained on full dataset (no accuracy test)")

# ----------------------------
# 4. Train model
# ----------------------------
#model = LogisticRegression(max_iter=1000)
#model.fit(X_train_vec, y_train)

# ----------------------------
# 5. Evaluate
# ----------------------------
#accuracy = model.score(X_test_vec, y_test)
#print(f"✅ Model trained with accuracy: {accuracy:.2f}")

# ----------------------------
# 6. Save model & vectorizer
# ----------------------------
joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("✅ Model and vectorizer saved as model.pkl & vectorizer.pkl")
