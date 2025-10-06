import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib
import sys

# ----------------------------
# 1. Load dataset (tolerant read)
# ----------------------------
df = pd.read_csv("evaluation.csv", sep=';', engine='python', on_bad_lines='skip', encoding='utf-8')

# normalize column names
df.columns = [c.strip().lower() for c in df.columns]
print("Columns:", df.columns.tolist())

# drop pure index column if present
if 'unnamed: 0' in df.columns:
    df = df.drop(columns=['unnamed: 0'])

# require text & label
if 'text' not in df.columns or 'label' not in df.columns:
    raise RuntimeError("CSV must contain 'text' and 'label' columns. Found: " + ", ".join(df.columns))

# drop rows missing text or label
df = df.dropna(subset=['text', 'label'])
df['text'] = df['text'].astype(str)

# ----------------------------
# 2. Inspect label values (very important)
# ----------------------------
raw_labels = df['label'].unique()
print("Raw unique label values (sample up to 50):", raw_labels[:50])

# show counts for debugging
print("Label value counts:")
print(df['label'].value_counts(dropna=False).head(50))

# ----------------------------
# 3. Normalize labels to 0/1
#    Accepts:
#      - 'FAKE' / 'REAL' (any case)
#      - numeric 0/1
#      - 'fake'/'real' etc.
# ----------------------------
def map_label(x):
    # handle numeric-like values first
    try:
        # If it is numeric string or numeric, try converting
        xi = float(x)
        if xi == 1.0:
            return 1
        if xi == 0.0:
            return 0
    except Exception:
        pass

    # otherwise handle textual labels
    s = str(x).strip().upper()
    if s in ("FAKE", "F", "1", "TRUE", "YES"):
        return 1
    if s in ("REAL", "R", "0", "FALSE", "NO"):
        return 0

    # unknown label -> return None to be dropped later
    return None

df['label_mapped'] = df['label'].apply(map_label)

# How many rows got mapped to None (unknown label)
unknown_count = df['label_mapped'].isna().sum()
if unknown_count:
    print(f"Warning: {unknown_count} rows have unknown labels and will be dropped. Example raw values:")
    print(df[df['label_mapped'].isna()]['label'].unique()[:20])

# drop unknowns
df = df.dropna(subset=['label_mapped'])
df['label_mapped'] = df['label_mapped'].astype(int)

print("Mapped label counts:")
print(df['label_mapped'].value_counts())

# If after mapping only one class exists, abort with a clear message
if df['label_mapped'].nunique() < 2:
    print("ERROR: After mapping, the dataset contains only one label class. Cannot train a binary classifier.")
    print("Please check 'evaluation.csv' label column. Here are the distinct raw label values:")
    print(raw_labels[:50])
    sys.exit(1)

# ----------------------------
# 4. Prepare X, y and split
# ----------------------------
X = df['text']
y = df['label_mapped']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ----------------------------
# 5. Vectorize (fit on train)
# ----------------------------
vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# ----------------------------
# 6. Train model
# ----------------------------
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

# ----------------------------
# 7. Evaluate
# ----------------------------
accuracy = model.score(X_test_vec, y_test)
print(f"✅ Model trained. Test accuracy: {accuracy:.4f}")

# ----------------
