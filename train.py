import pandas as pd
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from preprocess import clean_text

# ── 1. Load data ──────────────────────────────────────────
print("Loading dataset...")
df = pd.read_csv('data/IMDB Dataset.csv')
print(f"Dataset shape: {df.shape}")
print(df['sentiment'].value_counts())

# ── 2. Preprocess ─────────────────────────────────────────
print("\nCleaning text (this takes ~2 min for 50K rows)...")
df['clean_review'] = df['review'].apply(clean_text)

# Encode labels: positive=1, negative=0
df['label'] = df['sentiment'].map({'positive': 1, 'negative': 0})

# ── 3. Train / test split ─────────────────────────────────
X = df['clean_review']
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTrain size: {len(X_train)} | Test size: {len(X_test)}")

# ── 4. TF-IDF vectorization ───────────────────────────────
print("\nFitting TF-IDF vectorizer...")
tfidf = TfidfVectorizer(
    max_features=10000,   # keep top 10K words
    ngram_range=(1, 2),   # unigrams + bigrams
    min_df=5,             # ignore very rare words
    sublinear_tf=True     # apply log normalization
)
X_train_vec = tfidf.fit_transform(X_train)
X_test_vec  = tfidf.transform(X_test)

# ── 5. Train models ───────────────────────────────────────
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, C=1.0),
    'Naive Bayes':         MultinomialNB(alpha=0.1),
}

best_model = None
best_score = 0

for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train_vec, y_train)
    preds = model.predict(X_test_vec)
    acc   = accuracy_score(y_test, preds)
    print(f"Accuracy: {acc:.4f}")
    print(classification_report(y_test, preds,
          target_names=['Negative', 'Positive']))
    if acc > best_score:
        best_score = acc
        best_model = model
        best_name  = name

# ── 6. Save best model ────────────────────────────────────
os.makedirs('models', exist_ok=True)
with open('models/model.pkl', 'wb') as f:
    pickle.dump(best_model, f)
with open('models/tfidf.pkl', 'wb') as f:
    pickle.dump(tfidf, f)

print(f"\nBest model: {best_name} ({best_score:.4f})")
print("Saved to models/model.pkl and models/tfidf.pkl")