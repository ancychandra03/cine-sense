import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Keep negations — critical for sentiment
# "not good" should NOT become "good"
stop_words.discard('not')
stop_words.discard('no')
stop_words.discard("don't")
stop_words.discard("doesn't")
stop_words.discard("wasn't")

def clean_text(text):
    # 1. Lowercase
    text = text.lower()

    # 2. Remove HTML tags (IMDB reviews contain <br /> tags)
    text = re.sub(r'<.*?>', ' ', text)

    # 3. Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)

    # 4. Remove special characters, keep only letters
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # 5. Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    # 6. Tokenize, remove stopwords, lemmatize
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(w) for w in tokens
              if w not in stop_words and len(w) > 2]

    return ' '.join(tokens)


if __name__ == '__main__':
    # Quick test
    samples = [
        "This movie was absolutely AMAZING!!!",
        "Terrible film. Not good at all. Don't watch it.",
        "The <br /> acting was mediocre at best.",
    ]
    for s in samples:
        print(f"Original : {s}")
        print(f"Cleaned  : {clean_text(s)}")
        print()