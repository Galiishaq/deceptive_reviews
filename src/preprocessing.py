import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download("stopwords", quiet=True)
stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()

def prepare_dataset(df):
    df["cleaned_reviews"] = df["review"].apply(clean_text)
    X = df["cleaned_reviews"]
    y = df["label"].map({"truthful": 0, "deceptive": 1})
    
    return X, y

def clean_text(text):
    """Basic cleaning: lowercasing, punctuation removal, stopword elimination, and stemming."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    words = text.split()
    words = [stemmer.stem(word) for word in words if word not in stop_words]
    return " ".join(words)

def vectorize_data(df, max_feats, ngram=1):
    vectorizer = TfidfVectorizer(max_features=max_feats, ngram_range=(1, ngram))
    X, y = prepare_dataset(df)
    X_vec = vectorizer.fit_transform(X)
    return X_vec, y, vectorizer