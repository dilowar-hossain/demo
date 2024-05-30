import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
import joblib
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

# Ensure necessary NLTK data is downloaded
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load and preprocess data
def preprocess_text(text):
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t.isalpha()]
    stop_words = set(stopwords.words('english'))
    tokens = [t for t in tokens if t not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return ' '.join(tokens)

def prepare_supervised_data(df):
    df['cleaned_title'] = df['video_title'].apply(preprocess_text)
    df['cleaned_description'] = df['video_description'].apply(preprocess_text)
    X = df['cleaned_title'] + ' ' + df['cleaned_description']
    y = df['tags'].apply(lambda x: x.split(','))
    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(y)
    return X, y, mlb

file_path = 'youtube_data_feature_engineered.csv'
df = pd.read_csv(file_path)
X, y, mlb = prepare_supervised_data(df)

# Vectorize text data
vectorizer = TfidfVectorizer(max_features=1000)
X_tfidf = vectorizer.fit_transform(X)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_tfidf, y)

# Save model, vectorizer, and binarizer
joblib.dump(model, 'trained_model.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
joblib.dump(mlb, 'multilabel_binarizer.pkl')
