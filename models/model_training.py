import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer  # Added this line
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Bidirectional
from keras.callbacks import EarlyStopping
from keras.layers import Transformer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Load Data
def load_data(file_path):
    return pd.read_csv(file_path)

# Preprocess Data
def preprocess_text(text):
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t.isalpha()]
    stop_words = set(stopwords.words('english'))
    tokens = [t for t in tokens if t not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return ' '.join(tokens)

# Prepare Data for Supervised Learning
def prepare_supervised_data(df):
    df['cleaned_title'] = df['video_title'].apply(preprocess_text)
    df['cleaned_description'] = df['video_description'].apply(preprocess_text)
    X = df['cleaned_title'] + ' ' + df['cleaned_description']
    y = df['tags'].apply(lambda x: x.split(','))
    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(y)
    return X, y, mlb

# Train Supervised Learning Model
def train_supervised_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    vectorizer = TfidfVectorizer(max_features=1000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_tfidf, y_train)
    y_pred = model.predict(X_test_tfidf)
    print(classification_report(y_test, y_pred))
    return model, vectorizer

# Prepare Data for Sequence-to-Sequence Model
def prepare_seq2seq_data(df, max_seq_len=100):
    tokenizer = Tokenizer(num_words=10000)
    tokenizer.fit_on_texts(df['cleaned_title'] + ' ' + df['cleaned_description'])
    X = tokenizer.texts_to_sequences(df['cleaned_title'] + ' ' + df['cleaned_description'])
    X = pad_sequences(X, maxlen=max_seq_len, padding='post')
    y = df['tags'].apply(lambda x: x.split(','))
    y = tokenizer.texts_to_sequences(y)
    y = pad_sequences(y, maxlen=max_seq_len, padding='post')
    return X, y, tokenizer

# Train Sequence-to-Sequence Model_
def train_seq2seq_model(X, y, tokenizer, embedding_dim=100, max_seq_len=100):
    vocab_size = len(tokenizer.word_index) + 1
    model = Sequential([
        Embedding(vocab_size, embedding_dim, input_length=max_seq_len),
        Bidirectional(LSTM(64, return_sequences=True)),
        Dense(vocab_size, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    early_stopping = EarlyStopping(monitor='val_loss', patience=3)
    model.fit(X, y, epochs=10, validation_split=0.2, callbacks=[early_stopping], batch_size=32)
    return model

# Data Augmentation
def data_augmentation(df):
    # Placeholder for data augmentation techniques like back-translation
    # Implement techniques like back-translation or paraphrasing to create synthetic data
    return df

def main():
    file_path = 'youtube_data_feature_engineered.csv'
    df = load_data(file_path)

    # Data Augmentation
    df = data_augmentation(df)

    # Supervised Learning
    X, y, mlb = prepare_supervised_data(df)
    supervised_model, vectorizer = train_supervised_model(X, y)

    # Sequence-to-Sequence Model
    max_seq_len = 100
    X_seq, y_seq, tokenizer = prepare_seq2seq_data(df, max_seq_len)
    seq2seq_model = train_seq2seq_model(X_seq, y_seq, tokenizer, max_seq_len=max_seq_len)

if __name__ == '__main__':
    main()
