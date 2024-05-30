import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score, hamming_loss, make_scorer
from sklearn.feature_extraction.text import TfidfVectorizer

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

# Model Evaluation Metrics
def evaluate_model(y_true, y_pred):
    precision = precision_score(y_true, y_pred, average='micro')
    recall = recall_score(y_true, y_pred, average='micro')
    f1 = f1_score(y_true, y_pred, average='micro')
    hamming = hamming_loss(y_true, y_pred)
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1 Score: {f1}')
    print(f'Hamming Loss: {hamming}')
    return precision, recall, f1, hamming

# Cross-Validation
def cross_validate_model(model, X, y, cv=5):
    scores = cross_val_score(model, X, y, cv=cv, scoring=make_scorer(f1_score, average='micro'))
    print(f'Cross-Validation F1 Scores: {scores}')
    print(f'Mean F1 Score: {np.mean(scores)}')
    return scores

# Hyperparameter Tuning using Grid Search
def grid_search_tuning(model, param_grid, X, y):
    grid_search = GridSearchCV(model, param_grid, scoring=make_scorer(f1_score, average='micro'), cv=5)
    grid_search.fit(X, y)
    print(f'Best Parameters: {grid_search.best_params_}')
    print(f'Best Score: {grid_search.best_score_}')
    return grid_search.best_estimator_

# Hyperparameter Tuning using Random Search
def random_search_tuning(model, param_dist, X, y, n_iter=50):
    random_search = RandomizedSearchCV(model, param_distributions=param_dist, n_iter=n_iter, scoring=make_scorer(f1_score, average='micro'), cv=5, random_state=42)
    random_search.fit(X, y)
    print(f'Best Parameters: {random_search.best_params_}')
    print(f'Best Score: {random_search.best_score_}')
    return random_search.best_estimator_

def main():
    file_path = 'youtube_data_feature_engineered.csv'
    df = load_data(file_path)

    # Prepare data
    X, y, mlb = prepare_supervised_data(df)
    vectorizer = TfidfVectorizer(max_features=1000)
    X_tfidf = vectorizer.fit_transform(X)

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

    # Define the model
    model = LogisticRegression(max_iter=1000)

    # Cross-Validation
    cross_validate_model(model, X_tfidf, y)

    # Hyperparameter Tuning with Grid Search
    param_grid = {
        'C': [0.1, 1, 10],
        'penalty': ['l2']
    }
    best_model_grid = grid_search_tuning(model, param_grid, X_train, y_train)

    # Hyperparameter Tuning with Random Search
    param_dist = {
        'C': np.logspace(-3, 3, 10),
        'penalty': ['l2']
    }
    best_model_random = random_search_tuning(model, param_dist, X_train, y_train)

    # Evaluate the best model from Grid Search
    y_pred_grid = best_model_grid.predict(X_test)
    print('Evaluation Metrics for Grid Search Best Model:')
    evaluate_model(y_test, y_pred_grid)

    # Evaluate the best model from Random Search
    y_pred_random = best_model_random.predict(X_test)
    print('Evaluation Metrics for Random Search Best Model:')
    evaluate_model(y_test, y_pred_random)

if __name__ == '__main__':
    main()
