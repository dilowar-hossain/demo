import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

import spacy
spacy.cli.download('en_core_web_sm')
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from rake_nltk import Rake
import yake
import spacy
from gensim import corpora, models
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

# Keyword Extraction using TF-IDF
def extract_keywords_tfidf(texts, max_features=10):
    vectorizer = TfidfVectorizer(max_features=max_features)
    tfidf_matrix = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()
    dense = tfidf_matrix.todense()
    keyword_list = []
    for text in dense:
        sorted_keywords = [feature_names[i] for i in text.argsort()[::-1][:max_features]]
        keyword_list.append(sorted_keywords)
    return keyword_list

# Keyword Extraction using RAKE
def extract_keywords_rake(text):
    rake = Rake()
    rake.extract_keywords_from_text(text)
    return rake.get_ranked_phrases()[:10]

# Keyword Extraction using YAKE
def extract_keywords_yake(text, max_ngram_size=1, num_keywords=10):
    kw_extractor = yake.KeywordExtractor(n=max_ngram_size, top=num_keywords)
    keywords = kw_extractor.extract_keywords(text)
    return [keyword for keyword, score in keywords]

# Named Entity Recognition (NER)
def named_entity_recognition(text):
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(text)
    return [(ent.text, ent.label_) for ent in doc.ents]

# Topic Modeling using LDA
def topic_modeling(texts, num_topics=5, num_words=10):
    texts = [text.split() for text in texts]
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]
    lda_model = models.LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=10)
    topics = lda_model.print_topics(num_words=num_words)
    return topics

def main():
    file_path = 'youtube_data_feature_engineered.csv'
    df = load_data(file_path)
    
    # Preprocess the text data
    df['cleaned_title'] = df['video_title'].apply(preprocess_text)
    df['cleaned_description'] = df['video_description'].apply(preprocess_text)
    combined_text = df['cleaned_title'] + ' ' + df['cleaned_description']

    # Keyword Extraction using TF-IDF
    df['keywords_tfidf'] = extract_keywords_tfidf(combined_text)

    # Keyword Extraction using RAKE
    df['keywords_rake'] = df['cleaned_description'].apply(extract_keywords_rake)

    # Keyword Extraction using YAKE
    df['keywords_yake'] = df['cleaned_description'].apply(lambda x: extract_keywords_yake(x))

    # Named Entity Recognition (NER)
    df['named_entities'] = df['cleaned_description'].apply(named_entity_recognition)

    # Topic Modeling using LDA
    topics = topic_modeling(combined_text)
    print('LDA Topics:')
    for topic in topics:
        print(topic)
    
    # Save the processed DataFrame to a new CSV file
    df.to_csv('youtube_data_nlp_processed.csv', index=False)
    print('NLP processed data saved to youtube_data_nlp_processed.csv')

if __name__ == '__main__':
    main()
