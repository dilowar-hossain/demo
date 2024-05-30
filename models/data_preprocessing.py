import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

import pandas as pd
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer

def load_data(file_path):
    return pd.read_csv(file_path)

def clean_and_organize_data(df):
    # Remove duplicates
    df = df.drop_duplicates(subset='video_id')
    
    # Remove rows with missing metadata
    df = df.dropna(subset=['video_title', 'video_description'])
    
    # Normalize text data: convert to lowercase and remove special characters
    df['video_title'] = df['video_title'].apply(lambda x: x.lower().translate(str.maketrans('', '', string.punctuation)))
    df['video_description'] = df['video_description'].apply(lambda x: x.lower().translate(str.maketrans('', '', string.punctuation)))
    
    return df

def text_tokenization(text):
    return word_tokenize(text)

def remove_stopwords(tokens):
    stop_words = set(stopwords.words('english'))
    return [word for word in tokens if word not in stop_words]

def stemming(tokens):
    stemmer = PorterStemmer()
    return [stemmer.stem(word) for word in tokens]

def lemmatization(tokens):
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(word) for word in tokens]

def preprocess_text(text):
    tokens = text_tokenization(text)
    tokens = remove_stopwords(tokens)
    tokens = stemming(tokens)
    tokens = lemmatization(tokens)
    return ' '.join(tokens)

def preprocess_dataframe(df):
    df['cleaned_video_title'] = df['video_title'].apply(preprocess_text)
    df['cleaned_video_description'] = df['video_description'].apply(preprocess_text)
    return df

def main():
    file_path = 'youtube_data.csv'
    df = load_data(file_path)
    df = clean_and_organize_data(df)
    df = preprocess_dataframe(df)
    
    # Save the preprocessed data to a new CSV file
    df.to_csv('youtube_data_preprocessed.csv', index=False)
    print('Preprocessed data saved to youtube_data_preprocessed.csv')

if __name__ == '__main__':
    main()
