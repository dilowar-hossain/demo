import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import KeyedVectors

# Load pre-trained GloVe embeddings
def load_glove_embeddings(file_path):
    embeddings_index = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    return embeddings_index

def get_average_word_embeddings(text, embeddings_index, embedding_dim):
    words = text.split()
    valid_embeddings = [embeddings_index[word] for word in words if word in embeddings_index]
    if not valid_embeddings:
        return np.zeros(embedding_dim)
    return np.mean(valid_embeddings, axis=0)

def main():
    file_path = 'youtube_data_preprocessed.csv'
    glove_file_path = 'path/to/glove.6B.100d.txt'  # Update with your GloVe file path
    
    df = pd.read_csv(file_path)
    
    # 1. TF-IDF Calculation
    tfidf_vectorizer = TfidfVectorizer(max_features=1000)
    tfidf_matrix_title = tfidf_vectorizer.fit_transform(df['cleaned_video_title'])
    tfidf_matrix_description = tfidf_vectorizer.fit_transform(df['cleaned_video_description'])
    
    # Convert TF-IDF matrices to DataFrame
    tfidf_df_title = pd.DataFrame(tfidf_matrix_title.toarray(), columns=tfidf_vectorizer.get_feature_names_out())
    tfidf_df_description = pd.DataFrame(tfidf_matrix_description.toarray(), columns=tfidf_vectorizer.get_feature_names_out())
    
    # Concatenate TF-IDF DataFrames with original DataFrame
    df = pd.concat([df, tfidf_df_title.add_prefix('title_tfidf_'), tfidf_df_description.add_prefix('description_tfidf_')], axis=1)
    
    # 2. Word Embeddings (GloVe)
    embeddings_index = load_glove_embeddings(glove_file_path)
    embedding_dim = 100  # Update if using a different GloVe model
    
    # Calculate average word embeddings for titles and descriptions
    df['title_embedding'] = df['cleaned_video_title'].apply(lambda x: get_average_word_embeddings(x, embeddings_index, embedding_dim))
    df['description_embedding'] = df['cleaned_video_description'].apply(lambda x: get_average_word_embeddings(x, embeddings_index, embedding_dim))
    
    # Save the processed DataFrame to a new CSV file
    df.to_csv('youtube_data_feature_engineered.csv', index=False)
    print('Feature engineered data saved to youtube_data_feature_engineered.csv')

if __name__ == '__main__':
    main()
