import os
from models import data_collection, data_preprocessing, feature_engineering, model_training, nlp_algorithms, model_evaluation, deployment

def main():
    # Step 1: Data Collection
    print("Starting data collection...")
    raw_data_file = 'data/raw/youtube_data.csv'
    api_key = 'ABCD'
    if not os.path.exists(raw_data_file):
        data_collection.collect_data(api_key, raw_data_file)
    print("Data collection completed.")

    # Step 2: Data Preprocessing
    print("Starting data preprocessing...")
    processed_data_file = 'data/processed/youtube_data_clean.csv'
    data_preprocessing.preprocess_data(raw_data_file, processed_data_file)
    print("Data preprocessing completed.")

    # Step 3: Feature Engineering
    print("Starting feature engineering...")
    feature_engineered_file = 'data/processed/youtube_data_feature_engineered.csv'
    vectorizer_file = 'models/saved_models/tfidf_vectorizer.pkl'
    feature_engineering.engineer_features(processed_data_file, feature_engineered_file, vectorizer_file)
    print("Feature engineering completed.")

    # Step 4: Model Training
    print("Starting model training...")
    model_file = 'models/saved_models/trained_model.pkl'
    multilabel_binarizer_file = 'models/saved_models/multilabel_binarizer.pkl'
    model_training.train_model(feature_engineered_file, model_file, vectorizer_file, multilabel_binarizer_file)
    print("Model training completed.")

    # Step 5: NLP Algorithms (if applicable)
    print("Running NLP algorithms...")
    nlp_algorithms.run_nlp_algorithms(feature_engineered_file)
    print("NLP algorithms completed.")

    # Step 6: Model Evaluation and Tuning
    print("Starting model evaluation and tuning...")
    model_evaluation.evaluate_and_tune_model(feature_engineered_file, model_file, vectorizer_file, multilabel_binarizer_file)
    print("Model evaluation and tuning completed.")

    # Step 7: Deployment
    print("Starting deployment...")
    deployment.run_flask_app(model_file, vectorizer_file, multilabel_binarizer_file)
    print("Deployment completed.")

if __name__ == '__main__':
    main()
