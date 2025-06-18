import pandas as pd
import numpy as np
from data_loader import DataLoader
from feature_processor import FeatureProcessor
#from polarity_analysis import PolarityAnalysis
from topic_mining import TopicMining
from price_prediction import PricePrediction
import os
import matplotlib.pyplot as plt
import seaborn as sns
import ssl
import nltk

# Fix SSL certificate verification for NLTK
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Download required NLTK data
print("Downloading required NLTK data...")
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

def main():
    # Create necessary directories
    os.makedirs('polarity_ana_res', exist_ok=True)
    os.makedirs('topic_mining_res', exist_ok=True)
    os.makedirs('price_prediction_res', exist_ok=True)
    
    # Initialize data loader with correct BTC data path
    data_loader = DataLoader('cryptonews.csv', 'btcusd_1-min_data.csv')
    
    # Load and preprocess data
    data_loader.load_data()
    
    # Print data ranges
    print("\nData ranges:")
    print(f"BTC Price data range: {data_loader.btc_price['Timestamp'].min()} to {data_loader.btc_price['Timestamp'].max()}")
    print(f"News data range: {data_loader.cryptonews['date'].min()} to {data_loader.cryptonews['date'].max()}")
    
    # Merge data to get all features
    merged_df = data_loader.merge_data()
    
    # Filter BTC price data to match news data time window
    news_start_date = merged_df['date'].min()
    news_end_date = merged_df['date'].max()
    print(f"\nUsing data from {news_start_date} to {news_end_date}")
    
    # Calculate split date (80% for training, 20% for testing)
    split_date = news_start_date + (news_end_date - news_start_date) * 0.8
    print(f"Training data: {news_start_date} to {split_date}")
    print(f"Testing data: {split_date} to {news_end_date}")
    
    # Initialize topic mining with processed features
    topic_miner = TopicMining()
    
    # Run topic mining analysis first to get topic distributions
    print("\nRunning topic mining analysis...")
    texts = merged_df['text'].values
    dates = merged_df['date'].values
    
    # Run topic analysis with basic features first
    topic_info = topic_miner.run_analysis(texts, dates)
    
    # Load topic distributions and macro themes
    topic_distributions = pd.read_csv("topic_mining_res/topic_distributions.csv")
    
    # Get topic distribution matrix
    num_topics = len(topic_info)
    topic_dist_matrix = np.zeros((len(texts), num_topics))
    for i, row in topic_distributions.iterrows():
        topic_dist_matrix[i, int(row['topic'])] = row['probability']
    
    # Create macro theme one-hot encoding
    macro_themes = topic_distributions['macro_theme'].unique()
    macro_theme_matrix = np.zeros((len(texts), len(macro_themes)))
    for i, theme in enumerate(macro_themes):
        macro_theme_matrix[:, i] = (topic_distributions['macro_theme'] == theme).astype(int)
    
    # Initialize feature processor with all available features including topics and themes
    feature_columns = ['polarity', 'subjectivity']
    if 'Close' in merged_df.columns:
        feature_columns.extend(['Close', 'Volume'])
    
    print("\nProcessing features...")
    feature_processor = FeatureProcessor(merged_df, feature_columns)
    
    # Remove collinear features and get cleaned data
    selected_features = feature_processor.remove_collinear_features()
    cleaned_data = feature_processor.get_cleaned_data()
    
    print("\nSelected features after processing:", selected_features)
    
    # Combine all features: price data, sentiment, topics, and macro themes
    combined_features = np.hstack([
        cleaned_data.values,  # Price and sentiment features
        topic_dist_matrix,    # Topic distributions
        macro_theme_matrix    # Macro theme one-hot encoding
    ])
    
    # Ensure data alignment
    if len(combined_features) != len(merged_df):
        print("Warning: Feature length doesn't match data length")
        # Align data by date
        feature_df = pd.DataFrame(combined_features, index=dates)
        merged_df.set_index('date', inplace=True)
        aligned_data = merged_df.join(feature_df, how='inner')
        merged_df.reset_index(inplace=True)
        combined_features = aligned_data.iloc[:, -combined_features.shape[1]:].values
        texts = aligned_data['text'].values
        dates = aligned_data.index.values
    
    # Print detailed information about top topics and macro themes
    print("\nDetailed information about top topics:")
    for topic_id in topic_info['Topic'].head(10):
        print(f"\nTopic {topic_id}:")
        keywords = topic_miner.get_topic_keywords(topic_id)
        print("Keywords:", ", ".join([f"{word} ({score:.2f})" for word, score in keywords[:10]]))
    
    print("\nMacro themes distribution:")
    for i, theme in enumerate(macro_themes):
        count = np.sum(macro_theme_matrix[:, i])
        print(f"{theme}: {count} articles")
    
    # Initialize price prediction
    print("\nInitializing price prediction model...")
    price_predictor = PricePrediction(sequence_length=10)
    
    # Prepare data for price prediction
    prices = np.array(merged_df['Close'].values)
    
    # Save data ranges for reference
    data_ranges = pd.DataFrame({
        'dataset': ['BTC Price', 'News', 'Combined', 'Training', 'Testing'],
        'start_date': [
            data_loader.btc_price['Timestamp'].min(),
            data_loader.cryptonews['date'].min(),
            merged_df['date'].min(),
            news_start_date,
            split_date
        ],
        'end_date': [
            data_loader.btc_price['Timestamp'].max(),
            data_loader.cryptonews['date'].max(),
            merged_df['date'].max(),
            split_date,
            news_end_date
        ],
        'num_samples': [
            len(data_loader.btc_price),
            len(data_loader.cryptonews),
            len(merged_df),
            int(len(merged_df) * 0.8),
            int(len(merged_df) * 0.2)
        ]
    })
    data_ranges.to_csv('price_prediction_res/data_ranges.csv', index=False)
    
    # Save feature information
    feature_info = pd.DataFrame({
        'feature_type': ['Price/Sentiment', 'Topics', 'Macro Themes'],
        'num_features': [
            cleaned_data.shape[1],
            topic_dist_matrix.shape[1],
            macro_theme_matrix.shape[1]
        ],
        'feature_names': [
            list(cleaned_data.columns),
            [f'Topic_{i}' for i in range(topic_dist_matrix.shape[1])],
            list(macro_themes)
        ]
    })
    feature_info.to_csv('price_prediction_res/feature_info.csv', index=False)
    
    # Check if we have a saved model
    if os.path.exists('price_prediction_res/best_model.pth'):
        print("\nFound saved model. Loading...")
        if price_predictor.load_model(combined_features.shape[1]):
            print("Successfully loaded saved model")
            # Load losses from file if they exist
            if os.path.exists('price_prediction_res/losses.csv'):
                losses_df = pd.read_csv('price_prediction_res/losses.csv')
                train_losses = losses_df['train_loss'].tolist()
                test_losses = losses_df['test_loss'].tolist()
                price_predictor.plot_losses(train_losses, test_losses)
        else:
            print("Failed to load saved model. Training new model...")
            print("\nStarting model training...")
            print("This may take several minutes. Progress will be shown below.")
            train_losses, test_losses = price_predictor.train(
                prices=prices,
                topic_distributions=combined_features,
                epochs=100,
                learning_rate=0.001,
                patience=10
            )
            # Save losses to file
            losses_df = pd.DataFrame({
                'train_loss': train_losses,
                'test_loss': test_losses
            })
            losses_df.to_csv('price_prediction_res/losses.csv', index=False)
    else:
        print("\nNo saved model found. Training new model...")
        print("\nStarting model training...")
        print("This may take several minutes. Progress will be shown below.")
        train_losses, test_losses = price_predictor.train(
            prices=prices,
            topic_distributions=combined_features,
            epochs=100,
            learning_rate=0.001,
            patience=10
        )
        # Save losses to file
        losses_df = pd.DataFrame({
            'train_loss': train_losses,
            'test_loss': test_losses
        })
        losses_df.to_csv('price_prediction_res/losses.csv', index=False)
    
    # Plot losses if we have them
    if 'train_losses' in locals() and 'test_losses' in locals():
        price_predictor.plot_losses(train_losses, test_losses)
    
    # Make predictions for multiple timeframes
    print("\nMaking predictions for different timeframes...")
    timeframes = [30, 90]  # 1 month and 3 months
    predictions = price_predictor.predict_multiple_timeframes(prices[-10:], combined_features[-1], timeframes)
    
    print("\nPredictions for BTC price:")
    print(f"Current price: ${prices[-1]:.2f}")
    for days, pred_price in predictions.items():
        print(f"{days}-day prediction: ${pred_price:.2f}")
    
    print(f"\nNote: Model trained on data from {news_start_date} to {split_date}")
    print(f"Testing period: {split_date} to {news_end_date}")
    
    # Plot predictions
    print("\nGenerating prediction plots...")
    price_predictor.plot_predictions(
        actual_prices=prices[-100:],  # Last 100 days of actual prices
        predicted_prices=predictions,
        dates=merged_df['date'].values[-100:],
        timeframes=timeframes
    )
    
    # Print the most influential topics and themes for the last prediction
    print("\nMost influential topics for the last prediction:")
    topic_start_idx = cleaned_data.shape[1]
    top_topics_idx = np.argsort(combined_features[-1][topic_start_idx:topic_start_idx + num_topics])[-3:][::-1]
    for idx in top_topics_idx:
        try:
            topic_keywords = topic_miner.get_topic_keywords(idx)
            if isinstance(topic_keywords, list) and len(topic_keywords) > 0:
                keywords_str = ", ".join([f"{word}" for word, _ in topic_keywords[:5]])
                print(f"Topic {idx} (probability: {combined_features[-1][topic_start_idx + idx]:.3f}): {keywords_str}")
            else:
                print(f"Topic {idx} (probability: {combined_features[-1][topic_start_idx + idx]:.3f}): No keywords available")
        except Exception as e:
            print(f"Topic {idx} (probability: {combined_features[-1][topic_start_idx + idx]:.3f}): Error getting keywords")
    
    print("\nMost influential macro themes for the last prediction:")
    theme_start_idx = topic_start_idx + num_topics
    top_themes_idx = np.argsort(combined_features[-1][theme_start_idx:])[-3:][::-1]
    for idx in top_themes_idx:
        theme = macro_themes[idx]
        print(f"{theme} (weight: {combined_features[-1][theme_start_idx + idx]:.3f})")
    
    # Save prediction results
    results_df = pd.DataFrame({
        'date': dates[-1],
        'last_price': prices[-1],
        'predicted_price_30d': predictions[30],
        'predicted_price_90d': predictions[90],
        'top_topics': [top_topics_idx.tolist()],
        'topic_probabilities': [combined_features[-1][topic_start_idx:topic_start_idx + num_topics][top_topics_idx].tolist()],
        'top_themes': [macro_themes[top_themes_idx].tolist()],
        'theme_weights': [combined_features[-1][theme_start_idx:][top_themes_idx].tolist()],
        'model_training_start': news_start_date,
        'model_training_end': news_end_date
    })
    results_df.to_csv('price_prediction_res/last_prediction.csv', index=False)
    
    print("\nResults have been saved to price_prediction_res/")
    print("- last_prediction.csv: Detailed prediction results")
    print("- price_predictions.png: Plot of actual vs predicted prices")
    print("- losses.png: Training and test loss curves")

if __name__ == "__main__":
    main()
    