from data_loader import DataLoader
from topic_mining import TopicMining
import pandas as pd

def main():
    # Initialize data loader
    data_loader = DataLoader('cryptonews.csv', 'btc_price.csv')
    
    # Load and preprocess data
    data_loader.load_data()
    
    # Get cleaned texts and dates
    texts = data_loader.cryptonews['cleaned_text'].tolist()
    dates = data_loader.cryptonews['date'].tolist()
    
    # Initialize and run topic mining
    topic_miner = TopicMining()
    topic_info = topic_miner.run_analysis(texts, dates)
    
    # Print detailed information about top topics
    print("\nDetailed information about top topics:")
    for topic_id in topic_info['Topic'].head(10):
        print(f"\nTopic {topic_id}:")
        keywords = topic_miner.get_topic_keywords(topic_id)
        print("Keywords:", ", ".join([f"{word} ({score:.2f})" for word, score in keywords[:10]]))

if __name__ == "__main__":
    main() 