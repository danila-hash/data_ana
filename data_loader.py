import pandas as pd
import numpy as np
import ast
from functools import lru_cache
import os
import re
import unicodedata
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

class DataLoader:
    def __init__(self, cryptonews_path, btc_price_path):
        """
        Initialize the DataLoader with paths to required data files.
        
        Args:
            cryptonews_path (str): Path to the cryptonews CSV file
            btc_price_path (str): Path to the BTC price data CSV file
        """
        # Download required NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/wordnet')
        except LookupError:
            print("Downloading required NLTK data...")
            nltk.download('punkt')
            nltk.download('wordnet')
            nltk.download('averaged_perceptron_tagger')

        # Initialize lemmatizer
        self.lemmatizer = WordNetLemmatizer()
        
        # Construct paths relative to the data folder
        self.cryptonews_path = os.path.join('data', cryptonews_path)
        self.btc_price_path = os.path.join('data', btc_price_path)
        self.cryptonews = None
        self.btc_price = None
        self.btc_price_dict = None
        self.btc_volume_dict = None
        self.merged_df = None

        # Verify that the data files exist
        if not os.path.exists(self.cryptonews_path):
            raise FileNotFoundError(f"Crypto news data file not found at: {self.cryptonews_path}")
        if not os.path.exists(self.btc_price_path):
            raise FileNotFoundError(f"BTC price data file not found at: {self.btc_price_path}")

    def clean_text(self, text):
        """
        Clean text data by removing special characters, extra whitespace, etc.
        while preserving all words.
        
        Args:
            text (str): Text to clean
            
        Returns:
            str: Cleaned text
        """
        if not isinstance(text, str):
            return ""
            
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove special characters but keep spaces
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove unicode characters
        text = unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('utf-8')
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        return text

    def load_data(self):
        """Load and preprocess the data from CSV files."""
        # Read the data with optimized settings
        self.cryptonews = pd.read_csv(self.cryptonews_path, usecols=['date', 'sentiment', 'text'])
        self.btc_price = pd.read_csv(self.btc_price_path, usecols=['Timestamp', 'Close', 'Volume'])

        # Convert and align timestamps
        self.cryptonews['date'] = pd.to_datetime(self.cryptonews['date'], format='mixed', errors='coerce').dt.floor('min')
        self.btc_price['Timestamp'] = pd.to_datetime(self.btc_price['Timestamp'], unit='s', errors='coerce').dt.floor('min')

        # Clean text data
        print("Cleaning text data...")
        self.cryptonews['cleaned_text'] = self.cryptonews['text'].apply(self.clean_text)
        
        # Remove rows with empty cleaned text
        self.cryptonews = self.cryptonews[self.cryptonews['cleaned_text'].str.len() > 0]

        # Create lookup dictionaries for faster access
        self.btc_price_dict = self.btc_price.set_index('Timestamp')['Close'].to_dict()
        self.btc_volume_dict = self.btc_price.set_index('Timestamp')['Volume'].to_dict()

        # Process sentiment data
        self._process_sentiment()

        # Remove rows with null values
        self.cryptonews = self.cryptonews.dropna(subset=['date', 'polarity'])
        self.btc_price = self.btc_price.dropna(subset=['Timestamp', 'Volume'])

        return self

    @staticmethod
    @lru_cache(maxsize=10000)
    def _extract_sentiment(sentiment_str):
        """Extract polarity and subjectivity from sentiment string."""
        try:
            sentiment_dict = ast.literal_eval(sentiment_str)
            return float(sentiment_dict['polarity']), float(sentiment_dict['subjectivity'])
        except:
            return np.nan, np.nan

    def _process_sentiment(self):
        """Process sentiment data to extract polarity and subjectivity."""
        self.cryptonews[['polarity', 'subjectivity']] = pd.DataFrame(
            self.cryptonews['sentiment'].apply(self._extract_sentiment).tolist(),
            index=self.cryptonews.index
        )

    def merge_data(self):
        """Merge news data with price data."""
        if self.cryptonews is None or self.btc_price is None:
            raise ValueError("Data must be loaded first using load_data()")

        self.merged_df = pd.merge_asof(
            self.cryptonews.sort_values("date"),
            self.btc_price.sort_values("Timestamp"),
            left_on="date",
            right_on="Timestamp",
            direction="backward"
        )
        return self.merged_df

    def calculate_future_values(self, date, hours):
        """Calculate future price and volume for a given date and time period."""
        future_date = date + pd.Timedelta(hours=hours)
        return (
            self.btc_price_dict.get(future_date, np.nan),
            self.btc_volume_dict.get(future_date, np.nan)
        )

    def add_future_values(self, time_periods):
        """
        Add future price and volume changes for specified time periods.
        
        Args:
            time_periods (list): List of hours to calculate future values for
        """
        if self.merged_df is None:
            raise ValueError("Data must be merged first using merge_data()")

        for hours in time_periods:
            print(f"Calculating {hours}h future values...")
            future_values = self.merged_df['date'].apply(lambda x: self.calculate_future_values(x, hours))
            self.merged_df[f'future_price_{hours}h'] = future_values.apply(lambda x: x[0])
            self.merged_df[f'future_volume_{hours}h'] = future_values.apply(lambda x: x[1])
            
            # Calculate price change percentage
            self.merged_df[f'price_change_pct_{hours}h'] = (
                (self.merged_df[f'future_price_{hours}h'] - self.merged_df['Close']) / 
                self.merged_df['Close']
            ) * 100
            
            # Calculate volume change percentage, handling zero volumes
            self.merged_df[f'volume_change_pct_{hours}h'] = np.where(
                self.merged_df['Volume'] > 0,
                ((self.merged_df[f'future_volume_{hours}h'] - self.merged_df['Volume']) / 
                 self.merged_df['Volume']) * 100,
                np.nan
            )

        # Remove rows with null values in price and volume changes
        self.merged_df = self.merged_df.dropna(subset=[
            f'price_change_pct_{hours}h' for hours in time_periods
        ] + [
            f'volume_change_pct_{hours}h' for hours in time_periods
        ])

        return self.merged_df

    def get_sentiment_dataframes(self):
        """
        Split data into positive, negative, and neutral sentiment dataframes.
        
        Returns:
            tuple: (positive_news_df, negative_news_df, neutral_news_df)
        """
        if self.merged_df is None:
            raise ValueError("Data must be processed first")

        positive_news_df = self.merged_df[self.merged_df['polarity'] > 0.1].copy()
        negative_news_df = self.merged_df[self.merged_df['polarity'] < -0.1].copy()
        neutral_news_df = self.merged_df[
            (self.merged_df['polarity'] >= -0.1) & 
            (self.merged_df['polarity'] <= 0.1)
        ].copy()

        return positive_news_df, negative_news_df, neutral_news_df
