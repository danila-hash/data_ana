import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
from statsmodels.tsa.stattools import grangercausalitytests
import warnings
import os
from data_loader import DataLoader
from feature_processor import FeatureProcessor

# Create directory for saving plots if it doesn't exist
os.makedirs('polarity_ana_res', exist_ok=True)

# Suppress FutureWarning about verbose parameter
warnings.filterwarnings('ignore', category=FutureWarning)

# Initialize data loader and load data
data_loader = DataLoader('cryptonews.csv', 'btcusd_1-min_data.csv')
data_loader.load_data()
merged_df = data_loader.merge_data()

# Calculate future values for different time periods
time_periods = [1, 4, 24]
merged_df = data_loader.add_future_values(time_periods)

# Prepare feature sets for price and volume
price_features = [f'price_change_pct_{hours}h' for hours in time_periods]
volume_features = [f'volume_change_pct_{hours}h' for hours in time_periods]

# Get sentiment-based dataframes
positive_news_df, negative_news_df, neutral_news_df = data_loader.get_sentiment_dataframes()

# Calculate correlations for each news type using all features
print("\nCalculating correlations with all features...")
price_correlations = []
volume_correlations = []

for hours in time_periods:
    # Price correlations
    pos_price_corr = pearsonr(positive_news_df['polarity'].dropna(), 
                            positive_news_df[f'price_change_pct_{hours}h'].dropna())[0] if len(positive_news_df) > 0 else 0
    neg_price_corr = pearsonr(negative_news_df['polarity'].dropna(), 
                            negative_news_df[f'price_change_pct_{hours}h'].dropna())[0] if len(negative_news_df) > 0 else 0
    neu_price_corr = pearsonr(neutral_news_df['polarity'].dropna(), 
                            neutral_news_df[f'price_change_pct_{hours}h'].dropna())[0] if len(neutral_news_df) > 0 else 0
    price_correlations.append((pos_price_corr, neg_price_corr, neu_price_corr))

for hours in time_periods:
    # Volume correlations
    pos_vol_corr = pearsonr(positive_news_df['polarity'].dropna(), 
                          positive_news_df[f'volume_change_pct_{hours}h'].dropna())[0] if len(positive_news_df) > 0 else 0
    neg_vol_corr = pearsonr(negative_news_df['polarity'].dropna(), 
                          negative_news_df[f'volume_change_pct_{hours}h'].dropna())[0] if len(negative_news_df) > 0 else 0
    neu_vol_corr = pearsonr(neutral_news_df['polarity'].dropna(), 
                          neutral_news_df[f'volume_change_pct_{hours}h'].dropna())[0] if len(neutral_news_df) > 0 else 0
    volume_correlations.append((pos_vol_corr, neg_vol_corr, neu_vol_corr))

# Create subplots for price analysis with all features
fig, axes = plt.subplots(1, len(time_periods), figsize=(6*len(time_periods), 6))
if len(time_periods) == 1:
    axes = [axes]
fig.suptitle('News Polarity vs. BTC Price Changes (Using All Features)\nUsing Pearson Correlation: ρ = cov(X,Y)/(σₓσᵧ)', fontsize=16, y=1.05)

for idx, (hours, (pos_corr, neg_corr, neu_corr)) in enumerate(zip(time_periods, price_correlations)):
    # Create scatter plots for each news type
    axes[idx].scatter(positive_news_df['polarity'], positive_news_df[f'price_change_pct_{hours}h'], 
                     alpha=0.3, color='green', label='Positive News', s=20)
    axes[idx].scatter(negative_news_df['polarity'], negative_news_df[f'price_change_pct_{hours}h'], 
                     alpha=0.3, color='red', label='Negative News', s=20)
    axes[idx].scatter(neutral_news_df['polarity'], neutral_news_df[f'price_change_pct_{hours}h'], 
                     alpha=0.3, color='blue', label='Neutral News', s=20)
    
    # Add trend lines
    z_pos = np.polyfit(positive_news_df['polarity'].dropna(), 
                      positive_news_df[f'price_change_pct_{hours}h'].dropna(), 1) if len(positive_news_df) > 0 else [0, 0]
    z_neg = np.polyfit(negative_news_df['polarity'].dropna(), 
                      negative_news_df[f'price_change_pct_{hours}h'].dropna(), 1) if len(negative_news_df) > 0 else [0, 0]
    z_neu = np.polyfit(neutral_news_df['polarity'].dropna(), 
                      neutral_news_df[f'price_change_pct_{hours}h'].dropna(), 1) if len(neutral_news_df) > 0 else [0, 0]
    
    p_pos = np.poly1d(z_pos)
    p_neg = np.poly1d(z_neg)
    p_neu = np.poly1d(z_neu)
    
    axes[idx].plot(positive_news_df['polarity'], p_pos(positive_news_df['polarity']), 
                  color='green', linewidth=2, label='Positive Trend')
    axes[idx].plot(negative_news_df['polarity'], p_neg(negative_news_df['polarity']), 
                  color='red', linewidth=2, label='Negative Trend')
    axes[idx].plot(neutral_news_df['polarity'], p_neu(neutral_news_df['polarity']), 
                  color='blue', linewidth=2, label='Neutral Trend')
    
    # Customize the plot
    axes[idx].set_title(f'{hours}h Price Change\nPearson Correlation:\nPositive: {pos_corr:.3f}\nNegative: {neg_corr:.3f}\nNeutral: {neu_corr:.3f}')
    axes[idx].set_xlabel("News Polarity")
    axes[idx].set_ylabel(f"BTC Price Change ({hours}h)")
    axes[idx].grid(True, alpha=0.3)
    axes[idx].legend()

# Adjust layout
plt.tight_layout()
plt.savefig('polarity_ana_res/price_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

# Create subplots for volume analysis with all features
fig, axes = plt.subplots(1, len(time_periods), figsize=(6*len(time_periods), 6))
if len(time_periods) == 1:
    axes = [axes]
fig.suptitle('News Polarity vs. BTC Trading Volume (Using All Features)\nUsing Pearson Correlation: ρ = cov(X,Y)/(σₓσᵧ)', fontsize=16, y=1.05)

for idx, (hours, (pos_corr, neg_corr, neu_corr)) in enumerate(zip(time_periods, volume_correlations)):
    # Create scatter plots for each news type
    axes[idx].scatter(positive_news_df['polarity'], positive_news_df[f'volume_change_pct_{hours}h'], 
                     alpha=0.3, color='green', label='Positive News', s=20)
    axes[idx].scatter(negative_news_df['polarity'], negative_news_df[f'volume_change_pct_{hours}h'], 
                     alpha=0.3, color='red', label='Negative News', s=20)
    axes[idx].scatter(neutral_news_df['polarity'], neutral_news_df[f'volume_change_pct_{hours}h'], 
                     alpha=0.3, color='blue', label='Neutral News', s=20)
    
    # Add trend lines
    z_pos = np.polyfit(positive_news_df['polarity'].dropna(), 
                      positive_news_df[f'volume_change_pct_{hours}h'].dropna(), 1) if len(positive_news_df) > 0 else [0, 0]
    z_neg = np.polyfit(negative_news_df['polarity'].dropna(), 
                      negative_news_df[f'volume_change_pct_{hours}h'].dropna(), 1) if len(negative_news_df) > 0 else [0, 0]
    z_neu = np.polyfit(neutral_news_df['polarity'].dropna(), 
                      neutral_news_df[f'volume_change_pct_{hours}h'].dropna(), 1) if len(neutral_news_df) > 0 else [0, 0]
    
    p_pos = np.poly1d(z_pos)
    p_neg = np.poly1d(z_neg)
    p_neu = np.poly1d(z_neu)
    
    axes[idx].plot(positive_news_df['polarity'], p_pos(positive_news_df['polarity']), 
                  color='green', linewidth=2, label='Positive Trend')
    axes[idx].plot(negative_news_df['polarity'], p_neg(negative_news_df['polarity']), 
                  color='red', linewidth=2, label='Negative Trend')
    axes[idx].plot(neutral_news_df['polarity'], p_neu(neutral_news_df['polarity']), 
                  color='blue', linewidth=2, label='Neutral Trend')
    
    # Customize the plot
    axes[idx].set_title(f'{hours}h Volume Change\nPearson Correlation:\nPositive: {pos_corr:.3f}\nNegative: {neg_corr:.3f}\nNeutral: {neu_corr:.3f}')
    axes[idx].set_xlabel("News Polarity")
    axes[idx].set_ylabel(f"BTC Volume Change ({hours}h)")
    axes[idx].grid(True, alpha=0.3)
    axes[idx].legend()

# Adjust layout
plt.tight_layout()
plt.savefig('polarity_ana_res/volume_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

# Print correlation statistics
print("\nCorrelation Analysis - News Polarity vs Price:")
for hours, (pos_corr, neg_corr, neu_corr) in zip(time_periods, price_correlations):
    print(f"\n{hours}h Price Change:")
    print("Positive News:")
    print(f"Correlation: {pos_corr:.3f}")
    print("\nNegative News:")
    print(f"Correlation: {neg_corr:.3f}")
    print("\nNeutral News:")
    print(f"Correlation: {neu_corr:.3f}")

print("\nCorrelation Analysis - News Polarity vs Volume:")
for hours, (pos_corr, neg_corr, neu_corr) in zip(time_periods, volume_correlations):
    print(f"\n{hours}h Volume Change:")
    print("Positive News:")
    print(f"Correlation: {pos_corr:.3f}")
    print("\nNegative News:")
    print(f"Correlation: {neg_corr:.3f}")
    print("\nNeutral News:")
    print(f"Correlation: {neu_corr:.3f}")

# Granger causality tests
print("\nGranger Causality Tests - Polarity vs Price:")
for hours in time_periods:
    print(f"\n{hours}h Price Change:")
    print("Positive News:")
    if len(positive_news_df) > 0:
        results = grangercausalitytests(positive_news_df[['polarity', f'price_change_pct_{hours}h']].dropna(), 
                                      maxlag=1, verbose=False)
        p_value = results[1][0]['ssr_ftest'][1]
        print(f"P-value: {p_value:.4f}")
        print(f"Causality: {'Yes' if p_value < 0.05 else 'No'}")
    else:
        print("No data available")
    
    print("\nNegative News:")
    if len(negative_news_df) > 0:
        results = grangercausalitytests(negative_news_df[['polarity', f'price_change_pct_{hours}h']].dropna(), 
                                      maxlag=1, verbose=False)
        p_value = results[1][0]['ssr_ftest'][1]
        print(f"P-value: {p_value:.4f}")
        print(f"Causality: {'Yes' if p_value < 0.05 else 'No'}")
    else:
        print("No data available")
    
    print("\nNeutral News:")
    if len(neutral_news_df) > 0:
        results = grangercausalitytests(neutral_news_df[['polarity', f'price_change_pct_{hours}h']].dropna(), 
                                      maxlag=1, verbose=False)
        p_value = results[1][0]['ssr_ftest'][1]
        print(f"P-value: {p_value:.4f}")
        print(f"Causality: {'Yes' if p_value < 0.05 else 'No'}")
    else:
        print("No data available")

print("\nGranger Causality Tests - Polarity vs Volume:")
for hours in time_periods:
    print(f"\n{hours}h Volume Change:")
    print("Positive News:")
    if len(positive_news_df) > 0:
        results = grangercausalitytests(positive_news_df[['polarity', f'volume_change_pct_{hours}h']].dropna(), 
                                      maxlag=1, verbose=False)
        p_value = results[1][0]['ssr_ftest'][1]
        print(f"P-value: {p_value:.4f}")
        print(f"Causality: {'Yes' if p_value < 0.05 else 'No'}")
    else:
        print("No data available")
    
    print("\nNegative News:")
    if len(negative_news_df) > 0:
        results = grangercausalitytests(negative_news_df[['polarity', f'volume_change_pct_{hours}h']].dropna(), 
                                      maxlag=1, verbose=False)
        p_value = results[1][0]['ssr_ftest'][1]
        print(f"P-value: {p_value:.4f}")
        print(f"Causality: {'Yes' if p_value < 0.05 else 'No'}")
    else:
        print("No data available")
    
    print("\nNeutral News:")
    if len(neutral_news_df) > 0:
        results = grangercausalitytests(neutral_news_df[['polarity', f'volume_change_pct_{hours}h']].dropna(), 
                                      maxlag=1, verbose=False)
        p_value = results[1][0]['ssr_ftest'][1]
        print(f"P-value: {p_value:.4f}")
        print(f"Causality: {'Yes' if p_value < 0.05 else 'No'}")
    else:
        print("No data available")