import pandas as pd
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor

class FeatureProcessor:
    def __init__(self, df, features, vif_threshold=5.0, correlation_threshold=0.8):
        """
        Initialize the FeatureProcessor with data and feature names.
        
        Args:
            df (pd.DataFrame): DataFrame containing the features
            features (list): List of feature names to process
            vif_threshold (float): Threshold for VIF scores
            correlation_threshold (float): Threshold for correlation coefficients
        """
        self.df = df
        self.features = features
        self.vif_threshold = vif_threshold
        self.correlation_threshold = correlation_threshold
        self.selected_features = None

    def calculate_vif(self):
        """Calculate VIF scores for all features."""
        vif_data = pd.DataFrame()
        vif_data["Feature"] = self.features
        vif_data["VIF"] = [variance_inflation_factor(self.df[self.features].values, i) 
                          for i in range(self.df[self.features].shape[1])]
        return vif_data

    def remove_collinear_features(self):
        """Remove collinear features based on correlation and VIF scores."""
        # First, check correlation matrix
        correlation_matrix = self.df[self.features].corr().abs()
        upper = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper.columns if any(upper[column] > self.correlation_threshold)]
        
        if len(to_drop) > 0:
            print(f"\nDropping highly correlated features: {to_drop}")
            self.features = [f for f in self.features if f not in to_drop]
        
        # Then, check VIF scores
        while True:
            vif_data = self.calculate_vif()
            max_vif = vif_data['VIF'].max()
            if max_vif > self.vif_threshold:
                feature_to_drop = vif_data.loc[vif_data['VIF'].idxmax(), 'Feature']
                print(f"Dropping {feature_to_drop} with VIF = {max_vif:.2f}")
                self.features.remove(feature_to_drop)
            else:
                break
        
        self.selected_features = self.features
        return self.selected_features

    def get_cleaned_data(self):
        """Return the DataFrame with only the selected features."""
        if self.selected_features is None:
            self.remove_collinear_features()
        return self.df[self.selected_features] 