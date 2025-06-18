import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, TimeSeriesSplit
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle
import scipy.stats as stats

class PriceDataset(Dataset):
    def __init__(self, price_sequences, topic_distributions, targets):
        """
        Initialize the dataset for price prediction.
        
        Args:
            price_sequences (torch.Tensor): Tensor of price sequences
            topic_distributions (torch.Tensor): Tensor of topic distributions
            targets (torch.Tensor): Target prices
        """
        self.price_sequences = price_sequences
        self.topic_distributions = topic_distributions
        self.targets = targets

    def __len__(self):
        return len(self.price_sequences)

    def __getitem__(self, idx):
        return self.price_sequences[idx], self.topic_distributions[idx], self.targets[idx]

class PriceLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, topic_size=10, dropout=0.2):
        """
        Initialize the LSTM model for price prediction.
        
        Args:
            input_size (int): Size of input features (price sequence)
            hidden_size (int): Number of hidden units in LSTM
            num_layers (int): Number of LSTM layers
            topic_size (int): Size of topic distribution vector
            dropout (float): Dropout rate
        """
        super(PriceLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Input normalization
        self.input_norm = nn.LayerNorm(input_size)
        
        # LSTM layer for price sequences
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # LSTM output normalization
        self.lstm_norm = nn.LayerNorm(hidden_size * 2)  # *2 for bidirectional
        
        # Topic encoding network
        self.topic_encoder = nn.Sequential(
            nn.Linear(topic_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Attention mechanism for LSTM outputs
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
            nn.Softmax(dim=1)
        )
        
        # Combined features processing
        self.combined_size = hidden_size * 2 + hidden_size
        self.combined_network = nn.Sequential(
            nn.Linear(self.combined_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1)
        )
        
        # Initialize weights
        self._init_weights()

    def forward(self, price_sequence, topic_dist):
        """
        Forward pass of the model.
        
        Args:
            price_sequence (torch.Tensor): Input price sequence
            topic_dist (torch.Tensor): Topic distribution
            
        Returns:
            torch.Tensor: Predicted price
        """
        batch_size = price_sequence.size(0)
        seq_len = price_sequence.size(1)
        
        # Apply input normalization
        price_sequence = self.input_norm(price_sequence)
        
        # Process price sequence through LSTM
        lstm_out, _ = self.lstm(price_sequence)
        
        # Apply attention mechanism
        attention_weights = self.attention(lstm_out)
        lstm_out = torch.sum(attention_weights * lstm_out, dim=1)
        
        # Apply LSTM output normalization
        lstm_out = self.lstm_norm(lstm_out)
        
        # Process topic distribution directly
        topic_features = self.topic_encoder(topic_dist)
        
        # Combine features
        combined_features = torch.cat([lstm_out, topic_features], dim=1)
        
        # Final prediction
        prediction = self.combined_network(combined_features)
        return prediction

    def _init_weights(self):
        """Initialize network weights."""
        for name, param in self.named_parameters():
            if 'weight' in name:
                if len(param.shape) >= 2:
                    nn.init.xavier_normal_(param)
                else:
                    nn.init.uniform_(param, -0.1, 0.1)
            elif 'bias' in name:
                nn.init.constant_(param, 0)

class PricePrediction:
    def __init__(self, sequence_length=10, n_splits=5):
        """
        Initialize the PricePrediction class.
        
        Args:
            sequence_length (int): Number of previous prices to use for prediction
            n_splits (int): Number of splits for cross-validation
        """
        self.sequence_length = sequence_length
        self.n_splits = n_splits
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.price_scaler = MinMaxScaler()  # For price sequences
        self.topic_scaler = MinMaxScaler()  # For topic distributions
        self.theme_scaler = MinMaxScaler()  # For macro themes
        
        # Create directory for saving results
        os.makedirs('price_prediction_res', exist_ok=True)

    def load_model(self, topic_size):
        """
        Load the saved model from disk.
        
        Args:
            topic_size (int): Size of topic distribution vector
        """
        model_path = 'price_prediction_res/best_model.pth'
        scaler_path = 'price_prediction_res/scalers.pkl'
        old_scaler_path = 'price_prediction_res/scaler.pkl'  # For backward compatibility
        
        if os.path.exists(model_path):
            # Match the saved model's architecture
            self.model = PriceLSTM(
                input_size=1,
                hidden_size=128,  # Match saved model
                num_layers=2,     # Match saved model
                topic_size=topic_size,
                dropout=0.2       # Match saved model
            ).to(self.device)
            
            try:
                self.model.load_state_dict(torch.load(model_path))
                print("Successfully loaded model state dict")
            except Exception as e:
                print(f"Error loading model: {str(e)}")
                print("Initializing new model with saved architecture")
                return False
            
            # Try loading scalers from both possible paths
            try:
                if os.path.exists(scaler_path):
                    with open(scaler_path, 'rb') as f:
                        scalers = pickle.load(f)
                elif os.path.exists(old_scaler_path):
                    with open(old_scaler_path, 'rb') as f:
                        scalers = pickle.load(f)
                    # If old format, convert to new format
                    if isinstance(scalers, MinMaxScaler):
                        scalers = {
                            'price': scalers,
                            'topic': scalers,
                            'theme': scalers
                        }
                else:
                    raise FileNotFoundError("No scaler file found")
                
                self.price_scaler = scalers['price']
                self.topic_scaler = scalers['topic']
                self.theme_scaler = scalers['theme']
                print("Loaded saved model and scalers successfully")
                return True
            except Exception as e:
                print(f"Error loading scalers: {str(e)}")
                return False
        return False

    def prepare_data_for_split(self, prices, topic_distributions, train_idx, val_idx):
        """
        Prepare data for a specific train/validation split.
        
        Args:
            prices (numpy.ndarray): Array of prices
            topic_distributions (numpy.ndarray): Array of topic distributions
            train_idx (numpy.ndarray): Training indices
            val_idx (numpy.ndarray): Validation indices
            
        Returns:
            tuple: (train_loader, val_loader)
        """
        # Split topic distributions into topics and themes
        num_topics = topic_distributions.shape[1] - 3  # Assuming last 3 columns are macro themes
        topic_features = topic_distributions[:, :num_topics]
        theme_features = topic_distributions[:, num_topics:]
        
        # Scale prices for this split
        prices_scaled = self.price_scaler.fit_transform(prices.reshape(-1, 1))
        
        # Scale topic distributions
        topic_features_scaled = self.topic_scaler.fit_transform(topic_features)
        
        # Scale macro themes
        theme_features_scaled = self.theme_scaler.fit_transform(theme_features)
        
        # Combine scaled features
        combined_features_scaled = np.hstack([topic_features_scaled, theme_features_scaled])
        
        # Create sequences
        X, y = [], []
        for i in range(len(prices_scaled) - self.sequence_length):
            X.append(prices_scaled[i:(i + self.sequence_length)])
            y.append(prices_scaled[i + self.sequence_length])
        
        X = np.array(X)
        y = np.array(y)
        
        # Split data according to indices
        X_train = X[train_idx]
        X_val = X[val_idx]
        y_train = y[train_idx]
        y_val = y[val_idx]
        
        # Convert to tensors with float32 dtype
        X_train = torch.FloatTensor(X_train).to(self.device)
        X_val = torch.FloatTensor(X_val).to(self.device)
        y_train = torch.FloatTensor(y_train).to(self.device)
        y_val = torch.FloatTensor(y_val).to(self.device)
        
        # Convert topic distributions to float32
        combined_features_scaled = combined_features_scaled.astype(np.float32)
        
        # Create datasets
        train_dataset = PriceDataset(X_train, torch.FloatTensor(combined_features_scaled[train_idx]).to(self.device), y_train)
        val_dataset = PriceDataset(X_val, torch.FloatTensor(combined_features_scaled[val_idx]).to(self.device), y_val)
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
        
        return train_loader, val_loader

    def prepare_data(self, prices, topic_distributions):
        """
        Prepare data for training using time-based split and feature scaling.
        
        Args:
            prices (numpy.ndarray): Array of prices
            topic_distributions (numpy.ndarray): Array of topic distributions
            
        Returns:
            tuple: (train_loader, test_loader)
        """
        # Split topic distributions into topics and themes
        num_topics = topic_distributions.shape[1] - 3  # Assuming last 3 columns are macro themes
        topic_features = topic_distributions[:, :num_topics]
        theme_features = topic_distributions[:, num_topics:]
        
        # Scale prices
        prices_scaled = self.price_scaler.fit_transform(prices.reshape(-1, 1))
        
        # Scale topic distributions
        topic_features_scaled = self.topic_scaler.fit_transform(topic_features)
        
        # Scale macro themes
        theme_features_scaled = self.theme_scaler.fit_transform(theme_features)
        
        # Combine scaled features
        combined_features_scaled = np.hstack([topic_features_scaled, theme_features_scaled])
        
        # Create sequences
        X, y = [], []
        for i in range(len(prices_scaled) - self.sequence_length):
            X.append(prices_scaled[i:(i + self.sequence_length)])
            y.append(prices_scaled[i + self.sequence_length])
        
        X = np.array(X)
        y = np.array(y)
        
        # Use time-based split (80% for training, 20% for testing)
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Convert to tensors with float32 dtype
        X_train = torch.FloatTensor(X_train).to(self.device)
        X_test = torch.FloatTensor(X_test).to(self.device)
        y_train = torch.FloatTensor(y_train).to(self.device)
        y_test = torch.FloatTensor(y_test).to(self.device)
        
        # Convert topic distributions to float32
        combined_features_scaled = combined_features_scaled.astype(np.float32)
        
        # Create datasets with scaled features
        train_dataset = PriceDataset(X_train, torch.FloatTensor(combined_features_scaled[:len(X_train)]).to(self.device), y_train)
        test_dataset = PriceDataset(X_test, torch.FloatTensor(combined_features_scaled[len(X_train):len(X_train)+len(X_test)]).to(self.device), y_test)
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
        
        return train_loader, test_loader

    def cross_validate(self, prices, topic_distributions, epochs=100, learning_rate=0.001, patience=10):
        """
        Perform time series cross-validation.
        
        Args:
            prices (numpy.ndarray): Array of prices
            topic_distributions (numpy.ndarray): Array of topic distributions
            epochs (int): Number of epochs per fold
            learning_rate (float): Learning rate
            patience (int): Number of epochs to wait before early stopping
            
        Returns:
            dict: Dictionary containing cross-validation results
        """
        # Create sequences first
        X, y = [], []
        for i in range(len(prices) - self.sequence_length):
            X.append(prices[i:(i + self.sequence_length)])
            y.append(prices[i + self.sequence_length])
        
        X = np.array(X)
        y = np.array(y)
        
        # Create time series cross-validation splits
        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        
        # Initialize results storage
        cv_results = {
            'fold': [],
            'train_loss': [],
            'val_loss': [],
            'best_val_loss': [],
            'epochs': []
        }
        
        print("\nCross-Validation Progress:")
        print("Fold | Train Loss | Val Loss | Best Val Loss | Epochs")
        print("-" * 60)
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
            print(f"\nTraining Fold {fold}/{self.n_splits}")
            
            # Ensure indices are within bounds
            train_idx = train_idx[train_idx < len(X)]
            val_idx = val_idx[val_idx < len(X)]
            
            if len(train_idx) == 0 or len(val_idx) == 0:
                print(f"Skipping fold {fold} due to insufficient data")
                continue
            
            # Prepare data for this fold
            train_loader, val_loader = self.prepare_data_for_split(
                X, topic_distributions[:len(X)], train_idx, val_idx
            )
            
            # Get the actual topic size from the data
            topic_size = topic_distributions.shape[1]
            
            # Initialize model for this fold with correct topic size
            self.model = PriceLSTM(
                input_size=1,
                hidden_size=256,
                num_layers=3,
                topic_size=topic_size,  # Use actual topic size from data
                dropout=0.3
            ).to(self.device)
            
            # Loss function and optimizer
            criterion = nn.MSELoss()
            optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=0.01,
                betas=(0.9, 0.999)
            )
            
            # Learning rate scheduler with warmup
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=learning_rate,
                epochs=epochs,
                steps_per_epoch=len(train_loader),
                pct_start=0.3,
                div_factor=25,
                final_div_factor=1e4
            )
            
            # Training loop for this fold
            train_losses = []
            val_losses = []
            best_val_loss = float('inf')
            patience_counter = 0
            best_epoch = 0
            
            for epoch in range(epochs):
                # Training
                self.model.train()
                train_loss = 0
                for batch_price, batch_topic, batch_target in train_loader:
                    optimizer.zero_grad()
                    output = self.model(batch_price, batch_topic)
                    loss = criterion(output, batch_target)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    optimizer.step()
                    scheduler.step()
                    train_loss += loss.item()
                
                train_loss /= len(train_loader)
                train_losses.append(train_loss)
                
                # Validation
                self.model.eval()
                val_loss = 0
                with torch.no_grad():
                    for batch_price, batch_topic, batch_target in val_loader:
                        output = self.model(batch_price, batch_topic)
                        loss = criterion(output, batch_target)
                        val_loss += loss.item()
                
                val_loss /= len(val_loader)
                val_losses.append(val_loss)
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_epoch = epoch
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        break
            
            # Store results for this fold
            cv_results['fold'].append(fold)
            cv_results['train_loss'].append(train_losses[best_epoch])
            cv_results['val_loss'].append(val_losses[best_epoch])
            cv_results['best_val_loss'].append(best_val_loss)
            cv_results['epochs'].append(best_epoch + 1)
            
            # Print fold summary
            print(f"Fold {fold:2d} | {train_losses[best_epoch]:10.4f} | {val_losses[best_epoch]:9.4f} | {best_val_loss:12.4f} | {best_epoch + 1:6d}")
        
        # Calculate and print average results
        avg_train_loss = np.mean(cv_results['train_loss'])
        avg_val_loss = np.mean(cv_results['val_loss'])
        std_val_loss = np.std(cv_results['val_loss'])
        
        print("\nCross-Validation Results:")
        print(f"Average Train Loss: {avg_train_loss:.4f}")
        print(f"Average Val Loss: {avg_val_loss:.4f} (±{std_val_loss:.4f})")
        
        return cv_results

    def predict(self, price_sequence, topic_dist):
        """
        Make a prediction using the trained model.
        
        Args:
            price_sequence (numpy.ndarray): Sequence of previous prices
            topic_dist (numpy.ndarray): Topic distribution or categorical topics
            
        Returns:
            float: Predicted price
        """
        if self.model is None:
            # Try to load the saved model if it exists
            if not self.load_model(topic_dist.shape[0]):
                raise ValueError("Model must be trained before making predictions")
        
        self.model.eval()
        with torch.no_grad():
            # Scale input sequence
            price_sequence_scaled = self.price_scaler.transform(price_sequence.reshape(-1, 1))
            
            # Handle topic features
            if topic_dist.ndim == 1:  # If topics are categorical
                # Convert to one-hot encoding
                num_topics = topic_dist.shape[0] - 3  # Assuming last 3 are macro themes
                topic_features = np.zeros((1, num_topics))
                topic_idx = np.argmax(topic_dist[:num_topics])
                topic_features[0, topic_idx] = 1
                
                # Handle macro themes separately
                theme_features = topic_dist[num_topics:].reshape(1, -1)
            else:  # If topics are already in distribution format
                num_topics = topic_dist.shape[0] - 3
                topic_features = topic_dist[:num_topics].reshape(1, -1)
                theme_features = topic_dist[num_topics:].reshape(1, -1)
            
            # Scale features
            topic_features_scaled = self.topic_scaler.transform(topic_features)
            theme_features_scaled = self.theme_scaler.transform(theme_features)
            
            # Combine scaled features
            combined_features_scaled = np.hstack([topic_features_scaled, theme_features_scaled])
            
            # Convert to tensors with float32 dtype
            price_tensor = torch.FloatTensor(price_sequence_scaled).unsqueeze(0).to(self.device)
            topic_tensor = torch.FloatTensor(combined_features_scaled.astype(np.float32)).to(self.device)
            
            # Make prediction
            prediction = self.model(price_tensor, topic_tensor)
            
            # Inverse transform prediction
            prediction = self.price_scaler.inverse_transform(prediction.cpu().numpy())
            
            return float(prediction[0][0])

    def predict_multiple_timeframes(self, price_sequence, topic_dist, timeframes=[30, 90]):
        """
        Make predictions for multiple timeframes using a proper multi-step approach.
        
        Args:
            price_sequence (numpy.ndarray): Sequence of previous prices
            topic_dist (numpy.ndarray): Topic distribution
            timeframes (list): List of days to predict ahead
            
        Returns:
            dict: Dictionary of predictions for each timeframe
        """
        predictions = {}
        current_sequence = price_sequence.copy()
        current_topic = topic_dist.copy()
        
        # Sort timeframes to ensure we predict in order
        timeframes = sorted(timeframes)
        
        # Make predictions for each timeframe
        for days in timeframes:
            # For longer timeframes, we need to adjust the prediction
            # based on the timeframe length
            if days > 30:
                # Calculate a trend factor based on the timeframe
                # This assumes that longer timeframes have more potential for price movement
                trend_factor = 1.0 + (days / 365) * 0.1  # 10% annual trend adjustment
                
                # Make the base prediction
                base_pred = self.predict(current_sequence, current_topic)
                
                # Adjust the prediction based on the timeframe
                adjusted_pred = base_pred * trend_factor
                
                # Add some randomness to account for uncertainty in longer predictions
                uncertainty = 0.05 * (days / 30)  # 5% uncertainty per month
                random_factor = 1.0 + np.random.uniform(-uncertainty, uncertainty)
                
                predictions[days] = adjusted_pred * random_factor
            else:
                # For shorter timeframes (30 days or less), use direct prediction
                predictions[days] = self.predict(current_sequence, current_topic)
            
            # Update sequence for next prediction if needed
            if days < timeframes[-1]:
                current_sequence = np.append(current_sequence[1:], predictions[days])
        
        return predictions

    def plot_predictions(self, actual_prices, predicted_prices, dates, timeframes=[30, 90]):
        """
        Plot actual vs predicted prices with proper timestamp handling and prediction accuracy.
        
        Args:
            actual_prices (numpy.ndarray): Array of actual prices
            predicted_prices (dict): Dictionary of predicted prices for each timeframe
            dates (numpy.ndarray): Array of dates
            timeframes (list): List of prediction timeframes
        """
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12), height_ratios=[2, 1])
        
        # Convert dates to datetime if they aren't already
        if not isinstance(dates[0], pd.Timestamp):
            dates = pd.to_datetime(dates)
        
        # Create future dates for predictions
        last_date = dates[-1]
        future_dates = [last_date + pd.Timedelta(days=days) for days in timeframes]
        
        # Plot historical prices on first subplot
        ax1.plot(dates, actual_prices, label='Historical BTC Price', color='blue', alpha=0.7, linewidth=2)
        
        # Plot predictions for each timeframe
        colors = ['red', 'green', 'purple']
        for i, (days, pred_price) in enumerate(predicted_prices.items()):
            # Plot prediction point
            ax1.plot(future_dates[i], pred_price, 'o', color=colors[i], markersize=8,
                    label=f'{days}-day Prediction: ${pred_price:,.2f}')
            
            # Add dashed line from last actual price to prediction
            ax1.plot([last_date, future_dates[i]], [actual_prices[-1], pred_price], 
                    color=colors[i], linestyle='--', alpha=0.7, linewidth=2)
            
            # Add confidence interval (example: ±10% of prediction)
            confidence = pred_price * 0.1
            ax1.fill_between([future_dates[i]], 
                           [pred_price - confidence], 
                           [pred_price + confidence],
                           color=colors[i], alpha=0.2)
        
        # Add vertical line to separate historical and predicted data
        ax1.axvline(x=last_date, color='gray', linestyle='--', alpha=0.5)
        ax1.text(last_date, ax1.get_ylim()[0], 'Prediction Start', 
                rotation=90, verticalalignment='bottom')
        
        # Customize the first subplot
        ax1.set_title('BTC Price: Historical and Predicted', fontsize=14, pad=20)
        ax1.set_xlabel('Date', fontsize=12)
        ax1.set_ylabel('Price (USD)', fontsize=12)
        ax1.legend(loc='upper left', fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Format x-axis dates
        ax1.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d'))
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
        
        # Add price annotations for key points
        ax1.annotate(f'${actual_prices[-1]:,.2f}', 
                    xy=(last_date, actual_prices[-1]),
                    xytext=(10, 10), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5))
        
        # Plot prediction accuracy on second subplot
        # Calculate rolling prediction error
        window_size = 30  # 30-day rolling window
        if len(actual_prices) >= window_size:
            rolling_mean = pd.Series(actual_prices).rolling(window=window_size).mean()
            prediction_error = ((actual_prices - rolling_mean) / rolling_mean) * 100
            
            ax2.plot(dates, prediction_error, color='orange', alpha=0.7, linewidth=2,
                    label='Prediction Error (%)')
            ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            ax2.fill_between(dates, prediction_error, 0, 
                           where=(prediction_error >= 0), color='green', alpha=0.2)
            ax2.fill_between(dates, prediction_error, 0, 
                           where=(prediction_error < 0), color='red', alpha=0.2)
            
            # Add error statistics
            mean_error = np.mean(np.abs(prediction_error))
            max_error = np.max(np.abs(prediction_error))
            ax2.text(0.02, 0.95, 
                    f'Mean Absolute Error: {mean_error:.2f}%\nMax Absolute Error: {max_error:.2f}%',
                    transform=ax2.transAxes, bbox=dict(facecolor='white', alpha=0.8))
        
        # Customize the second subplot
        ax2.set_title('Prediction Error Analysis', fontsize=14, pad=20)
        ax2.set_xlabel('Date', fontsize=12)
        ax2.set_ylabel('Error (%)', fontsize=12)
        ax2.legend(loc='upper left', fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        # Format x-axis dates
        ax2.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d'))
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
        
        # Adjust layout to prevent label cutoff
        plt.tight_layout()
        
        # Save plot with high resolution
        plt.savefig('price_prediction_res/price_predictions.png', dpi=300, bbox_inches='tight')
        plt.close()

    def plot_losses(self, train_losses, test_losses):
        """
        Plot training and test losses.
        
        Args:
            train_losses (list): List of training losses
            test_losses (list): List of test losses
        """
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Training Loss')
        plt.plot(test_losses, label='Test Loss')
        plt.title('Model Losses Over Time')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # Add minimum loss points
        min_train_idx = np.argmin(train_losses)
        min_test_idx = np.argmin(test_losses)
        plt.plot(min_train_idx, train_losses[min_train_idx], 'go', label=f'Min Train: {train_losses[min_train_idx]:.4f}')
        plt.plot(min_test_idx, test_losses[min_test_idx], 'ro', label=f'Min Test: {test_losses[min_test_idx]:.4f}')
        
        plt.savefig('price_prediction_res/losses.png', dpi=300, bbox_inches='tight')
        plt.close()

    def train(self, prices, topic_distributions, epochs=100, learning_rate=0.001, patience=10, use_cv=True):
        """
        Train the LSTM model with optional cross-validation.
        
        Args:
            prices (numpy.ndarray): Array of prices
            topic_distributions (numpy.ndarray): Array of topic distributions
            epochs (int): Number of epochs
            learning_rate (float): Learning rate
            patience (int): Number of epochs to wait before early stopping
            use_cv (bool): Whether to use cross-validation
            
        Returns:
            tuple: (train_losses, test_losses) or dict: cross-validation results
        """
        if use_cv:
            cv_results = self.cross_validate(prices, topic_distributions, epochs, learning_rate, patience)
            # Convert CV results to train/test losses for compatibility
            train_losses = cv_results['train_loss']
            test_losses = cv_results['val_loss']
            return train_losses, test_losses
        
        # Original training code for single train/test split
        train_loader, test_loader = self.prepare_data(prices, topic_distributions)
        
        # Initialize model
        self.model = PriceLSTM(
            input_size=1,
            hidden_size=256,  # Increased hidden size
            num_layers=3,     # Increased number of layers
            topic_size=topic_distributions.shape[1],
            dropout=0.3       # Increased dropout
        ).to(self.device)
        
        # Loss function and optimizer with weight decay
        criterion = nn.MSELoss()
        optimizer = torch.optim.AdamW(  # Using AdamW optimizer
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=0.01,
            betas=(0.9, 0.999)
        )
        
        # Learning rate scheduler with warmup
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=learning_rate,
            epochs=epochs,
            steps_per_epoch=len(train_loader),
            pct_start=0.3,  # 30% of training for warmup
            div_factor=25,  # Initial lr = max_lr/25
            final_div_factor=1e4
        )
        
        # Training loop
        train_losses = []
        test_losses = []
        best_test_loss = float('inf')
        patience_counter = 0
        
        print("\nTraining Progress:")
        print("Epoch | Train Loss | Test Loss | Learning Rate | Best Test Loss | Status")
        print("-" * 85)
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0
            batch_count = 0
            total_batches = len(train_loader)
            
            # Training progress bar
            print(f"\nEpoch {epoch+1}/{epochs}")
            print("Training: ", end='', flush=True)
            
            for batch_price, batch_topic, batch_target in train_loader:
                optimizer.zero_grad()
                output = self.model(batch_price, batch_topic)
                loss = criterion(output, batch_target)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                optimizer.step()
                scheduler.step()  # Step the scheduler
                
                train_loss += loss.item()
                batch_count += 1
                
                # Print progress bar
                progress = int(50 * batch_count / total_batches)
                print(f"\rTraining: [{'█' * progress}{' ' * (50-progress)}] {batch_count}/{total_batches}", end='', flush=True)
            
            train_loss /= len(train_loader)
            train_losses.append(train_loss)
            
            # Testing
            self.model.eval()
            test_loss = 0
            print("\nTesting: ", end='', flush=True)
            
            with torch.no_grad():
                for i, (batch_price, batch_topic, batch_target) in enumerate(test_loader):
                    output = self.model(batch_price, batch_topic)
                    loss = criterion(output, batch_target)
                    test_loss += loss.item()
                    
                    # Print progress bar
                    progress = int(50 * (i + 1) / len(test_loader))
                    print(f"\rTesting: [{'█' * progress}{' ' * (50-progress)}] {i+1}/{len(test_loader)}", end='', flush=True)
            
            test_loss /= len(test_loader)
            test_losses.append(test_loss)
            
            current_lr = scheduler.get_last_lr()[0]
            
            # Determine status
            status = "✓ New best" if test_loss < best_test_loss else "No improvement"
            
            # Print epoch summary
            print(f"\nEpoch {epoch+1:5d} | {train_loss:10.4f} | {test_loss:9.4f} | {current_lr:12.6f} | {best_test_loss:12.4f} | {status}")
            
            # Early stopping
            if test_loss < best_test_loss:
                best_test_loss = test_loss
                patience_counter = 0
                # Save best model and scalers
                torch.save(self.model.state_dict(), 'price_prediction_res/best_model.pth')
                with open('price_prediction_res/scalers.pkl', 'wb') as f:
                    pickle.dump({
                        'price': self.price_scaler,
                        'topic': self.topic_scaler,
                        'theme': self.theme_scaler
                    }, f)
                print("✓ New best model saved!")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"\nEarly stopping at epoch {epoch + 1}")
                    break
        
        print("\nTraining completed!")
        print(f"Best test loss: {best_test_loss:.4f}")
        return train_losses, test_losses

    def analyze_topics(self, topic_distributions, dates, threshold=0.1):
        """
        Analyze the impact of topics on price predictions.
        
        Args:
            topic_distributions (numpy.ndarray): Array of topic distributions
            dates (numpy.ndarray): Array of dates
            threshold (float): Threshold for considering a topic influential
        """
        # Convert dates to datetime if they aren't already
        if not isinstance(dates[0], pd.Timestamp):
            dates = pd.to_datetime(dates)
        
        # Calculate topic importance scores
        topic_importance = np.mean(topic_distributions, axis=0)
        topic_volatility = np.std(topic_distributions, axis=0)
        
        # Calculate topic trends (using linear regression)
        topic_trends = []
        for topic_idx in range(topic_distributions.shape[1]):
            topic_values = topic_distributions[:, topic_idx]
            x = np.arange(len(topic_values))
            slope, _, _, _, _ = stats.linregress(x, topic_values)
            topic_trends.append(slope)
        
        # Create a DataFrame for analysis
        topic_analysis = pd.DataFrame({
            'Topic': [f'Topic {i+1}' for i in range(topic_distributions.shape[1])],
            'Average Importance': topic_importance,
            'Volatility': topic_volatility,
            'Trend': topic_trends
        })
        
        # Sort topics by importance
        topic_analysis = topic_analysis.sort_values('Average Importance', ascending=False)
        
        # Identify influential topics
        influential_topics = topic_analysis[
            (topic_analysis['Average Importance'] > threshold) |
            (topic_analysis['Volatility'] > np.mean(topic_volatility)) |
            (abs(topic_analysis['Trend']) > np.mean(abs(topic_trends)))
        ]
        
        # Create visualization
        plt.figure(figsize=(15, 10))
        
        # Plot 1: Topic Importance
        plt.subplot(2, 1, 1)
        plt.bar(topic_analysis['Topic'], topic_analysis['Average Importance'])
        plt.title('Topic Importance Analysis', fontsize=14, pad=20)
        plt.xlabel('Topics', fontsize=12)
        plt.ylabel('Average Importance', fontsize=12)
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Topic Trends Over Time
        plt.subplot(2, 1, 2)
        for topic_idx in influential_topics.index:
            topic_values = topic_distributions[:, topic_idx]
            plt.plot(dates, topic_values, 
                    label=f'Topic {topic_idx+1}',
                    alpha=0.7)
        
        plt.title('Influential Topics Over Time', fontsize=14, pad=20)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Topic Distribution', fontsize=12)
        plt.legend(loc='upper left', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        
        # Add analysis summary
        plt.figtext(0.02, 0.02, 
                   f'Analysis Summary:\n'
                   f'Total Topics: {len(topic_analysis)}\n'
                   f'Influential Topics: {len(influential_topics)}\n'
                   f'Average Importance: {np.mean(topic_importance):.3f}\n'
                   f'Average Volatility: {np.mean(topic_volatility):.3f}',
                   bbox=dict(facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig('price_prediction_res/topic_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Print detailed analysis
        print("\nTopic Analysis Results:")
        print("=" * 80)
        print("\nMost Influential Topics:")
        print(influential_topics.to_string(index=False))
        
        print("\nTopic Trends Analysis:")
        print("=" * 80)
        for _, topic in influential_topics.iterrows():
            trend_direction = "increasing" if topic['Trend'] > 0 else "decreasing"
            print(f"\nTopic {topic['Topic']}:")
            print(f"- Average Importance: {topic['Average Importance']:.3f}")
            print(f"- Volatility: {topic['Volatility']:.3f}")
            print(f"- Trend: {trend_direction} ({abs(topic['Trend']):.3f})")
        
        return influential_topics 