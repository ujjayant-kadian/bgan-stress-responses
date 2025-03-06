import torch
import torch.nn as nn
import torch.nn.functional as F
import torchbnn as bnn

class BayesianDiscriminator(nn.Module):
    def __init__(self, hidden_dim=256, window_seconds=30, num_conditions=3):
        """
        Conditional Bayesian Discriminator for physiological stress response validation.
        Args:
            hidden_dim (int): Dimension of hidden layers
            window_seconds (int): Length of time window in seconds
            num_conditions (int): Number of condition labels (default 3: Baseline, Stress, Amusement)
        """
        super(BayesianDiscriminator, self).__init__()
        
        # Set dimensions
        self.hidden_dim = hidden_dim
        self.window_seconds = window_seconds
        self.num_conditions = num_conditions
        
        # Sampling frequencies
        self.fs_brv = 64  # Hz (BVP/BRV sampling frequency)
        self.fs_eda = 4   # Hz (EDA sampling frequency)
        
        # Calculate sequence lengths based on sampling frequencies
        self.brv_seq_length = self.window_seconds * self.fs_brv  # 1920 samples for 30s
        self.eda_seq_length = self.window_seconds * self.fs_eda  # 120 samples for 30s
        
        # Condition embedding
        self.condition_embedding = nn.Embedding(num_conditions, hidden_dim // 2)
        
        # Statistical feature dimensions
        self.hrv_stats_dim = 5  # RMSSD, SDNN, LF, HF, LF_HF_ratio
        self.eda_stats_dim = 3  # mean_EDA, median_EDA, SCR_count
        
        # Time series dimensions
        self.brv_dim = 1  # BRV signal
        self.eda_dim = 1  # EDA signal
        
        # Feature extraction layers for statistical features
        self.hrv_stats_fc = bnn.BayesLinear(prior_mu=0, prior_sigma=0.1,
                                           in_features=self.hrv_stats_dim + hidden_dim // 2,  # features + condition
                                           out_features=hidden_dim // 2)
        
        self.eda_stats_fc = bnn.BayesLinear(prior_mu=0, prior_sigma=0.1,
                                           in_features=self.eda_stats_dim + hidden_dim // 2,  # features + condition
                                           out_features=hidden_dim // 2)
        
        # LSTM for time series processing (separate for each signal)
        self.lstm_hidden_dim = hidden_dim
        
        # BRV LSTM (higher sampling rate)
        self.brv_lstm = nn.LSTM(input_size=self.brv_dim + hidden_dim // 2,  # signal + condition
                               hidden_size=self.lstm_hidden_dim // 2,
                               num_layers=2,
                               batch_first=True,
                               bidirectional=True)
        
        # EDA LSTM (lower sampling rate)
        self.eda_lstm = nn.LSTM(input_size=self.eda_dim + hidden_dim // 2,  # signal + condition
                               hidden_size=self.lstm_hidden_dim // 2,
                               num_layers=2,
                               batch_first=True,
                               bidirectional=True)
        
        # Time series feature extraction heads
        self.brv_head = nn.Sequential(
            bnn.BayesLinear(prior_mu=0, prior_sigma=0.1,
                           in_features=self.lstm_hidden_dim,
                           out_features=hidden_dim // 2),
            nn.LeakyReLU()
        )
        
        self.eda_head = nn.Sequential(
            bnn.BayesLinear(prior_mu=0, prior_sigma=0.1,
                           in_features=self.lstm_hidden_dim,
                           out_features=hidden_dim // 2),
            nn.LeakyReLU()
        )
        
        # Combined processing layers
        self.fc1 = bnn.BayesLinear(prior_mu=0, prior_sigma=0.1,
                                  in_features=2*hidden_dim,  # Combined features
                                  out_features=hidden_dim)
        
        self.fc2 = bnn.BayesLinear(prior_mu=0, prior_sigma=0.1,
                                  in_features=hidden_dim,
                                  out_features=hidden_dim // 2)
        
        # Output layer for real/fake classification
        self.out = bnn.BayesLinear(prior_mu=0, prior_sigma=0.1,
                                  in_features=hidden_dim // 2,
                                  out_features=1)
        
        # Batch normalization layers
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim // 2)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.3)
    
    def _process_statistical_features(self, hrv_stats, eda_stats, condition_embedding):
        """
        Process statistical features with condition information.
        Args:
            hrv_stats (torch.Tensor): HRV statistical features
            eda_stats (torch.Tensor): EDA statistical features
            condition_embedding (torch.Tensor): Embedded condition vector
        """
        # Expand condition embedding for concatenation
        cond_expand = condition_embedding.unsqueeze(1).expand(-1, hrv_stats.size(1), -1)
        
        # Process HRV statistics with condition
        hrv_cond = torch.cat([hrv_stats, condition_embedding], dim=1)
        h1 = F.leaky_relu(self.hrv_stats_fc(hrv_cond))
        
        # Process EDA statistics with condition
        eda_cond = torch.cat([eda_stats, condition_embedding], dim=1)
        h2 = F.leaky_relu(self.eda_stats_fc(eda_cond))
        
        return torch.cat([h1, h2], dim=1)
    
    def _process_time_series(self, brv_series, eda_series, condition_embedding):
        """
        Process time series data with condition information.
        Args:
            brv_series (torch.Tensor): BRV time series
            eda_series (torch.Tensor): EDA time series
            condition_embedding (torch.Tensor): Embedded condition vector
        """
        # Expand condition embedding for time series
        brv_cond = condition_embedding.unsqueeze(1).expand(-1, self.brv_seq_length, -1)
        eda_cond = condition_embedding.unsqueeze(1).expand(-1, self.eda_seq_length, -1)
        
        # Process BRV signal (64 Hz) with condition
        brv_input = torch.cat([brv_series, brv_cond], dim=2)
        brv_lstm_out, _ = self.brv_lstm(brv_input)
        brv_weights = F.softmax(torch.sum(brv_lstm_out, dim=2, keepdim=True), dim=1)
        brv_context = torch.sum(brv_lstm_out * brv_weights, dim=1)
        brv_features = self.brv_head(brv_context)
        
        # Process EDA signal (4 Hz) with condition
        eda_input = torch.cat([eda_series, eda_cond], dim=2)
        eda_lstm_out, _ = self.eda_lstm(eda_input)
        eda_weights = F.softmax(torch.sum(eda_lstm_out, dim=2, keepdim=True), dim=1)
        eda_context = torch.sum(eda_lstm_out * eda_weights, dim=1)
        eda_features = self.eda_head(eda_context)
        
        return torch.cat([brv_features, eda_features], dim=1)
    
    def forward(self, x, condition, num_samples=1):
        """
        Forward pass of the discriminator.
        Args:
            x (dict): Input features containing:
                     - hrv_stats: Statistical HRV features
                     - eda_stats: Statistical EDA features
                     - brv_series: BRV time series (64 Hz)
                     - eda_series: EDA time series (4 Hz)
            condition (torch.Tensor): Condition labels (1=Baseline, 2=Stress, 3=Amusement)
            num_samples (int): Number of samples for uncertainty estimation
        Returns:
            tuple: (Probability samples, Mean probability, Std probability)
        """
        samples = []
        
        # Get condition embedding
        condition_embedding = self.condition_embedding(condition)
        
        for _ in range(num_samples):
            # Process statistical features with condition
            stats_features = self._process_statistical_features(
                x['hrv_stats'], x['eda_stats'], condition_embedding
            )
            
            # Process time series with condition
            series_features = self._process_time_series(
                x['brv_series'], x['eda_series'], condition_embedding
            )
            
            # Combine all features
            combined = torch.cat([stats_features, series_features], dim=1)
            
            # Common processing
            x_processed = F.leaky_relu(self.bn1(self.fc1(combined)))
            x_processed = self.dropout(x_processed)
            x_processed = F.leaky_relu(self.bn2(self.fc2(x_processed)))
            x_processed = self.dropout(x_processed)
            
            # Output probability
            out = torch.sigmoid(self.out(x_processed))
            samples.append(out)
        
        # Stack samples and compute statistics
        samples = torch.stack(samples)
        mean = torch.mean(samples, dim=0)
        std = torch.std(samples, dim=0)
        
        return samples, mean, std
    
    def sample_predictive(self, x, condition, num_samples=10):
        """
        Sample from the predictive distribution of the discriminator.
        Args:
            x (dict): Input features
            condition (torch.Tensor): Condition labels
            num_samples (int): Number of samples to generate
        Returns:
            tuple: (Samples, Mean probability, Standard deviation)
        """
        return self.forward(x, condition, num_samples)
