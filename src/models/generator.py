import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import torchbnn as bnn

class BayesianGenerator(nn.Module):
    def __init__(self, latent_dim=100, hidden_dim=256, window_seconds=30, num_conditions=3):
        """
        Conditional Bayesian Generator for physiological stress response simulation.
        Args:
            latent_dim (int): Dimension of input noise vector
            hidden_dim (int): Dimension of hidden layers
            window_seconds (int): Length of time window in seconds (default 30s)
            num_conditions (int): Number of condition labels (default 3: Baseline, Stress, Amusement)
        """
        super(BayesianGenerator, self).__init__()
        
        # Set dimensions
        self.latent_dim = latent_dim
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
        
        # Bayesian fully connected layers for statistical features
        self.stats_fc1 = bnn.BayesLinear(prior_mu=0, prior_sigma=0.1,
                                        in_features=latent_dim + hidden_dim // 2,  # noise + condition
                                        out_features=hidden_dim)
        
        self.stats_fc2 = bnn.BayesLinear(prior_mu=0, prior_sigma=0.1,
                                        in_features=hidden_dim,
                                        out_features=hidden_dim)
        
        # Statistical feature heads
        self.hrv_stats_head = bnn.BayesLinear(prior_mu=0, prior_sigma=0.1,
                                             in_features=hidden_dim,
                                             out_features=self.hrv_stats_dim)
        
        self.eda_stats_head = bnn.BayesLinear(prior_mu=0, prior_sigma=0.1,
                                             in_features=hidden_dim,
                                             out_features=self.eda_stats_dim)
        
        # Bayesian LSTMs for time series generation (separate for each signal)
        self.lstm_hidden_dim = hidden_dim
        
        # BRV LSTM (higher sampling rate)
        self.brv_lstm = nn.LSTM(input_size=latent_dim + hidden_dim // 2,  # noise + condition
                               hidden_size=self.lstm_hidden_dim,
                               num_layers=2,
                               batch_first=True,
                               bidirectional=True)
        
        # EDA LSTM (lower sampling rate)
        self.eda_lstm = nn.LSTM(input_size=latent_dim + hidden_dim // 2,  # noise + condition
                               hidden_size=self.lstm_hidden_dim,
                               num_layers=2,
                               batch_first=True,
                               bidirectional=True)
        
        # Time series generation heads
        self.brv_head = nn.Sequential(
            bnn.BayesLinear(prior_mu=0, prior_sigma=0.1,
                           in_features=2*self.lstm_hidden_dim,  # bidirectional
                           out_features=hidden_dim),
            nn.LeakyReLU(),
            bnn.BayesLinear(prior_mu=0, prior_sigma=0.1,
                           in_features=hidden_dim,
                           out_features=self.brv_dim)
        )
        
        self.eda_head = nn.Sequential(
            bnn.BayesLinear(prior_mu=0, prior_sigma=0.1,
                           in_features=2*self.lstm_hidden_dim,  # bidirectional
                           out_features=hidden_dim),
            nn.LeakyReLU(),
            bnn.BayesLinear(prior_mu=0, prior_sigma=0.1,
                           in_features=hidden_dim,
                           out_features=self.eda_dim)
        )
        
        # Batch normalization layers
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.3)
    
    def _generate_statistical_features(self, z, condition_embedding):
        """
        Generate statistical features from noise and condition.
        Args:
            z (torch.Tensor): Noise vector
            condition_embedding (torch.Tensor): Embedded condition vector
        """
        # Concatenate noise and condition
        x = torch.cat([z, condition_embedding], dim=1)
        
        # Common trunk
        x = F.leaky_relu(self.bn1(self.stats_fc1(x)))
        x = self.dropout(x)
        x = F.leaky_relu(self.bn2(self.stats_fc2(x)))
        x = self.dropout(x)
        
        # Generate HRV metrics with condition-specific ranges
        hrv_features = self.hrv_stats_head(x)
        hrv_features = torch.cat([
            F.softplus(hrv_features[:, :2]),  # RMSSD, SDNN (positive)
            F.softplus(hrv_features[:, 2:4]), # LF, HF (positive)
            F.softplus(hrv_features[:, 4:])   # LF/HF ratio (positive)
        ], dim=1)
        
        # Generate EDA features
        eda_features = self.eda_stats_head(x)
        eda_features = torch.cat([
            F.softplus(eda_features[:, :2]),  # mean_EDA, median_EDA (positive)
            F.softplus(eda_features[:, 2:])   # SCR_count (positive integer-like)
        ], dim=1)
        
        return hrv_features, eda_features
    
    def _generate_time_series(self, z, condition_embedding):
        """
        Generate time series data from noise and condition.
        Args:
            z (torch.Tensor): Noise vector
            condition_embedding (torch.Tensor): Embedded condition vector
        """
        batch_size = z.size(0)
        
        # Concatenate noise and condition
        z_cond = torch.cat([z, condition_embedding], dim=1)
        
        # Generate BRV signal (64 Hz)
        z_brv = z_cond.unsqueeze(1).expand(-1, self.brv_seq_length, -1)
        brv_lstm_out, _ = self.brv_lstm(z_brv)
        brv_signal = self.brv_head(brv_lstm_out)  # Shape: [batch, 1920, 1]
        
        # Generate EDA signal (4 Hz)
        z_eda = z_cond.unsqueeze(1).expand(-1, self.eda_seq_length, -1)
        eda_lstm_out, _ = self.eda_lstm(z_eda)
        eda_signal = self.eda_head(eda_lstm_out)  # Shape: [batch, 120, 1]
        
        # Apply physiological constraints with condition-specific ranges
        brv_signal = F.softplus(brv_signal)  # Ensure positive RR intervals
        eda_signal = F.softplus(eda_signal)  # Ensure positive EDA values
        
        # Additional constraints for BRV signal based on condition
        # Stress typically increases heart rate (shorter RR intervals)
        # Baseline and amusement have more normal ranges
        brv_signal = 0.6 + 0.6 * torch.sigmoid(brv_signal)  # Maps to [0.6, 1.2] range
        
        return brv_signal, eda_signal
    
    def forward(self, z, condition, num_samples=1):
        """
        Forward pass of the generator.
        Args:
            z (torch.Tensor): Input noise vector
            condition (torch.Tensor): Condition labels (1=Baseline, 2=Stress, 3=Amusement)
            num_samples (int): Number of samples for uncertainty estimation
        Returns:
            tuple: (Generated samples, Mean, Standard deviation) where each contains:
                  - Statistical features (HRV metrics, EDA features)
                  - Time series data (BRV signal, EDA signal)
        """
        samples = []
        
        # Get condition embedding
        condition_embedding = self.condition_embedding(condition)
        
        for _ in range(num_samples):
            # Generate statistical features
            hrv_stats, eda_stats = self._generate_statistical_features(z, condition_embedding)
            
            # Generate time series
            brv_series, eda_series = self._generate_time_series(z, condition_embedding)
            
            # Combine all outputs
            sample = {
                'hrv_stats': hrv_stats,
                'eda_stats': eda_stats,
                'brv_series': brv_series,
                'eda_series': eda_series
            }
            samples.append(sample)
        
        # Compute statistics across samples
        mean = {
            'hrv_stats': torch.mean(torch.stack([s['hrv_stats'] for s in samples]), dim=0),
            'eda_stats': torch.mean(torch.stack([s['eda_stats'] for s in samples]), dim=0),
            'brv_series': torch.mean(torch.stack([s['brv_series'] for s in samples]), dim=0),
            'eda_series': torch.mean(torch.stack([s['eda_series'] for s in samples]), dim=0)
        }
        
        std = {
            'hrv_stats': torch.std(torch.stack([s['hrv_stats'] for s in samples]), dim=0),
            'eda_stats': torch.std(torch.stack([s['eda_stats'] for s in samples]), dim=0),
            'brv_series': torch.std(torch.stack([s['brv_series'] for s in samples]), dim=0),
            'eda_series': torch.std(torch.stack([s['eda_series'] for s in samples]), dim=0)
        }
        
        return samples, mean, std
    
    def sample_predictive(self, z, condition, num_samples=10):
        """
        Sample from the predictive distribution of the generator.
        Args:
            z (torch.Tensor): Input noise vector
            condition (torch.Tensor): Condition labels
            num_samples (int): Number of samples to generate
        Returns:
            tuple: (Samples, Mean, Standard deviation)
        """
        return self.forward(z, condition, num_samples)
