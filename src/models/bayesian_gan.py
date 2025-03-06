import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

from .generator import BayesianGenerator
from .discriminator import BayesianDiscriminator
from .priors import GaussianPrior

class BayesianGAN:
    def __init__(self, latent_dim=100, hidden_dim=256, window_seconds=30, num_conditions=3, device='cuda', kl_scale=1.0):
        """
        Conditional Bayesian GAN for physiological stress response simulation.
        Args:
            latent_dim (int): Dimension of latent space
            hidden_dim (int): Dimension of hidden layers
            window_seconds (int): Length of time window in seconds (default 30s)
            num_conditions (int): Number of condition labels (default 3: Baseline, Stress, Amusement)
            device (str): Device to run the model on
            kl_scale (float): Scaling factor for KL divergence terms (default 1.0)
        """
        self.latent_dim = latent_dim
        self.device = device
        self.window_seconds = window_seconds
        self.num_conditions = num_conditions
        self.kl_scale = kl_scale
        
        # Initialize networks
        self.generator = BayesianGenerator(latent_dim=latent_dim,
                                         hidden_dim=hidden_dim,
                                         window_seconds=window_seconds,
                                         num_conditions=num_conditions).to(device)
        self.discriminator = BayesianDiscriminator(hidden_dim=hidden_dim,
                                                 window_seconds=window_seconds,
                                                 num_conditions=num_conditions).to(device)
        
        # Initialize optimizers
        self.g_optimizer = optim.Adam(self.generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.d_optimizer = optim.Adam(self.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        
        # Initialize prior
        self.prior = GaussianPrior(mu=0.0, sigma=1.0)
        
        # Initialize criterion
        self.criterion = nn.BCELoss()
    
    def _sample_noise(self, batch_size):
        """Generate random noise vectors."""
        return torch.randn(batch_size, self.latent_dim).to(self.device)
    
    def _get_labels(self, batch_size, real=True):
        """Generate labels for real/fake samples."""
        return torch.full((batch_size, 1), float(real), device=self.device)
    
    def _prepare_data_dict(self, data_batch):
        """
        Convert data batch to dictionary format.
        Args:
            data_batch (dict): Batch of data containing:
                             - hrv_stats: Statistical HRV features
                             - eda_stats: Statistical EDA features
                             - brv_series: BRV time series (64 Hz)
                             - eda_series: EDA time series (4 Hz)
                             - condition: Condition labels
        Returns:
            tuple: (Data dictionary with all tensors moved to device, Conditions)
        """
        return {
            'hrv_stats': data_batch['hrv_stats'].to(self.device),
            'eda_stats': data_batch['eda_stats'].to(self.device),
            'brv_series': data_batch['brv_series'].to(self.device),
            'eda_series': data_batch['eda_series'].to(self.device)
        }, data_batch['condition'].to(self.device)
    
    def train_discriminator(self, real_data, real_condition, num_samples=1):
        """
        Train the discriminator for one step.
        Args:
            real_data (dict): Batch of real physiological data
            real_condition (torch.Tensor): Condition labels for real data
            num_samples (int): Number of MC samples for uncertainty estimation
        Returns:
            float: Discriminator loss
        """
        batch_size = real_data['hrv_stats'].size(0)
        self.d_optimizer.zero_grad()
        
        # Generate fake samples with same conditions
        noise = self._sample_noise(batch_size)
        fake_samples, _, _ = self.generator(noise, real_condition, num_samples)
        
        # Initialize total loss
        d_loss = 0.0
        
        # Process each MC sample individually
        for i in range(num_samples):
            # Get current fake sample
            fake_data = {
                'hrv_stats': fake_samples[i]['hrv_stats'],
                'eda_stats': fake_samples[i]['eda_stats'],
                'brv_series': fake_samples[i]['brv_series'],
                'eda_series': fake_samples[i]['eda_series']
            }
            
            # Get discriminator predictions for both real and current fake sample
            real_pred_samples, real_pred_mean, _ = self.discriminator(real_data, real_condition, num_samples=1)
            fake_pred_samples, fake_pred_mean, _ = self.discriminator(fake_data, real_condition, num_samples=1)
            
            # Calculate loss for current MC sample
            real_labels = self._get_labels(batch_size, real=True)
            fake_labels = self._get_labels(batch_size, real=False)
            
            d_loss_real = self.criterion(real_pred_mean, real_labels)
            d_loss_fake = self.criterion(fake_pred_mean, fake_labels)
            d_loss += (d_loss_real + d_loss_fake) / num_samples
        
        # Add scaled KL divergence terms for Bayesian layers
        for layer in self.discriminator.modules():
            if hasattr(layer, 'kl_loss'):
                d_loss += self.kl_scale * layer.kl_loss() / batch_size
        
        d_loss.backward()
        self.d_optimizer.step()
        
        return d_loss.item()
    
    def train_generator(self, batch_size, condition, num_samples=1):
        """
        Train the generator for one step.
        Args:
            batch_size (int): Batch size
            condition (torch.Tensor): Condition labels to generate
            num_samples (int): Number of MC samples for uncertainty estimation
        Returns:
            float: Generator loss
        """
        self.g_optimizer.zero_grad()
        
        # Generate fake samples
        noise = self._sample_noise(batch_size)
        fake_samples, _, _ = self.generator(noise, condition, num_samples)
        
        # Initialize total loss
        g_loss = 0.0
        
        # Process each MC sample individually
        for i in range(num_samples):
            # Get current fake sample
            fake_data = {
                'hrv_stats': fake_samples[i]['hrv_stats'],
                'eda_stats': fake_samples[i]['eda_stats'],
                'brv_series': fake_samples[i]['brv_series'],
                'eda_series': fake_samples[i]['eda_series']
            }
            
            # Get discriminator predictions for current fake sample
            pred_samples, pred_mean, _ = self.discriminator(fake_data, condition, num_samples=1)
            
            # Calculate loss for current MC sample
            labels = self._get_labels(batch_size, real=True)  # We want generator to fool discriminator
            g_loss += self.criterion(pred_mean, labels) / num_samples
        
        # Add scaled KL divergence terms for Bayesian layers
        for layer in self.generator.modules():
            if hasattr(layer, 'kl_loss'):
                g_loss += self.kl_scale * layer.kl_loss() / batch_size
        
        g_loss.backward()
        self.g_optimizer.step()
        
        return g_loss.item()
    
    def train(self, dataloader, num_epochs, num_samples=1, log_interval=100):
        """
        Train the Bayesian GAN.
        Args:
            dataloader (DataLoader): DataLoader for training data
            num_epochs (int): Number of epochs to train
            num_samples (int): Number of MC samples for uncertainty estimation
            log_interval (int): How often to log training progress
        """
        for epoch in range(num_epochs):
            for batch_idx, data_batch in enumerate(dataloader):
                # Prepare data
                real_data, condition = self._prepare_data_dict(data_batch)
                batch_size = real_data['hrv_stats'].size(0)
                
                # Train discriminator
                d_loss = self.train_discriminator(real_data, condition, num_samples)
                
                # Train generator
                g_loss = self.train_generator(batch_size, condition, num_samples)
                
                if batch_idx % log_interval == 0:
                    print(f'Epoch [{epoch}/{num_epochs}] Batch [{batch_idx}/{len(dataloader)}] '
                          f'D_loss: {d_loss:.4f} G_loss: {g_loss:.4f}')
    
    def generate_samples(self, num_samples=100, condition=None, mc_samples=10):
        """
        Generate physiological stress response samples with uncertainty estimates.
        Args:
            num_samples (int): Number of samples to generate
            condition (int, optional): Specific condition to generate (1=Baseline, 2=Stress, 3=Amusement)
                                     If None, generates samples for all conditions
            mc_samples (int): Number of MC samples for uncertainty estimation
        Returns:
            tuple: (Samples, Mean predictions, Standard deviations) where each contains:
                  - Statistical features (HRV metrics, EDA features)
                  - Time series data (BRV signal at 64 Hz, EDA signal at 4 Hz)
        """
        self.generator.eval()
        with torch.no_grad():
            if condition is None:
                # Generate samples for all conditions
                all_samples, all_means, all_stds = [], [], []
                for c in range(self.num_conditions):
                    noise = self._sample_noise(num_samples)
                    condition_tensor = torch.full((num_samples,), c, dtype=torch.long).to(self.device)
                    samples, mean, std = self.generator(noise, condition_tensor, mc_samples)
                    
                    # Convert to numpy
                    samples_np = [{
                        'hrv_stats': s['hrv_stats'].cpu().numpy(),
                        'eda_stats': s['eda_stats'].cpu().numpy(),
                        'brv_series': s['brv_series'].cpu().numpy(),
                        'eda_series': s['eda_series'].cpu().numpy(),
                        'condition': c
                    } for s in samples]
                    
                    mean_np = {
                        'hrv_stats': mean['hrv_stats'].cpu().numpy(),
                        'eda_stats': mean['eda_stats'].cpu().numpy(),
                        'brv_series': mean['brv_series'].cpu().numpy(),
                        'eda_series': mean['eda_series'].cpu().numpy(),
                        'condition': c
                    }
                    
                    std_np = {
                        'hrv_stats': std['hrv_stats'].cpu().numpy(),
                        'eda_stats': std['eda_stats'].cpu().numpy(),
                        'brv_series': std['brv_series'].cpu().numpy(),
                        'eda_series': std['eda_series'].cpu().numpy(),
                        'condition': c
                    }
                    
                    all_samples.extend(samples_np)
                    all_means.append(mean_np)
                    all_stds.append(std_np)
                
                return all_samples, all_means, all_stds
            else:
                # Generate samples for specific condition
                noise = self._sample_noise(num_samples)
                condition_tensor = torch.full((num_samples,), condition, dtype=torch.long).to(self.device)
                samples, mean, std = self.generator(noise, condition_tensor, mc_samples)
                
                # Convert to numpy
                samples_np = [{
                    'hrv_stats': s['hrv_stats'].cpu().numpy(),
                    'eda_stats': s['eda_stats'].cpu().numpy(),
                    'brv_series': s['brv_series'].cpu().numpy(),
                    'eda_series': s['eda_series'].cpu().numpy(),
                    'condition': condition
                } for s in samples]
                
                mean_np = {
                    'hrv_stats': mean['hrv_stats'].cpu().numpy(),
                    'eda_stats': mean['eda_stats'].cpu().numpy(),
                    'brv_series': mean['brv_series'].cpu().numpy(),
                    'eda_series': mean['eda_series'].cpu().numpy(),
                    'condition': condition
                }
                
                std_np = {
                    'hrv_stats': std['hrv_stats'].cpu().numpy(),
                    'eda_stats': std['eda_stats'].cpu().numpy(),
                    'brv_series': std['brv_series'].cpu().numpy(),
                    'eda_series': std['eda_series'].cpu().numpy(),
                    'condition': condition
                }
                
                return samples_np, mean_np, std_np
    
    def save_model(self, path):
        """Save the model state."""
        torch.save({
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'g_optimizer_state_dict': self.g_optimizer.state_dict(),
            'd_optimizer_state_dict': self.d_optimizer.state_dict(),
        }, path)
    
    def load_model(self, path):
        """Load the model state."""
        checkpoint = torch.load(path)
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        self.g_optimizer.load_state_dict(checkpoint['g_optimizer_state_dict'])
        self.d_optimizer.load_state_dict(checkpoint['d_optimizer_state_dict'])
