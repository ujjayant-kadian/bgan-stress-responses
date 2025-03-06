import torch
import torch.distributions as dist

class GaussianPrior:
    def __init__(self, mu=0.0, sigma=1.0):
        """
        Gaussian prior distribution for weights.
        Args:
            mu (float): Mean of the Gaussian distribution
            sigma (float): Standard deviation of the Gaussian distribution
        """
        self.mu = mu
        self.sigma = sigma
    
    def log_prob(self, w):
        """Calculate log probability of weights under the prior."""
        return dist.Normal(self.mu, self.sigma).log_prob(w).sum()

class ScaleMixturePrior:
    def __init__(self, pi=0.5, sigma1=1.0, sigma2=2.0):
        """
        Scale mixture prior (mixture of two Gaussians) for more flexible weight priors.
        Args:
            pi (float): Mixing coefficient
            sigma1 (float): Standard deviation of first Gaussian
            sigma2 (float): Standard deviation of second Gaussian
        """
        self.pi = pi
        self.sigma1 = sigma1
        self.sigma2 = sigma2
    
    def log_prob(self, w):
        """Calculate log probability of weights under the mixture prior."""
        prob1 = torch.exp(dist.Normal(0.0, self.sigma1).log_prob(w))
        prob2 = torch.exp(dist.Normal(0.0, self.sigma2).log_prob(w))
        return torch.log(self.pi * prob1 + (1 - self.pi) * prob2).sum()

class LaplacePrior:
    def __init__(self, loc=0.0, scale=1.0):
        """
        Laplace prior distribution for sparsity-inducing priors.
        Args:
            loc (float): Location parameter
            scale (float): Scale parameter
        """
        self.loc = loc
        self.scale = scale
    
    def log_prob(self, w):
        """Calculate log probability of weights under the Laplace prior."""
        return dist.Laplace(self.loc, self.scale).log_prob(w).sum()
