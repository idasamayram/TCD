"""
GMM-based prototype discovery for temporal concepts.

Adapted from pcx_codes:
- experiments/global_understanding/plot_prototypes.py
- experiments/global_understanding/crp_plot_prototype.py

Follows PCX paper algorithm:
1. Collect concept relevance vectors ν^rel over training set
2. Filter to correctly predicted samples
3. Fit GMM per class: p^k = Σ λ_i^k N(μ_i^k, Σ_i^k)
4. Each Gaussian component = one prototype (prediction sub-strategy)
5. For new samples: log-likelihood L^k(ν), assign to argmax prototype
6. Deviations: Δ_i^k(ν) = ν - μ_i^k
"""

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.mixture import GaussianMixture
from typing import Dict, List, Tuple, Optional


class TemporalPrototypeDiscovery:
    """
    Discover prototypical prediction strategies via GMM clustering.
    
    Each Gaussian component in the mixture represents a distinct
    "prototype" - a characteristic pattern of concept activations
    that leads to a particular class prediction.
    
    Usage:
        discovery = TemporalPrototypeDiscovery(n_prototypes=4)
        discovery.fit(features, labels, outputs)
        prototype_samples = discovery.find_prototypes(top_k=6)
        likelihood = discovery.compute_likelihood(new_features)
    """
    
    def __init__(
        self,
        n_prototypes: int = 4,
        covariance_type: str = 'full',
        n_init: int = 1,
        max_iter: int = 10,
        random_state: int = 0
    ):
        """
        Initialize prototype discovery.
        
        Args:
            n_prototypes: Number of prototypes per class
            covariance_type: GMM covariance type ('full', 'tied', 'diag', 'spherical')
            n_init: Number of GMM initializations
            max_iter: Maximum GMM iterations
            random_state: Random seed
        """
        self.n_prototypes = n_prototypes
        self.covariance_type = covariance_type
        self.n_init = n_init
        self.max_iter = max_iter
        self.random_state = random_state
        
        self.gmms: Dict[int, GaussianMixture] = {}
        self.class_features: Dict[int, torch.Tensor] = {}
        self.class_sample_ids: Dict[int, np.ndarray] = {}
    
    def fit(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        outputs: torch.Tensor,
        sample_ids: Optional[np.ndarray] = None
    ):
        """
        Fit GMM prototypes per class.
        
        Filters to correctly predicted samples before fitting,
        as per PCX paper methodology.
        
        Args:
            features: Concept relevance vectors of shape (N, n_concepts)
            labels: True labels of shape (N,)
            outputs: Model output logits of shape (N, n_classes)
            sample_ids: Optional sample identifiers of shape (N,)
        """
        # Get predictions
        predictions = outputs.argmax(dim=1)
        
        # Get unique classes
        unique_classes = torch.unique(labels).cpu().numpy()
        
        for class_id in unique_classes:
            # Filter to correctly predicted samples of this class
            mask = (labels == class_id) & (predictions == class_id)
            class_features = features[mask]
            
            if class_features.shape[0] < self.n_prototypes:
                print(f"Warning: Class {class_id} has only {class_features.shape[0]} "
                      f"samples, need at least {self.n_prototypes} for GMM")
                continue
            
            # Store for later use
            self.class_features[class_id] = class_features
            if sample_ids is not None:
                self.class_sample_ids[class_id] = sample_ids[mask.cpu().numpy()]
            
            # Fit GMM
            gmm = GaussianMixture(
                n_components=self.n_prototypes,
                covariance_type=self.covariance_type,
                n_init=self.n_init,
                max_iter=self.max_iter,
                random_state=self.random_state,
                init_params='kmeans',
                verbose=2
            )
            
            gmm.fit(class_features.cpu().numpy())
            self.gmms[class_id] = gmm
            
            print(f"Class {class_id}: Fitted GMM with {self.n_prototypes} prototypes "
                  f"on {class_features.shape[0]} samples")
    
    def find_prototypes(
        self,
        class_id: int,
        top_k: int = 6
    ) -> Dict[int, np.ndarray]:
        """
        Find closest samples to each prototype center.
        
        From pcx_codes plot_prototypes.py:
        - Compute distances from all samples to all prototype means
        - For each prototype, return top-k closest samples
        
        Args:
            class_id: Class to find prototypes for
            top_k: Number of closest samples per prototype
            
        Returns:
            Dictionary mapping prototype_idx -> array of top-k sample indices
        """
        if class_id not in self.gmms:
            raise ValueError(f"No GMM fitted for class {class_id}")
        
        gmm = self.gmms[class_id]
        features = self.class_features[class_id].cpu().numpy()
        
        # Compute distances: (n_samples, n_prototypes)
        distances = np.linalg.norm(
            features[:, None, :] - gmm.means_[None, :, :],
            axis=2
        )
        
        # Find top-k closest samples for each prototype
        prototype_samples = {}
        for proto_idx in range(self.n_prototypes):
            # Get indices of top-k closest samples to this prototype
            closest_indices = np.argsort(distances[:, proto_idx])[:top_k]
            prototype_samples[proto_idx] = closest_indices
        
        return prototype_samples
    
    def compute_likelihood(
        self,
        features: torch.Tensor,
        class_id: int
    ) -> np.ndarray:
        """
        Compute log-likelihood L^k(ν) = log p^k(ν) for samples.
        
        Args:
            features: Concept relevance vectors of shape (N, n_concepts)
            class_id: Class ID to compute likelihood for
            
        Returns:
            Log-likelihoods of shape (N,)
        """
        if class_id not in self.gmms:
            raise ValueError(f"No GMM fitted for class {class_id}")
        
        gmm = self.gmms[class_id]
        log_likelihood = gmm.score_samples(features.cpu().numpy())
        
        return log_likelihood
    
    def compute_deviation(
        self,
        features: torch.Tensor,
        class_id: int,
        prototype_idx: int
    ) -> torch.Tensor:
        """
        Compute deviation Δ_i^k(ν) = ν - μ_i^k.
        
        From PCX paper: deviation shows how a sample differs from
        the prototype center in concept space.
        
        Args:
            features: Concept relevance vectors of shape (N, n_concepts)
            class_id: Class ID
            prototype_idx: Prototype index within class
            
        Returns:
            Deviations of shape (N, n_concepts)
        """
        if class_id not in self.gmms:
            raise ValueError(f"No GMM fitted for class {class_id}")
        
        gmm = self.gmms[class_id]
        mean = torch.from_numpy(gmm.means_[prototype_idx]).float()
        
        deviations = features - mean[None, :]
        return deviations
    
    def assign_prototype(
        self,
        features: torch.Tensor,
        class_id: int
    ) -> np.ndarray:
        """
        Assign each sample to closest prototype.
        
        Args:
            features: Concept relevance vectors of shape (N, n_concepts)
            class_id: Class ID
            
        Returns:
            Prototype assignments of shape (N,) - indices 0 to n_prototypes-1
        """
        if class_id not in self.gmms:
            raise ValueError(f"No GMM fitted for class {class_id}")
        
        gmm = self.gmms[class_id]
        assignments = gmm.predict(features.cpu().numpy())
        
        return assignments
    
    def get_prototype_coverage(
        self,
        class_id: int
    ) -> np.ndarray:
        """
        Compute percentage of samples assigned to each prototype.
        
        Args:
            class_id: Class ID
            
        Returns:
            Coverage percentages of shape (n_prototypes,)
        """
        if class_id not in self.gmms:
            raise ValueError(f"No GMM fitted for class {class_id}")
        
        features = self.class_features[class_id]
        assignments = self.assign_prototype(features, class_id)
        
        counts = np.bincount(assignments, minlength=self.n_prototypes)
        percentages = (counts / counts.sum()) * 100
        
        return percentages
    
    def get_mean_cosine_similarity(
        self,
        class_id: int
    ) -> np.ndarray:
        """
        Compute mean cosine similarity of prototypes to class center.
        
        From pcx_codes plot_prototypes.py - measures how representative
        each prototype is of the overall class distribution.
        
        Args:
            class_id: Class ID
            
        Returns:
            Cosine similarities of shape (n_prototypes,)
        """
        if class_id not in self.gmms:
            raise ValueError(f"No GMM fitted for class {class_id}")
        
        gmm = self.gmms[class_id]
        features = self.class_features[class_id]
        
        # Class center (mean over all samples)
        class_center = features.mean(dim=0)
        
        # Cosine similarity between each prototype mean and class center
        prototype_means = torch.from_numpy(gmm.means_).float()
        cosine_sims = F.cosine_similarity(
            prototype_means,
            class_center[None, :],
            dim=1
        ).numpy()
        
        return cosine_sims


if __name__ == "__main__":
    # Test prototype discovery on synthetic data
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Synthetic concept features: 100 samples, 16 concepts, 2 classes
    features_class0 = torch.randn(50, 16) + torch.tensor([1.0] * 8 + [0.0] * 8)
    features_class1 = torch.randn(50, 16) + torch.tensor([0.0] * 8 + [1.0] * 8)
    features = torch.cat([features_class0, features_class1])
    
    labels = torch.cat([torch.zeros(50), torch.ones(50)]).long()
    
    # Simulate perfect predictions
    outputs = torch.zeros(100, 2)
    outputs[:50, 0] = 2.0
    outputs[50:, 1] = 2.0
    
    # Fit prototypes
    discovery = TemporalPrototypeDiscovery(n_prototypes=2)
    discovery.fit(features, labels, outputs)
    
    # Find representative samples
    proto_samples = discovery.find_prototypes(class_id=0, top_k=3)
    print(f"Prototype samples for class 0: {proto_samples}")
    
    # Compute coverage
    coverage = discovery.get_prototype_coverage(class_id=0)
    print(f"Prototype coverage: {coverage}")
    
    # Compute cosine similarity
    cosine_sims = discovery.get_mean_cosine_similarity(class_id=0)
    print(f"Cosine similarities: {cosine_sims}")
    
    print("✓ Prototype discovery test passed!")
