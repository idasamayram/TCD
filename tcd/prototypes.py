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
from typing import Dict, List, Tuple, Optional, Union


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
        n_prototypes: Union[int, Dict[int, int]] = 4,
        covariance_type: str = 'diag',
        n_init: int = 5,
        max_iter: int = 200,
        random_state: int = 0,
        balance_method: str = 'none'
    ):
        """
        Initialize prototype discovery.
        
        Args:
            n_prototypes: Number of prototypes per class. Can be an int (same for all
                         classes) or a dict mapping class_id -> n_prototypes for
                         per-class control.
            covariance_type: GMM covariance type ('full', 'tied', 'diag', 'spherical')
                           Default 'diag' is better for high-dimensional data (64+ dims)
            n_init: Number of GMM initializations (default 5 for better convergence)
            max_iter: Maximum GMM iterations (default 200 for convergence in 64-dim space)
            random_state: Random seed
            balance_method: Method for handling class imbalance ('downsample' or 'oversample')
                          'downsample' (default): Sample from majority to match minority
                          'oversample': Replicate minority samples (with jitter)
        """
        self.n_prototypes = n_prototypes
        # Build per-class lookup; resolved to a concrete int per class during fit
        if isinstance(n_prototypes, dict):
            self.n_prototypes_per_class: Dict[int, int] = {int(k): int(v) for k, v in n_prototypes.items()}
        else:
            self.n_prototypes_per_class = {}  # resolved dynamically in fit()
        self.covariance_type = covariance_type
        self.n_init = n_init
        self.max_iter = max_iter
        self.random_state = random_state
        self.balance_method = balance_method
        
        self.gmms: Dict[int, GaussianMixture] = {}
        self.class_features: Dict[int, torch.Tensor] = {}
        self.class_sample_ids: Dict[int, np.ndarray] = {}
    
    def fit(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        outputs: torch.Tensor,
        sample_ids: Optional[np.ndarray] = None,
        class_weights: Optional[torch.Tensor] = None
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
            class_weights: Optional class weights for sample weighting (shape: n_classes)
                When provided, minority class samples are given higher weight during GMM fitting
        """
        # Get predictions
        predictions = outputs.argmax(dim=1)
        
        # Get unique classes
        unique_classes = torch.unique(labels).cpu().numpy()
        
        # First pass: collect class sizes for downsampling
        class_sizes = {}
        class_data = {}
        for class_id in unique_classes:
            # Filter to correctly predicted samples of this class
            mask = (labels == class_id) & (predictions == class_id)
            class_features = features[mask]
            
            # Resolve n_prototypes for this class
            n_proto_check = int(self.n_prototypes_per_class.get(int(class_id), self.n_prototypes))
            
            if class_features.shape[0] < n_proto_check:
                print(f"Warning: Class {class_id} has only {class_features.shape[0]} "
                      f"samples, need at least {n_proto_check} for GMM")
                continue
            
            class_sizes[class_id] = class_features.shape[0]
            class_data[class_id] = {
                'features': class_features,
                'mask': mask
            }
        
        # Determine target size for balancing
        if class_weights is not None and self.balance_method == 'downsample':
            # Find minority class size (highest weight means smallest class)
            minority_size = min(class_sizes.values())
            print(f"\nBalancing method: {self.balance_method}")
            print(f"Target size: {minority_size} (minority class size)")
        
        for class_id in unique_classes:
            if class_id not in class_data:
                continue
            
            class_features = class_data[class_id]['features']
            mask = class_data[class_id]['mask']
            
            # Resolve n_prototypes for this class
            n_proto = int(self.n_prototypes_per_class.get(int(class_id), self.n_prototypes))
            
            # Store for later use
            self.class_features[class_id] = class_features
            if sample_ids is not None:
                self.class_sample_ids[class_id] = sample_ids[mask.cpu().numpy()]
            
            # Handle class balancing
            features_for_gmm = class_features.cpu().numpy()
            
            if class_weights is not None:
                if self.balance_method == 'downsample':
                    # Downsample majority class to minority class size
                    if class_sizes[class_id] > minority_size:
                        rng = np.random.RandomState(self.random_state)
                        sample_indices = rng.choice(
                            class_sizes[class_id], 
                            size=minority_size, 
                            replace=False
                        )
                        features_for_gmm = features_for_gmm[sample_indices]
                        print(f"Class {class_id}: Downsampled from {class_sizes[class_id]} to {minority_size} samples")
                    else:
                        print(f"Class {class_id}: No downsampling needed ({class_sizes[class_id]} samples)")
                
                elif self.balance_method == 'oversample' and class_weights[class_id] > 1.0:
                    # Oversample minority class with jittered duplication (backward compatibility)
                    weight = class_weights[class_id].item()
                    oversample_factor = max(1, int(round(weight)))
                    
                    if oversample_factor > 1:
                        n_samples = features_for_gmm.shape[0]
                        oversampled = [features_for_gmm]
                        rng = np.random.RandomState(self.random_state)
                        for _ in range(oversample_factor - 1):
                            # Add small noise to avoid identical samples
                            jittered = features_for_gmm + rng.normal(0, 1e-5, features_for_gmm.shape)
                            oversampled.append(jittered)
                        features_for_gmm = np.concatenate(oversampled, axis=0)
                        print(f"Class {class_id}: Oversampled by {oversample_factor}x due to class weight {weight:.2f}")
            
            # Fit GMM
            gmm = GaussianMixture(
                n_components=n_proto,
                covariance_type=self.covariance_type,
                n_init=self.n_init,
                max_iter=self.max_iter,
                random_state=self.random_state,
                init_params='kmeans',
                verbose=2
            )
            
            gmm.fit(features_for_gmm)
            self.gmms[class_id] = gmm
            
            print(f"Class {class_id}: Fitted GMM with {n_proto} prototypes "
                  f"on {features_for_gmm.shape[0]} samples (original: {class_features.shape[0]})")
    
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
        for proto_idx in range(gmm.n_components):
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
        
        gmm = self.gmms[class_id]
        features = self.class_features[class_id]
        assignments = self.assign_prototype(features, class_id)
        
        counts = np.bincount(assignments, minlength=gmm.n_components)
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
    
    @staticmethod
    def select_optimal_n_prototypes(
        features: torch.Tensor,
        min_prototypes: int = 1,
        max_prototypes: int = 10,
        covariance_type: str = 'diag',
        n_init: int = 5,
        max_iter: int = 200,
        random_state: int = 0,
        criterion: str = 'bic'
    ) -> Tuple[int, Dict[int, float]]:
        """
        Select optimal number of prototypes using BIC or AIC.
        
        This helps avoid hardcoding n_prototypes and instead finds
        the number that best fits the data.
        
        Args:
            features: Concept relevance vectors of shape (N, n_concepts)
            min_prototypes: Minimum number of prototypes to try
            max_prototypes: Maximum number of prototypes to try
            covariance_type: GMM covariance type
            n_init: Number of GMM initializations
            max_iter: Maximum GMM iterations
            random_state: Random seed
            criterion: 'bic' (Bayesian Information Criterion) or 'aic' (Akaike IC)
            
        Returns:
            optimal_n: Optimal number of prototypes
            scores: Dictionary mapping n_prototypes -> BIC/AIC score
        """
        features_np = features.cpu().numpy()
        scores = {}
        
        print(f"\nSelecting optimal n_prototypes using {criterion.upper()}...")
        print(f"Testing range: {min_prototypes} to {max_prototypes}")
        
        for n in range(min_prototypes, max_prototypes + 1):
            gmm = GaussianMixture(
                n_components=n,
                covariance_type=covariance_type,
                n_init=n_init,
                max_iter=max_iter,
                random_state=random_state,
                init_params='kmeans'
            )
            
            gmm.fit(features_np)
            
            if criterion == 'bic':
                score = gmm.bic(features_np)
            elif criterion == 'aic':
                score = gmm.aic(features_np)
            else:
                raise ValueError(f"Unknown criterion: {criterion}. Use 'bic' or 'aic'")
            
            scores[n] = score
            print(f"  n={n}: {criterion.upper()}={score:.2f}")
        
        # Lower BIC/AIC is better
        optimal_n = min(scores, key=scores.get)
        print(f"\nOptimal n_prototypes: {optimal_n} ({criterion.upper()}={scores[optimal_n]:.2f})")
        
        return optimal_n, scores


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
