"""
TCD Variant C: Learned Cluster / PCX-Style Concepts

STRUCTURAL SKELETON WITH DOCSTRINGS AND TODOs
Main research goal - direct adaptation of PCX algorithm for 1D.

Directly follows PCX paper algorithm adapted for temporal data:
1. Collect CRP concept relevance vectors ν^rel at chosen layer
2. Filter to correctly predicted samples, group by class
3. Fit GMM per class: p^k = Σ λ_i^k N(μ_i^k, Σ_i^k)
4. Each Gaussian component = one temporal concept prototype
5. New predictions: log-likelihood L^k(ν), assign to argmax prototype
6. Deviations: Δ_i^k(ν) = ν - μ_i^k
7. Intervention: suppress concept activations, observe prediction change

This is the most direct adaptation of PCX to 1D time series.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from tcd.prototypes import TemporalPrototypeDiscovery


class LearnedClusterTCD:
    """
    PCX-style learned concept prototypes for temporal data.
    
    This variant directly implements the PCX algorithm for 1D:
    - Concepts = GMM cluster centers in CRP concept space
    - Each cluster = distinct prediction sub-strategy
    - Enables prototype visualization and deviation analysis
    
    Key difference from PCX: operates on 1D temporal data instead of images.
    
    Algorithm (from PCX paper):
    1. For each sample x, compute concept relevance vector ν^rel ∈ ℝ^C
       where C = number of filters at analyzed layer
    2. Collect {ν_i^k} for all correctly predicted samples of class k
    3. Fit GMM: p^k(ν) = Σ_j λ_j^k N(ν | μ_j^k, Σ_j^k)
    4. Each component j is a "prototype" concept pattern
    5. Assign new samples to prototypes via argmax_j,k p^k_j(ν)
    6. Visualize: Find samples closest to each μ_j^k
    7. Intervene: Zero out concept channels, measure prediction change
    
    Usage (TODO - full implementation):
        # Step 1: Collect concept relevance vectors (see scripts/run_analysis.py)
        features, labels, outputs = collect_concept_features(model, dataset, layer_name)
        
        # Step 2: Fit GMM prototypes
        tcd = LearnedClusterTCD(n_prototypes=4)
        tcd.fit(features, labels, outputs)
        
        # Step 3: Analyze prototypes
        prototype_samples = tcd.find_prototypes(class_id=0)
        coverage = tcd.get_coverage(class_id=0)
        
        # Step 4: Test new samples
        new_prototype = tcd.assign_prototype(new_features, class_id=0)
        deviation = tcd.compute_deviation(new_features, class_id=0, prototype_idx=0)
        
        # Step 5: Intervention
        intervened_output = tcd.intervene(model, x, layer_name, prototype_idx=0)
    """
    
    def __init__(
        self,
        n_prototypes: int = 4,
        layer_name: str = 'conv1',
        covariance_type: str = 'full',
        n_init: int = 1,
        max_iter: int = 10
    ):
        """
        Initialize learned cluster TCD.
        
        Args:
            n_prototypes: Number of prototypes (GMM components) per class
            layer_name: Layer to extract concepts from
            covariance_type: GMM covariance ('full', 'tied', 'diag', 'spherical')
            n_init: Number of GMM initializations
            max_iter: Maximum GMM iterations
        """
        self.n_prototypes = n_prototypes
        self.layer_name = layer_name
        
        # Use TemporalPrototypeDiscovery for GMM fitting
        self.prototype_discovery = TemporalPrototypeDiscovery(
            n_prototypes=n_prototypes,
            covariance_type=covariance_type,
            n_init=n_init,
            max_iter=max_iter
        )
        
        self.fitted = False
    
    def fit(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        outputs: torch.Tensor,
        sample_ids: Optional[np.ndarray] = None
    ):
        """
        Fit GMM prototypes from concept relevance vectors.
        
        Follows PCX methodology: filter to correctly predicted samples,
        then fit GMM per class.
        
        Args:
            features: Concept relevance vectors of shape (N, n_concepts)
                These come from ChannelConcept.attribute(layer_relevance)
            labels: True labels of shape (N,)
            outputs: Model output logits of shape (N, n_classes)
            sample_ids: Optional sample identifiers
        """
        self.prototype_discovery.fit(features, labels, outputs, sample_ids)
        self.fitted = True
        
        print(f"✓ Fitted {self.n_prototypes} prototypes per class")
    
    def find_prototypes(
        self,
        class_id: int,
        top_k: int = 6
    ) -> Dict[int, np.ndarray]:
        """
        Find representative samples for each prototype.
        
        From pcx_codes plot_prototypes.py: returns samples closest
        to each Gaussian mean μ_j^k.
        
        Args:
            class_id: Class to find prototypes for
            top_k: Number of closest samples per prototype
            
        Returns:
            Dict mapping prototype_idx -> sample_indices
        """
        if not self.fitted:
            raise ValueError("Must call fit() before find_prototypes()")
        
        return self.prototype_discovery.find_prototypes(class_id, top_k)
    
    def assign_prototype(
        self,
        features: torch.Tensor,
        class_id: int
    ) -> np.ndarray:
        """
        Assign samples to prototypes.
        
        Args:
            features: Concept relevance vectors of shape (N, n_concepts)
            class_id: Class ID
            
        Returns:
            Prototype assignments of shape (N,)
        """
        if not self.fitted:
            raise ValueError("Must call fit() before assign_prototype()")
        
        return self.prototype_discovery.assign_prototype(features, class_id)
    
    def compute_deviation(
        self,
        features: torch.Tensor,
        class_id: int,
        prototype_idx: int
    ) -> torch.Tensor:
        """
        Compute deviation from prototype: Δ = ν - μ_j^k.
        
        From PCX paper: shows how sample differs from prototype center
        in concept space. Useful for understanding prediction rationale.
        
        Args:
            features: Concept relevance vectors
            class_id: Class ID
            prototype_idx: Prototype index
            
        Returns:
            Deviations of shape (N, n_concepts)
        """
        if not self.fitted:
            raise ValueError("Must call fit() before compute_deviation()")
        
        return self.prototype_discovery.compute_deviation(features, class_id, prototype_idx)
    
    def get_coverage(
        self,
        class_id: int
    ) -> np.ndarray:
        """
        Get percentage of samples assigned to each prototype.
        
        Args:
            class_id: Class ID
            
        Returns:
            Coverage percentages of shape (n_prototypes,)
        """
        if not self.fitted:
            raise ValueError("Must call fit() before get_coverage()")
        
        return self.prototype_discovery.get_prototype_coverage(class_id)
    
    def intervene(
        self,
        model: torch.nn.Module,
        x: torch.Tensor,
        prototype_idx: int,
        method: str = 'suppress'
    ) -> torch.Tensor:
        """
        Intervene on prototype concepts and measure effect.
        
        TODO: Implement full intervention pipeline:
        1. Identify which filter channels contribute most to prototype
        2. Suppress/ablate those channels during forward pass
        3. Measure change in prediction
        
        This tests the causal role of prototype concepts.
        
        Args:
            model: PyTorch model
            x: Input data of shape (B, C, T)
            prototype_idx: Prototype to intervene on
            method: 'suppress', 'ablate', or 'amplify'
            
        Returns:
            Intervened model output
        """
        # TODO: Implement
        # 1. Get prototype center μ_j^k
        # 2. Find top-k concept dimensions with highest values
        # 3. Use ConceptInterventionHook to suppress those channels
        # 4. Run forward pass and return output
        
        print("TODO: Implement LearnedClusterTCD.intervene()")
        raise NotImplementedError()
    
    def visualize_prototype(
        self,
        class_id: int,
        prototype_idx: int,
        dataset,
        n_samples: int = 6
    ):
        """
        Visualize samples belonging to a prototype.
        
        TODO: Implement visualization of:
        1. Representative samples (closest to μ_j^k)
        2. Their heatmaps
        3. Top contributing concepts
        4. Deviation patterns
        
        Args:
            class_id: Class ID
            prototype_idx: Prototype to visualize
            dataset: Dataset to load samples from
            n_samples: Number of samples to show
        """
        # TODO: Implement
        # 1. Get prototype samples
        # 2. Load their signals and heatmaps
        # 3. Plot with tcd.visualization functions
        # 4. Show concept contributions
        
        print("TODO: Implement LearnedClusterTCD.visualize_prototype()")
        raise NotImplementedError()
    
    def get_concept_labels(self) -> List[str]:
        """Get concept labels (filter indices)."""
        # Concepts = layer filters, labeled by index
        # Exact number depends on layer architecture
        return [f"Filter-{i}" for i in range(64)]  # Placeholder


if __name__ == "__main__":
    print("LearnedClusterTCD is a structural skeleton with working GMM core.")
    print("The TemporalPrototypeDiscovery backend is fully functional.")
    print("\nTODO for full implementation:")
    print("  - intervene(): Implement concept suppression during forward pass")
    print("  - visualize_prototype(): Implement prototype visualization pipeline")
    print("\nCore functionality (GMM fitting, prototype discovery) is ready to use.")
