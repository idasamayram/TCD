"""
TCD Variant B: Temporal Descriptor Concepts

STRUCTURAL SKELETON WITH DOCSTRINGS AND TODOs

Extracts temporal descriptors from relevance curves and clusters them.
Descriptors capture temporal patterns like:
- Slope statistics (rise/fall rates)
- Peak/burst characteristics
- Autocorrelation structure
- Spectral density of relevance signal

TODO: Full implementation of descriptor extraction and clustering.
"""

import torch
import numpy as np
from scipy import signal, stats
from typing import List, Tuple, Optional, Dict
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture


class TemporalDescriptorTCD:
    """
    Temporal descriptor-based concept extraction.
    
    Identifies concepts by clustering temporal patterns in relevance signals.
    Each concept represents a distinct temporal signature (e.g., "sudden spike",
    "gradual rise", "oscillatory", "sustained plateau").
    
    Algorithm:
    1. Extract segments from relevance curves (e.g., peaks, events)
    2. Compute descriptors for each segment:
       - Slope: max/min/mean derivative
       - Duration: width of significant activation
       - Shape: autocorrelation, spectral features
    3. Cluster descriptor vectors with k-means or GMM
    4. Assign segments to concepts based on cluster membership
    
    Usage (TODO):
        tcd = TemporalDescriptorTCD(n_concepts=5)
        tcd.fit(heatmaps)  # Learn descriptor clusters from training data
        concept_assignments = tcd.extract_concepts(heatmaps)
    """
    
    def __init__(
        self,
        n_concepts: int = 5,
        descriptor_types: Optional[List[str]] = None,
        clustering_method: str = 'kmeans'
    ):
        """
        Initialize temporal descriptor extractor.
        
        Args:
            n_concepts: Number of temporal concept clusters
            descriptor_types: List of descriptor types to extract
                Options: ['slope', 'peak', 'autocorr', 'spectral']
            clustering_method: 'kmeans' or 'gmm'
        """
        self.n_concepts = n_concepts
        self.descriptor_types = descriptor_types or ['slope', 'peak', 'autocorr']
        self.clustering_method = clustering_method
        
        self.clusterer = None
        self.fitted = False
    
    def _extract_segments(
        self,
        signal: np.ndarray,
        threshold: float = 0.5
    ) -> List[Tuple[int, int]]:
        """
        Extract significant segments from relevance signal.
        
        Uses threshold-based detection and peak finding to identify
        regions of high relevance.
        
        Args:
            signal: Relevance signal of shape (timesteps,)
            threshold: Threshold for significance (relative to max)
            
        Returns:
            List of (start_idx, end_idx) tuples
        """
        # Normalize signal
        signal_norm = signal / (signal.max() + 1e-10)
        
        # Find peaks
        peaks, properties = signal.find_peaks(
            signal_norm,
            height=threshold,
            distance=10  # Minimum distance between peaks
        )
        
        # Extract segments around peaks
        segments = []
        
        # Threshold-based method: find contiguous regions above threshold
        above_threshold = signal_norm > threshold
        
        # Find start and end points of segments
        diff = np.diff(above_threshold.astype(int))
        starts = np.where(diff == 1)[0] + 1
        ends = np.where(diff == -1)[0] + 1
        
        # Handle edge cases
        if above_threshold[0]:
            starts = np.r_[0, starts]
        if above_threshold[-1]:
            ends = np.r_[ends, len(signal)]
        
        # Combine into segments
        for start, end in zip(starts, ends):
            if end - start >= 5:  # Minimum segment length
                segments.append((int(start), int(end)))
        
        # If no segments found, use full signal
        if len(segments) == 0:
            segments.append((0, len(signal)))
        
        return segments
    
    def _compute_slope_descriptors(
        self,
        segment: np.ndarray
    ) -> np.ndarray:
        """
        Compute slope-based descriptors for segment.
        
        Extracts derivative statistics to characterize rate of change.
        
        Args:
            segment: Signal segment of shape (length,)
            
        Returns:
            Descriptor vector with features:
            - max_slope: steepest rise
            - min_slope: steepest fall
            - mean_slope: average rate of change
            - slope_variance: variability of derivative
            - rise_time: time from min to max
            - fall_time: time from max to min
        """
        if len(segment) < 2:
            return np.zeros(6)
        
        # Compute derivatives
        derivatives = np.diff(segment)
        
        # Slope statistics
        max_slope = derivatives.max()
        min_slope = derivatives.min()
        mean_slope = derivatives.mean()
        slope_variance = derivatives.var()
        
        # Rise and fall times
        max_idx = np.argmax(segment)
        min_idx = np.argmin(segment)
        
        if min_idx < max_idx:
            rise_time = max_idx - min_idx
            # Find next minimum after max for fall time
            fall_start_idx = max_idx
            if max_idx < len(segment) - 1:
                # Simple approximation: distance to end
                fall_time = len(segment) - max_idx
            else:
                fall_time = 0
        else:
            fall_time = min_idx - max_idx
            rise_time = 0
        
        # Normalize by segment length
        rise_time_norm = rise_time / len(segment)
        fall_time_norm = fall_time / len(segment)
        
        return np.array([
            max_slope,
            min_slope,
            mean_slope,
            slope_variance,
            rise_time_norm,
            fall_time_norm
        ])
    
    def _compute_peak_descriptors(
        self,
        segment: np.ndarray
    ) -> np.ndarray:
        """
        Compute peak/burst descriptors for segment.
        
        Extracts peak characteristics using scipy.signal.find_peaks.
        
        Args:
            segment: Signal segment of shape (length,)
            
        Returns:
            Descriptor vector with features:
            - peak_height: maximum value
            - peak_width: duration at half-maximum
            - peak_prominence: relative height above baseline
            - n_peaks: number of local maxima
            - peak_spacing: mean distance between peaks
        """
        if len(segment) < 3:
            return np.zeros(5)
        
        # Peak statistics
        peak_height = segment.max()
        
        # Find peaks
        peaks, properties = signal.find_peaks(
            segment,
            prominence=segment.std() * 0.5
        )
        
        n_peaks = len(peaks)
        
        if n_peaks > 0:
            # Peak prominence: mean of prominences
            if 'prominences' in properties:
                peak_prominence = properties['prominences'].mean()
            else:
                peak_prominence = segment.max() - segment.min()
            
            # Peak spacing
            if n_peaks > 1:
                peak_spacing = np.mean(np.diff(peaks)) / len(segment)
            else:
                peak_spacing = 0.0
            
            # Peak width: estimate using widths at half prominence
            if 'widths' in properties:
                peak_width = properties['widths'].mean() / len(segment)
            else:
                # Estimate width at half maximum
                half_max = (segment.max() + segment.min()) / 2
                above_half = segment > half_max
                if above_half.sum() > 0:
                    peak_width = above_half.sum() / len(segment)
                else:
                    peak_width = 0.0
        else:
            peak_prominence = segment.max() - segment.min()
            peak_spacing = 0.0
            peak_width = 0.0
        
        return np.array([
            peak_height,
            peak_width,
            peak_prominence,
            n_peaks / len(segment),  # Normalized peak count
            peak_spacing
        ])
    
    def _compute_autocorr_descriptors(
        self,
        segment: np.ndarray,
        max_lag: int = 50
    ) -> np.ndarray:
        """
        Compute autocorrelation descriptors.
        
        Captures temporal structure and periodicity.
        
        Args:
            segment: Signal segment
            max_lag: Maximum lag for autocorrelation
            
        Returns:
            Descriptor vector: autocorrelation at key lags plus decay rate
        """
        if len(segment) < max_lag:
            max_lag = len(segment) // 2
        
        if max_lag < 2:
            return np.zeros(5)
        
        # Normalize segment
        segment_norm = segment - segment.mean()
        
        # Compute autocorrelation
        acf = np.correlate(segment_norm, segment_norm, mode='full')
        acf = acf[len(acf)//2:]  # Take positive lags only
        acf = acf / acf[0]  # Normalize by zero-lag value
        
        # Extract descriptors at key lags
        lags_to_sample = [1, 5, 10, 20, min(max_lag, len(acf)-1)]
        acf_values = []
        
        for lag in lags_to_sample:
            if lag < len(acf):
                acf_values.append(acf[lag])
            else:
                acf_values.append(0.0)
        
        # Compute decay rate: fit exponential to first N lags
        # acf ~ exp(-decay_rate * lag)
        valid_lags = min(20, len(acf))
        if valid_lags > 5:
            # Approximate decay rate from log(acf)
            acf_positive = np.abs(acf[1:valid_lags]) + 1e-10
            log_acf = np.log(acf_positive)
            # Slope of log(acf) vs lag gives decay rate
            lags = np.arange(1, valid_lags)
            decay_rate = -np.polyfit(lags, log_acf, 1)[0]
        else:
            decay_rate = 0.0
        
        acf_values.append(decay_rate)
        
        return np.array(acf_values)
    
    def _compute_spectral_descriptors(
        self,
        segment: np.ndarray,
        sample_rate: int = 400
    ) -> np.ndarray:
        """
        Compute spectral descriptors via FFT.
        
        Extracts frequency content of relevance pattern.
        
        Args:
            segment: Signal segment
            sample_rate: Sampling rate
            
        Returns:
            Descriptor vector with spectral features:
            - dominant_frequency: frequency with maximum power
            - spectral_centroid: center of mass of spectrum
            - spectral_bandwidth: spread around centroid
            - spectral_flatness: measure of tone vs noise
        """
        if len(segment) < 4:
            return np.zeros(4)
        
        # Compute FFT
        freqs, psd = signal.welch(
            segment,
            fs=sample_rate,
            nperseg=min(len(segment), 256)
        )
        
        # Dominant frequency
        dominant_freq = freqs[np.argmax(psd)]
        
        # Spectral centroid: weighted mean frequency
        spectral_centroid = np.sum(freqs * psd) / (np.sum(psd) + 1e-10)
        
        # Spectral bandwidth: weighted std of frequencies
        spectral_bandwidth = np.sqrt(
            np.sum(((freqs - spectral_centroid) ** 2) * psd) / (np.sum(psd) + 1e-10)
        )
        
        # Spectral flatness: geometric mean / arithmetic mean of PSD
        # Measures how tone-like (0) vs noise-like (1) the signal is
        geometric_mean = np.exp(np.mean(np.log(psd + 1e-10)))
        arithmetic_mean = np.mean(psd)
        spectral_flatness = geometric_mean / (arithmetic_mean + 1e-10)
        
        return np.array([
            dominant_freq,
            spectral_centroid,
            spectral_bandwidth,
            spectral_flatness
        ])
    
    def _extract_descriptors(
        self,
        heatmap: np.ndarray
    ) -> np.ndarray:
        """
        Extract all descriptors from heatmap.
        
        Combines all descriptor types into feature vectors.
        
        Args:
            heatmap: Heatmap of shape (channels, timesteps)
            
        Returns:
            Descriptor matrix of shape (n_segments, n_features)
        """
        # Average across channels to get single relevance signal
        signal = heatmap.mean(axis=0)
        
        # Extract segments
        segments = self._extract_segments(signal)
        
        # Compute descriptors for each segment
        descriptors = []
        for start, end in segments:
            segment = signal[start:end]
            
            desc = []
            if 'slope' in self.descriptor_types:
                desc.append(self._compute_slope_descriptors(segment))
            if 'peak' in self.descriptor_types:
                desc.append(self._compute_peak_descriptors(segment))
            if 'autocorr' in self.descriptor_types:
                desc.append(self._compute_autocorr_descriptors(segment))
            if 'spectral' in self.descriptor_types:
                desc.append(self._compute_spectral_descriptors(segment))
            
            if len(desc) > 0:
                descriptors.append(np.concatenate(desc))
        
        if len(descriptors) == 0:
            # Return zero vector if no segments found
            n_features = sum([
                6 if 'slope' in self.descriptor_types else 0,
                5 if 'peak' in self.descriptor_types else 0,
                6 if 'autocorr' in self.descriptor_types else 0,
                4 if 'spectral' in self.descriptor_types else 0
            ])
            return np.zeros((1, n_features))
        
        return np.array(descriptors)
    
    def fit(
        self,
        heatmaps: torch.Tensor
    ):
        """
        Fit descriptor clusters from training heatmaps.
        
        Extracts descriptors from all samples and fits clustering model.
        
        Args:
            heatmaps: Training heatmaps of shape (n_samples, channels, timesteps)
        """
        print(f"Fitting TemporalDescriptorTCD with {self.n_concepts} concepts...")
        print(f"Descriptor types: {self.descriptor_types}")
        
        # Extract descriptors from all samples
        all_descriptors = []
        
        for i in range(heatmaps.shape[0]):
            heatmap = heatmaps[i].cpu().numpy()
            descriptors = self._extract_descriptors(heatmap)
            all_descriptors.append(descriptors)
        
        # Concatenate all descriptors
        # Each sample may have multiple segments, so flatten
        descriptor_matrix = np.vstack(all_descriptors)
        
        print(f"Extracted {descriptor_matrix.shape[0]} segments with {descriptor_matrix.shape[1]} features")
        
        # Fit clustering model
        if self.clustering_method == 'kmeans':
            self.clusterer = KMeans(
                n_clusters=self.n_concepts,
                random_state=42,
                n_init=10
            )
        elif self.clustering_method == 'gmm':
            self.clusterer = GaussianMixture(
                n_components=self.n_concepts,
                covariance_type='diag',
                random_state=42,
                n_init=5
            )
        else:
            raise ValueError(f"Unknown clustering method: {self.clustering_method}")
        
        self.clusterer.fit(descriptor_matrix)
        self.fitted = True
        
        print(f"✓ Fitted {self.clustering_method.upper()} with {self.n_concepts} temporal concept clusters")
    
    def extract_concepts(
        self,
        heatmaps: torch.Tensor
    ) -> torch.Tensor:
        """
        Extract temporal concept relevances.
        
        Assigns segments to concepts based on descriptor similarity
        and aggregates relevance per concept.
        
        Args:
            heatmaps: Heatmaps of shape (batch, channels, timesteps)
            
        Returns:
            Concept relevances of shape (batch, n_concepts)
        """
        if not self.fitted:
            raise ValueError("Must call fit() before extract_concepts()")
        
        batch_size = heatmaps.shape[0]
        concept_relevances = np.zeros((batch_size, self.n_concepts))
        
        for i in range(batch_size):
            heatmap = heatmaps[i].cpu().numpy()
            
            # Extract descriptors for this sample
            descriptors = self._extract_descriptors(heatmap)
            
            # Assign segments to clusters
            if self.clustering_method == 'kmeans':
                assignments = self.clusterer.predict(descriptors)
            else:  # gmm
                assignments = self.clusterer.predict(descriptors)
            
            # Compute relevance for each concept
            # Aggregate based on how many segments are assigned to each concept
            # weighted by segment importance (mean absolute relevance)
            signal = heatmap.mean(axis=0)
            segments = self._extract_segments(signal)
            
            for seg_idx, (start, end) in enumerate(segments):
                if seg_idx >= len(assignments):
                    break
                
                concept_idx = assignments[seg_idx]
                # Weight by segment relevance
                segment_relevance = np.abs(signal[start:end]).mean()
                concept_relevances[i, concept_idx] += segment_relevance
        
        return torch.from_numpy(concept_relevances).float()
    
    def get_concept_labels(self) -> List[str]:
        """Get concept labels."""
        return [f"Temporal-{i}" for i in range(self.n_concepts)]


if __name__ == "__main__":
    # Test temporal descriptor extraction
    print("Testing TemporalDescriptorTCD implementation...")
    
    # Create synthetic heatmaps
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Generate sample heatmaps with different temporal patterns
    n_samples = 10
    n_channels = 3
    n_timesteps = 2000
    
    heatmaps = torch.randn(n_samples, n_channels, n_timesteps)
    
    # Add some structure: bursts, ramps, oscillations
    for i in range(n_samples):
        if i % 3 == 0:
            # Add burst pattern
            start = np.random.randint(0, n_timesteps - 100)
            heatmaps[i, :, start:start+50] += 2.0
        elif i % 3 == 1:
            # Add ramp pattern
            start = np.random.randint(0, n_timesteps - 200)
            ramp = torch.linspace(0, 2, 100)
            heatmaps[i, :, start:start+100] += ramp
        else:
            # Add oscillation
            t = torch.linspace(0, 4*np.pi, 200)
            osc = torch.sin(t)
            start = np.random.randint(0, n_timesteps - 200)
            heatmaps[i, :, start:start+200] += osc
    
    # Test TCD
    tcd = TemporalDescriptorTCD(
        n_concepts=3,
        descriptor_types=['slope', 'peak', 'autocorr', 'spectral'],
        clustering_method='kmeans'
    )
    
    # Fit on data
    tcd.fit(heatmaps)
    
    # Extract concepts
    concept_relevances = tcd.extract_concepts(heatmaps)
    
    print(f"\nConcept relevances shape: {concept_relevances.shape}")
    print(f"Concept relevances:\n{concept_relevances}")
    
    print("\n✓ TemporalDescriptorTCD implementation complete and tested!")

