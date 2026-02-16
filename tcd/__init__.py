"""TCD (Temporal Concept Discovery) package initialization."""

from .attribution import TimeSeriesCondAttribution
from .concepts import ChannelConcept, FilterBankConcept
from .composites import get_composite, EpsilonPlusFlat, EpsilonComposite, GradientComposite
from .prototypes import TemporalPrototypeDiscovery
from .intervention import ConceptInterventionHook, compute_intervention_effect, measure_concept_importance
from .evaluation import evaluate_concept_quality, compute_faithfulness, compute_stability, compute_concept_purity
from .visualization import plot_ts_heatmap, plot_concept_relevance, plot_prototype_grid, plot_deviation_matrix

__version__ = '0.1.0'

__all__ = [
    # Attribution
    'TimeSeriesCondAttribution',
    
    # Concepts
    'ChannelConcept',
    'FilterBankConcept',
    
    # Composites
    'get_composite',
    'EpsilonPlusFlat',
    'EpsilonComposite',
    'GradientComposite',
    
    # Prototypes
    'TemporalPrototypeDiscovery',
    
    # Intervention
    'ConceptInterventionHook',
    'compute_intervention_effect',
    'measure_concept_importance',
    
    # Evaluation
    'evaluate_concept_quality',
    'compute_faithfulness',
    'compute_stability',
    'compute_concept_purity',
    
    # Visualization
    'plot_ts_heatmap',
    'plot_concept_relevance',
    'plot_prototype_grid',
    'plot_deviation_matrix',
]
