"""TCD (Temporal Concept Discovery) package initialization."""

from .attribution import TimeSeriesCondAttribution
from .concepts import ChannelConcept, FilterBankConcept
from .composites import get_composite, EpsilonComposite, GradientComposite, CNCValidatedComposite
from .prototypes import TemporalPrototypeDiscovery
from .intervention import (
    ConceptInterventionHook, 
    PrototypeInterventionHook,
    compute_intervention_effect, 
    measure_concept_importance,
    prototype_intervention_analysis
)
from .evaluation import (evaluate_concept_quality, compute_faithfulness, compute_stability,
                         compute_concept_purity, compute_faithfulness_prototype_level,
                         compute_incremental_faithfulness)
from .visualization import (
    plot_ts_heatmap, 
    plot_concept_relevance, 
    plot_prototype_grid, 
    plot_deviation_matrix,
    plot_prototype_samples,
    plot_prototype_gallery,
    plot_prototype_comparison,
    plot_attribution_graph,
    plot_robustness_summary
)
from .robustness import (
    noise_robustness,
    shift_robustness,
    channel_dropout_robustness,
    run_robustness_analysis,
    print_robustness_report,
    robustness_deviation_analysis
)
from .interpretation import ConceptInterpreter, extract_sample_features

__version__ = '0.1.0'

__all__ = [
    # Attribution
    'TimeSeriesCondAttribution',
    
    # Concepts
    'ChannelConcept',
    'FilterBankConcept',
    
    # Composites
    'get_composite',
    'EpsilonComposite',
    'GradientComposite',
    'CNCValidatedComposite',
    
    # Prototypes
    'TemporalPrototypeDiscovery',
    
    # Intervention
    'ConceptInterventionHook',
    'PrototypeInterventionHook',
    'compute_intervention_effect',
    'measure_concept_importance',
    'prototype_intervention_analysis',
    
    # Evaluation
    'evaluate_concept_quality',
    'compute_faithfulness',
    'compute_stability',
    'compute_concept_purity',
    'compute_faithfulness_prototype_level',
    'compute_incremental_faithfulness',
    
    # Visualization
    'plot_ts_heatmap',
    'plot_concept_relevance',
    'plot_prototype_grid',
    'plot_deviation_matrix',
    'plot_prototype_samples',
    'plot_prototype_gallery',
    'plot_prototype_comparison',
    'plot_attribution_graph',
    'plot_robustness_summary',
    
    # Robustness
    'noise_robustness',
    'shift_robustness',
    'channel_dropout_robustness',
    'run_robustness_analysis',
    'print_robustness_report',
    'robustness_deviation_analysis',
    
    # Interpretation
    'ConceptInterpreter',
    'extract_sample_features',
]
