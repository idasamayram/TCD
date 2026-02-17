#!/usr/bin/env python3
"""
Step 1: CRP Analysis - Collect concept features across dataset.

Adapted from pcx_codes/experiments/preprocessing/collect_features_globally.py

Computes CRP attribution for entire dataset and saves:
- Per-layer concept relevance vectors (HDF5)
- Model outputs (PyTorch)
- Sample IDs (PyTorch)
- Input-level heatmaps (HDF5)

Usage:
    python scripts/run_analysis.py --config configs/default.yaml --output results/crp_features
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import os
from pathlib import Path
import yaml
import torch
import h5py
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.cnn1d_model import CNN1D_Wide, VibrationDataset, get_layer_names
from tcd.attribution import TimeSeriesCondAttribution
from tcd.concepts import ChannelConcept
from tcd.composites import get_composite


def load_config(config_path: str) -> dict:
    """Load YAML configuration."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def run_crp_analysis(
    model,
    dataset,
    layer_names,
    composite,
    output_path,
    batch_size=4,   # CRP needs full backward graph, so keep this very small
    device='cuda' if torch.cuda.is_available() else 'cpu',
    use_class_weights=False
):
    """
    Run CRP analysis on dataset and save concept features.
    
    Args:
        model: PyTorch model
        dataset: Dataset to analyze
        layer_names: List of layer names to analyze
        composite: LRP composite
        output_path: Directory to save results
        batch_size: Batch size
        device: Device to use
        use_class_weights: If True, weight concept relevances by inverse class frequency
    """
    model.to(device)
    model.eval()
    print('device in run_crp_analysis:', device)
    
    # Get class weights from dataset if requested
    class_weights = None
    if use_class_weights and hasattr(dataset, 'weights'):
        class_weights = dataset.weights.to(device)
        print(f"Using class weights: {class_weights.cpu().numpy()}")
    
    # Initialize attribution and concept
    attributor = TimeSeriesCondAttribution(model)
    cc = ChannelConcept()
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False
    )
    
    # Storage for features per class
    class_features = {class_id: {layer: [] for layer in layer_names} for class_id in [0, 1]}
    class_outputs = {class_id: [] for class_id in [0, 1]}
    class_sample_ids = {class_id: [] for class_id in [0, 1]}
    class_heatmaps = {class_id: [] for class_id in [0, 1]}

    print(f"Running CRP analysis on {len(dataset)} samples...")
    sample_idx = 0

    # NOTE: Do NOT use torch.no_grad() here!
    # CRP/LRP requires gradient computation graph to propagate relevance.
    # model.eval() already handles dropout/batchnorm for inference.
    for data, labels in tqdm(dataloader, desc="Processing batches"):
        # Ensure data is on the correct device
        data = data.to(device)
        batch_size_actual = data.shape[0]

        # Get model outputs (for storing predictions)
        with torch.no_grad():
            outputs = model(data)

        # Prepare conditions for each sample
        conditions = [{"y": label.item()} for label in labels]

        # Compute CRP attribution with layer recording
        # CRITICAL: Explicitly move data_with_grad to the same device as model
        data_with_grad = data.clone().requires_grad_(True).to(device)
        
        try:
            attr = attributor(
                data_with_grad,
                conditions,
                composite,
                record_layer=layer_names
            )
            
            # Safety check: verify attr is the expected namedtuple
            if not hasattr(attr, 'relevances'):
                raise AttributeError(
                    f"attributor() returned unexpected type: {type(attr)}. "
                    f"Expected namedtuple with 'relevances' attribute."
                )
            
            # Safety check: verify relevances is a dict
            if not isinstance(attr.relevances, dict):
                raise TypeError(
                    f"attr.relevances has unexpected type: {type(attr.relevances)}. "
                    f"Expected dict mapping layer_name -> tensor."
                )
                
        except Exception as e:
            print(f"\n⚠ CRP attribution failed for batch starting at sample {sample_idx}")
            print(f"Error: {e}")
            print(f"Batch size: {batch_size_actual}, Device: {device}")
            print("Skipping this batch and continuing...")
            sample_idx += batch_size_actual
            continue
            
        # attr is a namedtuple: (heatmap, activations, relevances, prediction)
        heatmap = attr.heatmap

        # Extract concept relevances per layer
        eps_relevances = {}
        for layer in layer_names:
            if layer in attr.relevances:
                layer_rel = attr.relevances[layer]
                # Use ChannelConcept to get per-filter relevance
                concept_rel = cc.attribute(layer_rel, abs_norm=True)
                eps_relevances[layer] = concept_rel.detach().cpu()

        # Store outputs from the attribution result
        for i in range(batch_size_actual):
            class_id = labels[i].item()
            
            # Apply class weight if enabled
            weight = class_weights[class_id].item() if class_weights is not None else 1.0

            for layer in layer_names:
                if layer in eps_relevances:
                    # Scale relevances by class weight to give minority class more influence
                    weighted_relevance = eps_relevances[layer][i] * weight
                    class_features[class_id][layer].append(weighted_relevance)

            class_outputs[class_id].append(attr.prediction[i].detach().cpu())
            class_sample_ids[class_id].append(sample_idx + i)
            class_heatmaps[class_id].append(heatmap[i].detach().cpu().numpy())

        # After storing all results for this batch...
        sample_idx += batch_size_actual

        # Critical: free the computation graph and cached tensors
        del attr, heatmap, data_with_grad, eps_relevances
        if torch.cuda.is_available():
            torch.cuda.empty_cache()



    # Save results per class
    os.makedirs(output_path, exist_ok=True)
    
    for class_id in [0, 1]:
        print(f"\nSaving class {class_id} features...")
        
        # Save per-layer concept relevances as HDF5
        h5_path = os.path.join(output_path, f"eps_relevances_class_{class_id}.hdf5")
        with h5py.File(h5_path, 'w') as f:
            for layer in layer_names:
                if class_features[class_id][layer]:
                    features = torch.stack(class_features[class_id][layer]).numpy()
                    f.create_dataset(layer, data=features)
                    print(f"  Layer {layer}: {features.shape}")
        
        # Save outputs
        outputs_path = os.path.join(output_path, f"outputs_class_{class_id}.pt")
        torch.save(class_outputs[class_id], outputs_path)
        
        # Save sample IDs
        ids_path = os.path.join(output_path, f"sample_ids_class_{class_id}.pt")
        torch.save(class_sample_ids[class_id], ids_path)
        
        # Save heatmaps
        heatmaps_path = os.path.join(output_path, f"heatmaps_class_{class_id}.hdf5")
        with h5py.File(heatmaps_path, 'w') as f:
            heatmaps = np.array(class_heatmaps[class_id])
            f.create_dataset('heatmaps', data=heatmaps)
            print(f"  Heatmaps: {heatmaps.shape}")
        
        print(f"✓ Saved {len(class_outputs[class_id])} samples for class {class_id}")
    
    print(f"\n✓ CRP analysis complete. Results saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Run CRP analysis on dataset")
    parser.add_argument('--config', type=str, default='../configs/default.yaml',
                       help='Path to config file')
    parser.add_argument('--model', type=str, default=None,
                       help='Path to model checkpoint (overrides config)')
    parser.add_argument('--data', type=str, default=None,
                       help='Path to data directory (overrides config)')
    parser.add_argument('--output', type=str, default='results/crp_features',
                       help='Output directory for features')
    parser.add_argument('--batch-size', type=int, default=None,
                       help='Batch size (overrides config)')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Override config with command line args
    model_path = args.model or config['model']['path']
    data_path = args.data or config['data']['path']
    batch_size = args.batch_size or config['analysis']['batch_size']
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    print(f"Loading model from {model_path}...")
    model = CNN1D_Wide()
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print(f"Warning: Model checkpoint not found at {model_path}")
        print("Using randomly initialized model (for testing only)")
    
    # Get layer names
    layer_names = config['analysis']['layers']
    print(f"Analyzing layers: {layer_names}")
    
    # Load dataset
    print(f"Loading dataset from {data_path}...")
    if os.path.exists(data_path):
        dataset = VibrationDataset(data_path)
        print(f"Loaded {len(dataset)} samples")
    else:
        print(f"Warning: Data path not found at {data_path}")
        print("Cannot proceed without data. Please provide valid data path.")
        return
    
    # Get composite
    composite_name = config['analysis']['composite']
    composite = get_composite(composite_name)
    print(f"Using LRP composite: {composite_name}")
    
    # Get class weights option
    use_class_weights = config['analysis'].get('use_class_weights', False)
    if use_class_weights:
        print("Class weighting enabled for concept relevances")
    
    # Run analysis
    run_crp_analysis(
        model=model,
        dataset=dataset,
        layer_names=layer_names,
        composite=composite,
        output_path=args.output,
        batch_size=batch_size,
        device=device,
        use_class_weights=use_class_weights
    )


if __name__ == "__main__":
    main()
