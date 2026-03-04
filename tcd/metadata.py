"""
Metadata-Driven Prototype Validation for TCD.

Parses CNC sample filenames to extract machine/operation/date metadata
and cross-references with discovered GMM prototypes.
"""

import os
import re
import warnings
from pathlib import Path
from typing import Optional, Dict, List

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency


# Pattern:  M{machine}_Aug_2019_OP{operation}_{sample_id}_window_{window_id}
_FILENAME_RE = re.compile(
    r'^(?P<machine>M\d+)_[A-Za-z]+_\d{4}_(?P<operation>OP\d+)_(?P<sample_id>[^_]+)_window_(?P<window_id>\d+)'
)


class MetadataAnalyzer:
    """
    Analyze CNC sample metadata in relation to GMM prototype assignments.

    The filenames stored in ``VibrationDataset.file_paths`` follow the pattern::

        M{machine}_{month}_{year}_OP{operation}_{sample_id}_window_{window_id}_downsampled.h5

    Any filename that does not match this pattern is marked as ``"unknown"``.
    """

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    def parse_filenames(
        self,
        dataset,
        prototype_assignments: Optional[np.ndarray] = None,
    ) -> pd.DataFrame:
        """
        Parse dataset file paths and return a tidy metadata DataFrame.

        Args:
            dataset: A ``VibrationDataset`` instance with ``.file_paths`` and
                ``.labels`` attributes.
            prototype_assignments: Optional integer array of shape ``(N,)``
                mapping each sample to a prototype index.

        Returns:
            DataFrame with columns:
            ``sample_idx``, ``filename``, ``machine``, ``operation``,
            ``sample_id``, ``window_id``, ``label``, ``prototype_assignment``.
        """
        rows = []
        for idx, (fp, label) in enumerate(zip(dataset.file_paths, dataset.labels)):
            stem = Path(fp).stem
            m = _FILENAME_RE.match(stem)
            if m:
                machine = m.group('machine')
                operation = m.group('operation')
                sample_id = m.group('sample_id')
                window_id = int(m.group('window_id'))
            else:
                machine = operation = sample_id = 'unknown'
                window_id = -1

            proto = (
                int(prototype_assignments[idx])
                if prototype_assignments is not None
                else -1
            )
            rows.append({
                'sample_idx': idx,
                'filename': str(fp),
                'machine': machine,
                'operation': operation,
                'sample_id': sample_id,
                'window_id': window_id,
                'label': int(label),
                'prototype_assignment': proto,
            })

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------

    def analyze_prototype_metadata(
        self,
        metadata_df: pd.DataFrame,
        prototype_assignments: Optional[np.ndarray] = None,
    ) -> Dict:
        """
        For each prototype, compute metadata distributions and chi-squared tests.

        Args:
            metadata_df: DataFrame from :meth:`parse_filenames`.
            prototype_assignments: If provided, overrides the
                ``prototype_assignment`` column in *metadata_df*.

        Returns:
            Dict mapping prototype_id → analysis results.
        """
        if prototype_assignments is not None:
            metadata_df = metadata_df.copy()
            metadata_df['prototype_assignment'] = prototype_assignments

        df = metadata_df[metadata_df['prototype_assignment'] >= 0].copy()
        results = {}

        for proto_id, group in df.groupby('prototype_assignment'):
            proto_id = int(proto_id)
            total = len(group)

            machine_dist = (
                group['machine'].value_counts(normalize=True).mul(100).round(1).to_dict()
            )
            operation_dist = (
                group['operation'].value_counts(normalize=True).mul(100).round(1).to_dict()
            )
            label_dist = (
                group['label'].value_counts(normalize=True).mul(100).round(1).to_dict()
            )

            # Chi-squared tests (prototype assignment vs metadata)
            chi2_machine = self._chi2_test(df, 'machine', proto_id)
            chi2_operation = self._chi2_test(df, 'operation', proto_id)

            results[proto_id] = {
                'n_samples': total,
                'pct_of_total': round(100.0 * total / len(df), 1),
                'label_distribution': label_dist,
                'machine_distribution': machine_dist,
                'operation_distribution': operation_dist,
                'chi2_machine': chi2_machine,
                'chi2_operation': chi2_operation,
            }

        return results

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def plot_prototype_metadata(
        self,
        metadata_df: pd.DataFrame,
        output_dir: str,
    ) -> None:
        """
        Generate and save prototype metadata visualisation plots.

        Three files are written to *output_dir*:
        - ``prototype_machine_distribution.png``
        - ``prototype_operation_distribution.png``
        - ``prototype_operation_heatmap.png``

        Args:
            metadata_df: DataFrame from :meth:`parse_filenames`.
            output_dir: Directory where plots are saved.
        """
        os.makedirs(output_dir, exist_ok=True)
        df = metadata_df[metadata_df['prototype_assignment'] >= 0].copy()

        # --- stacked bar: prototype × machine ---
        self._stacked_bar(
            df,
            col='machine',
            title='Prototype–Machine Distribution',
            filename=os.path.join(output_dir, 'prototype_machine_distribution.png'),
        )

        # --- stacked bar: prototype × operation ---
        self._stacked_bar(
            df,
            col='operation',
            title='Prototype–Operation Distribution',
            filename=os.path.join(output_dir, 'prototype_operation_distribution.png'),
        )

        # --- heatmap: prototype × operation (sample counts) ---
        self._operation_heatmap(
            df,
            filename=os.path.join(output_dir, 'prototype_operation_heatmap.png'),
        )

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------

    def generate_report(self, metadata_df: pd.DataFrame) -> None:
        """
        Print a human-readable summary of prototype metadata.

        Args:
            metadata_df: DataFrame from :meth:`parse_filenames`.
        """
        analysis = self.analyze_prototype_metadata(metadata_df)
        label_names = {0: 'OK', 1: 'NOK'}

        print("\n" + "=" * 60)
        print("PROTOTYPE METADATA REPORT")
        print("=" * 60)

        for proto_id, info in sorted(analysis.items()):
            # Dominant class label
            label_dist = info['label_distribution']
            dominant_label_id = max(label_dist, key=label_dist.get)
            dominant_label = label_names.get(dominant_label_id, str(dominant_label_id))
            dominant_pct = label_dist[dominant_label_id]

            print(
                f"\nPrototype {proto_id} ({dominant_label}, "
                f"{info['pct_of_total']:.0f}% of samples):"
            )

            # Machines
            mach_str = ', '.join(
                f"{m} ({p:.0f}%)" for m, p in sorted(info['machine_distribution'].items())
            )
            print(f"  Machines: {mach_str}")

            # Top operations
            op_items = sorted(
                info['operation_distribution'].items(), key=lambda x: -x[1]
            )
            top_ops = op_items[:3]
            other_pct = sum(v for _, v in op_items[3:])
            op_str = ', '.join(f"{op} ({p:.0f}%)" for op, p in top_ops)
            if other_pct > 0:
                op_str += f", other ({other_pct:.0f}%)"
            print(f"  Operations: {op_str}")

            # Chi-squared hints
            chi2_op = info['chi2_operation']
            if chi2_op and chi2_op['p_value'] < 0.05:
                dom_op = op_items[0][0] if op_items else 'unknown'
                print(f"  → Likely associated with {dom_op} fault type "
                      f"(p={chi2_op['p_value']:.3f})")

        print("=" * 60 + "\n")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _chi2_test(df: pd.DataFrame, col: str, proto_id: int) -> Optional[Dict]:
        """Binary contingency table: 'this prototype' vs 'all others'."""
        try:
            tmp = df.copy()
            tmp['is_proto'] = (tmp['prototype_assignment'] == proto_id).astype(int)
            contingency = pd.crosstab(tmp[col], tmp['is_proto'])
            if contingency.shape[0] < 2 or contingency.shape[1] < 2:
                return None
            chi2, p, dof, _ = chi2_contingency(contingency.values)
            return {'chi2': float(chi2), 'p_value': float(p), 'dof': int(dof)}
        except Exception:
            return None

    @staticmethod
    def _stacked_bar(
        df: pd.DataFrame,
        col: str,
        title: str,
        filename: str,
    ) -> None:
        """Create and save a stacked bar chart: prototypes on x-axis, col as stacks."""
        pivot = (
            df.groupby(['prototype_assignment', col])
            .size()
            .unstack(fill_value=0)
        )
        pivot = pivot.apply(pd.to_numeric, errors='coerce').fillna(0)  # ensure numeric
        pivot_pct = pivot.div(pivot.sum(axis=1), axis=0) * 100
        pivot_pct = pivot_pct.astype(float)

        fig, ax = plt.subplots(figsize=(max(6, len(pivot) * 1.2), 4))
        pivot_pct.plot(kind='bar', stacked=True, ax=ax, colormap='tab10')
        ax.set_xlabel('Prototype')
        ax.set_ylabel('Percentage (%)')
        ax.set_title(title)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        plt.tight_layout()
        fig.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved {filename}")

    @staticmethod
    def _operation_heatmap(df: pd.DataFrame, filename: str) -> None:
        """Create and save a heatmap of prototype × operation sample counts."""
        pivot = (
            df.groupby(['prototype_assignment', 'operation'])
            .size()
            .unstack(fill_value=0)
        )

        fig, ax = plt.subplots(
            figsize=(max(6, pivot.shape[1] * 0.7), max(4, pivot.shape[0] * 0.7))
        )
        im = ax.imshow(pivot.values, aspect='auto', cmap='YlOrRd')
        ax.set_xticks(range(pivot.shape[1]))
        ax.set_xticklabels(pivot.columns, rotation=45, ha='right', fontsize=8)
        ax.set_yticks(range(pivot.shape[0]))
        ax.set_yticklabels([f'Proto {p}' for p in pivot.index], fontsize=8)
        ax.set_title('Sample Counts: Prototype × Operation')
        plt.colorbar(im, ax=ax, label='Count')
        plt.tight_layout()
        fig.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved {filename}")
