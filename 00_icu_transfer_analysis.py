#!/usr/bin/env python3
"""
CLIF ICU Transfer Analysis Script

Identifies hospitalizations coded as outside-hospital (OSH) admissions that are
actually transfers linked to a prior encounter at this site (discharged <=6 hours
before the OSH admission), then compares ICU LOS, hospital LOS, ICU/hospital type,
mortality, and discharge category before vs. after the transfer.

This script processes CLIF data to:
1. Identify the in-system OSH cohort and its direct-to-ICU subset
2. Compare ICU length of stay (first stay and cumulative) before vs. after transfer
3. Compare hospital length of stay before vs. after transfer
4. Compare ICU type and hospital type before vs. after transfer
5. Compare mortality rate across cohorts
6. Compare discharge category before vs. after transfer
"""

import sys
import os
import traceback
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from clifpy import ClifOrchestrator

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config.load_config import load_config


# ============================================================================
# DATAVIZ CONSTANTS
# ============================================================================

# validated categorical palette (dataviz skill reference palette, fixed hue order)
CATEGORICAL_HUES = [
    '#2a78d6', '#eb6834', '#1baf7a', '#eda100',
    '#e87ba4', '#008300', '#4a3aa7', '#e34948',
]
OTHER_HUE = '#898781'  # muted ink - fallback once distinct categories exceed the 8-slot budget


# ============================================================================
# ARGUMENT PARSING
# ============================================================================

def parse_arguments():
    """Parse command-line arguments for the ICU transfer analysis"""
    parser = argparse.ArgumentParser(
        description='CLIF ICU Transfer Analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default settings
  python 02_icu_transfer_analysis.py

  # Specify custom config and output paths
  python 02_icu_transfer_analysis.py --config-path /path/to/config.json --output-dir /path/to/output
        """
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        metavar='PATH',
        help='Output directory for results (default: output_directory from config)'
    )

    parser.add_argument(
        '--config-path',
        type=str,
        default='config/config.json',
        metavar='PATH',
        help='Path to CLIF configuration JSON file (default: config/config.json)'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output with detailed progress messages'
    )

    parser.add_argument(
        '--log-file',
        type=str,
        default=None,
        metavar='PATH',
        help='Optional log file path to save console output'
    )

    args = parser.parse_args()

    # Get script directory for relative path resolution
    script_dir = Path(__file__).parent.resolve()

    # Resolve config path
    config_path = Path(args.config_path)
    if not config_path.is_absolute():
        args.config_path = str((script_dir / args.config_path).resolve())

    # Resolve output directory (may also be set from config later if not provided)
    if args.output_dir is not None:
        output_dir = Path(args.output_dir)
        if not output_dir.is_absolute():
            args.output_dir = (script_dir / output_dir).resolve()
        else:
            args.output_dir = output_dir

    return args


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def print_section(title):
    """Print a section header"""
    print(f"\n{'=' * 70}")
    print(title)
    print('=' * 70)


def print_summary(title, data_dict):
    """Print a formatted summary"""
    print()
    print("=" * 60)
    print(title)
    print("=" * 60)
    for key, value in data_dict.items():
        print(f"{key}: {value}")
    print("=" * 60)
    print()


def print_los_summary(label, df, value_col):
    """Print n + median for a before/current ICU LOS slice."""
    metric_label = value_col.replace('_days', '').replace('_', ' ')
    print(f"{label}: {len(df)} hospitalizations")
    print(f"Median {metric_label} (days): {df[value_col].median()}")


# ============================================================================
# STAGE 1: COHORT IDENTIFICATION
# ============================================================================

def identify_in_system_osh_cohort(hosp, verbose=True):
    """
    Identify OSH admissions preceded by a prior admission with a gap <= 6 hours
    (i.e. the "outside hospital" transfer is actually a prior encounter we already
    have at this site).

    Args:
        hosp: Hospitalization dataframe
        verbose: Print progress messages

    Returns:
        Tuple of (hosp, osh_cohort, in_system_osh_cohort) - hosp is returned because it gains
        prev_hospitalization_id / prev_discharge_dttm / time_diff_hours columns
    """
    if verbose:
        print("2.1 Identify in-system OSH cohort (linked OSH transfers)...")

    # sort by patient_id, admission_dttm ascending so "previous" always means chronologically prior
    hosp = hosp.sort_values(['patient_id', 'admission_dttm']).reset_index(drop=True)

    # previous hospitalization's discharge time for the same patient (encounter-to-encounter gap)
    hosp['prev_hospitalization_id'] = hosp.groupby('patient_id')['hospitalization_id'].shift(1)
    hosp['prev_discharge_dttm'] = hosp.groupby('patient_id')['discharge_dttm'].shift(1)
    hosp['time_diff_hours'] = (hosp['admission_dttm'] - hosp['prev_discharge_dttm']).dt.total_seconds() / 3600

    # OSH admissions, case-insensitive
    osh_mask = hosp['admission_type_category'].str.lower() == 'osh'
    osh_cohort = hosp[osh_mask].copy()

    if osh_cohort.empty:
        raise ValueError("No OSH admissions found - check admission_type_category values")

    osh_cohort['has_previous_admission'] = osh_cohort['prev_hospitalization_id'].notna()
    osh_cohort['is_osh_transfer'] = osh_cohort['has_previous_admission'] & (osh_cohort['time_diff_hours'] <= 6)

    in_system_osh_cohort = osh_cohort[osh_cohort['is_osh_transfer']].copy()

    if in_system_osh_cohort.empty:
        raise ValueError("No linked-transfer OSH hospitalizations found after filtering")

    n_osh_total = len(osh_cohort)
    n_in_system_osh_cohort = len(in_system_osh_cohort)

    if verbose:
        print(f"   → {n_in_system_osh_cohort:,} hospitalizations in in-system OSH cohort (of {n_osh_total:,} total OSH)")

    return hosp, osh_cohort, in_system_osh_cohort


def identify_direct_icu_transfers(adt, in_system_osh_cohort, verbose=True):
    """
    Among the in-system OSH cohort, find hospitalizations whose first ADT record after
    transfer lands directly in the ICU.

    Args:
        adt: ADT dataframe
        in_system_osh_cohort: In-system OSH cohort dataframe
        verbose: Print progress messages

    Returns:
        Tuple of (direct_icu_transfer_hosp_ids, cohort_summary)
    """
    if verbose:
        print("2.2 Identify direct-to-ICU transfers...")

    first_adt = (
        adt[adt['hospitalization_id'].isin(in_system_osh_cohort['hospitalization_id'])]
        .sort_values(['hospitalization_id', 'in_dttm'])
        .drop_duplicates('hospitalization_id', keep='first')
    )
    first_adt['first_adt_is_icu'] = first_adt['location_category'].str.lower() == 'icu'

    direct_icu_transfer_hosp_ids = first_adt.loc[first_adt['first_adt_is_icu'], 'hospitalization_id'].tolist()

    if len(direct_icu_transfer_hosp_ids) == 0:
        raise ValueError("No direct-to-ICU transfers found in in-system OSH cohort")

    n_in_system_osh_cohort = len(in_system_osh_cohort)
    n_direct_icu = len(direct_icu_transfer_hosp_ids)
    pct_direct_icu = np.round(100 * n_direct_icu / n_in_system_osh_cohort, 1) if n_in_system_osh_cohort else float('nan')

    if verbose:
        print(f"   → {n_direct_icu:,} direct-to-ICU transfers ({pct_direct_icu}% of linked OSH)")

    return direct_icu_transfer_hosp_ids


def build_cohort_summary(osh_cohort, in_system_osh_cohort, direct_icu_transfer_hosp_ids):
    """
    Build the cohort identification summary table.

    Args:
        osh_cohort: All OSH admissions
        in_system_osh_cohort: Linked-transfer OSH admissions
        direct_icu_transfer_hosp_ids: Direct-to-ICU hospitalization_ids within in_system_osh_cohort

    Returns:
        cohort_summary DataFrame
    """
    n_osh_total = len(osh_cohort)
    n_in_system_osh_cohort = len(in_system_osh_cohort)
    n_direct_icu = len(direct_icu_transfer_hosp_ids)

    pct_in_system_osh_cohort = np.round(100 * n_in_system_osh_cohort / n_osh_total, 1) if n_osh_total else float('nan')
    pct_direct_icu = np.round(100 * n_direct_icu / n_in_system_osh_cohort, 1) if n_in_system_osh_cohort else float('nan')

    return pd.DataFrame({
        'metric': [
            'Total OSH hospitalizations',
            'OSH hospitalizations with a linked prior encounter (discharged <=6h before OSH admission)',
            'Direct-to-ICU transfers among linked OSH hospitalizations (first ADT record after transfer is ICU)',
        ],
        'n': [n_osh_total, n_in_system_osh_cohort, n_direct_icu],
        'pct': [np.nan, pct_in_system_osh_cohort, pct_direct_icu],
        'pct_of': ['-', 'total OSH', 'linked OSH'],
    })


def build_cohort_pairs(osh_cohort, verbose=True):
    """
    Build current/previous hospitalization_id pair tables.

    Args:
        osh_cohort: All OSH admissions (with is_osh_transfer flag)
        verbose: Print progress messages

    Returns:
        Tuple of (osh_cohort_pairs, in_system_osh_cohort_pairs)
    """
    if verbose:
        print("2.3 Build cohort pairs...")

    # osh_cohort_pairs: current_hospitalization_id / prev_hospitalization_id for every OSH
    # admission. prev_hospitalization_id is only kept when it is a validated same-site
    # transfer (is_osh_transfer); otherwise there is no trustworthy "before" encounter, so it's NaN.
    osh_cohort_pairs = osh_cohort[['hospitalization_id']].rename(
        columns={'hospitalization_id': 'current_hospitalization_id'}
    )
    osh_cohort_pairs['prev_hospitalization_id'] = np.where(
        osh_cohort['is_osh_transfer'].to_numpy(), osh_cohort['prev_hospitalization_id'].to_numpy(), np.nan
    )

    # in_system_osh_cohort_pairs: the subset of osh_cohort_pairs with a validated prior encounter at
    # this site - the key used throughout to join "before transfer" data onto "current/after transfer" data
    in_system_osh_cohort_pairs = osh_cohort_pairs.dropna(subset=['prev_hospitalization_id']).reset_index(drop=True)

    return osh_cohort_pairs, in_system_osh_cohort_pairs


def get_cohort_ids(in_system_osh_cohort, in_system_osh_cohort_pairs, osh_cohort, direct_icu_transfer_hosp_ids, verbose=True):
    """
    Derive all hospitalization_id / patient_id lists used throughout the analysis.

    Args:
        in_system_osh_cohort: In-system OSH cohort dataframe
        in_system_osh_cohort_pairs: In-system OSH cohort pairs dataframe
        osh_cohort: All OSH admissions dataframe
        direct_icu_transfer_hosp_ids: Direct-to-ICU hospitalization_ids
        verbose: Print progress messages

    Returns:
        Dict of id lists: in_system_osh_cohort_hosp_ids, in_system_osh_cohort_pat_ids, osh_hosp_ids,
        in_system_osh_cohort_prev_hosp_ids, direct_icu_prev_hosp_ids
    """
    in_system_osh_cohort_hosp_ids = in_system_osh_cohort['hospitalization_id'].tolist()
    in_system_osh_cohort_pat_ids = in_system_osh_cohort['patient_id'].tolist()
    osh_hosp_ids = osh_cohort['hospitalization_id'].unique().tolist()
    in_system_osh_cohort_prev_hosp_ids = in_system_osh_cohort_pairs['prev_hospitalization_id'].tolist()

    if verbose:
        print(f"In-system OSH cohort: {len(in_system_osh_cohort_hosp_ids)} hospitalizations, "
              f"{len(set(in_system_osh_cohort_pat_ids))} unique patients")
        print(f"In-system OSH cohort previous hospitalization_ids: {len(in_system_osh_cohort_prev_hosp_ids)}")

    # direct_icu_transfer_hosp_ids' "before transfer" counterpart, derived the same way
    # in_system_osh_cohort_pairs itself was derived from osh_cohort_pairs
    direct_icu_pairs = in_system_osh_cohort_pairs[
        in_system_osh_cohort_pairs['current_hospitalization_id'].isin(direct_icu_transfer_hosp_ids)
    ].reset_index(drop=True)
    direct_icu_prev_hosp_ids = direct_icu_pairs['prev_hospitalization_id'].tolist()

    if verbose:
        print(f"Direct ICU transfer cohort: {len(direct_icu_transfer_hosp_ids)} hospitalizations")

    return {
        'in_system_osh_cohort_hosp_ids': in_system_osh_cohort_hosp_ids,
        'in_system_osh_cohort_pat_ids': in_system_osh_cohort_pat_ids,
        'osh_hosp_ids': osh_hosp_ids,
        'in_system_osh_cohort_prev_hosp_ids': in_system_osh_cohort_prev_hosp_ids,
        'direct_icu_prev_hosp_ids': direct_icu_prev_hosp_ids,
    }


# ============================================================================
# STAGE 2: ICU LOS
# ============================================================================

def compute_icu_los(adt_df, hosp_ids):
    """
    ICU LOS for a set of hospitalization_ids: sum ICU location-segment durations per
    hospitalization (a hospitalization can have multiple non-contiguous ICU stays).

    Args:
        adt_df: ADT dataframe
        hosp_ids: List of hospitalization_ids to compute LOS for

    Returns:
        Tuple of (icu_segments_df, icu_los_df) - icu_segments_df carries per-segment
        los_hours and is reused by first_icu_los and the ICU type analysis.
    """
    subset = adt_df[adt_df['hospitalization_id'].isin(hosp_ids)].copy()
    icu_segments = subset[subset['location_category'].str.lower() == 'icu'].copy()
    icu_segments['los_hours'] = (
        (icu_segments['out_dttm'] - icu_segments['in_dttm']).dt.total_seconds() / 3600
    )

    icu_los = (
        icu_segments.groupby('hospitalization_id')['los_hours']
        .sum()
        .rename('icu_los_hours')
        .reset_index()
    )
    icu_los['icu_los_days'] = np.round(icu_los['icu_los_hours'] / 24, 2)

    # keep every hospitalization_id, even with no ICU time (LOS = 0, not missing)
    # dtype='object' avoids an empty list defaulting to float64, which would fail to
    # merge against the string hospitalization_id column
    icu_los = (
        pd.DataFrame({'hospitalization_id': pd.array(hosp_ids, dtype='object')})
        .drop_duplicates()
        .merge(icu_los, on='hospitalization_id', how='left')
        .fillna({'icu_los_hours': 0, 'icu_los_days': 0})
    )
    return icu_segments, icu_los


def first_icu_los(icu_segments_df, hosp_ids):
    """
    First (chronologically earliest) ICU segment per hospitalization_id - same
    fill-zero pattern as compute_icu_los, but selects the FIRST icu segment
    chronologically instead of summing.

    Args:
        icu_segments_df: Per-segment ICU dataframe from compute_icu_los
        hosp_ids: List of hospitalization_ids

    Returns:
        DataFrame with first_icu_los_hours / first_icu_los_days per hospitalization_id
    """
    first = (
        icu_segments_df.sort_values('in_dttm')
        .drop_duplicates('hospitalization_id', keep='first')
        .rename(columns={'los_hours': 'first_icu_los_hours'})
        [['hospitalization_id', 'first_icu_los_hours']]
    )
    first['first_icu_los_days'] = np.round(first['first_icu_los_hours'] / 24, 2)

    first = (
        pd.DataFrame({'hospitalization_id': pd.array(hosp_ids, dtype='object')})
        .drop_duplicates()
        .merge(first, on='hospitalization_id', how='left')
        .fillna({'first_icu_los_hours': 0, 'first_icu_los_days': 0})
    )
    return first


def run_icu_los_analysis(adt, osh_hosp_ids, in_system_osh_cohort_hosp_ids, in_system_osh_cohort_prev_hosp_ids,
                          direct_icu_transfer_hosp_ids, direct_icu_prev_hosp_ids, verbose=True):
    """
    Compute ICU LOS (first-stay and cumulative) for the overall OSH baseline, in-system
    OSH cohort, and direct ICU transfer cohort.

    Args:
        adt: ADT dataframe
        osh_hosp_ids, in_system_osh_cohort_hosp_ids, in_system_osh_cohort_prev_hosp_ids,
        direct_icu_transfer_hosp_ids, direct_icu_prev_hosp_ids: hospitalization_id lists
        verbose: Print progress messages

    Returns:
        Dict of intermediate dataframes plus the long-format icu_los_summary table
    """
    if verbose:
        print("3.1 Compute ICU LOS (first-stay and cumulative)...")

    prev_icu_segments, prev_icu_los = compute_icu_los(adt, in_system_osh_cohort_prev_hosp_ids)
    curr_icu_segments, curr_icu_los = compute_icu_los(adt, in_system_osh_cohort_hosp_ids)

    prev_first_icu = first_icu_los(prev_icu_segments, in_system_osh_cohort_prev_hosp_ids)
    curr_first_icu = first_icu_los(curr_icu_segments, in_system_osh_cohort_hosp_ids)

    # overall OSH cohort baseline: current = every OSH admission's own ICU LOS; before =
    # the prior-encounter ICU LOS, available only for the linked-transfer subset
    osh_curr_icu_segments, osh_curr_icu_los = compute_icu_los(adt, osh_hosp_ids)
    osh_curr_first_icu = first_icu_los(osh_curr_icu_segments, osh_hosp_ids)

    # direct ICU transfer cohort: first ICU stay LOS, before transfer vs current
    direct_prev_first_icu = prev_first_icu[prev_first_icu['hospitalization_id'].isin(direct_icu_prev_hosp_ids)]
    direct_curr_first_icu = curr_first_icu[curr_first_icu['hospitalization_id'].isin(direct_icu_transfer_hosp_ids)]

    # direct ICU transfer cohort: cumulative ICU LOS, before transfer vs current
    direct_prev_icu_los = prev_icu_los[prev_icu_los['hospitalization_id'].isin(direct_icu_prev_hosp_ids)]
    direct_curr_icu_los = curr_icu_los[curr_icu_los['hospitalization_id'].isin(direct_icu_transfer_hosp_ids)]

    if verbose:
        print("=== Overall OSH cohort (baseline) ===")
        print_los_summary("Current (all OSH admissions)", osh_curr_first_icu, 'first_icu_los_days')
        print("=== In-system OSH cohort: first ICU LOS ===")
        print_los_summary("Before transfer", prev_first_icu, 'first_icu_los_days')
        print_los_summary("Current", curr_first_icu, 'first_icu_los_days')
        print("=== Direct ICU transfer cohort: first ICU LOS ===")
        print_los_summary("Before transfer", direct_prev_first_icu, 'first_icu_los_days')
        print_los_summary("Current", direct_curr_first_icu, 'first_icu_los_days')
        print("=== In-system OSH cohort: cumulative ICU LOS ===")
        print_los_summary("Before transfer", prev_icu_los, 'icu_los_days')
        print_los_summary("Current", curr_icu_los, 'icu_los_days')
        print("=== Direct ICU transfer cohort: cumulative ICU LOS ===")
        print_los_summary("Before transfer", direct_prev_icu_los, 'icu_los_days')
        print_los_summary("Current", direct_curr_icu_los, 'icu_los_days')

    # ICU LOS results (printed together for review above) as one long-format table
    los_summary_rows = [
        {'cohort': 'overall_osh', 'los_type': 'first_icu', 'period': 'current',
         'n': len(osh_curr_first_icu), 'median_los_days': osh_curr_first_icu['first_icu_los_days'].median()},
        {'cohort': 'in_system_osh', 'los_type': 'first_icu', 'period': 'before_transfer',
         'n': len(prev_first_icu), 'median_los_days': prev_first_icu['first_icu_los_days'].median()},
        {'cohort': 'in_system_osh', 'los_type': 'first_icu', 'period': 'current',
         'n': len(curr_first_icu), 'median_los_days': curr_first_icu['first_icu_los_days'].median()},
        {'cohort': 'direct_icu_transfer', 'los_type': 'first_icu', 'period': 'before_transfer',
         'n': len(direct_prev_first_icu), 'median_los_days': direct_prev_first_icu['first_icu_los_days'].median()},
        {'cohort': 'direct_icu_transfer', 'los_type': 'first_icu', 'period': 'current',
         'n': len(direct_curr_first_icu), 'median_los_days': direct_curr_first_icu['first_icu_los_days'].median()},
        {'cohort': 'in_system_osh', 'los_type': 'cumulative_icu', 'period': 'before_transfer',
         'n': len(prev_icu_los), 'median_los_days': prev_icu_los['icu_los_days'].median()},
        {'cohort': 'in_system_osh', 'los_type': 'cumulative_icu', 'period': 'current',
         'n': len(curr_icu_los), 'median_los_days': curr_icu_los['icu_los_days'].median()},
        {'cohort': 'direct_icu_transfer', 'los_type': 'cumulative_icu', 'period': 'before_transfer',
         'n': len(direct_prev_icu_los), 'median_los_days': direct_prev_icu_los['icu_los_days'].median()},
        {'cohort': 'direct_icu_transfer', 'los_type': 'cumulative_icu', 'period': 'current',
         'n': len(direct_curr_icu_los), 'median_los_days': direct_curr_icu_los['icu_los_days'].median()},
    ]
    icu_los_summary = pd.DataFrame(los_summary_rows)

    return {
        'prev_icu_segments': prev_icu_segments,
        'curr_icu_segments': curr_icu_segments,
        'icu_los_summary': icu_los_summary,
    }


# ============================================================================
# STAGE 3: HOSPITAL LOS
# ============================================================================

def hospital_los(hosp_df, hosp_ids):
    """
    Whole-encounter hospital LOS (admission_dttm -> discharge_dttm) for a set of
    hospitalization_ids.

    Args:
        hosp_df: Hospitalization dataframe
        hosp_ids: List of hospitalization_ids

    Returns:
        DataFrame with hosp_los_hours / hosp_los_days per hospitalization_id
    """
    subset = hosp_df[hosp_df['hospitalization_id'].isin(hosp_ids)][
        ['hospitalization_id', 'admission_dttm', 'discharge_dttm']
    ].copy()
    subset['hosp_los_hours'] = (
        (subset['discharge_dttm'] - subset['admission_dttm']).dt.total_seconds() / 3600
    )
    subset['hosp_los_days'] = np.round(subset['hosp_los_hours'] / 24, 2)
    return subset


def run_hospital_los_analysis(hosp, osh_hosp_ids, in_system_osh_cohort_hosp_ids, in_system_osh_cohort_prev_hosp_ids,
                               direct_icu_transfer_hosp_ids, direct_icu_prev_hosp_ids, verbose=True):
    """
    Compute whole-encounter hospital LOS across cohorts.

    Args:
        hosp: Hospitalization dataframe
        osh_hosp_ids, in_system_osh_cohort_hosp_ids, in_system_osh_cohort_prev_hosp_ids,
        direct_icu_transfer_hosp_ids, direct_icu_prev_hosp_ids: hospitalization_id lists
        verbose: Print progress messages

    Returns:
        hosp_los_summary DataFrame
    """
    if verbose:
        print("4.1 Compute hospital LOS...")

    overall_hosp_los = hospital_los(hosp, hosp['hospitalization_id'].tolist())
    osh_hosp_los = hospital_los(hosp, osh_hosp_ids)
    in_system_osh_cohort_before_hosp_los = hospital_los(hosp, in_system_osh_cohort_prev_hosp_ids)
    in_system_osh_cohort_after_hosp_los = hospital_los(hosp, in_system_osh_cohort_hosp_ids)
    direct_icu_before_hosp_los = hospital_los(hosp, direct_icu_prev_hosp_ids)
    direct_icu_after_hosp_los = hospital_los(hosp, direct_icu_transfer_hosp_ids)

    hosp_los_summary = pd.DataFrame({
        'cohort': [
            'overall (all hospitalizations)',
            'overall OSH cohort',
            'in-system OSH cohort (before transfer)',
            'in-system OSH cohort (after transfer)',
            'direct ICU transfer (before transfer)',
            'direct ICU transfer (after transfer)',
        ],
        'n': [
            len(overall_hosp_los), len(osh_hosp_los),
            len(in_system_osh_cohort_before_hosp_los), len(in_system_osh_cohort_after_hosp_los),
            len(direct_icu_before_hosp_los), len(direct_icu_after_hosp_los),
        ],
        'median_hosp_los_days': [
            overall_hosp_los['hosp_los_days'].median(),
            osh_hosp_los['hosp_los_days'].median(),
            in_system_osh_cohort_before_hosp_los['hosp_los_days'].median(),
            in_system_osh_cohort_after_hosp_los['hosp_los_days'].median(),
            direct_icu_before_hosp_los['hosp_los_days'].median(),
            direct_icu_after_hosp_los['hosp_los_days'].median(),
        ],
    })

    if verbose:
        print(hosp_los_summary.to_string(index=False))

    return hosp_los_summary


# ============================================================================
# STAGE 4: ICU TYPE & HOSPITAL TYPE
# ============================================================================

def first_icu_detail(icu_segments_df, hosp_ids):
    """
    First (chronologically earliest) ICU segment per hospitalization_id, carrying its
    location_type (ICU type) and hospital_id (hospital id) - the shared basis for
    both the ICU type and hospital type analyses.

    Args:
        icu_segments_df: Per-segment ICU dataframe from compute_icu_los
        hosp_ids: List of hospitalization_ids

    Returns:
        DataFrame with first_icu_type / first_icu_hospital_id per hospitalization_id
    """
    first = (
        icu_segments_df.sort_values('in_dttm')
        .drop_duplicates('hospitalization_id', keep='first')
        [['hospitalization_id', 'location_type', 'hospital_id']]
        .rename(columns={'location_type': 'first_icu_type', 'hospital_id': 'first_icu_hospital_id'})
    )
    return (
        pd.DataFrame({'hospitalization_id': pd.array(hosp_ids, dtype='object')})
        .drop_duplicates()
        .merge(first, on='hospitalization_id', how='left')
    )


def merge_before_after(pairs_df, prev_df, curr_df, id_col='hospitalization_id'):
    """
    Merge a per-hospitalization detail df onto a current/prev pairs table (e.g.
    in_system_osh_cohort_pairs), prefixing every non-id column with prev_/curr_ for each side -
    reused for ICU LOS, ICU type, hospital type, and discharge category.

    Args:
        pairs_df: current_hospitalization_id / prev_hospitalization_id pairs table
        prev_df: Per-hospitalization detail df for the "before" side
        curr_df: Per-hospitalization detail df for the "current" side
        id_col: Shared hospitalization_id column name in prev_df/curr_df

    Returns:
        Merged DataFrame with prev_*/curr_* columns
    """
    value_cols = [c for c in prev_df.columns if c != id_col]
    prev_renamed = prev_df.rename(
        columns={id_col: 'prev_hospitalization_id', **{c: f'prev_{c}' for c in value_cols}}
    )
    curr_renamed = curr_df.rename(
        columns={id_col: 'current_hospitalization_id', **{c: f'curr_{c}' for c in value_cols}}
    )
    return (
        pairs_df
        .merge(prev_renamed, on='prev_hospitalization_id', how='left')
        .merge(curr_renamed, on='current_hospitalization_id', how='left')
    )


def category_counts(detail_df, value_col, cohort_label, missing_label='missing'):
    """
    Count/percentage of a categorical column within a per-hospitalization detail df -
    reused for ICU type, hospital type, and discharge category.

    Args:
        detail_df: Per-hospitalization detail dataframe
        value_col: Categorical column to summarize
        cohort_label: Label for the 'cohort' column in the output
        missing_label: Fill value for missing categories

    Returns:
        DataFrame with cohort / value_col / n / pct
    """
    counts = (
        detail_df[value_col]
        .fillna(missing_label)
        .value_counts()
        .rename_axis(value_col)
        .reset_index(name='n')
    )
    counts['pct'] = np.round(100 * counts['n'] / len(detail_df), 1) if len(detail_df) else np.nan
    counts.insert(0, 'cohort', cohort_label)
    return counts


def hex_to_rgba(hex_color, alpha):
    """Convert a hex color string to an rgba() string with the given alpha."""
    hex_color = hex_color.lstrip('#')
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    return f'rgba({r},{g},{b},{alpha})'


def make_sankey(before_after_df, prev_col, curr_col, title, missing_label='missing'):
    """
    Before -> current Sankey over a categorical column, one node color per category
    shared across before/after sides (identity, not rank). Reused for ICU type,
    hospital type, and discharge category.

    Args:
        before_after_df: DataFrame with prev_col / curr_col columns
        prev_col: Column name for the "before" category
        curr_col: Column name for the "current" category
        title: Figure title
        missing_label: Fill value for missing categories

    Returns:
        plotly.graph_objects.Figure
    """
    df = before_after_df[[prev_col, curr_col]].copy()
    df[prev_col] = df[prev_col].fillna(missing_label)
    df[curr_col] = df[curr_col].fillna(missing_label)

    flow_counts = df.groupby([prev_col, curr_col]).size().rename('n').reset_index()

    categories_seen = sorted(set(flow_counts[prev_col]) | set(flow_counts[curr_col]))
    color_map = {
        c: (CATEGORICAL_HUES[i] if i < len(CATEGORICAL_HUES) else OTHER_HUE)
        for i, c in enumerate(categories_seen)
    }
    before_index = {c: i for i, c in enumerate(categories_seen)}
    after_index = {c: i + len(categories_seen) for i, c in enumerate(categories_seen)}

    fig = go.Figure(go.Sankey(
        node=dict(
            label=[f"Before: {c}" for c in categories_seen] + [f"After: {c}" for c in categories_seen],
            color=[color_map[c] for c in categories_seen] * 2,
            pad=15,
            thickness=18,
            line=dict(color='rgba(11,11,11,0.10)', width=1),
        ),
        link=dict(
            source=[before_index[c] for c in flow_counts[prev_col]],
            target=[after_index[c] for c in flow_counts[curr_col]],
            value=flow_counts['n'],
            color=[hex_to_rgba(color_map[c], 0.4) for c in flow_counts[prev_col]],
        ),
    ))
    fig.update_layout(
        title_text=title,
        font_size=12,
        plot_bgcolor='#fcfcfb',
        paper_bgcolor='#fcfcfb',
    )
    return fig


def run_icu_type_hospital_type_analysis(prev_icu_segments, curr_icu_segments, in_system_osh_cohort_pairs,
                                         in_system_osh_cohort_prev_hosp_ids, in_system_osh_cohort_hosp_ids,
                                         direct_icu_prev_hosp_ids, direct_icu_transfer_hosp_ids,
                                         verbose=True):
    """
    Compare ICU type and hospital type before vs. after transfer, for the in-system
    OSH cohort and its direct ICU transfer subset.

    Args:
        prev_icu_segments, curr_icu_segments: Per-segment ICU dataframes from run_icu_los_analysis
        in_system_osh_cohort_pairs: In-system OSH cohort pairs dataframe
        in_system_osh_cohort_prev_hosp_ids, in_system_osh_cohort_hosp_ids,
        direct_icu_prev_hosp_ids, direct_icu_transfer_hosp_ids: hospitalization_id lists
        verbose: Print progress messages

    Returns:
        Dict of summary tables and sankey figures
    """
    if verbose:
        print("5.1 Compare ICU type and hospital type before vs. after transfer...")

    prev_icu_detail = first_icu_detail(prev_icu_segments, in_system_osh_cohort_prev_hosp_ids)
    curr_icu_detail = first_icu_detail(curr_icu_segments, in_system_osh_cohort_hosp_ids)

    icu_type_before_after = merge_before_after(in_system_osh_cohort_pairs, prev_icu_detail, curr_icu_detail)

    # In-system OSH cohort: first ICU's type, before transfer vs current (number/percentage)
    icu_type_summary_in_system_osh = pd.concat([
        category_counts(prev_icu_detail, 'first_icu_type', 'before_transfer', missing_label='no_icu_stay'),
        category_counts(curr_icu_detail, 'first_icu_type', 'current', missing_label='no_icu_stay'),
    ], ignore_index=True)

    # direct ICU transfer cohort: first ICU's type, before transfer vs current (number/percentage)
    direct_prev_icu_detail = prev_icu_detail[prev_icu_detail['hospitalization_id'].isin(direct_icu_prev_hosp_ids)]
    direct_curr_icu_detail = curr_icu_detail[curr_icu_detail['hospitalization_id'].isin(direct_icu_transfer_hosp_ids)]

    icu_type_summary_direct_icu = pd.concat([
        category_counts(direct_prev_icu_detail, 'first_icu_type', 'before_transfer', missing_label='no_icu_stay'),
        category_counts(direct_curr_icu_detail, 'first_icu_type', 'current', missing_label='no_icu_stay'),
    ], ignore_index=True)

    # In-system OSH cohort: first ICU's type, before transfer -> current (sankey)
    fig_icu_type_in_system_osh = make_sankey(
        icu_type_before_after, 'prev_first_icu_type', 'curr_first_icu_type',
        title="In-system OSH cohort: first ICU's type, before transfer -> current",
        missing_label='no_icu_stay',
    )

    # direct ICU transfer cohort: first ICU's type, before transfer -> current (sankey)
    direct_icu_type_before_after = icu_type_before_after[
        icu_type_before_after['current_hospitalization_id'].isin(direct_icu_transfer_hosp_ids)
    ]
    fig_icu_type_direct_icu = make_sankey(
        direct_icu_type_before_after, 'prev_first_icu_type', 'curr_first_icu_type',
        title="Direct ICU transfer cohort: first ICU's type, before transfer -> current",
        missing_label='no_icu_stay',
    )

    # hospital type of the first ICU stay - reuses the same first-ICU-segment detail
    # computed for ICU type above (prev_icu_detail/curr_icu_detail already carry
    # first_icu_hospital_id from that same segment)
    prev_hospital_detail = prev_icu_detail
    curr_hospital_detail = curr_icu_detail
    hospital_id_before_after = icu_type_before_after

    # In-system OSH cohort: first ICU's hospital type, before transfer vs current (number/percentage)
    hospital_id_summary_in_system_osh = pd.concat([
        category_counts(prev_hospital_detail, 'first_icu_hospital_id', 'before_transfer', missing_label='no_icu_stay'),
        category_counts(curr_hospital_detail, 'first_icu_hospital_id', 'current', missing_label='no_icu_stay'),
    ], ignore_index=True)

    # direct ICU transfer cohort: first ICU's hospital id, before transfer vs current (number/percentage)
    direct_prev_hospital_detail = prev_hospital_detail[prev_hospital_detail['hospitalization_id'].isin(direct_icu_prev_hosp_ids)]
    direct_curr_hospital_detail = curr_hospital_detail[curr_hospital_detail['hospitalization_id'].isin(direct_icu_transfer_hosp_ids)]

    hospital_id_summary_direct_icu = pd.concat([
        category_counts(direct_prev_hospital_detail, 'first_icu_hospital_id', 'before_transfer', missing_label='no_icu_stay'),
        category_counts(direct_curr_hospital_detail, 'first_icu_hospital_id', 'current', missing_label='no_icu_stay'),
    ], ignore_index=True)

    # In-system OSH cohort: first ICU's hospital id, before transfer -> current (sankey)
    fig_hospital_type_in_system_osh = make_sankey(
        hospital_id_before_after, 'prev_first_icu_hospital_id', 'curr_first_icu_hospital_id',
        title="In-system OSH cohort: first ICU's hospital type, before transfer -> current",
        missing_label='no_icu_stay',
    )

    # direct ICU transfer cohort: first ICU's hospital type, before transfer -> current (sankey)
    direct_hospital_id_before_after = hospital_id_before_after[
        hospital_id_before_after['current_hospitalization_id'].isin(direct_icu_transfer_hosp_ids)
    ]
    fig_hospital_type_direct_icu = make_sankey(
        direct_hospital_id_before_after, 'prev_first_icu_hospital_id', 'curr_first_icu_hospital_id',
        title="Direct ICU transfer cohort: first ICU's hospital id, before transfer -> current",
        missing_label='no_icu_stay',
    )

    return {
        'icu_type_summary_in_system_osh': icu_type_summary_in_system_osh,
        'icu_type_summary_direct_icu': icu_type_summary_direct_icu,
        'hospital_id_summary_in_system_osh': hospital_id_summary_in_system_osh,
        'hospital_id_summary_direct_icu': hospital_id_summary_direct_icu,
        'fig_icu_type_in_system_osh': fig_icu_type_in_system_osh,
        'fig_icu_type_direct_icu': fig_icu_type_direct_icu,
        'fig_hospital_type_in_system_osh': fig_hospital_type_in_system_osh,
        'fig_hospital_type_direct_icu': fig_hospital_type_direct_icu,
    }


# ============================================================================
# STAGE 5: MORTALITY RATE
# ============================================================================

def compute_mortality(hosp_df, hosp_ids=None):
    """
    Mortality rate: discharge_category == 'expired', case-insensitive.

    Args:
        hosp_df: Hospitalization dataframe
        hosp_ids: Optional list of hospitalization_ids to restrict to

    Returns:
        pd.Series with n_total / n_expired / pct_expired
    """
    subset = hosp_df if hosp_ids is None else hosp_df[hosp_df['hospitalization_id'].isin(hosp_ids)]
    n_total = len(subset)
    n_expired = (subset['discharge_category'].str.lower() == 'expired').sum()
    pct_expired = np.round(100 * n_expired / n_total, 1) if n_total > 0 else float('nan')
    return pd.Series({'n_total': n_total, 'n_expired': n_expired, 'pct_expired': pct_expired})


def run_mortality_analysis(hosp, osh_hosp_ids, in_system_osh_cohort_hosp_ids, direct_icu_transfer_hosp_ids, verbose=True):
    """
    Compare mortality rate across cohorts.

    Args:
        hosp: Hospitalization dataframe
        osh_hosp_ids, in_system_osh_cohort_hosp_ids, direct_icu_transfer_hosp_ids: hospitalization_id lists
        verbose: Print progress messages

    Returns:
        mortality_summary DataFrame
    """
    if verbose:
        print("6.1 Compute mortality rate...")

    overall_mortality = compute_mortality(hosp)
    osh_mortality = compute_mortality(hosp, osh_hosp_ids)
    in_system_osh_mortality = compute_mortality(hosp, in_system_osh_cohort_hosp_ids)
    direct_icu_mortality = compute_mortality(hosp, direct_icu_transfer_hosp_ids)

    mortality_summary = (
        pd.DataFrame({
            'overall': overall_mortality,
            'overall_osh': osh_mortality,
            'in_system_osh': in_system_osh_mortality,
            'direct_icu_transfer': direct_icu_mortality,
        })
        .T.rename_axis('cohort').reset_index()
    )

    if verbose:
        print(mortality_summary.to_string(index=False))

    return mortality_summary


# ============================================================================
# STAGE 6: DISCHARGE CATEGORY SUMMARY
# ============================================================================

def discharge_detail(hosp_df, hosp_ids):
    """
    hospitalization_id + discharge_category lookup, restricted to a set of ids -
    feeds category_counts (number/percentage tables).

    Args:
        hosp_df: Hospitalization dataframe
        hosp_ids: List of hospitalization_ids

    Returns:
        DataFrame with hospitalization_id / discharge_category
    """
    return hosp_df[hosp_df['hospitalization_id'].isin(hosp_ids)][['hospitalization_id', 'discharge_category']]


def run_discharge_analysis(hosp, osh_hosp_ids, in_system_osh_cohort_prev_hosp_ids, in_system_osh_cohort_hosp_ids,
                            direct_icu_prev_hosp_ids, direct_icu_transfer_hosp_ids,
                            in_system_osh_cohort_pairs, verbose=True):
    """
    Compare discharge category before vs. after transfer.

    Args:
        hosp: Hospitalization dataframe
        osh_hosp_ids, in_system_osh_cohort_prev_hosp_ids, in_system_osh_cohort_hosp_ids,
        direct_icu_prev_hosp_ids, direct_icu_transfer_hosp_ids: hospitalization_id lists
        in_system_osh_cohort_pairs: In-system OSH cohort pairs dataframe
        verbose: Print progress messages

    Returns:
        Dict of summary tables and the discharge category sankey figures
    """
    if verbose:
        print("7.1 Compare discharge category before vs. after transfer...")

    # overall OSH cohort: discharge category, before transfer (in-system-OSH subset only) vs
    # current (all OSH admissions) (number/percentage)
    overall_osh_discharge_summary = pd.concat([
        category_counts(discharge_detail(hosp, in_system_osh_cohort_prev_hosp_ids), 'discharge_category', 'before_transfer'),
        category_counts(discharge_detail(hosp, osh_hosp_ids), 'discharge_category', 'current'),
    ], ignore_index=True)

    # In-system OSH cohort: discharge category, before transfer vs current (number/percentage)
    in_system_osh_discharge_summary = pd.concat([
        category_counts(discharge_detail(hosp, in_system_osh_cohort_prev_hosp_ids), 'discharge_category', 'before_transfer'),
        category_counts(discharge_detail(hosp, in_system_osh_cohort_hosp_ids), 'discharge_category', 'current'),
    ], ignore_index=True)

    # direct ICU transfer cohort: discharge category, before transfer vs current (number/percentage)
    direct_icu_discharge_summary = pd.concat([
        category_counts(discharge_detail(hosp, direct_icu_prev_hosp_ids), 'discharge_category', 'before_transfer'),
        category_counts(discharge_detail(hosp, direct_icu_transfer_hosp_ids), 'discharge_category', 'current'),
    ], ignore_index=True)

    discharge_lookup = hosp[['hospitalization_id', 'discharge_category']]
    discharge_category_before_after = merge_before_after(in_system_osh_cohort_pairs, discharge_lookup, discharge_lookup)

    # In-system OSH cohort: discharge category, before transfer -> current (sankey)
    fig_discharge_category_in_system_osh = make_sankey(
        discharge_category_before_after, 'prev_discharge_category', 'curr_discharge_category',
        title="In-system OSH cohort: discharge category, before transfer -> current",
    )

    # direct ICU transfer cohort: discharge category, before transfer -> current (sankey)
    direct_discharge_category_before_after = discharge_category_before_after[
        discharge_category_before_after['current_hospitalization_id'].isin(direct_icu_transfer_hosp_ids)
    ]
    fig_discharge_category_direct_icu = make_sankey(
        direct_discharge_category_before_after, 'prev_discharge_category', 'curr_discharge_category',
        title="Direct ICU transfer cohort: discharge category, before transfer -> current",
    )

    return {
        'overall_osh_discharge_summary': overall_osh_discharge_summary,
        'in_system_osh_discharge_summary': in_system_osh_discharge_summary,
        'direct_icu_discharge_summary': direct_icu_discharge_summary,
        'fig_discharge_category_in_system_osh': fig_discharge_category_in_system_osh,
        'fig_discharge_category_direct_icu': fig_discharge_category_direct_icu,
    }


# ============================================================================
# SAVE OUTPUTS
# ============================================================================

def save_outputs(output_dir, site_name, cohort_summary, icu_los_summary, hosp_los_summary,
                  icu_type_hospital_type_results, mortality_summary, discharge_results, verbose=True):
    """
    Save every summary DataFrame to CSV and every Sankey figure to PNG.

    Args:
        output_dir: Output directory
        site_name: Site name (used in output filenames)
        cohort_summary, icu_los_summary, hosp_los_summary, mortality_summary: Summary DataFrames
        icu_type_hospital_type_results: Dict returned by run_icu_type_hospital_type_analysis
        discharge_results: Dict returned by run_discharge_analysis
        verbose: Print progress messages

    Returns:
        None
    """
    if verbose:
        print("8.1 Save summary tables and figures...")

    summary_tables = {
        'cohort_summary': cohort_summary,
        'icu_los_summary': icu_los_summary,
        'hospital_los_summary': hosp_los_summary,
        'icu_type_summary_in_system_osh': icu_type_hospital_type_results['icu_type_summary_in_system_osh'],
        'icu_type_summary_direct_icu': icu_type_hospital_type_results['icu_type_summary_direct_icu'],
        'hospital_type_summary_in_system_osh_cohort': icu_type_hospital_type_results['hospital_id_summary_in_system_osh'],
        'hospital_type_summary_direct_icu': icu_type_hospital_type_results['hospital_id_summary_direct_icu'],
        'mortality_summary': mortality_summary,
        'discharge_summary_overall_osh': discharge_results['overall_osh_discharge_summary'],
        'discharge_summary_in_system_osh_cohort': discharge_results['in_system_osh_discharge_summary'],
        'discharge_summary_direct_icu': discharge_results['direct_icu_discharge_summary'],
    }

    for name, df in summary_tables.items():
        out_path = os.path.join(output_dir, f"{name}_{site_name}.csv")
        df.to_csv(out_path, index=False)
        print(f"✅ Saved → {out_path}")

    sankey_figures = {
        'sankey_icu_type_in_system_osh_cohort': icu_type_hospital_type_results['fig_icu_type_in_system_osh'],
        'sankey_icu_type_direct_icu': icu_type_hospital_type_results['fig_icu_type_direct_icu'],
        'sankey_hospital_type_in_system_osh_cohort': icu_type_hospital_type_results['fig_hospital_type_in_system_osh'],
        'sankey_hospital_type_direct_icu': icu_type_hospital_type_results['fig_hospital_type_direct_icu'],
        'sankey_discharge_category_in_system_osh_cohort': discharge_results['fig_discharge_category_in_system_osh'],
        'sankey_discharge_category_direct_icu': discharge_results['fig_discharge_category_direct_icu'],
    }

    for name, fig in sankey_figures.items():
        out_path = os.path.join(output_dir, f"{name}_{site_name}.png")
        fig.write_image(out_path)
        print(f"✅ Saved → {out_path}")


# ============================================================================
# MAIN EXECUTION FUNCTION
# ============================================================================

def main():
    """
    Main execution function for the ICU transfer analysis pipeline.

    Returns:
        int: Exit code (0 for success, 1 for failure)
    """
    args = parse_arguments()

    # Setup logging if requested
    log_file = None
    original_stdout = sys.stdout
    original_stderr = sys.stderr

    if args.log_file:
        log_path = Path(args.log_file)

        # If user provided a directory (no file extension), auto-generate filename
        if log_path.suffix == '' or log_path.is_dir():
            from datetime import datetime
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_path = log_path / f"icu_transfer_analysis_{timestamp}.log"

        log_path.parent.mkdir(parents=True, exist_ok=True)

        class Tee:
            """Write to multiple file objects simultaneously."""
            def __init__(self, *files):
                self.files = files

            def write(self, obj):
                for f in self.files:
                    f.write(obj)
                    f.flush()

            def flush(self):
                for f in self.files:
                    f.flush()

        log_file = open(log_path, 'w')
        sys.stdout = Tee(original_stdout, log_file)
        sys.stderr = Tee(original_stderr, log_file)

    try:
        config_path = args.config_path

        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")

        verbose = args.verbose

        print_section("CLIF ICU Transfer Analysis Pipeline")
        print(f"Config path: {config_path}")
        print(f"Verbose: {verbose}")

        # Load config
        config = load_config()
        site_name = config['site_name'].lower()

        output_dir = args.output_dir if args.output_dir is not None else config['output_directory']
        output_dir = Path(output_dir)
        if not output_dir.is_absolute():
            output_dir = (Path(__file__).parent / output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"Site name: {site_name}")
        print(f"Output directory: {output_dir}")

        co = ClifOrchestrator(config_path=config_path)

        # ====================================================================
        # STAGE 1: LOAD TABLES
        # ====================================================================
        print_section("Stage 1: Load tables")

        co.load_table('hospitalization')
        hosp = co.hospitalization.df.copy()
        hosp['admission_dttm'] = pd.to_datetime(hosp['admission_dttm'])
        hosp['discharge_dttm'] = pd.to_datetime(hosp['discharge_dttm'])

        co.load_table('adt')
        adt = co.adt.df.copy()
        adt['in_dttm'] = pd.to_datetime(adt['in_dttm'])
        adt['out_dttm'] = pd.to_datetime(adt['out_dttm'])

        if verbose:
            print(f"   → hospitalization: {len(hosp):,} rows")
            print(f"   → adt: {len(adt):,} rows")

        # ====================================================================
        # STAGE 2: COHORT IDENTIFICATION
        # ====================================================================
        print_section("Stage 2: Cohort identification")

        hosp, osh_cohort, in_system_osh_cohort = identify_in_system_osh_cohort(hosp, verbose)
        direct_icu_transfer_hosp_ids = identify_direct_icu_transfers(adt, in_system_osh_cohort, verbose)
        cohort_summary = build_cohort_summary(osh_cohort, in_system_osh_cohort, direct_icu_transfer_hosp_ids)
        osh_cohort_pairs, in_system_osh_cohort_pairs = build_cohort_pairs(osh_cohort, verbose)
        cohort_ids = get_cohort_ids(in_system_osh_cohort, in_system_osh_cohort_pairs, osh_cohort,
                                     direct_icu_transfer_hosp_ids, verbose)

        # ====================================================================
        # STAGE 3: ICU LOS
        # ====================================================================
        print_section("Stage 3: ICU LOS analysis")

        icu_los_results = run_icu_los_analysis(
            adt, cohort_ids['osh_hosp_ids'], cohort_ids['in_system_osh_cohort_hosp_ids'],
            cohort_ids['in_system_osh_cohort_prev_hosp_ids'], direct_icu_transfer_hosp_ids,
            cohort_ids['direct_icu_prev_hosp_ids'], verbose
        )

        # ====================================================================
        # STAGE 4: HOSPITAL LOS
        # ====================================================================
        print_section("Stage 4: Hospital LOS analysis")

        hosp_los_summary = run_hospital_los_analysis(
            hosp, cohort_ids['osh_hosp_ids'], cohort_ids['in_system_osh_cohort_hosp_ids'],
            cohort_ids['in_system_osh_cohort_prev_hosp_ids'], direct_icu_transfer_hosp_ids,
            cohort_ids['direct_icu_prev_hosp_ids'], verbose
        )

        # ====================================================================
        # STAGE 5: ICU TYPE & HOSPITAL TYPE
        # ====================================================================
        print_section("Stage 5: ICU type & hospital type analysis")

        icu_type_hospital_type_results = run_icu_type_hospital_type_analysis(
            icu_los_results['prev_icu_segments'], icu_los_results['curr_icu_segments'],
            in_system_osh_cohort_pairs, cohort_ids['in_system_osh_cohort_prev_hosp_ids'],
            cohort_ids['in_system_osh_cohort_hosp_ids'], cohort_ids['direct_icu_prev_hosp_ids'],
            direct_icu_transfer_hosp_ids, verbose
        )

        # ====================================================================
        # STAGE 6: MORTALITY RATE
        # ====================================================================
        print_section("Stage 6: Mortality rate")

        mortality_summary = run_mortality_analysis(
            hosp, cohort_ids['osh_hosp_ids'], cohort_ids['in_system_osh_cohort_hosp_ids'],
            direct_icu_transfer_hosp_ids, verbose
        )

        # ====================================================================
        # STAGE 7: DISCHARGE CATEGORY SUMMARY
        # ====================================================================
        print_section("Stage 7: Discharge category summary")

        discharge_results = run_discharge_analysis(
            hosp, cohort_ids['osh_hosp_ids'], cohort_ids['in_system_osh_cohort_prev_hosp_ids'],
            cohort_ids['in_system_osh_cohort_hosp_ids'], cohort_ids['direct_icu_prev_hosp_ids'],
            direct_icu_transfer_hosp_ids, in_system_osh_cohort_pairs, verbose
        )

        # ====================================================================
        # STAGE 8: SAVE OUTPUTS
        # ====================================================================
        print_section("Stage 8: Save outputs")

        save_outputs(
            output_dir, site_name, cohort_summary, icu_los_results['icu_los_summary'],
            hosp_los_summary, icu_type_hospital_type_results, mortality_summary,
            discharge_results, verbose
        )

        print_section("ANALYSIS COMPLETED SUCCESSFULLY")
        print(f"Output files saved to: {output_dir}")

        return 0

    except FileNotFoundError as e:
        print(f"\nERROR: File not found: {e}")
        return 1

    except KeyError as e:
        print(f"\nERROR: Missing required data field: {e}")
        traceback.print_exc()
        return 1

    except ValueError as e:
        print(f"\nERROR: Invalid data or configuration: {e}")
        traceback.print_exc()
        return 1

    except Exception as e:
        print(f"\nUNEXPECTED ERROR: {type(e).__name__}: {e}")
        traceback.print_exc()
        return 1

    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        if log_file:
            log_file.close()
            print(f"\nLog saved to: {args.log_file}")


# ============================================================================
# SCRIPT ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    sys.exit(main())
