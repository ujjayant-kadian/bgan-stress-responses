import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set up Matplotlib for better visuals
plt.rcParams.update({'figure.figsize': (10, 5), 'axes.grid': True})

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PROCESSED = PROJECT_ROOT / "data/processed"
FIGURES_DIR = PROJECT_ROOT / "reports/figures"
# Create figures directory if it doesn't exist
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
#                     LOAD PREPROCESSED DATA FUNCTION
# =============================================================================
def load_preprocessed_data(file_path):
    """
    Loads preprocessed physiological data from a pickle file.
    
    Args:
        file_path (Path or str): Path to the preprocessed data file.
    
    Returns:
        data (list): List of dictionaries with segmented time-series samples.
    """
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    return data

# =============================================================================
#                     PLOT TIME-SERIES RESPONSES
# =============================================================================
def plot_time_series(sample, save=True):
    """
    Modified plotting function to properly handle different sampling rates and save plots
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # Create proper time axes based on sampling rates
    eda_fs = 4  # EDA sampling rate from preprocessing.py
    bvp_fs = 64  # BVP sampling rate from preprocessing.py
    
    eda_time = np.arange(len(sample["EDA_series"])) / eda_fs
    bvp_time = np.arange(len(sample["BVP_series"])) / bvp_fs
    
    # Plot EDA
    ax1.plot(eda_time, sample["EDA_series"], color='blue', label="EDA (Z-scored)")
    ax1.set_ylabel("EDA (normalized)")
    ax1.grid(True)
    ax1.legend()
    
    # Plot BVP
    ax2.plot(bvp_time, sample["BVP_series"], color='red', label="BVP")
    ax2.set_xlabel("Time (seconds)")
    ax2.set_ylabel("BVP Signal")
    ax2.grid(True)
    ax2.legend()
    
    # Add label to the title
    label = sample.get("label", "Unknown")
    plt.suptitle(f"EDA and BVP Time-Series (Label: {label})")
    plt.tight_layout()
    
    if save:
        plt.savefig(FIGURES_DIR / f'time_series_plot_label_{label}.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

# =============================================================================
#             PLOT STATISTICAL DISTRIBUTIONS (EDA, HRV METRICS)
# =============================================================================
def plot_feature_distributions(data, save=True):
    """
    Plots histograms of EDA and HRV features and optionally saves them
    """
    eda_means = [sample['EDA_features']['mean_EDA'] for sample in data if 'EDA_features' in sample]
    rmssd_values = [sample['HRV_metrics']['RMSSD'] for sample in data if 'HRV_metrics' in sample]
    sdnn_values = [sample['HRV_metrics']['SDNN'] for sample in data if 'HRV_metrics' in sample]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    sns.histplot(eda_means, bins=30, kde=True, ax=axes[0], color="blue")
    axes[0].set_title("EDA Mean Distribution")
    
    sns.histplot(rmssd_values, bins=30, kde=True, ax=axes[1], color="green")
    axes[1].set_title("HRV RMSSD Distribution")
    
    sns.histplot(sdnn_values, bins=30, kde=True, ax=axes[2], color="red")
    axes[2].set_title("HRV SDNN Distribution")

    plt.tight_layout()
    
    if save:
        plt.savefig(FIGURES_DIR / 'feature_distributions.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

# =============================================================================
#                     CORRELATION HEATMAP OF FEATURES
# =============================================================================
def plot_feature_correlation(data, save=True):
    """
    Generates and optionally saves a correlation heatmap of extracted physiological features
    """
    feature_list = []
    for sample in data:
        features = {}
        if "HRV_metrics" in sample:
            features.update(sample["HRV_metrics"])
        if "EDA_features" in sample:
            features.update(sample["EDA_features"])
        feature_list.append(features)

    feature_df = pd.DataFrame(feature_list).dropna()
    corr_matrix = feature_df.corr()

    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1)
    plt.title("Feature Correlation Heatmap")
    
    if save:
        plt.savefig(FIGURES_DIR / 'correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

# =============================================================================
#             ADDITIONAL TESTING FUNCTIONS FOR PREPROCESSING QUALITY
# =============================================================================
def test_preprocessing_quality(data):
    """
    Performs quality tests on the preprocessed data.
    Prints summary statistics and detects potential issues that
    could indicate problems with preprocessing.
    """
    total_samples = len(data)
    print(f"Total samples loaded: {total_samples}")
    
    # Compute lengths of EDA & BVP series for each sample
    eda_lengths = [len(sample.get("EDA_series", [])) for sample in data]
    bvp_lengths = [len(sample.get("BVP_series", [])) for sample in data]
    
    print(f"Avg EDA series length: {np.mean(eda_lengths):.2f} (min: {np.min(eda_lengths)}, max: {np.max(eda_lengths)})")
    print(f"Avg BVP series length: {np.mean(bvp_lengths):.2f} (min: {np.min(bvp_lengths)}, max: {np.max(bvp_lengths)})")
    
    # Check for NaNs in HRV metrics within each sample
    nan_count = 0
    for sample in data:
        if "HRV_metrics" in sample:
            hrv_vals = list(sample["HRV_metrics"].values())
            if any(np.isnan(val) for val in hrv_vals):
                nan_count += 1
    print(f"Samples with NaN HRV metrics: {nan_count} ({(nan_count/total_samples)*100:.1f}%)")
    
    # Check sampling rate consistency
    expected_ratio = 64/4  # BVP_fs/EDA_fs
    actual_ratio = np.mean(bvp_lengths) / np.mean(eda_lengths)
    print(f"BVP/EDA length ratio: {actual_ratio:.2f} (expected: {expected_ratio:.2f})")
    
    # Check time alignment
    for sample in data:
        eda_duration = len(sample["EDA_series"]) / 4  # seconds
        bvp_duration = len(sample["BVP_series"]) / 64  # seconds
        if abs(eda_duration - bvp_duration) > 0.3:  # allow 0.3s difference
            print(f"Warning: Duration mismatch - EDA: {eda_duration:.2f}s, BVP: {bvp_duration:.2f}s")

def save_preprocessing_report(data):
    """
    Saves preprocessing quality test results to a text file
    """
    report_path = FIGURES_DIR / 'preprocessing_report.txt'
    
    total_samples = len(data)
    eda_lengths = [len(sample.get("EDA_series", [])) for sample in data]
    bvp_lengths = [len(sample.get("BVP_series", [])) for sample in data]
    
    nan_count = sum(1 for sample in data if "HRV_metrics" in sample and 
                   any(np.isnan(val) for val in sample["HRV_metrics"].values()))
    
    with open(report_path, 'w') as f:
        f.write("Preprocessing Quality Report\n")
        f.write("==========================\n\n")
        f.write(f"Total samples: {total_samples}\n")
        f.write(f"Average EDA series length: {np.mean(eda_lengths):.2f}\n")
        f.write(f"Average BVP series length: {np.mean(bvp_lengths):.2f}\n")
        f.write(f"Samples with NaN HRV metrics: {nan_count} ({(nan_count/total_samples)*100:.1f}%)\n")
        
        # Add time alignment check
        misaligned = 0
        for sample in data:
            eda_duration = len(sample["EDA_series"]) / 4
            bvp_duration = len(sample["BVP_series"]) / 64
            if abs(eda_duration - bvp_duration) > 0.3:
                misaligned += 1
        f.write(f"Samples with timing misalignment: {misaligned}\n")

# =============================================================================
#                     MAIN FUNCTION TO RUN VISUALIZATIONS & TESTS
# =============================================================================
if __name__ == '__main__':
    # Load combined dataset (change file path if needed)
    file_path = DATA_PROCESSED / "WESAD_combined_time_series.pkl"
    
    if not file_path.exists():
        print(f"Error: File {file_path} not found.")
    else:
        print(f"Loading data from {file_path} ...")
        data = load_preprocessed_data(file_path)
        
        # Create figures directory
        print(f"\nSaving figures to {FIGURES_DIR}")
        
        # Run and save preprocessing quality report
        print("Generating preprocessing report...")
        save_preprocessing_report(data)
        
        # Generate and save plots
        print("Generating and saving plots...")
        plot_time_series(data[1000])
        plot_feature_distributions(data)
        plot_feature_correlation(data)
        
        print(f"\nAll visualizations have been saved to {FIGURES_DIR}")
