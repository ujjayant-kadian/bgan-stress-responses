# Preprocessing and Feature Analysis Report

## Dataset Overview
- Total number of samples: 2329
- Label distribution:
  * Label 1.0: 1252 samples (53.8%)
  * Label 2.0: 699 samples (30.0%)
  * Label 3.0: 378 samples (16.2%)

## Signal Properties
### EDA (Electrodermal Activity)
- Sampling rate: 4 Hz
- Processing steps applied:
  * Butterworth low-pass filter
  * Outlier removal
  * Z-score normalization
- Distribution characteristics:
  * Bimodal distribution suggesting distinct physiological states
  * Range: approximately -2 to 4 standard deviations
  * Two peaks: one around -1 and another around 2

### BVP (Blood Volume Pulse)
- Sampling rate: 64 Hz
- Processing steps applied:
  * Peak detection for RR intervals
  * HRV feature extraction

## Feature Analysis
### Heart Rate Variability (HRV) Metrics
1. RMSSD (Root Mean Square of Successive Differences)
   - Normal distribution centered around 0.3 seconds
   - Indicates beat-to-beat variability
   - Range: 0.0 to 0.6 seconds

2. SDNN (Standard Deviation of NN Intervals)
   - Normal distribution centered around 0.2 seconds
   - Represents overall heart rate variability
   - Slightly right-skewed distribution

## Feature Correlations
### Strong Correlations (> 0.7)
- RMSSD & SDNN (0.92): Expected as both measure HRV
- mean_EDA & median_EDA (1.00): Different statistics of same signal

### Moderate Correlations (0.3 - 0.7)
- HF with RMSSD (0.61) and SDNN (0.62)
- SCR_count with mean_EDA/median_EDA (0.46/0.45)

### Weak Correlations (< 0.3)
- Most EDA features with HRV features
- LF_HF_ratio with most other features

## Data Quality Metrics
- Signal length consistency:
  * EDA: 110.5 $\pm$ 0.5 samples
  * BVP: 1768.4 $\pm$ 0.5 samples

## Recommendations
1. Consider feature selection due to high correlations between some metrics
2. The bimodal EDA distribution suggests potential for effective stress state classification
3. Independent nature of EDA and HRV features suggests value in using both for classification
4. Consider analyzing the relationship between labels and feature distributions
