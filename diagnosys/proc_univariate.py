import pandas as pd
import numpy as np
from scipy import stats

def proc_univariate(data, variable_name="Variable"):
    """
    Replicates SAS PROC UNIVARIATE procedure for a pandas Series or array-like data.
    Prints formatted results similar to SAS output and returns dictionary of statistics.
    
    Parameters:
    -----------
    data : pandas.Series or array-like
        The data to analyze (single column/variable only)
    variable_name : str, default "Variable"
        Name of the variable for display purposes
    
    Returns:
    --------
    dict : Dictionary containing all computed statistics
    """
    
    # Convert to pandas Series if not already
    if not isinstance(data, pd.Series):
        data = pd.Series(data)
    
    # Remove missing values
    clean_data = data.dropna()
    n = len(clean_data)
    
    if n == 0:
        raise ValueError("No valid observations found")
    
    # Sort data for quantile calculations
    sorted_data = np.sort(clean_data)
    
    # Basic statistics
    mean = clean_data.mean()
    median = clean_data.median()
    mode_result = stats.mode(clean_data, keepdims=True)
    mode = mode_result.mode[0] if len(mode_result.mode) > 0 else np.nan
    
    # Variability measures
    std_dev = clean_data.std(ddof=1)  # Sample standard deviation
    variance = clean_data.var(ddof=1)  # Sample variance
    data_range = clean_data.max() - clean_data.min()
    q1 = clean_data.quantile(0.25)
    q3 = clean_data.quantile(0.75)
    iqr = q3 - q1
    
    # Sum of weights (assuming equal weights)
    sum_weights = n
    sum_observations = clean_data.sum()
    
    # Moments
    skewness = stats.skew(clean_data, bias=False)
    kurtosis = stats.kurtosis(clean_data, bias=False, fisher=False)  # Pearson's kurtosis
    
    # Sum of squares
    uncorrected_ss = np.sum(clean_data**2)
    corrected_ss = np.sum((clean_data - mean)**2)
    
    # Coefficient of variation
    coeff_variation = (std_dev / mean) * 100 if mean != 0 else np.nan
    
    # Standard error of mean
    std_error_mean = std_dev / np.sqrt(n)
    
    # Tests for location (H0: μ = 0)
    # Student's t-test
    t_stat = mean / std_error_mean
    t_pvalue = 2 * (1 - stats.t.cdf(abs(t_stat), n-1))
    
    # Sign test
    positive_count = np.sum(clean_data > 0)
    sign_pvalue = 2 * stats.binom.cdf(min(positive_count, n - positive_count), n, 0.5)
    
    # Signed rank test (Wilcoxon)
    try:
        signed_rank_stat, signed_rank_pvalue = stats.wilcoxon(clean_data, alternative='two-sided')
    except:
        signed_rank_stat, signed_rank_pvalue = np.nan, np.nan
    
    # Quantiles
    quantiles = {
        '0% Min': clean_data.min(),
        '1%': clean_data.quantile(0.01),
        '5%': clean_data.quantile(0.05),
        '10%': clean_data.quantile(0.10),
        '25% Q1': q1,
        '50% Median': median,
        '75% Q3': q3,
        '90%': clean_data.quantile(0.90),
        '95%': clean_data.quantile(0.95),
        '99%': clean_data.quantile(0.99),
        '100% Max': clean_data.max()
    }
    
    # Extreme observations
    # Get 5 lowest and 5 highest values with their indices
    sorted_with_index = clean_data.sort_values()
    lowest_extreme = []
    highest_extreme = []
    
    # Get 5 lowest values
    for i in range(min(5, len(sorted_with_index))):
        value = sorted_with_index.iloc[i]
        obs_index = sorted_with_index.index[i]
        lowest_extreme.append({'value': value, 'obs': obs_index})
    
    # Get 5 highest values
    for i in range(min(5, len(sorted_with_index))):
        value = sorted_with_index.iloc[-(i+1)]
        obs_index = sorted_with_index.index[-(i+1)]
        highest_extreme.append({'value': value, 'obs': obs_index})
    
    # Reverse highest_extreme to show in descending order
    highest_extreme.reverse()
    
    # Print results in SAS PROC UNIVARIATE format
    print(f"The UNIVARIATE Procedure")
    print(f"Variable: {variable_name}")
    print(f"{'':>50}Moments")
    print(f"{'N':<15} {n:>15} {'Sum Weights':<15} {sum_weights:>15}")
    print(f"{'Mean':<15} {mean:>15.1f} {'Sum Observations':<15} {sum_observations:>15.0f}")
    print(f"{'Std Deviation':<15} {std_dev:>15.6f} {'Variance':<15} {variance:>15.0f}")
    print(f"{'Skewness':<15} {skewness:>15.8f} {'Kurtosis':<15} {kurtosis:>15.8f}")
    print(f"{'Uncorrected SS':<15} {uncorrected_ss:>15.0f} {'Corrected SS':<15} {corrected_ss:>15.0f}")
    print(f"{'Coeff Variation':<15} {coeff_variation:>15.6f} {'Std Error Mean':<15} {std_error_mean:>15.6f}")
    
    print(f"\n{'':>25}Basic Statistical Measures")
    print(f"{'':>15}Location{'':>20}Variability")
    print(f"{'Mean':<8} {mean:>10.2f} {'Std Deviation':<15} {std_dev:>10.0f}")
    print(f"{'Median':<8} {median:>10.2f} {'Variance':<15} {variance:>10.0f}")
    print(f"{'Mode':<8} {mode:>10.2f} {'Range':<15} {data_range:>10.0f}")
    print(f"{'':>25} {'Interquartile Range':<15} {iqr:>10.0f}")
    
    print(f"\n{'':>15}Tests for Location: Mu0=0")
    print(f"{'Test':<15} {'Statistic':<15} {'p Value':<15}")
    print(f"{'Student\\'s t':<15} t {t_stat:>10.5f} {'Pr > |t|':<10} {t_pvalue:>8.4f}")
    print(f"{'Sign':<15} M {positive_count:>10.0f} {'Pr >= |M|':<10} {sign_pvalue:>8.4f}")
    if not np.isnan(signed_rank_stat):
        print(f"{'Signed Rank':<15} S {signed_rank_stat:>10.0f} {'Pr >= |S|':<10} {signed_rank_pvalue:>8.4f}")
    
    print(f"\n{'':>15}Quantiles (Definition 5)")
    print(f"{'Level':<15} {'Quantile':<15}")
    for level, value in quantiles.items():
        print(f"{level:<15} {value:>10.0f}")
    
    print(f"\n{'':>15}Extreme Observations")
    print(f"{'':>10}Lowest{'':>20}Highest")
    print(f"{'Value':<10} {'Obs':<10} {'Value':<10} {'Obs':<10}")
    for i in range(5):
        if i < len(lowest_extreme) and i < len(highest_extreme):
            low = lowest_extreme[i]
            high = highest_extreme[i]
            print(f"{low['value']:<10.0f} {low['obs']:<10} {high['value']:<10.0f} {high['obs']:<10}")
        elif i < len(lowest_extreme):
            low = lowest_extreme[i]
            print(f"{low['value']:<10.0f} {low['obs']:<10} {'':>20}")
        elif i < len(highest_extreme):
            high = highest_extreme[i]
            print(f"{'':>20} {high['value']:<10.0f} {high['obs']:<10}")
    
    # Return dictionary with all results
    return {
        'variable_name': variable_name,
        'n': n,
        'mean': mean,
        'std_dev': std_dev,
        'skewness': skewness,
        'uncorrected_ss': uncorrected_ss,
        'coeff_variation': coeff_variation,
        'sum_weights': sum_weights,
        'sum_observations': sum_observations,
        'variance': variance,
        'kurtosis': kurtosis,
        'corrected_ss': corrected_ss,
        'std_error_mean': std_error_mean,
        'median': median,
        'mode': mode,
        'range': data_range,
        'iqr': iqr,
        't_statistic': t_stat,
        't_pvalue': t_pvalue,
        'sign_statistic': positive_count,
        'sign_pvalue': sign_pvalue,
        'signed_rank_statistic': signed_rank_stat,
        'signed_rank_pvalue': signed_rank_pvalue,
        'quantiles': quantiles,
        'extreme_observations': {
            'lowest': lowest_extreme,
            'highest': highest_extreme
        }
    }

# Example usage:
# results = proc_univariate(df['loan_amount'], "LOAN")
# results = proc_univariate(data_array, "VARIABLE_NAME")