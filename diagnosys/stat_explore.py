import numpy as np
import pandas as pd
from scipy import stats


def print_stat_explore_results(results_df):
    """
    Print results in a format similar to SAS StatExplore output
    """
    print("Variable Analysis Summary")
    print("=" * 120)
    
    # Print header
    header = f"{'Variable':<12} {'Role':<8} {'Mean':<12} {'Standard':<12} {'Non':<8} {'Missing':<8} {'Minimum':<12} {'Median':<12} {'Maximum':<12} {'Skewness':<12} {'Kurtosis':<12}"
    print(header)
    
    subheader = f"{'':>12} {'':>8} {'':>12} {'Deviation':<12} {'Missing':<8} {'':>8} {'':>12} {'':>12} {'':>12} {'':>12} {'':>12}"
    print(subheader)
    
    print("-" * 120)
    
    # Print each row
    for _, row in results_df.iterrows():
        # Format numbers appropriately
        mean_str = f"{row['Mean']:.5f}" if not pd.isna(row['Mean']) else "."
        std_str = f"{row['Standard Deviation']:.5f}" if not pd.isna(row['Standard Deviation']) else "."
        min_str = f"{row['Minimum']:.0f}" if not pd.isna(row['Minimum']) else "."
        median_str = f"{row['Median']:.0f}" if not pd.isna(row['Median']) else "."
        max_str = f"{row['Maximum']:.0f}" if not pd.isna(row['Maximum']) else "."
        skew_str = f"{row['Skewness']:.5f}" if not pd.isna(row['Skewness']) else "."
        kurt_str = f"{row['Kurtosis']:.5f}" if not pd.isna(row['Kurtosis']) else "."
        
        line = f"{row['Variable']:<12} {row['Role']:<8} {mean_str:<12} {std_str:<12} {row['Non Missing']:<8} {row['Missing']:<8} {min_str:<12} {median_str:<12} {max_str:<12} {skew_str:<12} {kurt_str:<12}"
        print(line)
    
    print("-" * 120)
    print(f"Total variables analyzed: {len(results_df)}")


def stat_explore(df, role_column=None):
    """
    Replicates SAS StatExplore procedure for analyzing numeric variables in a DataFrame.
    Provides summary statistics for all numeric columns similar to SAS output.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame to analyze
    role_column : str, optional
        Name of a column to use as 'Role' (will be set to 'INPUT' for all variables)
    
    Returns:
    --------
    pandas.DataFrame : DataFrame containing summary statistics for all numeric variables
    """
    
    # Get only numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) == 0:
        raise ValueError("No numeric columns found in DataFrame")
    
    # Initialize results list
    results = []
    
    for col in numeric_cols:
        data = df[col]
        clean_data = data.dropna()
        
        # Basic counts
        n_total = len(data)
        n_valid = len(clean_data)
        n_missing = n_total - n_valid
        
        if n_valid == 0:
            # Handle case where all values are missing
            stats_dict = {
                'Variable': col,
                'Role': 'INPUT',
                'Mean': np.nan,
                'Standard Deviation': np.nan,
                'Non Missing': 0,
                'Missing': n_missing,
                'Minimum': np.nan,
                'Median': np.nan,
                'Maximum': np.nan,
                'Skewness': np.nan,
                'Kurtosis': np.nan
            }
        else:
            # Calculate statistics
            mean_val = clean_data.mean()
            std_val = clean_data.std(ddof=1)  # Sample standard deviation
            min_val = clean_data.min()
            median_val = clean_data.median()
            max_val = clean_data.max()
            skewness_val = stats.skew(clean_data, bias=False)
            kurtosis_val = stats.kurtosis(clean_data, bias=False, fisher=False)  # Pearson's kurtosis
            
            stats_dict = {
                'Variable': col,
                'Role': 'INPUT',
                'Mean': mean_val,
                'Standard Deviation': std_val,
                'Non Missing': n_valid,
                'Missing': n_missing,
                'Minimum': min_val,
                'Median': median_val,
                'Maximum': max_val,
                'Skewness': skewness_val,
                'Kurtosis': kurtosis_val
            }
        
        results.append(stats_dict)
    
    # Create DataFrame from results
    results_df = pd.DataFrame(results)
    
    # Print results in SAS-like format
    print_stat_explore_results(results_df)
    
    return results_df

# Example usage:
# results = stat_explore(df)
# 
# # Access specific statistics
# print(results[['Variable', 'Mean', 'Standard Deviation']])
# 
# # Get statistics for a specific variable
# loan_stats = results[results['Variable'] == 'LOAN']