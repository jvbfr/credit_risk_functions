import numpy as np
import pandas as pd


def class_conditional_distribution(df: pd.DataFrame, target_col: str, analysis_col: str) -> pd.DataFrame:
    """
    Generates a distributional analysis of a numeric column, grouped by a binary target variable.

    This function replicates a common SAS output, providing key descriptive statistics
    for a variable, segmented by a target variable's levels (0 and 1), plus an
    overall summary.

    Args:
        df (pd.DataFrame): The input DataFrame containing the data.
        target_col (str): The name of the binary target column (should contain 0s and 1s).
        analysis_col (str): The name of the numeric column to be analyzed.

    Returns:
        pd.DataFrame: A DataFrame containing the distributional analysis with statistics
                      like mean, median, standard deviation, skewness, kurtosis, etc.,
                      for each target level and overall.
    """
    # --- Define the aggregation functions ---
    # We use a dictionary to specify the calculations for the .agg() method.
    aggregations = {
        'Median': ('median'),
        'Minimum': ('min'),
        'Maximum': ('max'),
        'Mean': ('mean'),
        'Standard Deviation': ('std'),
        'Skewness': ('skew'),
        # kurt() is the pandas equivalent of kurtosis
        'Kurtosis': (lambda x: x.kurt()),
    }

    # --- Perform grouped analysis using groupby() and agg() ---
    # This calculates all the statistics for each target level at once.
    grouped_analysis = df.groupby(target_col)[analysis_col].agg(**aggregations).reset_index()

    # --- Calculate Missing and Non-Missing counts separately ---
    # The .agg() method doesn't have a simple way to count nulls, so we do it here.
    missing_counts = df.groupby(target_col)[analysis_col].apply(lambda x: x.isnull().sum()).reset_index(name='Missing')
    non_missing_counts = df.groupby(target_col)[analysis_col].apply(lambda x: x.notnull().sum()).reset_index(name='Non Missing')

    # --- Merge the calculated stats into one DataFrame ---
    result_df = pd.merge(grouped_analysis, missing_counts, on=target_col)
    result_df = pd.merge(result_df, non_missing_counts, on=target_col)

    # Rename the target column to 'Target Level' for clarity
    result_df = result_df.rename(columns={target_col: 'Target Level'})
    result_df.insert(0, 'Target', target_col) # Add the 'Target' column name

    # --- Calculate Overall Statistics for the entire column ---
    overall_stats = {
        'Target': target_col,
        'Target Level': '_OVERALL_',
        'Median': df[analysis_col].median(),
        'Missing': df[analysis_col].isnull().sum(),
        'Non Missing': df[analysis_col].notnull().sum(),
        'Minimum': df[analysis_col].min(),
        'Maximum': df[analysis_col].max(),
        'Mean': df[analysis_col].mean(),
        'Standard Deviation': df[analysis_col].std(),
        'Skewness': df[analysis_col].skew(),
        'Kurtosis': df[analysis_col].kurtosis()
    }
    overall_df = pd.DataFrame([overall_stats])

    # --- Combine the grouped results with the overall results ---
    final_df = pd.concat([result_df, overall_df], ignore_index=True)

    # --- Reorder columns to match the desired SAS output format ---
    column_order = [
        'Target', 'Target Level', 'Median', 'Missing', 'Non Missing',
        'Minimum', 'Maximum', 'Mean', 'Standard Deviation', 'Skewness', 'Kurtosis'
    ]
    final_df = final_df[column_order]

    return final_df

# --- Example Usage ---
# Call the analysis function
# analysis_result = analyze_distribution(sample_df, target_col='BAD', analysis_col='CLAGE')

# Print the resulting analysis table
# print("--- Distribution Analysis Result ---")
# print(analysis_result.to_string())



