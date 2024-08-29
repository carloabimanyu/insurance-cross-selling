import os
import joblib
import json
import yaml
import pandas as pd
from typing import List, Dict, Union

def get_config():
    """
    Retrieves the configuration from the config.yaml file.

    Returns:
        dict: The configuration dictionary.
    """
    config_path = os.path.join(os.path.dirname(__file__), "../config/config.yaml")
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def move_target_to_last(data: Union[pd.DataFrame, pd.Series], target_cols_name: str) -> pd.DataFrame:
    """
    Efficiently move the specified target column to the last position in a DataFrame
    optimized for DataFrames with many rows but few columns.

    Args:
        data (Union[pd.DataFrame, pd.Series]): The input DataFrame or Series.
        target_cols_name (str): The name of the column to be moved to the last position.

    Returns:
        pd.DataFrame: A new DataFrame with the target column as the last column.

    Raises:
        TypeError: If the input data is not a pandas DataFrame or Series.
        ValueError: If the target column is not found in the DataFrame.
        ValueError: If the input is a Series and doesn't match the target column name.
    """
    # Check if input is a pandas DataFrame or Series
    if not isinstance(data, (pd.DataFrame, pd.Series)):
        raise TypeError("Input data must be a pandas DataFrame or Series.")

    # Handle Series input
    if isinstance(data, pd.Series):
        if data.name != target_cols_name:
            raise ValueError(f"Series name '{data.name}' does not match target column name '{target_cols_name}'.")
        return pd.DataFrame(data)

    # Check if target column exists in the DataFrame
    if target_cols_name not in data.columns:
        raise ValueError(f"Column '{target_cols_name}' not found in the dataset.")

    # If the target column is already at the end, return the DataFrame as is
    if data.columns[-1] == target_cols_name:
        return data

    # For few columns, direct indexing is efficient
    cols = data.columns.tolist()
    cols.append(cols.pop(cols.index(target_cols_name)))

    # Use loc for efficient column reordering
    return data.loc[:, cols]

def initial_check(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform an initial check on a pandas DataFrame, analyzing each column's properties.

    This function examines each column in the input DataFrame and collects information
    about its data type, null values, unique values, and other type-specific properties.

    Args:
        df (pd.DataFrame): The input DataFrame to be analyzed.

    Returns:
        pd.DataFrame: A DataFrame containing the analysis results for each column.

    The returned DataFrame includes the following columns:
    - Column: Name of the column in the original DataFrame.
    - Type: Data type of the column.
    - Null Count: Number of null values in the column.
    - Unique Count: Number of unique values in the column.
    - Min: Minimum value (for numeric columns only).
    - Max: Maximum value (for numeric columns only).
    - Empty String Count: Number of empty strings (for string columns only).
    """
    result: List[Dict] = []
    
    # Precompute null counts and unique counts for all columns
    null_counts = df.isnull().sum()
    unique_counts = df.nunique()

    for col in df.columns:
        col_data = df[col]
        dtype = col_data.dtype
        
        column_info = {
            'Column': col,
            'Type': dtype,
            'Null Count': null_counts[col],
            'Unique Count': unique_counts[col]
        }

        if pd.api.types.is_numeric_dtype(dtype):
            column_info.update({
                'Min': col_data.min(),
                'Max': col_data.max()
            })
        elif pd.api.types.is_string_dtype(dtype):
            column_info['Empty String Count'] = (col_data == '').sum()

        result.append(column_info)
    
    return pd.DataFrame(result)