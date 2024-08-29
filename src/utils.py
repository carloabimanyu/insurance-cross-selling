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