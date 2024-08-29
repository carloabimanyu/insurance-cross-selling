import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from src import utils

def preprocess_data(data, config=None):
    """
    Preprocess the data by setting the index and encoding categorical columns.

    Parameters:
    data (pd.DataFrame): The raw data to be processed.
    config (dict): Configuration dictionary containing preprocessing settings.

    Returns:
    pd.DataFrame: The processed data with encoded categorical variables.
    """
    if config is None:
        config = utils.get_config()

    # Set the index
    id_column = config["raw_data"]["id_column"]
    data = data.set_index(id_column)

    # Handle categorical columns
    categorical_cols = config["raw_data"]["categorical_cols"]

    # Encode categorical columns
    enc = OneHotEncoder(drop="first", sparse_output=False)
    encoded_data = enc.fit_transform(data[categorical_cols])
    
    # Create a DataFrame with the encoded columns
    encoded_df = pd.DataFrame(
        encoded_data, 
        index=data.index, 
        columns=enc.get_feature_names_out(categorical_cols)
    )
    
    # Drop the original categorical columns and add the encoded ones
    data = data.drop(columns=categorical_cols)
    data = pd.concat([data, encoded_df], axis=1)

    # Make the target to be the last columns
    target_cols = config["raw_data"]["target_cols"]
    data = utils.move_target_to_last(data, target_cols)

    return data
