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

    # Separate binary and non-binary categorical columns
    binary_cols = [col for col in categorical_cols if data[col].nunique() == 2]
    non_binary_cols = [col for col in categorical_cols if col not in binary_cols and col != "Vehicle_Age"]

    # Handle binary columns
    for col in binary_cols:
        data[col] = (data[col] == data[col].value_counts().index[0]).astype(int)

    # Encode non-binary categorical columns
    if non_binary_cols:
        enc = OneHotEncoder(sparse_output=False)
        encoded_data = enc.fit_transform(data[non_binary_cols])
        feature_names = enc.get_feature_names_out(non_binary_cols)
        encoded_df = pd.DataFrame(encoded_data, index=data.index, columns=feature_names)
        data = pd.concat([data.drop(columns=non_binary_cols), encoded_df], axis=1)

    # This code reduces memory usage. Reference: https://www.kaggle.com/code/jmascacibar/optimizing-memory-usage-with-insurance-cross-sell/notebook
    data['Vehicle_Age'] = data['Vehicle_Age'].map({"< 1 Year": 0, "1-2 Year": 1, "> 2 Years": 2}).astype('int8')
    data['Gender'] = data['Gender'].astype('int8')
    data['Vehicle_Damage'] = data['Vehicle_Damage'].astype('int8')
    data['Age'] = data['Age'].astype('int8')
    data['Driving_License'] = data['Driving_License'].astype('int8')
    data['Region_Code'] = data['Region_Code'].astype('int8')
    data['Previously_Insured'] = data['Previously_Insured'].astype('int8')
    data['Annual_Premium'] = data['Annual_Premium'].astype('int32')
    data['Policy_Sales_Channel'] = data['Policy_Sales_Channel'].astype('int16')
    data['Vintage'] = data['Vintage'].astype('int16')
    
    if config["target_cols"] in data.columns:
        # Make the target to be the last columns
        target_cols = config["target_cols"]
        data = utils.move_target_to_last(data, target_cols)
        data[target_cols] = data[target_cols].astype('int8')

    return data
