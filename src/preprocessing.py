import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from src import utils

def preprocess_data(data, config=None):
    # """
    # Preprocess the data by setting the index and encoding categorical columns.

    # Parameters:
    # data (pd.DataFrame): The raw data to be processed.
    # config (dict): Configuration dictionary containing preprocessing settings.

    # Returns:
    # pd.DataFrame: The processed data with encoded categorical variables.
    # """
    # if config is None:
    #     config = utils.get_config()

    # # Set the index
    # id_column = config["raw_data"]["id_column"]
    # data = data.set_index(id_column)

    # # Handle categorical columns
    # categorical_cols = config["raw_data"]["categorical_cols"]

    # # Encode categorical columns
    # enc = OneHotEncoder(drop="first", sparse_output=False)
    # encoded_data = enc.fit_transform(data[categorical_cols])
    
    # # Create a DataFrame with the encoded columns
    # encoded_df = pd.DataFrame(
    #     encoded_data, 
    #     index=data.index, 
    #     columns=enc.get_feature_names_out(categorical_cols)
    # )
    
    # # Drop the original categorical columns and add the encoded ones
    # data = data.drop(columns=categorical_cols)
    # data = pd.concat([data, encoded_df], axis=1)

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
    non_binary_cols = [col for col in categorical_cols if col not in binary_cols]

    # Handle binary columns
    for col in binary_cols:
        data[col] = (data[col] == data[col].value_counts().index[0]).astype(int)

    # Encode non-binary categorical columns
    if non_binary_cols:
        # enc = OneHotEncoder(drop="first", sparse_output=False)
        enc = OneHotEncoder(sparse_output=False)
        encoded_data = enc.fit_transform(data[non_binary_cols])
        feature_names = enc.get_feature_names_out(non_binary_cols)
        encoded_df = pd.DataFrame(encoded_data, index=data.index, columns=feature_names)
        data = pd.concat([data.drop(columns=non_binary_cols), encoded_df], axis=1)

    # Make the target to be the last columns
    target_cols = config["raw_data"]["target_cols"]
    data = utils.move_target_to_last(data, target_cols)

    # # This code reduces memory usage. Reference: https://www.kaggle.com/code/jmascacibar/optimizing-memory-usage-with-insurance-cross-sell/notebook
    data['Vehicle_Age_1-2 Year'] = data['Vehicle_Age_1-2 Year'].astype('int8')
    data['Vehicle_Age_< 1 Year'] = data['Vehicle_Age_< 1 Year'].astype('int8') 
    data['Vehicle_Age_> 2 Years'] = data['Vehicle_Age_> 2 Years'].astype('int8')
    data['Gender'] = data['Gender'].astype('int8')
    data['Vehicle_Damage'] = data['Vehicle_Damage'].astype('int8')
    data['Age'] = data['Age'].astype('int8')
    data['Driving_License'] = data['Driving_License'].astype('int8')
    data['Region_Code'] = data['Region_Code'].astype('int8')
    data['Previously_Insured'] = data['Previously_Insured'].astype('int8')
    data['Annual_Premium'] = data['Annual_Premium'].astype('int32')
    data['Policy_Sales_Channel'] = data['Policy_Sales_Channel'].astype('int16')
    data['Vintage'] = data['Vintage'].astype('int16')
    
    if 'Response' in data.columns:
        data['Response'] = data['Response'].astype('int8')

    return data
