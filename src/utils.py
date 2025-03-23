import numpy as np
def extract_num_columns(df):
    """
    extract numerical column for numerical analysis 
    input: Pandas DataFrame
    output: A Pandas Index object
    """
    num_cols = df.columns[df.dtypes!='object']
    return num_cols


def extract_cat_columns(df):
    """
    extract categorical column for analysis 
    input: Pandas DataFrame
    output: A Pandas Index object
    """
    cat_cols = df.columns[df.dtypes=='object']
    return cat_cols

def make_float(df, columns):
    """Converts the integer data to the float type
    Input:
    df: Pandas DataFrame
    columns: pandas index object
    Output: pandas DataFrame with float type
    """
    for col in columns:
        df[col] = df[col].astype(float)
    return df


def transform_column(column):
    """Reducing the skewness by applying the transformation
    Input: pandas column
    Output: pandas column
    """
    return np.pow(column, 0.55)


def encode_fit_transform(df, columns):
    # Initialize a dictionary to save the encoders
    encoders = {}
    # Apply the encoding to the training data.
    for column in columns:
        if column != "SubscriptionType":
            encoder = LabelEncoder()
            df[column] = encoder.fit_transform(df[column])
        else:
            encoder = OrdinalEncoder(categories='auto')
            df[column] = encoder.fit_transform(df[[column]]) #fit transform expects a 2d array.
    
    
        encoders[column] = encoder  # Save the individual encoder
        return df , encoders


def encode_transform(df, columns, encoders):
     # Apply the encoding to the training data.
    for column in columns:
        encoder = encoders[column]
        if column != "SubscriptionType":
            df[column] = encoder.transform(df[column])
        else:
            df[column] = encoder.transform(df[[column]]) #fit transform expects a 2d array.
    
    return df 


    


    