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


    