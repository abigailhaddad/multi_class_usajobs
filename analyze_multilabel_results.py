# -*- coding: utf-8 -*-
"""
Created on Sat Jul 22 19:16:46 2023

@author: abiga
"""

import pickle
from typing import Any, List
import pandas as pd

def read_data_from_file(file_path: str) -> Any:
    """
    Reads data from the specified file path.

    Args:
        file_path (str): The path of the file to read the data from.

    Returns:
        Any: The data read from the file.
    """
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    return data




def capitalize_words(s):
    """Capitalize the first letter of each word."""
    return ' '.join(word.capitalize() for word in s.split())

def get_results(df):
    # Capitalize occupations
    df['occupation'] = df['occupation'].apply(lambda lst: [capitalize_words(item.lower()) for item in lst])
    

    # Reshape data to long format
    long_df = df.explode('occupation').reset_index()[['occupation', 'index']].drop_duplicates()

    # Get occupation counts
    occupation_counts = long_df['occupation'].value_counts()

    cumulative_counts = []
    top_occupations_so_far = []

    for occupation in occupation_counts.index:
        top_occupations_so_far.append(occupation)
        subset_df = long_df[long_df['occupation'].isin(top_occupations_so_far)]
        unique_rows_count = subset_df['index'].nunique()
        cumulative_counts.append(unique_rows_count)
        print(f"Occupation: {occupation}, Current Count: {subset_df.shape[0]}, Unique Rows: {unique_rows_count}")

    result_df = pd.DataFrame({
    'Count': occupation_counts.values,
    'Cumulative Rows': cumulative_counts
}, index=occupation_counts.index)
    return(result_df)



df=read_data_from_file("../data/all_cols_sample.pkl")
result = get_results(df)