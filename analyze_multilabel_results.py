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

    # Calculate average number of top N occupations in each row
    # Calculate average number of top N occupations in each row
    averages = []
    max_occupations = len(occupation_counts)
    for N in range(1, max_occupations + 1):
        top_N = occupation_counts.index[:N].tolist()
    
        # Rows containing at least one of the top N occupations
        rows_with_top_N = df['occupation'].apply(lambda x: any(occ in top_N for occ in x))
        relevant_rows = df[rows_with_top_N]
    
        # Average number of top N occupations in those rows
        counts_per_row = relevant_rows['occupation'].apply(lambda x: sum(1 for occ in x if occ in top_N))
        avg = counts_per_row.sum() / len(relevant_rows)
        averages.append(avg)

    result_df = pd.DataFrame({
        'Count': occupation_counts.values,
        'Cumulative Rows': cumulative_counts,
        'Average Occupations per Listing': averages
    }, index=occupation_counts.index)
    
    result_df['Adjusted Average'] = result_df['Average Occupations per Listing'] * result_df['Cumulative Rows'] / len(df)


    return result_df

df = read_data_from_file("../data/all_cols_sample.pkl")
result = get_results(df)
print(result)
