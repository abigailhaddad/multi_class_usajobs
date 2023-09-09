# -*- coding: utf-8 -*-
"""
Created on Sat Jul 22 19:16:46 2023

@author: abiga
"""

import pickle
from typing import Any
import pandas as pd
import matplotlib.pyplot as plt

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
    long_df = df[['occupation']].explode('occupation').reset_index()[['occupation', 'index']].drop_duplicates()

    # Get occupation counts
    occupation_counts = long_df['occupation'].value_counts()

    cumulative_counts = []
    top_occupations_so_far = []

    for occupation in occupation_counts.index:
        top_occupations_so_far.append(occupation)
        subset_df = long_df[long_df['occupation'].isin(top_occupations_so_far)]
        unique_rows_count = subset_df['index'].nunique()
        cumulative_counts.append(unique_rows_count)

    result_df = pd.DataFrame({
       'Count': occupation_counts.values,
       'Cumulative Rows': cumulative_counts,
   }, index=occupation_counts.index)
    result_df['cum_count'] = result_df['Count'].cumsum()
   
    result_df['Average']=result_df['cum_count']/len(df)
    #result_df=result_df.drop(columns=['cum_count'])
    return result_df




def plot_scatter(dataframe):
    """Generate scatter plot of verage Average vs. Cumulative Rows."""
    plt.figure(figsize=(10, 6))
    plt.scatter(dataframe.index, dataframe['Average'], color='blue', marker='o')
    plt.title('Adjusted Average vs. Number of Occupations')
    plt.xlabel('Number of Occupation')
    plt.ylabel('Average')
    plt.grid(True)
    plt.tight_layout()
    plt.show()




df = read_data_from_file("../data/all_cols_sample.pkl")
result = get_results(df)
print(result)
# Sample code to call the plotting function:
plot_scatter(result)