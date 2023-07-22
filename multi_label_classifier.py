# -*- coding: utf-8 -*-
"""
Created on Sat Jul 22 17:02:13 2023

@author: abiga
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Apr  9 18:22:00 2023

@author: abiga
"""

import pickle
from typing import Any, List, Tuple

import numpy as np
import openai
import pandas as pd
import json

from pydantic import BaseModel


class OccupationResponse(BaseModel):
    occupations: List[str]

# Load the API key from the file (change this to reflect where you've put
# your API key)
with open("../key/key.txt", "r") as key_file:
    api_key = key_file.read().strip()
    openai.api_key = api_key


def gen_list_of_occupations() -> List[str]:
    """
    Generates a list of occupation codes (OPM OCC codes) along with their corresponding
    occupational series names as comments.

    Returns:
    list: A list of formatted occupation codes with leading zeros if necessary.
    """

    list_of_raw_occupations = [
        "1529",  # Mathematician
        "1550",  # Computer Scientist
        "1515",  # Operations Research Analyst
        "0110",  # Economist
        "2210",  # Information Technology Management
        "1520",  # Mathematician Statistician
        "1530",  # Statistician
        "0800",  # Engineering
        "0100",  # Social Science, Psychology, and Welfare
        "1510",  # Actuary
        "0150",  # Geography
        "0343",  # Management and Program Analyst
        "0601",  # General Health Science
        "0400",  # Natural Resources Management and Biological Sciences
        "1300",  # General Physical Science
        "0130",  # Foreign Affairs
        "1517",  # Digital Forensics Examiner
        "1340",  # Meteorologist
        "1516",  # Cryptanalyst
        "1531",  # Statistician/Data Scientist
        "0401",  # General Natural Resources Management and Biological Sciences
        "1370",  # Cartographer
        "1372",  # Geodesist
        "1160",  # Financial Analysis
        "8960",  # Production Control
        "0500",  # Accounting and Budget
        "0340",  # Program Management
        "1306",  # Health Physicist
        "0685",  # Public Health Program Specialist
        "0187",  # Social Insurance Administrator
        "1330",  # Physical Scientist
        "1320",  # Chemist
        "1313",  # Geophysics
        "0560",  # Budget Analysis
        "0200",  # Human Resources Management
        "0391",  # Telecommunications
        "1035",  # Public Affairs
        "0690",  # Industrial Hygiene
        "0890",  # Agricultural Commodity Grading
        "1410",  # Librarian
        "1701",  # General Education and Training
        "0801",  # General Engineering
        "2010",  # Inventory Management
        "0809",  # Construction Control
        "1321",  # Metallurgist
        "1371",  # Cartographic Technician
        "0101",  # Social Science
        "0180",  # Psychology
        "0201",  # Human Resources Management
        "0341",  # Administrative Officer
        "0454",  # Soil Conservation
        "0696",  # Consumer Safety
        "0804",  # Fire Protection Engineering
        "0828",  # Equipment Services
        "0850",  # Electrical Engineering
        "0854",  # Computer Engineering
        "1301",  # General Physical Scientist
        "1308",  # Environmental Health
        "0501",  # Financial Administration and Program
        "0570",  # Financial Institutions Examining
        "0106",  # Insurance Accounts
        "0814",  # Mine Safety and Health
        "0193",  # Social Services
    ]
    list_of_occupations = [add_leading_zero(
        i) for i in list_of_raw_occupations]
    return list_of_occupations


def add_leading_zero(occupation_code: str) -> str:
    """
    Adds a leading zero to occupation codes with less than 4 digits.

    Args:
    occupation_code (str): The occupation code to format.

    Returns:
    str: The formatted occupation code with leading zeros if necessary.
    """
    return occupation_code.zfill(4)


def concatenate_columns(data_frame, columns, new_column_name='info'):
    """
    Concatenates the specified columns of a DataFrame into a new 'info' column.

    Args:
        data_frame (pd.DataFrame): The input DataFrame.
        columns (list): A list of column names to concatenate.
        new_column_name: A string with the name of the new column name

    Returns:
        pd.DataFrame: The DataFrame with the new 'new_column_name' column.
    """
    for column in columns:
        if column not in data_frame.columns:
            raise ValueError(f"Column '{column}' not found in the DataFrame.")

    # Convert each column to string and concatenate them into a new column
    data_frame[new_column_name] = data_frame[columns].astype(str).apply(' '.join, axis=1)

    return data_frame


def process_prompt(prompt: str, engine: str, temperature: float, function_name: str = None) -> Any:
    """
    Processes a given prompt using the specified engine, temperature, and function name.

    Args:
        prompt (str): The input prompt.
        engine (str): The engine to be used for processing the prompt.
        temperature (float): The temperature to be used in processing the prompt.
        function_name (str, optional): The function to be called for structured output.

    Returns:
        Any: The structured response generated by the engine for the given function call or plain text if no function_name is provided.
    """
    messages = [{"role": "user", "content": prompt}]

    if function_name:
        try:
            response = openai.ChatCompletion.create(
                model=engine,
                messages=messages,
                functions=[{
                    "name": function_name,
                    "description": "Get top 5 occupations for the job posting",
                    "parameters": OccupationResponse.schema()
                }],
                function_call={"name": function_name},
                max_tokens=1024,
                temperature=temperature
            )
            # Extracting structured result
            output = json.loads(response.choices[0]["message"]["function_call"]["arguments"])
            return OccupationResponse(**output).occupations
        except Exception as e:
            print(f"Error processing prompt with function call. Engine: {engine}, Prompt: {prompt}, Error: {str(e)[:100]}")
            return []
    else:
        try:
            response = openai.ChatCompletion.create(
                model=engine,
                messages=messages,
                max_tokens=1024,
                temperature=temperature
            )
            return response.choices[0]['message']['content']
        except Exception as e:
            print(f"Error processing prompt. Engine: {engine}, Prompt: {prompt}, Error: {str(e)[:100]}")
            return ''






def gpt_calls(sample: pd.DataFrame) -> pd.DataFrame:
    """
    Processes a sample DataFrame by calling the GPT engine for each row, generating a filtered DataFrame with additional columns for occupation, job duties, and job qualifications.

    Args:
        sample (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: The filtered DataFrame with additional columns for occupation, job duties, and job qualifications.
    """

    engine = 'gpt-3.5-turbo-0613'
    temperature = 0.1

    # Create empty lists to store results for both prompts
    results_prompt_1 = []

    # Iterate through the dataframe and process each prompt
    for _, row in sample.iterrows():

        # Process first prompt
        # Example of calling the function
        response = process_prompt(f"Please label this job posting with the top 5 occupations that match it: {row['info']}", engine, temperature, "get_top_5_occupations")
        results_prompt_1.append(response)

    sample['occupation'] = results_prompt_1

    return sample



def sample_data(data, n=30):
    """
    Returns a random sample of n rows from the input DataFrame.

    Args:
        data (pd.DataFrame): The input DataFrame.
        n (int): The number of rows to sample.

    Returns:
        pd.DataFrame: The randomly sampled DataFrame with n rows.
    """

    random_sample = data.sample(n, random_state=1)
    return (random_sample)


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



def filter_func(x, list_of_occupations):
    """
    Filters a list of dictionaries `x` to include only those dictionaries whose 'Code' value 
    matches one of the codes in the list `listofoccupations`.

    Parameters:
    -----------
    x : list
        A list of dictionaries, each of which contains a 'Code' key.
    list_of_occupations : list
        A list of occupation codes to filter on.

    Returns:
    --------
    bool
        Returns True if any of the 'Code' values in `x` match any of the codes in `listofoccupations`, 
        False otherwise.
    """
    return any(code in [entry['Code'] for entry in x] for code in list_of_occupations)

def main():
    current_data = read_data_from_file("../data/currentResults.pkl")
    # Define the columns to be concatenated
    columns = [
    "JobSummary",
        "MajorDuties",]
    # Concatenate the specified columns into a new 'info' column in the
    # DataFrame
    current_data = concatenate_columns(current_data, columns)
    # Generate the list of occupations (OPM OCC codes)
    list_of_occupations = gen_list_of_occupations()
    # Define a custom lambda function to check if a code from the list exists
    # in the value
    # Apply the custom function and filter the DataFrame based on the
    # occupation codes
    filtered_data_occ = current_data[current_data['JobCategory'].apply(filter_func, list_of_occupations=list_of_occupations)]
    # Filter the DataFrame to include only rows where the 'info' column
    # contains the word "data"
    jobs_with_data = filtered_data_occ.loc[filtered_data_occ['info'].str.lower(
    ).str.count("data") >= 2].head(5)
    # Process the  DataFrame using the GPT engine and return the final
    # DataFrame with additional columns
    data_frame = gpt_calls(jobs_with_data)
    data_frame.to_pickle("../data/all_cols_sample.pkl")
    # Define the columns to be kept and written out
    return(data_frame)

df=main()

    
