# Import necessary libraries
import pickle
import openai
import pandas as pd
import json
from pydantic import BaseModel
from typing import List, Any


class OccupationResponse(BaseModel):
    occupations: List[str]

# Load the API key from the file (change this to reflect where you've put your API key)
with open("../key/key.txt", "r") as key_file:
    api_key = key_file.read().strip()
    openai.api_key = api_key

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
                    "description": "Get no more than 3 occupations for the job posting",
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

    # Create empty lists to store results for the prompts
    results_prompt_1 = []

    # Iterate through the dataframe and process each prompt
    for _, row in sample.iterrows():
        # Process first prompt
        response = process_prompt(f"Please label this job posting with UP TO 3 occupational names that match it - not topic areas, but job names: {row['duties_var']}", engine, temperature, "get_up_to_3_occupations")
        results_prompt_1.append(response)

    sample['occupation'] = results_prompt_1

    return sample

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

def do_multi_labels(n: int = None, num_repeats: int = 1) -> pd.DataFrame:
    """
    Main function to read the data, sample it, and process it using GPT engine with repeated calls.

    Args:
        n (int, optional): Number of rows to sample. If None, the entire dataset is used.
        num_repeats (int): Number of times GPT is called for each job listing.

    Returns:
        pd.DataFrame: Processed DataFrame with additional columns for each repeated call.
    """
    
    current_data = read_data_from_file("../data/historical_joa.pkl")
    
    if n:
        current_data = current_data.sample(n, random_state=1)

    # Process the DataFrame using the GPT engine with repeated calls
    for i in range(num_repeats):
        data_frame = gpt_calls(current_data)
        current_data[f'occupation_{i+1}'] = data_frame['occupation']
    
    current_data=current_data.drop(columns=['occupation'])
    current_data.to_pickle("../data/all_cols_sample.pkl")
    
    return current_data


