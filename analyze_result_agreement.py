import pandas as pd
from collections import defaultdict
import datetime

def compute_agreement(row: pd.Series, num_repeats: int) -> float:
    occupation_counts = {}
    
    for i in range(num_repeats):
        for occupation in row[f'occupation_{i+1}']:
            occupation_counts[occupation] = occupation_counts.get(occupation, 0) + 1

    consistent_occupations = sum(1 for count in occupation_counts.values() if count == num_repeats)
    return consistent_occupations / len(occupation_counts)

def construct_count_dataframe(data, num_repeats):
    all_titles = set()
    rows = []

    for _, row in data.iterrows():
        occupation_counts = defaultdict(int)

        for i in range(num_repeats):
            for occupation in row[f'occupation_{i+1}']:
                occupation_counts[occupation] += 1
                all_titles.add(occupation)

        rows.append(occupation_counts)

    count_df = pd.DataFrame(rows).fillna(0).astype(int)
    return count_df, all_titles

def main(file_path: str, num_repeats: int):
    data = pd.read_pickle(file_path)
    
    agreements = []
    listings_with_consistent_titles = 0

    for _, row in data.iterrows():
        agreement = compute_agreement(row, num_repeats)
        agreements.append(agreement)
        if agreement > 0: 
            listings_with_consistent_titles += 1

    count_df, _ = construct_count_dataframe(data, num_repeats)
    
    log_entries = []
    log_entries.append(f"Number of rows processed: {len(data)}")
    log_entries.append(f"Number of 'raters' (GPT calls per listing): {num_repeats}")
    log_entries.append("\nAgreement Scores:")
    for i, agreement in enumerate(agreements, 1):
        log_entries.append(f"Listing {i} Agreement Score: {agreement:.2f}")
    log_entries.append(f"\nPercentage of listings with at least one occupation present in all {num_repeats} pulls: {listings_with_consistent_titles / len(data) * 100:.2f}%")
    
    log_df = pd.DataFrame(log_entries, columns=["Log"])
    
    current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    file_name = f"../results/log_{current_time}.xlsx"
    
    # Write to Excel with multiple sheets
    with pd.ExcelWriter(file_name, engine='openpyxl') as writer:
        log_df.to_excel(writer, sheet_name="Log", index=False)
        count_df.to_excel(writer, sheet_name="Count", index=False)
        data.to_excel(writer, sheet_name="Data", index=False)

main("../data/all_cols_sample.pkl", num_repeats=3)
