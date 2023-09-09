import pandas as pd
from collections import defaultdict

def compute_agreement(row: pd.Series, num_repeats: int) -> float:
    occupation_counts = {}
    
    for i in range(num_repeats):
        for occupation in row[f'occupation_{i+1}']:
            occupation_counts[occupation] = occupation_counts.get(occupation, 0) + 1

    consistent_occupations = sum(1 for count in occupation_counts.values() if count == num_repeats)
    
    return consistent_occupations / len(occupation_counts)


def construct_count_dataframe(file_path: str, num_repeats: int) -> pd.DataFrame:
    data = pd.read_pickle(file_path)
    
    all_titles = set()
    rows = []

    for _, row in data.iterrows():
        occupation_counts = defaultdict(int)
        
        for i in range(num_repeats):
            for occupation in row[f'occupation_{i+1}']:
                occupation_counts[occupation] += 1
                all_titles.add(occupation)

        rows.append(occupation_counts)

    count_df = pd.DataFrame(rows, columns=list(all_titles)).fillna(0).astype(int)
    return count_df


def main():
    file_path = "../data/all_cols_sample.pkl"
    num_repeats = 3

    # Compute and redirect output to txt file
    with open("../results/output.txt", "w") as f:
        agreements = []
        listings_with_consistent_titles = 0

        data = pd.read_pickle(file_path)
        for _, row in data.iterrows():
            agreement = compute_agreement(row, num_repeats)
            agreements.append(agreement)
            if agreement > 0:  # If there's at least one occupation that appeared in all n calls
                listings_with_consistent_titles += 1

        # Write results to the file
        for i, agreement in enumerate(agreements, 1):
            f.write(f"Listing {i} Agreement Score: {agreement:.2f}\n")
        f.write(f"\nPercentage of listings with at least one occupation present in all {num_repeats} pulls: {listings_with_consistent_titles / len(data) * 100:.2f}%\n")

    # Construct and save count DataFrame to csv
    count_df = construct_count_dataframe(file_path, num_repeats)
    count_df.to_csv("../results/count_data.csv", index=False)

main()
