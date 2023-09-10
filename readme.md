# Historical Job Listing Analysis for Occupational Code 1560 (Data Scientist)

This repository is focused on analyzing historical job listings specifically for the occupational code 1560 (Data Scientist) from USAJobs using the historical jobs API. The scripts fetch, classify, cluster, and visually represent these listings, offering insights into job title frequencies and their related groupings.


## Workflow

1. **Fetching Historical Data**: `pull_historical.py`
   - Fetches historical job listing data for the occupational code 1560 (Data Scientist) from USAJobs and writes it out for further processing.
   
2. **Multilabel Classification**: `multi_label_classifier.py`
   - Uses machine learning (including potential calls to GPT-based models) to label each job listing with multiple job titles.
   
3. **Clustering**: `clustering.py`
   - Clusters the data to group similar job roles together.

4. **Graphics & Visualization**: `graphics_frequency_job_title.py`
   - Produces a graphical representation, like a word cloud, of the most frequent job titles in the dataset.

5. **Analysis of Result Agreement**: `analyze_result_agreement.py`
   - If the `do_multi_labels` script performs more than one GPT API call for each listing, this script analyzes the agreement between the results. It then outputs a log file with the agreement scores.

## Running the Scripts

Execute the entire workflow by running the main script main.py.

## Outputs

1. Historical USAJobs job data for occupational code 1560.
2. Labeled job listings with potential multiple titles.
3. Data grouped by similar job roles using BERT.
4. Visuals showcasing job title frequencies.
5. A log file with agreement scores (if applicable).
