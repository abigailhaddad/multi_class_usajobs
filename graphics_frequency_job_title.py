# -*- coding: utf-8 -*-
"""
Created on Sat Sep  9 23:01:17 2023
@author: abiga
"""

import pandas as pd
import matplotlib.pyplot as plt
import re
from wordcloud import WordCloud


def extract_link_number(link_list):
    numbers = []
    for link in link_list:
        href = link.get('href', '')
        match = re.search(r'(\d+)$', href)
        if match:
            numbers.append(int(match.group(1)))
    return numbers


def generate_wordcloud(title_frequencies_dict, save_path=None):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(title_frequencies_dict)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis('off')
    if save_path:
        plt.savefig(save_path)
    plt.show()


def process_dataframe(df):
    df = df.loc[df['occupation_1'].str.len() > 0]
    df['_links_numbers'] = df['_links'].apply(extract_link_number)
    df['occupation_str'] = df['occupation_1'].str.join(', ')

    summary_df = df.groupby('occupation_str')['_links_numbers'].agg(lambda x: [item for sublist in x for item in sublist]).reset_index()
    summary_df['counts'] = summary_df['_links_numbers'].apply(len)
    summary_df = summary_df.sort_values(by='counts', ascending=False).reset_index(drop=True)

    return summary_df


def frequencies_wordcloud():
    df = pd.read_pickle("../data/all_cols_sample.pkl")

    # Extract and process titles for wordcloud
    all_titles = df['occupation_1'].explode().str.title()
    title_frequencies_dict = all_titles.value_counts().to_dict()
    generate_wordcloud(title_frequencies_dict, save_path='wordcloud.png')

    summary_df = process_dataframe(df)
    print(summary_df.head())
    return(summary_df)


