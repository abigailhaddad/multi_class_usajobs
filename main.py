# -*- coding: utf-8 -*-
"""
Created on Sat Sep  9 23:27:40 2023

@author: abiga
"""

from pull_historical import fetch_and_write_out_historical
from multi_label_classifier import do_multi_labels
from clustering import do_the_clustering
from graphics_frequency_job_title import frequencies_wordcloud


fetch_and_write_out_historical()
current_data = do_multi_labels()
clustered_data = do_the_clustering()
summary_df = frequencies_wordcloud()

# challenging!

challenges= ["Data Scientist, Data Analyst, Software Engineer", 
             'Geospatial Analyst, Remote Sensing Specialist, Software Developer',
             'Data Scientist, Data Analyst, Business Intelligence Analyst, Machine Learning Engineer, Artificial Intelligence Specialist, Data Engineer',
             'Application Developer, Software Engineer, Cloud Engineer']

challenge_summary = summary_df.loc[summary_df['occupation_str'].isin(challenges)]