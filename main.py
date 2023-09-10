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
