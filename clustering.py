import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def prepare_data(df):
    """Convert job titles to lowercase and create a concatenated column."""
    df['all_titles'] = df['occupation_1'].apply(lambda x: ','.join([title.lower() for title in x]))
    return df

def get_tfidf_representation(df):
    """Vectorize the concatenated job titles."""
    vectorizer = TfidfVectorizer(analyzer=lambda x: x.split(','))
    X = vectorizer.fit_transform(df['all_titles'])
    return X, vectorizer

def find_optimal_clusters(data, max_k=10):
    """Find the optimal number of clusters using the elbow method."""
    iters = range(2, max_k+1, 2)
    sse = []
    
    for k in iters:
        sse.append(KMeans(n_clusters=k, random_state=42).fit(data).inertia_)
        
    differences = np.diff(sse)
    optimal_clusters = np.where(differences > np.roll(differences, shift=-1))[0][0] + 2

    return optimal_clusters

def perform_clustering(data, optimal_clusters):
    """Perform KMeans clustering and return cluster labels."""
    kmeans = KMeans(n_clusters=optimal_clusters, init='k-means++', max_iter=300, n_init=10, random_state=0)
    y_kmeans = kmeans.fit_predict(data)
    return kmeans, y_kmeans

def create_summary(df, y_kmeans, X, vectorizer):
    """Create a summary dataframe showing the percentage presence of each job title in each cluster."""
    features = vectorizer.get_feature_names_out()
    df_tfidf = pd.DataFrame(X.todense(), columns=features)
    df_tfidf['Cluster'] = y_kmeans
    
    summary_df = pd.DataFrame()
    for i in range(y_kmeans.max()+1):  # Loop through each cluster
        cluster_data = df_tfidf[df_tfidf['Cluster'] == i].drop('Cluster', axis=1)
        proportions = cluster_data.mean() * 100
        summary_df[f'Cluster {i + 1}'] = proportions
    
    summary_df = summary_df.transpose()
    return summary_df


def get_top_n_titles(summary_df, n=5):
    """Get the top n job titles by percentage for each cluster."""
    dict_clusters={}
    for row_number in range(0, len(summary_df)):
        row=  summary_df.iloc[row_number]
        dict_values=row.sort_values(ascending=False).head(n).to_dict()
        dict_clusters[row_number]=dict_values
    return dict_clusters
        


def main():
    df = pd.read_pickle("../data/all_cols_sample.pkl")
    df = prepare_data(df)
    X, vectorizer = get_tfidf_representation(df)
    optimal_clusters = find_optimal_clusters(X)
    print(optimal_clusters)
    kmeans, y_kmeans = perform_clustering(X, optimal_clusters)
    
    # Add cluster labels to the df dataframe
    df['cluster'] = y_kmeans

    summary_df = create_summary(df, y_kmeans, X, vectorizer)
    top_titles_dict = get_top_n_titles(summary_df, n=10)
    print(top_titles_dict)
    
    # If you want to see which job postings belong to, let's say, cluster 0:
    cluster_0_jobs = df[df['cluster'] == 0]
    print(cluster_0_jobs)
    
    return df, summary_df

# Call the main function
if __name__ == '__main__':
    df, summary_df = main()
