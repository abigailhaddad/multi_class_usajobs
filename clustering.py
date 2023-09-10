import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from transformers import BertTokenizer, BertModel
import torch

# Load pre-trained model and tokenizer
model_name = "bert-base-uncased"
model = BertModel.from_pretrained(model_name)
tokenizer = BertTokenizer.from_pretrained(model_name)


def find_optimal_clusters(data, max_k=10):
    """Find the optimal number of clusters using the elbow method."""
    iters = range(2, max_k+1, 2)
    sse = []
    
    for k in iters:
        sse.append(KMeans(n_clusters=k, random_state=42).fit(data).inertia_)
        
    differences = np.diff(sse)
    optimal_clusters = np.where(differences > np.roll(differences, shift=-1))[0][0] + 2

    return optimal_clusters

def get_bert_embedding(phrase_list):
    """
    Get the BERT embedding for a list of phrases.
    """
    if not phrase_list:
        return np.zeros(model.config.hidden_size)
    
    # Embed each phrase in the list
    embeddings = []
    for phrase in phrase_list:
        inputs = tokenizer(phrase, return_tensors="pt", truncation=True, padding=True, max_length=128)
        outputs = model(**inputs)
        # Take the average of the token embeddings to get a single vector for the phrase
        phrase_embedding = outputs.last_hidden_state.mean(dim=1).squeeze().detach().numpy()
        embeddings.append(phrase_embedding)
    
    # Average the embeddings of all phrases to get a single embedding for the entire list
    return np.mean(embeddings, axis=0)

def create_presence_df(df):
    """Create a DataFrame that indicates the presence (1) or absence (0) of each job title."""
    # Flatten all job titles and create a list of unique titles
    all_titles = list({title for sublist in df['occupation_1'].tolist() for title in sublist})
    presence_df = pd.DataFrame(columns=all_titles)
    
    for idx, row in df.iterrows():
        for title in all_titles:
            if title in row['occupation_1']:
                presence_df.at[idx, title] = 1
            else:
                presence_df.at[idx, title] = 0
    
    return presence_df


def get_title_proportions(df, y_kmeans):
    """Get proportions of each title in each cluster."""
    presence_df = create_presence_df(df)
    presence_df['Cluster'] = y_kmeans

    proportions_df = presence_df.groupby('Cluster').mean() * 100
    return proportions_df

def get_top_n_titles(summary_df, n=5):
    """Get the top n job titles by percentage for each cluster and drop 0s."""
    dict_clusters = {}
    
    for row_number in range(0, len(summary_df)):
        row = summary_df.iloc[row_number]
        # Round the values to integers and filter out zeros
        dict_values = {k: int(v) for k, v in row.sort_values(ascending=False).head(n).to_dict().items() if v > 0}
        dict_clusters[row_number] = dict_values
        
    return dict_clusters


def do_the_clustering():
    df = pd.read_pickle("../data/all_cols_sample.pkl")
    
    #we're dropping 2 here - I think the API failed - I need to fix error handling so that doesn't happen
    
    df = df.loc[df['occupation_1'].str.len()>0]
    
    # Get BERT embeddings for each job title
    df['embedding'] = df['occupation_1'].apply(get_bert_embedding)

    
    embeddings_matrix = np.array(df['embedding'].tolist())

    optimal_clusters = find_optimal_clusters(embeddings_matrix)
    print(optimal_clusters)

    kmeans = KMeans(n_clusters=optimal_clusters, init='k-means++', max_iter=300, n_init=10, random_state=0)
    y_kmeans = kmeans.fit_predict(embeddings_matrix)
    
    df['cluster'] = y_kmeans
    print(df['cluster'].value_counts())

    summary_df = get_title_proportions(df, y_kmeans)
    top_titles_dict = get_top_n_titles(summary_df)
    print(top_titles_dict)
    
    
    return df

