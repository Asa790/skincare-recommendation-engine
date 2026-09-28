import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity

def load_data(df_path, matrix_path):
    df = pd.read_pickle(df_path)
    with open(matrix_path, 'rb') as f:
        tfidf, tfidf_matrix = pickle.load(f)
    return df, tfidf, tfidf_matrix

def get_recommendations(product_name, df, tfidf_matrix, top_n=5):
    if product_name not in df['product_name'].values:
        return []
    
    idx = df[df['product_name'] == product_name].index[0]
    sim_scores = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
    
    similar_indices = sim_scores.argsort()[-(top_n+1):-1][::-1]
    return df.iloc[similar_indices][['product_name', 'product_type', 'price']]