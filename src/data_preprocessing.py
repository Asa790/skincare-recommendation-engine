import pandas as pd
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

def clean_ingredient_text(val):
    try:
        items = ast.literal_eval(val)
        return " ".join([i.replace(" ", "_") for i in items])
    except:
        return ""

def preprocess_and_save(file_path, output_df_path, output_matrix_path):
    df = pd.read_csv(file_path)
    df['processed_ingreds'] = df['clean_ingreds'].apply(clean_ingredient_text)
    
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(df['processed_ingreds'])
    
    df.to_pickle(output_df_path)
    with open(output_matrix_path, 'wb') as f:
        pickle.dump((tfidf, tfidf_matrix), f)

if __name__ == "__main__":
    preprocess_and_save(
        'data/skincare_products_clean.csv', 
        'data/processed_skincare.pkl', 
        'data/tfidf_data.pkl'
    )