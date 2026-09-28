import streamlit as st
import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Skincare Prototipe", layout="wide")

st.markdown("""
<style>
.stApp {
    background: repeating-linear-gradient(
        90deg,
        #ffcce6,
        #ffcce6 40px,
        #ffffff 40px,
        #ffffff 80px
    );
}
.block-container {
    background-color: rgba(255, 255, 255, 0.95);
    border: 8px double #cccccc;
    padding: 3rem !important;
    border-radius: 10px;
    margin-top: 3rem;
}
header[data-testid="stHeader"] {
    background-color: #111111 !important;
}
h1 {
    font-family: 'Brush Script MT', 'Lucida Handwriting', cursive;
    color: #111111 !important;
    text-align: center;
    font-size: 4rem !important;
    margin-bottom: 1rem;
}
p, span, label, div {
    color: #333333 !important;
    font-family: 'Trebuchet MS', sans-serif;
}
div[data-baseweb="select"] > div {
    background-color: #ffffff;
    border: 2px solid #111111;
    border-radius: 0;
}
.stButton>button {
    background-color: #ffb3d9;
    color: #111111;
    border: 2px solid #111111;
    border-radius: 0;
    text-transform: uppercase;
    letter-spacing: 2px;
    width: 100%;
    padding: 0.75rem;
}
.stButton>button:hover {
    background-color: #111111;
    color: #ffffff;
    border: 2px solid #111111;
}
</style>
""", unsafe_allow_html=True)

def load_data():
    df = pd.read_pickle('data/processed_skincare.pkl')
    with open('data/tfidf_data.pkl', 'rb') as f:
        tfidf, tfidf_matrix = pickle.load(f)
    return df, tfidf, tfidf_matrix

df, tfidf, tfidf_matrix = load_data()

st.title("Skincare Matches")

product_list = df['product_name'].tolist()
selected_product = st.selectbox("Choose a reference product", product_list)

if st.button("Find Matches"):
    idx = df[df['product_name'] == selected_product].index[0]
    sim_scores = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
    similar_indices = sim_scores.argsort()[-6:-1][::-1]
    
    results = df.iloc[similar_indices][['product_name', 'product_type', 'price']]
    st.table(results)