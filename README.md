# 🧴 Skincare Recommendation Engine

A content-based skincare recommendation system and interactive web application powered by **Streamlit**, **TF-IDF**, and **Cosine Similarity**.

This application analyzes cosmetic ingredient formulations across skincare products to find accurate product alternatives ("dupes" or matches) sharing similar active and base ingredients.

---

## 🌟 Key Features

- **Ingredient-Based Recommendations**: Uses chemical/ingredient profiles rather than simple brand or category matching, identifying true functional alternatives.
- **Compound Preservation**: Preserves multi-word ingredient names (e.g., `sodium_hyaluronate`, `cetyl_alcohol`) during tokenization for precise chemical feature extraction.
- **Interactive Streamlit UI**: A clean, stylish web interface allowing users to select any reference product and instantly view top matches.
- **Product Metadata**: Displays recommendations alongside product types and prices to help users find budget-friendly or formulaic equivalents.
- **Data Exploration Notebook**: Includes an in-depth exploratory data analysis (EDA) notebook detailing ingredient counts, product categories, and price distributions.

---

## 🧠 How It Works

```
                                      +--------------------+
                                      | skincare_products  |
                                      |     _clean.csv     |
                                      +---------+----------+
                                                |
                                                v
+------------------------+            +---------+----------+
|  clean_ingredient_text | ---------> | Compound Tokenizer |
+------------------------+            +---------+----------+
                                                |
                                                v
                                      +---------+----------+
                                      |  TF-IDF Vectorizer |
                                      +---------+----------+
                                                |
                                                v
                                      +---------+----------+
+-----------------------+             |   TF-IDF Matrix    |
| Selected Product Name | ----------> |         +          |
+-----------------------+             | Cosine Similarity  |
                                      +---------+----------+
                                                |
                                                v
                                      +---------+----------+
                                      | Top-5 Recommendations
                                      +--------------------+
```

1. **Preprocessing (`src/data_preprocessing.py`)**:
   - Parses the stringified ingredient list into Python lists.
   - Replaces whitespaces in multi-word ingredient names with underscores (`_`) so phrases like `"salicylic acid"` are treated as single tokens (`"salicylic_acid"`).
   - Generates and fits a **TF-IDF Vectorizer** across all processed ingredient strings.
   - Serializes the processed dataframe and TF-IDF matrix into `.pkl` files.

2. **Similarity Engine (`src/recommendation_model.py` & `app.py`)**:
   - Calculates the **Cosine Similarity** between the target product's TF-IDF vector and the entire catalog.
   - Ranks the most similar products, excluding the queried product itself, and returns the top 5 matches.

---

## 📁 Project Structure

```plaintext
skincare-recommendation-engine/
│
├── app.py                             # Streamlit interactive web application
├── requirements.txt                   # Project dependencies
├── .gitignore                         # Git exclusion rules
│
├── data/
│   ├── skincare_products_clean.csv    # Source product & ingredient dataset
│   ├── processed_skincare.pkl         # Preprocessed dataframe cache
│   └── tfidf_data.pkl                 # Serialized TF-IDF vectorizer & matrix
│
├── notebooks/
│   └── 01_data_exploration.ipynb      # EDA on ingredients, prices & categories
│
└── src/
    ├── data_preprocessing.py          # Ingredient cleaning & TF-IDF pipeline
    └── recommendation_model.py        # Cosine similarity recommendation logic
```

---

## 🚀 Getting Started

### 1. Prerequisites

- Python 3.9+ installed on your machine.

### 2. Clone the Repository

```bash
git clone https://github.com/Asa790/skincare-recommendation-engine.git
cd skincare-recommendation-engine
```

### 3. Set Up a Virtual Environment

```bash
# Windows (PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. (Optional) Rebuild Preprocessed Data & Matrix

If you modify `skincare_products_clean.csv` or the preprocessing logic:

```bash
python src/data_preprocessing.py
```

### 6. Run the Application

Launch the Streamlit app:

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`.

---

## 📊 Exploratory Data Analysis (EDA)

The `notebooks/01_data_exploration.ipynb` notebook covers:

- Cleaning and converting currency/price strings into numerical values.
- Analyzing product distribution across categories (Moisturisers, Serums, Cleansers, etc.).
- Distribution of ingredient counts per product.
- Correlation and statistical distributions using `matplotlib` and `seaborn`.

To view or run the notebook:

```bash
jupyter notebook notebooks/01_data_exploration.ipynb
```

---

## 🛠️ Tech Stack

- **Frontend & App Framework**: [Streamlit](https://streamlit.io/)
- **Data Manipulation**: [Pandas](https://pandas.pydata.org/)
- **Machine Learning & NLP**: [Scikit-Learn](https://scikit-learn.org/) (TF-IDF Vectorizer, Cosine Similarity)
- **Data Visualization**: [Matplotlib](https://matplotlib.org/), [Seaborn](https://seaborn.pydata.org/)
- **Serialization**: Python `pickle`

---

## 🔮 Future Enhancements

- [ ] Filter recommendations by skin type (Dry, Oily, Sensitive, Combination).
- [ ] Filter by price range / budget constraints.
- [ ] Highlight shared key active ingredients (e.g. Niacinamide, Retinol, Ceramides) in the UI.
- [ ] Allergy / ingredient exclusion filters (e.g., fragrance-free, paraben-free).
