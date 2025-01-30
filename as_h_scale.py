import streamlit as st
import pandas as pd
import re
import io
import openpyxl
import xlsxwriter
import logging
import tempfile
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter

def clean_data(df):
    df_cleaned = df.copy()
    df_cleaned.columns = [col.strip().lower().replace(" ", "_").replace("\n", "_") for col in df_cleaned.columns]
    text_columns = df_cleaned.select_dtypes(include=['object']).columns
    df_cleaned[text_columns] = df_cleaned[text_columns].apply(lambda x: x.str.lower().str.strip())
    df_cleaned.fillna("", inplace=True)
    required_columns = ['case_reference', 'assigned_solicitor']
    if all(col in df_cleaned.columns for col in required_columns):
        df_cleaned.drop_duplicates(subset=required_columns, inplace=True)
    numerical_columns = df_cleaned.select_dtypes(include=['number']).columns
    df_cleaned[numerical_columns] = df_cleaned[numerical_columns].apply(pd.to_numeric, errors='coerce')
    df_cleaned[text_columns] = df_cleaned[text_columns].replace(r'[^\w\s]', '', regex=True).replace(r'\s+', ' ', regex=True)
    return df_cleaned

def extract_keywords_tfidf(df, ngram_range=(1,2), top_n=20):
    text_series = df.select_dtypes(include=['object']).apply(lambda x: ' '.join(x), axis=1)
    vectorizer = TfidfVectorizer(ngram_range=ngram_range, stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(text_series)
    feature_names = vectorizer.get_feature_names_out()
    scores = tfidf_matrix.sum(axis=0).A1
    keyword_scores = dict(zip(feature_names, scores))
    top_keywords = dict(Counter(keyword_scores).most_common(top_n))
    return top_keywords

def compute_text_similarity(df, threshold=0.5, top_n=5):
    text_series = df.select_dtypes(include=['object']).apply(lambda x: ' '.join(x), axis=1)
    if len(text_series) < 2:
        return "Not enough text entries for similarity comparison."
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(text_series)
    similarity_matrix = cosine_similarity(tfidf_matrix)
    similar_pairs = []
    for i in range(len(text_series)):
        for j in range(i + 1, len(text_series)):
            if similarity_matrix[i, j] > threshold:
                similar_pairs.append((i, j, similarity_matrix[i, j]))
    if not similar_pairs:
        return "No similar text pairs found. Try adjusting the threshold."
    similar_pairs = sorted(similar_pairs, key=lambda x: x[2], reverse=True)[:top_n]
    results = []
    for i, j, score in similar_pairs:
        results.append({
            'Text 1': text_series[i],
            'Text 2': text_series[j],
            'Similarity Score': round(score, 2)
        })
    return results

def extract_named_entities(text_series):
    entity_patterns = {
        'LEGAL_TERMS': r'\b(contract|agreement|negotiation|settlement|lawsuit|bill|litigation|damages|contractual|breach)\b'
    }
    entities = {key: [] for key in entity_patterns}
    for text in text_series:
        for entity, pattern in entity_patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            entities[entity].extend(matches)
    return {key: list(set(value)) for key, value in entities.items()}

logging.basicConfig(level=logging.INFO)
st.sidebar.title("Settings")

uploaded_file = st.sidebar.file_uploader("Upload your Excel file", type=["xlsx"])

if uploaded_file is not None:
    try:
        df = pd.read_excel(uploaded_file, engine='openpyxl')
        st.write("### Raw Data:")
        st.dataframe(df.head())
        
        df_cleaned = clean_data(df)
        st.write("### Cleaned Data:")
        st.dataframe(df_cleaned.head())
        
        named_entities = extract_named_entities(df_cleaned.select_dtypes(include=['object']).apply(lambda x: ' '.join(x), axis=1))
        st.write("### Named Entity Recognition (NER)")
        st.write(named_entities)
        
        top_keywords = extract_keywords_tfidf(df_cleaned)
        st.write("### Top Keywords (TF-IDF)")
        st.write(top_keywords)
        
        st.write("### Top Similar Texts (Cosine Similarity)")
        similarity_threshold = st.slider("Set Cosine Similarity Threshold", min_value=0.1, max_value=1.0, value=0.5, step=0.1)
        similarity_results = compute_text_similarity(df_cleaned, threshold=similarity_threshold)
        st.write(similarity_results)
        
        st.markdown("""
        <div title='TF-IDF (Term Frequency-Inverse Document Frequency) ranks words based on their importance. 
        It gives higher scores to words that appear frequently in a document but not in many other documents. 
        This helps identify key topics while filtering out common words like "the" or "and".'>
        ℹ️ **What is TF-IDF?** (Hover to learn more)
        </div>
        <div title='Named Entity Recognition (NER) is a method to identify specific terms in text, such as names of legal concepts, organizations, or locations. 
        In this case, it focuses on extracting legal terms like "contract", "negotiation", "settlement", etc.'>
        ℹ️ **What is NER?** (Hover to learn more)
        </div>
        <div title='Cosine Similarity is a way to measure how similar two pieces of text are. 
        It works by converting text into numerical vectors and calculating the angle between them. 
        A similarity score closer to 1 means the texts are very similar, while a score closer to 0 means they are different. 
        This section displays the **top similar text pairs** based on a similarity threshold.'>
        ℹ️ **What is Cosine Similarity?** (Hover to learn more)
        </div>
        """, unsafe_allow_html=True)
        
        logging.info("File processed and ready for download.")
    except Exception as e:
        st.error(f"Error loading file: {e}")
        logging.error(f"Error encountered: {e}")
