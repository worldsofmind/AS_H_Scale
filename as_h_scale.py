import streamlit as st
import pandas as pd
import re
import io
import openpyxl
import xlsxwriter
import logging
import tempfile
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter

def clean_data(df):
    # Standardize column names
    df_cleaned = df.copy()
    df_cleaned.columns = [col.strip().lower().replace(" ", "_").replace("\n", "_") for col in df_cleaned.columns]
    
    # Convert all text columns to lowercase and remove leading/trailing spaces
    text_columns = df_cleaned.select_dtypes(include=['object']).columns
    df_cleaned[text_columns] = df_cleaned[text_columns].apply(lambda x: x.str.lower().str.strip())
    
    # Handle missing values
    df_cleaned.fillna("", inplace=True)
    
    # Remove duplicates if case_reference and assigned_solicitor exist
    required_columns = ['case_reference', 'assigned_solicitor']
    if all(col in df_cleaned.columns for col in required_columns):
        df_cleaned.drop_duplicates(subset=required_columns, inplace=True)
    
    # Auto-detect and convert numerical columns
    numerical_columns = df_cleaned.select_dtypes(include=['number']).columns
    df_cleaned[numerical_columns] = df_cleaned[numerical_columns].apply(pd.to_numeric, errors='coerce')
    
    # Use vectorized Pandas operations for text cleaning
    df_cleaned[text_columns] = df_cleaned[text_columns].replace(r'[^\w\s]', '', regex=True).replace(r'\s+', ' ', regex=True)
    
    return df_cleaned

def preprocess_text(text):
    """Remove numbers, case references, and extra spaces."""
    text = re.sub(r'\b\d+\b', '', text)  # Remove numeric values
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    return text

def extract_keywords_tfidf(df, ngram_range=(1,2), top_n=20):
    text_series = df.select_dtypes(include=['object']).apply(lambda x: ' '.join(x), axis=1)  # Combine all text columns
    text_series = text_series.apply(preprocess_text)  # Preprocess text
    
    vectorizer = TfidfVectorizer(ngram_range=ngram_range, stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(text_series)
    feature_names = vectorizer.get_feature_names_out()
    
    # Sum up the TF-IDF scores for each word
    scores = tfidf_matrix.sum(axis=0).A1
    keyword_scores = dict(zip(feature_names, scores))
    
    # Get the top N keywords
    top_keywords = dict(Counter(keyword_scores).most_common(top_n))
    return top_keywords

# Initialize logging
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
        
        # Optimize file download handling
        with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as temp_file:
            with pd.ExcelWriter(temp_file.name, engine='xlsxwriter') as writer:
                df_cleaned.to_excel(writer, index=False, sheet_name="Cleaned Data")
            
            with open(temp_file.name, "rb") as f:
                st.download_button(
                    label="Download Cleaned Data",
                    data=f,
                    file_name="cleaned_data.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
        
        # Extract TF-IDF Keywords from entire document
        top_keywords = extract_keywords_tfidf(df_cleaned)
        st.write("### Top Keywords (TF-IDF)")
        st.write(top_keywords)
        
        logging.info("File processed and ready for download.")
    except Exception as e:
        st.error(f"Error loading file: {e}")
        logging.error(f"Error encountered: {e}")
