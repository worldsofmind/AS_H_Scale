import streamlit as st
import pandas as pd
import io
import re
import unidecode
import contractions
import spacy
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

nlp = spacy.load("en_core_web_sm")
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_data(df):
    """Function to clean and preprocess the data."""
    # Drop duplicates
    df = df.drop_duplicates()
    
    # Drop rows with all missing values
    df = df.dropna(how='all')
    
    # Fill missing values with empty strings
    df = df.fillna('')
    
    # Normalize text columns (lowercasing, stripping spaces, removing accents, removing special characters, expanding contractions)
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].str.strip().str.lower()
        df[col] = df[col].apply(lambda x: unidecode.unidecode(x))  # Remove accents
        df[col] = df[col].apply(lambda x: contractions.fix(x))  # Expand contractions
        df[col] = df[col].apply(lambda x: re.sub(r'[^\w\s]', '', x))  # Remove special characters
        df[col] = df[col].apply(lambda x: re.sub(r'\d+', '', x))  # Remove numbers
        df[col] = df[col].apply(lambda x: ' '.join([word for word in x.split() if word not in stop_words]))  # Remove stopwords
        df[col] = df[col].apply(lambda x: ' '.join([lemmatizer.lemmatize(word) for word in x.split()]))  # Lemmatization
        df[col] = df[col].apply(lambda x: re.sub(r'\s+', ' ', x).strip())  # Remove extra spaces
        
        # Named Entity Recognition (NER) Masking
        def mask_entities(text):
            doc = nlp(text)
            return ' '.join(["[ENTITY]" if token.ent_type_ else token.text for token in doc])
        df[col] = df[col].apply(mask_entities)
    
    # Remove leading and trailing whitespace from all column names
    df.columns = df.columns.str.strip()
    
    return df

st.title("Data Cleaning & Processing App")

uploaded_file = st.file_uploader("Upload an Excel file", type=["xlsx", "xls"])

if uploaded_file is not None:
    if uploaded_file.name.endswith(".xls"):
        df = pd.read_excel(uploaded_file, engine="xlrd")
    else:
        df = pd.read_excel(uploaded_file, engine="openpyxl")
    
    st.write("Raw Data:")
    st.dataframe(df.head())
    
    cleaned_df = clean_data(df)
    st.write("Cleaned Data:")
    st.dataframe(cleaned_df.head())
    
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        cleaned_df.to_excel(writer, index=False, sheet_name='Cleaned Data')
    output.seek(0)
    
    st.download_button(label="Download Cleaned Data",
                       data=output,
                       file_name="cleaned_data.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
