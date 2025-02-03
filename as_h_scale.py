import streamlit as st
import pandas as pd
import re
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Stopwords list
STOP_WORDS = set(stopwords.words('english'))

# Enhanced Data Cleaning Function
def clean_text(text):
    if not isinstance(text, str):
        return "No data available"
    
    # 1. Lowercase text
    text = text.lower().strip()
    
    # 2. Remove special characters and numbers
    text = re.sub(r'[^a-z\s]', '', text)
    
    # 3. Remove extra whitespaces
    text = re.sub(r'\s+', ' ', text)
    
    # 4. Remove stopwords
    words = word_tokenize(text)
    filtered_words = [word for word in words if word not in STOP_WORDS]
    
    return ' '.join(filtered_words)

# Extract key phrases using TF-IDF
def extract_key_phrases(text_list):
    vectorizer = TfidfVectorizer(ngram_range=(1, 3), stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(text_list)
    feature_array = vectorizer.get_feature_names_out()

    tfidf_scores = tfidf_matrix.sum(axis=0).tolist()[0]
    sorted_indices = sorted(range(len(tfidf_scores)), key=lambda i: tfidf_scores[i], reverse=True)
    key_phrases = [feature_array[i] for i in sorted_indices[:10]]

    return key_phrases

# Dynamic Reason Extraction Function
def extract_reasons(text):
    reasons = []

    # Sentence Tokenization
    sentences = sent_tokenize(text)

    # Extract reasons based on cause-effect phrases
    cause_effect_keywords = ['because', 'due to', 'as a result of', 'therefore', 'resulting from']
    for sentence in sentences:
        if any(phrase in sentence for phrase in cause_effect_keywords):
            reasons.append(sentence)

    # Key Phrase Extraction
    key_phrases = extract_key_phrases([text])

    # Frequent Words Analysis
    words = word_tokenize(text)
    word_freq = Counter(words)
    common_words = word_freq.most_common(10)

    # Combine results
    output = {
        "Extracted Reasons": reasons if reasons else ["No explicit reasons found."],
        "Key Phrases": key_phrases,
        "Frequent Words": [f"{word}: {count}" for word, count in common_words]
    }

    return output

# Streamlit UI
st.title("Dynamic Legal Case Analysis Tool (Without spaCy)")

# Text Area for Input
st.subheader("Analyze Specific Text")
input_text = st.text_area("Paste the legal text here:", height=300)

# File Upload Option
st.subheader("Or Upload an Excel File")
uploaded_file = st.file_uploader("Upload an Excel file", type=["xlsx"])

dataset_texts = []

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    column_name = "Comms between AS, PMU LE, DLA and BRC on billing (Part 1)"

    if column_name in df.columns:
        df[column_name] = df[column_name].astype(str).apply(clean_text)
        dataset_texts = df[column_name].tolist()

# Analyze Button
if st.button("Analyze Text"):
    if input_text.strip() != "":
        cleaned_text = clean_text(input_text)
        analysis_result = extract_reasons(cleaned_text)

        st.write("### Analysis Results")
        for category, results in analysis_result.items():
            st.subheader(category)
            for item in results:
                st.write(f"- {item}")
    else:
        st.warning("Please input some text or upload a file for analysis.")
