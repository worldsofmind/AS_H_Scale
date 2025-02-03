import streamlit as st
import pandas as pd
import torch
import string
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter

# Cache the model to avoid reloading
@st.cache_data
def load_model():
    return SentenceTransformer('paraphrase-MiniLM-L6-v2')

model = load_model()

# Text Cleaning Function
def clean_text(text):
    if not isinstance(text, str):
        return "No data available"
    text = text.lower().strip()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

# Data Cleaning Function
def clean_data(df, column_name):
    df = df.copy()
    df[column_name] = df[column_name].astype(str).apply(clean_text)
    df[column_name] = df[column_name].fillna("No data available")
    return df

# Extract Key Themes Using TF-IDF & N-grams
def extract_themes(text_list):
    vectorizer = TfidfVectorizer(ngram_range=(2, 3), stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(text_list)
    feature_array = vectorizer.get_feature_names_out()
    
    tfidf_scores = tfidf_matrix.sum(axis=0).tolist()[0]
    sorted_indices = sorted(range(len(tfidf_scores)), key=lambda i: tfidf_scores[i], reverse=True)
    common_phrases = [feature_array[i] for i in sorted_indices[:10]]
    
    return common_phrases

# Text Analysis Function
def analyze_text(text, dataset_texts):
    analysis = {
        "Key Themes": [],
        "Frequent Words": [],
        "Semantic Similarity Analysis": []
    }
    
    # Extracting Dominant Themes
    extracted_themes = extract_themes(dataset_texts)
    analysis["Key Themes"] = extracted_themes if extracted_themes else ["No dominant themes detected"]
    
    # Tokenization & Word Frequency
    words = text.split()
    stop_words = set(string.punctuation)
    filtered_words = [word for word in words if word not in stop_words]
    
    word_freq = Counter(filtered_words)
    common_words = word_freq.most_common(10)
    analysis["Frequent Words"] = [f"{word}: {count}" for word, count in common_words]

    # Semantic Similarity Analysis
    with st.spinner("Encoding dataset..."):
        dataset_embeddings = model.encode(dataset_texts, convert_to_tensor=True)
    text_embedding = model.encode(text, convert_to_tensor=True)

    similarities = util.pytorch_cos_sim(text_embedding, dataset_embeddings)[0]
    most_similar_index = torch.argmax(similarities).item()
    similarity_score = similarities[most_similar_index].item()

    analysis["Semantic Similarity Analysis"].append(f"Closest match in dataset with score {similarity_score:.2f}")

    return analysis

# Streamlit App UI
st.title("Legal Case Analysis Tool (Filtered by Case Reference)")

# Upload Excel File
uploaded_file = st.file_uploader("Upload an Excel file", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    column_name = "Comms between AS, PMU LE, DLA and BRC on billing (Part 1)"
    case_reference_col = "Case reference"  # Make sure this matches your Excel column name
    
    if column_name in df.columns and case_reference_col in df.columns:
        df = clean_data(df, column_name)
        
        # Filter by Case Reference
        selected_case = st.selectbox("Select a Case Reference", df[case_reference_col].unique())
        
        # Extract and Combine Text Related to the Selected Case
        case_texts = df[df[case_reference_col] == selected_case][column_name].tolist()
        combined_text = " ".join(case_texts)
        
        # Analysis
        analysis_result = analyze_text(combined_text, df[column_name].tolist())
        
        # Display Results
        st.write("### Analysis Results")
        for category, results in analysis_result.items():
            st.subheader(category)
            st.write(results if results else "No findings in this category.")
    else:
        st.error("The required columns are missing from the file. Please upload a valid dataset.")
