import streamlit as st
import pandas as pd
import torch
import string
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter

# Cache the model to avoid reloading on every interaction
@st.cache_data
def load_model():
    return SentenceTransformer('paraphrase-MiniLM-L6-v2')

model = load_model()

# Function to clean text
def clean_text(text):
    if not isinstance(text, str):
        return "No data available"
    text = text.lower().strip()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

# Function to clean dataset
def clean_data(df, column_name):
    df = df.copy()
    df[column_name] = df[column_name].astype(str).apply(clean_text)
    df[column_name] = df[column_name].fillna("No data available")
    return df

# Function to extract dominant themes using TF-IDF & N-grams
def extract_themes(text_list):
    vectorizer = TfidfVectorizer(ngram_range=(2, 3), stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(text_list)
    feature_array = vectorizer.get_feature_names_out()
    
    # Extract the most common key phrases
    tfidf_scores = tfidf_matrix.sum(axis=0).tolist()[0]
    sorted_indices = sorted(range(len(tfidf_scores)), key=lambda i: tfidf_scores[i], reverse=True)
    common_phrases = [feature_array[i] for i in sorted_indices[:10]]
    
    return common_phrases

# Function to perform structured analysis dynamically
def analyze_text(text, dataset_texts):
    analysis = {
        "Key Themes": [],
        "Frequent Words": [],
        "Semantic Similarity Analysis": []
    }
    
    # Extract dominant themes from dataset
    extracted_themes = extract_themes(dataset_texts)
    analysis["Key Themes"] = extracted_themes if extracted_themes else ["No dominant themes detected"]
    
    # Tokenization and counting word frequency
    words = text.split()
    stop_words = set(string.punctuation)
    filtered_words = [word for word in words if word not in stop_words]
    
    word_freq = Counter(filtered_words)
    common_words = word_freq.most_common(10)
    analysis["Frequent Words"] = [f"{word}: {count}" for word, count in common_words]

    # Perform semantic similarity analysis
    with st.spinner("Encoding dataset..."):
        dataset_embeddings = model.encode(dataset_texts, convert_to_tensor=True)
    text_embedding = model.encode(text, convert_to_tensor=True)

    similarities = util.pytorch_cos_sim(text_embedding, dataset_embeddings)[0]
    most_similar_index = torch.argmax(similarities).item()
    similarity_score = similarities[most_similar_index].item()

    analysis["Semantic Similarity Analysis"].append(f"Closest match in dataset with score {similarity_score:.2f}")

    return analysis

# Streamlit UI
st.title("Legal Case Analysis Tool (Dynamic Semantic Analysis)")

# File uploader
uploaded_file = st.file_uploader("Upload an Excel file", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    column_name = "Comms between AS, PMU LE, DLA and BRC on billing (Part 1)"
    
    if column_name in df.columns:
        df = clean_data(df, column_name)
        
        analysis_option = st.radio(
            "Choose analysis method:",
            ("Analyze each row independently", "Analyze entire dataset"))
        
        dataset_texts = df[column_name].tolist()  # Extract text from entire dataset
        
        if analysis_option == "Analyze each row independently":
            selected_case = st.selectbox("Select a case reference", df["Case reference"].tolist())
            selected_row = df[df["Case reference"] == selected_case]
            analysis_result = analyze_text(selected_row[column_name].values[0], dataset_texts)
        else:
            combined_text = " ".join(dataset_texts)
            analysis_result = analyze_text(combined_text, dataset_texts)
        
        st.write("### Analysis Results")
        for category, results in analysis_result.items():
            st.subheader(category)
            st.write(results if results else "No findings in this category.")
    else:
        st.error("The required column is missing from the file. Please upload a valid dataset.")
