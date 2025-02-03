import streamlit as st
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
import re

# Cache the model for efficiency
@st.cache_data
def load_model():
    return SentenceTransformer('paraphrase-MiniLM-L6-v2')

model = load_model()

# Function to clean text
def clean_text(text):
    if not isinstance(text, str):
        return "No data available"
    text = text.lower().strip()
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    return text

# Data cleaning
def clean_data(df, column_name):
    df = df.copy()
    df[column_name] = df[column_name].astype(str).apply(clean_text)
    df[column_name] = df[column_name].fillna("No data available")
    return df

# Extract dominant themes using TF-IDF
def extract_themes(text_list):
    vectorizer = TfidfVectorizer(ngram_range=(2, 3), stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(text_list)
    feature_array = vectorizer.get_feature_names_out()
    tfidf_scores = tfidf_matrix.sum(axis=0).tolist()[0]
    sorted_indices = sorted(range(len(tfidf_scores)), key=lambda i: tfidf_scores[i], reverse=True)
    common_phrases = [feature_array[i] for i in sorted_indices[:10]]
    return common_phrases

# Enhanced Reason Extraction Function
def extract_reasons(text):
    patterns = {
        "Significant Workload and Time Commitment": [
            r"\b(time spent|hours|extensive work|significant workload)\b",
            r"\b(drafting documents|court attendance|legal correspondences)\b"
        ],
        "Complexity of the Case": [
            r"\b(complex legal issues|multi-layered case|involving different religions)\b",
            r"\b(ancillary proceedings|interim applications|cross-referencing affidavits)\b"
        ],
        "High Volume of Communications": [
            r"\b(emails exchanged|letters|correspondence with court|frequent communications)\b"
        ],
        "Urgency and Procedural Challenges": [
            r"\b(urgent hearing|procedural delays|last-minute submissions|urgent application)\b"
        ],
        "Client-Related Difficulties": [
            r"\b(difficult client|uncooperative client|late instructions|challenging behavior)\b"
        ],
        "Administrative and Logistical Efforts": [
            r"\b(managing documents|court filings|logistical arrangements|document reviews)\b"
        ]
    }

    reasons = {key: [] for key in patterns}

    for key, regex_list in patterns.items():
        for regex in regex_list:
            matches = re.findall(regex, text)
            if matches:
                reasons[key].extend(matches)

    # Clean up empty categories
    reasons = {k: v for k, v in reasons.items() if v}
    return reasons

# Text Analysis Function
def analyze_text(text, dataset_texts):
    analysis = {
        "Key Themes": [],
        "Frequent Words": [],
        "Contextual Reasons": [],
        "Semantic Similarity Analysis": []
    }

    # Extracting Dominant Themes
    extracted_themes = extract_themes(dataset_texts)
    analysis["Key Themes"] = extracted_themes if extracted_themes else ["No dominant themes detected"]

    # Frequent Words
    words = text.split()
    word_freq = Counter(words)
    common_words = word_freq.most_common(10)
    analysis["Frequent Words"] = [f"{word}: {count}" for word, count in common_words]

    # Extract Contextual Reasons
    contextual_reasons = extract_reasons(text)
    if contextual_reasons:
        for reason, examples in contextual_reasons.items():
            analysis["Contextual Reasons"].append(f"{reason}: {', '.join(set(examples))}")
    else:
        analysis["Contextual Reasons"].append("No significant contextual reasons identified.")

    # Semantic Similarity
    dataset_embeddings = model.encode(dataset_texts, convert_to_tensor=True)
    text_embedding = model.encode(text, convert_to_tensor=True)
    similarities = util.pytorch_cos_sim(text_embedding, dataset_embeddings)[0]
    most_similar_index = torch.argmax(similarities).item()
    similarity_score = similarities[most_similar_index].item()
    analysis["Semantic Similarity Analysis"].append(f"Closest match in dataset with score {similarity_score:.2f}")

    return analysis

# Streamlit App UI
st.title("Legal Case Analysis Tool (Advanced Contextual Analysis)")

uploaded_file = st.file_uploader("Upload an Excel file", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    column_name = "Comms between AS, PMU LE, DLA and BRC on billing (Part 1)"
    case_reference_col = "Case reference"

    if column_name in df.columns and case_reference_col in df.columns:
        df = clean_data(df, column_name)
        selected_case = st.selectbox("Select a Case Reference", df[case_reference_col].unique())
        case_texts = df[df[case_reference_col] == selected_case][column_name].tolist()
        combined_text = " ".join(case_texts)

        analysis_result = analyze_text(combined_text, df[column_name].tolist())

        st.write("### Analysis Results")
        for category, results in analysis_result.items():
            st.subheader(category)
            st.write(results if results else "No findings in this category.")
    else:
        st.error("The required columns are missing from the file. Please upload a valid dataset.")
