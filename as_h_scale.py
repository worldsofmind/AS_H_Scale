import streamlit as st
import pandas as pd
import re
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer

# Clean text
def clean_text(text):
    if not isinstance(text, str):
        return "No data available"
    return text.lower().strip()

# Extract dominant themes using TF-IDF
def extract_themes(text_list):
    vectorizer = TfidfVectorizer(ngram_range=(2, 3), stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(text_list)
    feature_array = vectorizer.get_feature_names_out()

    tfidf_scores = tfidf_matrix.sum(axis=0).tolist()[0]
    sorted_indices = sorted(range(len(tfidf_scores)), key=lambda i: tfidf_scores[i], reverse=True)
    common_phrases = [feature_array[i] for i in sorted_indices[:10]]

    return common_phrases

# Advanced NLP Analysis to Identify Key Reasons
def analyze_reasons(text):
    reasons = {
        "Significant Workload and Time Commitment": [],
        "Complexity of the Case": [],
        "High Volume of Communications": [],
        "Urgency and Procedural Challenges": [],
        "Client-Related Difficulties": [],
        "Administrative and Logistical Efforts": []
    }

    # Define regex patterns for each category
    patterns = {
        "Significant Workload and Time Commitment": r"(hours|drafting|attendances|correspondence|documents|workload)",
        "Complexity of the Case": r"(complexity|litigation|case conferences|court hearings|multi-layered)",
        "High Volume of Communications": r"(emails|correspondences|letters|communication|exchange)",
        "Urgency and Procedural Challenges": r"(urgent|immediate|child custody|protection orders|emergency)",
        "Client-Related Difficulties": r"(uncooperative|late instructions|difficult client|challenging)",
        "Administrative and Logistical Efforts": r"(filings|logistics|submissions|administrative|documentation)"
    }

    # Sentence-level analysis
    sentences = re.split(r'(?<=[.!?]) +', text)
    for sentence in sentences:
        for category, pattern in patterns.items():
            if re.search(pattern, sentence, re.IGNORECASE):
                reasons[category].append(sentence)

    # Filter out empty categories
    reasons = {k: v for k, v in reasons.items() if v}

    return reasons

# Streamlit UI
st.title("Legal Case Analysis Tool (Without spaCy)")

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
        analysis_result = analyze_reasons(input_text)

        st.write("### Key Reasons Identified")
        for category, sentences in analysis_result.items():
            st.subheader(category)
            for sentence in sentences:
                st.write(f"- {sentence}")
    else:
        st.warning("Please input some text or upload a file for analysis.")
