import streamlit as st
import pandas as pd
import spacy
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer

# Load spaCy model
@st.cache_data
def load_nlp_model():
    return spacy.load("en_core_web_sm")

nlp = load_nlp_model()

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
    doc = nlp(text)
    reasons = {
        "Significant Workload and Time Commitment": [],
        "Complexity of the Case": [],
        "High Volume of Communications": [],
        "Urgency and Procedural Challenges": [],
        "Client-Related Difficulties": [],
        "Administrative and Logistical Efforts": []
    }

    # Keyword sets for pattern matching
    workload_keywords = ['hours', 'drafting', 'client attendances', 'correspondence', 'documents']
    complexity_keywords = ['complexity', 'prolonged litigation', 'case conferences', 'court hearings']
    communication_keywords = ['emails', 'correspondences', 'letters', 'communication']
    urgency_keywords = ['urgent', 'immediate', 'child custody', 'protection orders']
    client_difficulties_keywords = ['uncooperative', 'late instructions', 'difficult client']
    admin_keywords = ['filings', 'logistics', 'submissions', 'administrative burden']

    # Named Entity Recognition (NER) and Dependency Parsing
    for sent in doc.sents:
        sent_text = sent.text.lower()

        # Workload and Time Commitment
        if any(word in sent_text for word in workload_keywords):
            reasons["Significant Workload and Time Commitment"].append(sent.text)

        # Complexity of the Case
        if any(word in sent_text for word in complexity_keywords):
            reasons["Complexity of the Case"].append(sent.text)

        # High Volume of Communications
        if any(word in sent_text for word in communication_keywords):
            reasons["High Volume of Communications"].append(sent.text)

        # Urgency and Procedural Challenges
        if any(word in sent_text for word in urgency_keywords):
            reasons["Urgency and Procedural Challenges"].append(sent.text)

        # Client-Related Difficulties
        if any(word in sent_text for word in client_difficulties_keywords):
            reasons["Client-Related Difficulties"].append(sent.text)

        # Administrative and Logistical Efforts
        if any(word in sent_text for word in admin_keywords):
            reasons["Administrative and Logistical Efforts"].append(sent.text)

    # Filter out empty categories
    reasons = {k: v for k, v in reasons.items() if v}

    return reasons

# Streamlit UI
st.title("Legal Case Analysis Tool (Advanced NLP)")

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
