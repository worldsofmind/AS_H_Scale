import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
import re

# Function to clean text
def clean_text(text):
    if not isinstance(text, str):
        return "No data available"
    text = text.lower().strip()
    text = re.sub(r'[^\w\s]', '', text)
    return text

# Function to extract email-like sections
def extract_email_sections(text):
    emails = re.findall(r'(From:.*?)(?=From:|$)', text, re.DOTALL | re.IGNORECASE)
    return emails if emails else [text]

# Data cleaning function
def clean_data(df, text_columns):
    for col in text_columns:
        df[col] = df[col].astype(str).apply(clean_text)
    return df

# Extract dominant themes using NMF
def extract_themes(text_list, num_topics=5, num_words=10):
    filtered_texts = []
    for text in text_list:
        filtered_texts.extend(extract_email_sections(text))

    filtered_texts = [text for text in filtered_texts if text.strip()]

    if not filtered_texts:
        return ["No valid data available for theme extraction."]

    vectorizer = TfidfVectorizer(ngram_range=(2, 3), stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(filtered_texts)

    nmf_model = NMF(n_components=num_topics, random_state=42)
    nmf_model.fit(tfidf_matrix)

    feature_names = vectorizer.get_feature_names_out()
    topics = []

    for topic_idx, topic in enumerate(nmf_model.components_):
        top_words = [feature_names[i] for i in topic.argsort()[:-num_words - 1:-1]]
        topics.append(f"Topic {topic_idx + 1}: {', '.join(top_words)}")

    return topics

# Enhanced Reason Extraction Function
def extract_reasons(text):
    patterns = {
        "Significant Workload and Time Commitment": [r"\b(time spent|hours|extensive work|significant workload)\b"],
        "Complexity of the Case": [r"\b(complex|multi-layered|difficulty|complicated)\b"],
        "High Volume of Communications": [r"\b(emails exchanged|letters|correspondence|communications)\b"],
        "Urgency and Procedural Challenges": [r"\b(urgent|procedural delays|last-minute|urgent application)\b"],
        "Client-Related Difficulties": [r"\b(difficult client|uncooperative client|late instructions|challenging behavior)\b"],
        "Administrative and Logistical Efforts": [r"\b(managing documents|court filings|logistical arrangements|document reviews)\b"]
    }

    reasons = {key: [] for key in patterns}

    for key, regex_list in patterns.items():
        for regex in regex_list:
            matches = re.findall(regex, text)
            if matches:
                reasons[key].extend(matches)

    reasons = {k: v for k, v in reasons.items() if v}
    return reasons

# Text Analysis Function
def analyze_text(text, dataset_texts):
    analysis = {
        "Key Themes": [],
        "Contextual Reasons": []
    }

    extracted_themes = extract_themes(dataset_texts)
    analysis["Key Themes"] = extracted_themes if extracted_themes else ["No dominant themes detected"]

    contextual_reasons = extract_reasons(text)
    if contextual_reasons:
        for reason, examples in contextual_reasons.items():
            analysis["Contextual Reasons"].append(f"{reason}: {', '.join(set(examples))}")
    else:
        analysis["Contextual Reasons"].append("No significant contextual reasons identified.")

    return analysis

# Streamlit App UI
st.set_page_config(page_title="Legal Case Analysis Tool", layout="wide")

st.title("📊 AS Fee Deviation Analysis")

uploaded_file = st.file_uploader("📥 Upload your cleaned Excel file", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)

    text_columns = ['Comms_Part1', 'Comms_Part2', 'Comms_Part3']
    df = clean_data(df, text_columns)

    analysis_mode = st.sidebar.radio("Choose analysis mode:", ("Analyze by Case Reference", "Analyze All Cases Together"))

    if analysis_mode == "Analyze by Case Reference":
        selected_case = st.sidebar.selectbox("Select a Case Reference", df["Case_Ref"].unique())
        selected_data = df[df["Case_Ref"] == selected_case]
    else:
        selected_case = "All Cases"
        selected_data = df

    combined_text = ' '.join(selected_data[text_columns].values.flatten())
    dataset_texts = df[text_columns].values.flatten().tolist()

    analysis_result = analyze_text(combined_text, dataset_texts)

    st.markdown("### 🗂️ Analysis Results")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Key Themes")
        st.write(analysis_result.get("Key Themes", "No dominant themes detected."))
    with col2:
        st.subheader("Contextual Reasons")
        st.write(analysis_result.get("Contextual Reasons", "No significant contextual reasons identified."))
