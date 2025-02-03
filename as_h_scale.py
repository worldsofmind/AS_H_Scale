import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from collections import Counter
import re

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

    reasons = {k: v for k, v in reasons.items() if v}
    return reasons

# Text Analysis Function
def analyze_text(text, dataset_texts):
    analysis = {
        "Key Themes": [],
        "Contextual Reasons": []
    }

    # Extracting Dominant Themes
    extracted_themes = extract_themes(dataset_texts)
    analysis["Key Themes"] = extracted_themes if extracted_themes else ["No dominant themes detected"]

    # Extract Contextual Reasons
    contextual_reasons = extract_reasons(text)
    if contextual_reasons:
        for reason, examples in contextual_reasons.items():
            analysis["Contextual Reasons"].append(f"{reason}: {', '.join(set(examples))}")
    else:
        analysis["Contextual Reasons"].append("No significant contextual reasons identified.")

    return analysis

# Topic Modeling Function using LDA
def topic_modeling(text_list, num_topics=5):
    vectorizer = CountVectorizer(stop_words='english')
    doc_term_matrix = vectorizer.fit_transform(text_list)

    lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
    lda.fit(doc_term_matrix)

    topics = []
    feature_names = vectorizer.get_feature_names_out()

    for idx, topic in enumerate(lda.components_):
        top_words = [feature_names[i] for i in topic.argsort()[:-6:-1]]
        topics.append(f"Topic {idx + 1}: {', '.join(top_words)}")

    return topics

# Streamlit App UI
st.title("Legal Case Analysis Tool (With Topic Modeling)")

uploaded_file = st.file_uploader("Upload an Excel file", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    
    # Allow user to select the column for analysis
    column_name = st.selectbox("Select the column to analyze", df.columns.tolist())
    
    if "Case reference" in df.columns:
        df = clean_data(df, column_name)
        
        # Option to analyze case-by-case or all cases
        analysis_mode = st.radio("Choose analysis mode:", ("Analyze by Case Reference", "Analyze All Cases Together"))

        if analysis_mode == "Analyze by Case Reference":
            selected_case = st.selectbox("Select a Case Reference", df["Case reference"].unique())
            case_texts = df[df["Case reference"] == selected_case][column_name].tolist()
            combined_text = " ".join(case_texts)
            analysis_result = analyze_text(combined_text, df[column_name].tolist())
        else:
            combined_text = " ".join(df[column_name].tolist())
            analysis_result = analyze_text(combined_text, df[column_name].tolist())

        # Topic Modeling Option
        if st.checkbox("Perform Topic Modeling"):
            num_topics = st.slider("Select number of topics:", 2, 10, 5)
            topics = topic_modeling(df[column_name].tolist(), num_topics)
            analysis_result["Discovered Topics"] = topics

        # Display Results
        st.write("### Analysis Results")
        for category, results in analysis_result.items():
            st.subheader(category)
            st.write(results if results else "No findings in this category.")
    else:
        st.error("The 'Case reference' column is missing from the file. Please upload a valid dataset.")
