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
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    return text

# Data cleaning
def clean_data(df, column_name):
    df = df.copy()
    df[column_name] = df[column_name].astype(str).apply(clean_text)
    df[column_name] = df[column_name].fillna("No data available")
    return df

# Extract dominant themes using NMF
def extract_themes(text_list, num_topics=5, num_words=10):
    vectorizer = TfidfVectorizer(ngram_range=(2, 3), stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(text_list)

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

# Streamlit App UI
st.set_page_config(page_title="Legal Case Analysis Tool", layout="wide")

st.markdown("""
    <style>
        .main {background-color: #f9f9f9; padding: 20px; border-radius: 10px;}
        h1, h2, h3, h4 {color: #333;}
        .stButton button {background-color: #4CAF50; color: white;}
    </style>
""", unsafe_allow_html=True)

st.title("📊 Legal Case Analysis Tool")

uploaded_file = st.file_uploader("📥 Upload an Excel file", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)

    st.sidebar.header("⚙️ Analysis Settings")
    column_name = st.sidebar.selectbox("Select the column to analyze", df.columns.tolist())

    if "Case reference" in df.columns:
        df = clean_data(df, column_name)

        analysis_mode = st.sidebar.radio("Choose analysis mode:", ("Analyze by Case Reference", "Analyze All Cases Together"))

        if analysis_mode == "Analyze by Case Reference":
            selected_case = st.sidebar.selectbox("Select a Case Reference", df["Case reference"].unique())
            case_texts = df[df["Case reference"] == selected_case][column_name].tolist()
            combined_text = " ".join(case_texts)
            analysis_result = analyze_text(combined_text, df[column_name].tolist())
        else:
            combined_text = " ".join(df[column_name].tolist())
            analysis_result = analyze_text(combined_text, df[column_name].tolist())

        st.markdown("### 🗂️ Analysis Results")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Key Themes")
            st.write(analysis_result.get("Key Themes", "No dominant themes detected."))
        with col2:
            st.subheader("Contextual Reasons")
            st.write(analysis_result.get("Contextual Reasons", "No significant contextual reasons identified."))

    else:
        st.error("The 'Case reference' column is missing from the file. Please upload a valid dataset."
