import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
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

# Extract email exchanges from text
def extract_emails(text):
    # Enhanced regex to capture common email patterns and handle multi-line emails
    email_pattern = r"(From:.*?)(?=(From:|Sent:|-----Original Message-----|$))"
    emails = re.findall(email_pattern, text, re.DOTALL)
    return [email[0].strip() for email in emails if email[0].strip()]

# Topic Modeling Function for Emails Only
def topic_modeling_on_emails(email_texts, num_topics=5):
    if not email_texts:
        return ["No email exchanges detected for topic modeling."]

    vectorizer = CountVectorizer(stop_words='english')
    doc_term_matrix = vectorizer.fit_transform(email_texts)

    lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
    lda.fit(doc_term_matrix)

    topics = []
    feature_names = vectorizer.get_feature_names_out()

    for idx, topic in enumerate(lda.components_):
        top_words = [feature_names[i] for i in topic.argsort()[:-6:-1]]
        topics.append(f"Topic {idx + 1}: {', '.join(top_words)}")

    return topics

# Streamlit App UI
st.title("Legal Case Analysis Tool (With Topic Modeling for Emails Only)")

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
        else:
            case_texts = df[column_name].tolist()

        # Topic Modeling on Emails
        if st.checkbox("Perform Topic Modeling (Emails Only)"):
            num_topics = st.slider("Select number of topics:", 2, 10, 5)
            email_texts = []
            for text in case_texts:
                email_texts.extend(extract_emails(text))  # Extract emails from each entry

            if email_texts:
                topics = topic_modeling_on_emails(email_texts, num_topics)
                st.subheader("Email Topics Identified:")
                for topic in topics:
                    st.write(topic)
            else:
                st.warning("No emails found in the selected column.")

    else:
        st.error("The 'Case reference' column is missing from the file. Please upload a valid dataset.")

