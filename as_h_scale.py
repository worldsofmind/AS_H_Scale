import streamlit as st
import pandas as pd
from collections import Counter
import string

# Function to clean the data
def clean_data(df, column_name):
    df = df.copy()
    df[column_name] = df[column_name].astype(str).str.strip().str.lower()
    df[column_name] = df[column_name].fillna("No data available")
    return df

# Function to perform structured analysis dynamically
def analyze_text(text):
    analysis = {
        "Key Themes": [],
        "Frequent Words": [],
        "Contextual Insights": []
    }
    
    # Tokenization and preprocessing
    words = text.split()
    stop_words = set(string.punctuation)
    filtered_words = [word for word in words if word not in stop_words]
    
    # Identify frequent words
    word_freq = Counter(filtered_words)
    common_words = word_freq.most_common(10)
    analysis["Frequent Words"] = [f"{word}: {count}" for word, count in common_words]
    
    # Extract key themes based on word patterns
    themes = []
    if any(word in text for word in ["urgent", "time-sensitive", "immediate"]):
        themes.append("Urgency of Case")
    if any(word in text for word in ["complex", "multi-layered", "challenging"]):
        themes.append("Case Complexity")
    if any(word in text for word in ["negotiation", "counter-offer", "fee discussion"]):
        themes.append("Fee Negotiations")
    if any(word in text for word in ["court hearing", "multiple hearings", "sessions"]):
        themes.append("High Number of Hearings")
    if any(word in text for word in ["delays", "extended proceedings", "slow process"]):
        themes.append("Delays and Prolonged Proceedings")
    if any(word in text for word in ["additional work", "extra hours", "excess workload"]):
        themes.append("Significant Workload")
    
    analysis["Key Themes"] = themes if themes else ["No dominant themes detected"]
    
    # Extract contextual insights
    contextual_insights = []
    if "offer too low" in text or "counter-proposal" in text:
        contextual_insights.append("Solicitor challenged fee offered.")
    if "complex legal matter" in text or "multiple stakeholders" in text:
        contextual_insights.append("Case required handling multiple legal aspects.")
    
    analysis["Contextual Insights"] = contextual_insights if contextual_insights else ["No significant contextual insights."]
    
    return analysis

# Streamlit UI
st.title("Legal Case Analysis Tool")

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
        
        if analysis_option == "Analyze each row independently":
            selected_case = st.selectbox("Select a case reference", df["Case reference"].tolist())
            selected_row = df[df["Case reference"] == selected_case]
            analysis_result = analyze_text(selected_row[column_name].values[0])
        else:
            combined_text = " ".join(df[column_name].tolist())
            analysis_result = analyze_text(combined_text)
        
        st.write("### Analysis Results")
        for category, results in analysis_result.items():
            st.subheader(category)
            st.write(results if results else "No findings in this category.")
    else:
        st.error("The required column is missing from the file. Please upload a valid dataset.")
