import streamlit as st
import pandas as pd

# Function to clean the data
def clean_data(df, column_name):
    df = df.copy()
    df[column_name] = df[column_name].astype(str).str.strip().str.lower()
    df[column_name] = df[column_name].fillna("No data available")
    return df

# Function to perform structured analysis
def analyze_text(text):
    analysis = {
        "Text Segmentation": [],
        "Keyword Identification": [],
        "Contextual Analysis": [],
        "Pattern Matching": [],
        "Logical Consistency Check": []
    }
    
    # Text Segmentation
    if "bill of costs" in text or "invoice" in text:
        analysis["Text Segmentation"].append("Contains billing details")
    if "email" in text or "correspondence" in text:
        analysis["Text Segmentation"].append("Includes email exchanges")
    if "offer" in text or "counter-offer" in text:
        analysis["Text Segmentation"].append("Mentions negotiation details")
    
    # Keyword Identification
    if "complex" in text or "difficult" in text:
        analysis["Keyword Identification"].append("Complex case")
    if "additional work" in text or "extra hours" in text:
        analysis["Keyword Identification"].append("Additional work required")
    if "urgent" in text or "short turnaround" in text:
        analysis["Keyword Identification"].append("Urgency of the case")
    
    # Contextual Analysis
    if "low offer" in text or "counter-proposal" in text:
        analysis["Contextual Analysis"].append("Solicitor disagreed with offered fees")
    if "justification" in text:
        analysis["Contextual Analysis"].append("Solicitor provided justification for fee increase")
    
    # Pattern Matching
    if "multiple hearings" in text or "many rounds" in text:
        analysis["Pattern Matching"].append("Prolonged court proceedings")
    if "submissions" in text or "affidavits" in text:
        analysis["Pattern Matching"].append("Extensive documentation required")
    
    # Logical Consistency Check
    if any(term in text for term in ["many hours", "significant effort"]):
        analysis["Logical Consistency Check"].append("Time commitment aligns with workload claim")
    
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
