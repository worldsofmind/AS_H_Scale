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
        "1. Extensive Work and Time Spent on the Case": [],
        "2. Complexity of the Case": [],
        "3. Urgency and Uncooperative Parties": [],
        "4. High Number of Hearings and Procedural Complexity": [],
        "5. Justification for Higher Fees": [],
        "Final Conclusion": []
    }
    
    # Step 1: Identifying extensive work and time spent
    if any(term in text for term in ["many hours", "time-consuming", "significant effort", "workload", "extensive work"]):
        analysis["1. Extensive Work and Time Spent on the Case"].append("Solicitor emphasized significant time commitment.")
    if any(term in text for term in ["emails", "correspondence", "documents", "drafting", "filing"]):
        analysis["1. Extensive Work and Time Spent on the Case"].append("Numerous documents filed, including affidavits, applications, and legal correspondences.")
    
    # Step 2: Assessing complexity of the case
    if any(term in text for term in ["complex", "interlinked issues", "difficult case", "multiple applications"]):
        analysis["2. Complexity of the Case"].append("Case involved multiple interlinked legal issues requiring additional legal work.")
    if any(term in text for term in ["self-represented", "frequent lawyer changes", "multiple firms"]):
        analysis["2. Complexity of the Case"].append("Frequent changes in legal representation caused delays and increased complexity.")
    
    # Step 3: Identifying urgency and uncooperative parties
    if any(term in text for term in ["urgent", "last minute", "short turnaround", "immediate response"]):
        analysis["3. Urgency and Uncooperative Parties"].append("Urgent nature of case required fast legal responses.")
    if any(term in text for term in ["difficult client", "late instructions", "last-minute changes"]):
        analysis["3. Urgency and Uncooperative Parties"].append("Client provided late instructions, increasing solicitor workload.")
    if any(term in text for term in ["delays", "prolonged", "extended proceedings"]):
        analysis["3. Urgency and Uncooperative Parties"].append("Defendant prolonged proceedings unnecessarily.")
    
    # Step 4: Evaluating procedural complexity and hearings
    if any(term in text for term in ["court hearings", "multiple hearings", "sessions", "mediation", "proceedings"]):
        analysis["4. High Number of Hearings and Procedural Complexity"].append("Multiple hearings and mediations increased workload.")
    if any(term in text for term in ["affidavits", "court filings", "submissions"]):
        analysis["4. High Number of Hearings and Procedural Complexity"].append("Extensive documentation and legal filings required.")
    
    # Step 5: Justification for higher fees
    if any(term in text for term in ["offer too low", "counter-offer", "fee negotiation", "higher fees"]):
        analysis["5. Justification for Higher Fees"].append("Solicitor argued that the offered compensation was insufficient.")
    if any(term in text for term in ["precedent cases", "similar cases", "comparative analysis"]):
        analysis["5. Justification for Higher Fees"].append("Solicitor compared the case complexity with previous cases.")
    
    # Final Conclusion
    analysis["Final Conclusion"] = [
        "The assigned solicitor wants to deviate from the H-scale due to:",
        "- High case complexity with interlinked legal issues.",
        "- Significant time investment in drafting, filing, and court appearances.",
        "- Urgency of case requiring quick responses and last-minute changes.",
        "- Difficult client and prolonged proceedings causing additional work.",
        "- Argument that standard Legal Aid compensation does not adequately cover workload."
    ]
    
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

