import streamlit as st
import pandas as pd
import re
import io
import openpyxl
import xlsxwriter

def clean_data(df):
    # Make a copy to avoid modifying original data
    df_cleaned = df.copy()

    # Standardize column names (convert to lowercase, replace spaces with underscores)
    df_cleaned.columns = [col.strip().lower().replace(" ", "_").replace("\n", "_") for col in df_cleaned.columns]

    # Remove leading/trailing spaces from text columns
    for col in df_cleaned.select_dtypes(include=['object']).columns:
        df_cleaned[col] = df_cleaned[col].str.strip()

    # Convert all text to lowercase for uniformity
    df_cleaned = df_cleaned.applymap(lambda x: x.lower() if isinstance(x, str) else x)

    # Handle missing values: Fill NaNs with an empty string for text analytics
    df_cleaned.fillna("", inplace=True)

    # Remove duplicates based on 'case_reference' and 'assigned_solicitor'
    if 'case_reference' in df_cleaned.columns and 'assigned_solicitor' in df_cleaned.columns:
        df_cleaned.drop_duplicates(subset=['case_reference', 'assigned_solicitor'], inplace=True)

    # Ensure numerical columns are properly formatted
    numerical_columns = [
        'initial_fees_billed_by_as_(100%_for_sections_1-3)',
        'final_fees_agreed_upon_(100%_for_sections_1-3)'
    ]
    for col in numerical_columns:
        if col in df_cleaned.columns:
            df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors='coerce')

    # Remove special characters and extra spaces from text fields
    text_columns = df_cleaned.select_dtypes(include=['object']).columns
    for col in text_columns:
        df_cleaned[col] = df_cleaned[col].apply(lambda x: re.sub(r'\s+', ' ', re.sub(r'[^\w\s]', '', x)).strip())

    return df_cleaned

# Streamlit UI
st.title("Data Cleaning for Text Analytics")

uploaded_file = st.file_uploader("Upload your Excel file", type=["xlsx"])

if uploaded_file is not None:
    df = pd.read_excel(uploaded_file, sheet_name=0, engine='openpyxl')  # Load first sheet
    st.write("### Raw Data:")
    st.dataframe(df.head())
    
    df_cleaned = clean_data(df)
    
    st.write("### Cleaned Data:")
    st.dataframe(df_cleaned.head())
    
    # Option to download cleaned data as Excel
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df_cleaned.to_excel(writer, index=False, sheet_name='Cleaned Data')
        writer.close()
    output.seek(0)
    
    st.download_button(
        label="Download Cleaned Data",
        data=output,
        file_name="cleaned_data.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
