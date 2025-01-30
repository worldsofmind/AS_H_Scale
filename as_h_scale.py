import streamlit as st
import pandas as pd
import re
import io
import openpyxl
import xlsxwriter

def clean_data(df):
    # Ensure all column names are standardized
    df_cleaned = df.copy()
    df_cleaned.columns = [col.strip().lower().replace(" ", "_").replace("\n", "_") for col in df_cleaned.columns]

    # Convert all text to lowercase for consistency
    df_cleaned = df_cleaned.astype(str).apply(lambda x: x.str.lower().str.strip())

    # Handle missing values by replacing NaNs with an empty string
    df_cleaned.fillna("", inplace=True)

    # Check if required columns exist before processing
    required_columns = ['case_reference', 'assigned_solicitor']
    for col in required_columns:
        if col not in df_cleaned.columns:
            st.warning(f"Warning: Column '{col}' is missing from the dataset!")

    # Remove duplicates if required columns exist
    if all(col in df_cleaned.columns for col in required_columns):
        df_cleaned.drop_duplicates(subset=required_columns, inplace=True)

    # Convert numeric columns to proper format if they exist
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
    try:
        df = pd.read_excel(uploaded_file, sheet_name=0, engine='openpyxl')  # Load first sheet
        st.write("### Raw Data:")
        st.dataframe(df.head())

        # Clean data
        df_cleaned = clean_data(df)
        st.write("### Cleaned Data:")
        st.dataframe(df_cleaned.head())

        # Option to download cleaned data as Excel
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df_cleaned.to_excel(writer, index=False, sheet_name='Cleaned Data')
        output.seek(0)

        st.download_button(
            label="Download Cleaned Data",
            data=output,
            file_name="cleaned_data.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    
    except Exception as e:
        st.error(f"Error loading file: {e}")
