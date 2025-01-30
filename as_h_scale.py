import streamlit as st
import pandas as pd
import re
import io
import openpyxl
import xlsxwriter
import logging
import tempfile

def clean_data(df, columns_to_clean):
    # Standardize column names
    df_cleaned = df.copy()
    df_cleaned.columns = [col.strip().lower().replace(" ", "_").replace("\n", "_") for col in df_cleaned.columns]
    
    # Select only columns chosen for cleaning
    df_cleaned = df_cleaned[columns_to_clean]
    
    # Convert all text to lowercase and remove leading/trailing spaces
    df_cleaned = df_cleaned.astype(str).apply(lambda x: x.str.lower().str.strip())
    
    # Handle missing values
    df_cleaned.fillna("", inplace=True)
    
    # Remove duplicates if case_reference and assigned_solicitor exist
    required_columns = ['case_reference', 'assigned_solicitor']
    if all(col in df_cleaned.columns for col in required_columns):
        df_cleaned.drop_duplicates(subset=required_columns, inplace=True)
    
    # Auto-detect and convert numerical columns
    numerical_columns = df_cleaned.select_dtypes(include=['number']).columns
    df_cleaned[numerical_columns] = df_cleaned[numerical_columns].apply(pd.to_numeric, errors='coerce')
    
    # Use vectorized Pandas operations for text cleaning
    text_columns = df_cleaned.select_dtypes(include=['object']).columns
    df_cleaned[text_columns] = df_cleaned[text_columns].replace(r'[^\w\s]', '', regex=True).replace(r'\s+', ' ', regex=True).str.strip()
    
    return df_cleaned

# Initialize logging
logging.basicConfig(level=logging.INFO)
st.sidebar.title("Settings")

uploaded_file = st.sidebar.file_uploader("Upload your Excel file", type=["xlsx"])

if uploaded_file is not None:
    try:
        xls = pd.ExcelFile(uploaded_file)
        sheet_name = st.sidebar.selectbox("Select Sheet", xls.sheet_names)
        df = pd.read_excel(xls, sheet_name=sheet_name, engine='openpyxl')
        
        st.write("### Raw Data:")
        st.dataframe(df.head())
        
        # Allow user to select columns for cleaning
        columns_to_clean = st.sidebar.multiselect("Select Columns to Clean", df.columns, default=df.columns)
        
        df_cleaned = clean_data(df, columns_to_clean)
        
        st.write("### Cleaned Data:")
        st.dataframe(df_cleaned.head())
        
        # Display summary statistics
        st.write("### Data Summary")
        st.write(df_cleaned.describe(include='all'))
        
        # Optimize file download handling
        with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as temp_file:
            with pd.ExcelWriter(temp_file.name, engine='xlsxwriter') as writer:
                df_cleaned.to_excel(writer, index=False, sheet_name="Cleaned Data")
            
            st.download_button(
                label="Download Cleaned Data",
                data=open(temp_file.name, "rb"),
                file_name="cleaned_data.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        
        logging.info("File processed and ready for download.")
    except Exception as e:
        st.error(f"Error loading file: {e}")
        logging.error(f"Error encountered: {e}")
