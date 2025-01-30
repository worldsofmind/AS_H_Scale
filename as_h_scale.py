import streamlit as st
import pandas as pd
import io

def clean_data(df):
    """Function to clean and preprocess the data."""
    # Drop duplicates
    df = df.drop_duplicates()
    
    # Drop rows with all missing values
    df = df.dropna(how='all')
    
    # Fill missing values with empty strings
    df = df.fillna('')
    
    # Normalize text columns (lowercasing, stripping spaces)
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].str.strip().str.lower()
    
    return df

st.title("Data Cleaning & Processing App")

uploaded_file = st.file_uploader("Upload an Excel file", type=["xlsx", "xls"])

if uploaded_file is not None:
    if uploaded_file.name.endswith(".xls"):
        df = pd.read_excel(uploaded_file, engine="xlrd")
    else:
        df = pd.read_excel(uploaded_file, engine="openpyxl")
    
    st.write("Raw Data:")
    st.dataframe(df.head())
    
    cleaned_df = clean_data(df)
    st.write("Cleaned Data:")
    st.dataframe(cleaned_df.head())
    
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        cleaned_df.to_excel(writer, index=False, sheet_name='Cleaned Data')
    output.seek(0)
    
    st.download_button(label="Download Cleaned Data",
                       data=output,
                       file_name="cleaned_data.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

