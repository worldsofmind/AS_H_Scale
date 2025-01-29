import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import shap
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from flair.models import TextClassifier
from flair.data import Sentence
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.cluster import KMeans

# Streamlit App Header
st.title("Honoria Scale Negotiation Prediction App")

# Step 1: Upload Dataset and Clean Data
st.header("Step 1: Upload and Clean Dataset")
uploaded_file = st.file_uploader("Upload Excel file", type=["xlsx"])
if uploaded_file:
    try:
        df = pd.read_excel(uploaded_file, engine="openpyxl")  # Explicitly specify openpyxl engine

        # Step 2: Rename columns for easier handling
        df.columns = ['Case_Reference', 'Assigned_Solicitor', 'Negotiation_Rounds',
                      'Billing_Communication', 'Billing_Communication_Part2', 'Initial_Fees', 'Final_Fees',
                      'Pre_H_Scale_Guidelines']

        # Step 3: Convert numerical columns to numeric types
        df['Initial_Fees'] = pd.to_numeric(df['Initial_Fees'], errors='coerce')
        df['Final_Fees'] = pd.to_numeric(df['Final_Fees'], errors='coerce')
        df['Negotiation_Rounds'] = pd.to_numeric(df['Negotiation_Rounds'], errors='coerce')

        # Step 4: Handle missing values
        df['Billing_Communication'].fillna("No Communication", inplace=True)
        df['Billing_Communication_Part2'].fillna("No Communication", inplace=True)
        df['Negotiation_Rounds'].fillna(0, inplace=True)

        # Step 5: Define the target variable based on negotiation behavior
        df['Negotiation_Outcome'] = df.apply(
            lambda row: 'Negotiated' if pd.notnull(row['Initial_Fees']) and pd.notnull(row['Final_Fees']) and row['Initial_Fees'] > row['Final_Fees']
            else 'Accepted' if pd.notnull(row['Initial_Fees']) and pd.notnull(row['Final_Fees']) and row['Initial_Fees'] == row['Final_Fees']
            else 'Unknown', axis=1)

        # Encode target variable
        label_encoder = LabelEncoder()
        df['Negotiation_Outcome'] = label_encoder.fit_transform(df['Negotiation_Outcome'])

        # Display cleaned dataset
        st.subheader("Cleaned Dataset Preview")
        st.dataframe(df)

        # Allow user to download the cleaned dataset
        cleaned_csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Cleaned Dataset", cleaned_csv, "cleaned_dataset.csv", "text/csv")

        # Step 6: Extract Factors Causing Deviation
        st.header("Step 2: Identify Factors Causing Deviation from H Scale")
        df['Combined_Communication'] = df['Billing_Communication'] + " " + df['Billing_Communication_Part2']

        # Text Feature Extraction
        vectorization_method = st.selectbox("Choose text feature extraction method", ["TF-IDF", "Count Vectorizer", "LDA", "NMF"])
        
        if vectorization_method == "TF-IDF":
            vectorizer = TfidfVectorizer(stop_words='english', max_features=500)
        elif vectorization_method == "Count Vectorizer":
            vectorizer = CountVectorizer(stop_words='english', max_features=500)
        elif vectorization_method == "LDA":
            vectorizer = TfidfVectorizer(stop_words='english', max_features=500)
            tfidf_matrix = vectorizer.fit_transform(df['Combined_Communication'])
            lda_model = LatentDirichletAllocation(n_components=5, random_state=42)
            lda_topics = lda_model.fit_transform(tfidf_matrix)
            df['Topic'] = np.argmax(lda_topics, axis=1)
        elif vectorization_method == "NMF":
            vectorizer = TfidfVectorizer(stop_words='english', max_features=500)
            tfidf_matrix = vectorizer.fit_transform(df['Combined_Communication'])
            nmf_model = NMF(n_components=5, random_state=42)
            nmf_topics = nmf_model.fit_transform(tfidf_matrix)
            df['Topic'] = np.argmax(nmf_topics, axis=1)
        
        num_clusters = 5  # Ensure num_clusters is defined globally
        if vectorization_method in ["TF-IDF", "Count Vectorizer"]:
            text_features = vectorizer.fit_transform(df['Combined_Communication'])
            kmeans = KMeans(n_clusters=num_clusters, random_state=42)
            df['Deviation_Cluster'] = kmeans.fit_predict(text_features)

        # Display Factor Analysis
        st.subheader("Identified Factors Leading to Deviation")
        for cluster in range(num_clusters):
            top_words = [vectorizer.get_feature_names_out()[i] for i in np.argsort(kmeans.cluster_centers_[cluster])[-10:]]
            st.write(f"**Category {cluster}:** {', '.join(top_words)}")

        # Store results for later use
        st.session_state['factors_identified'] = True
        st.session_state['df_with_factors'] = df.copy()

        # Step 7: Proceed to Classification Model
        if st.button("Proceed to Classification Model"):
            st.session_state['classification_ready'] = True

    except Exception as e:
        st.error(f"Error processing the uploaded file: {e}")

