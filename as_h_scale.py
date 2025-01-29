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
import spacy

# Load spaCy model (ensure it's pre-installed in requirements.txt)
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    st.error("spaCy model 'en_core_web_sm' is not available. Ensure it's pre-installed in requirements.txt.")
    nlp = None

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

        # Step 6: Extract Reasons for Deviation from H Scale
        st.header("Step 2: Analyze Reasons for Deviation")
        df['Combined_Communication'] = df['Billing_Communication'] + " " + df['Billing_Communication_Part2']

        # Enhanced Semantic Analysis to Extract Key Reasons
        def extract_reasons(text):
            if nlp is None:
                return "NLP model not available"
            doc = nlp(text)
            reasons = []
            for ent in doc.ents:
                if ent.label_ in ['LAW', 'ORG', 'MONEY', 'GPE']:  # Extract legal/financial entities
                    reasons.append(ent.text)
            for token in doc:
                if token.dep_ in ['dobj', 'pobj', 'attr', 'nsubj'] and token.pos_ in ['NOUN', 'PROPN']:
                    reasons.append(token.text)
            return ", ".join(set(reasons))
        
        df['Deviation_Reasons'] = df['Combined_Communication'].apply(extract_reasons)
        
        # Display Extracted Reasons
        st.subheader("Identified Reasons for Deviation")
        st.dataframe(df[['Case_Reference', 'Assigned_Solicitor', 'Deviation_Reasons']])
        
        # Store extracted reasons for later use
        st.session_state['reasons_identified'] = True
        st.session_state['df_with_reasons'] = df.copy()

        # Step 7: Classification Model
        if st.session_state.get('classification_ready'):
            st.header("Step 3: Classification Model")
            
            # Feature Engineering
            categorical_features = ['Assigned_Solicitor']
            numerical_features = ['Negotiation_Rounds', 'Initial_Fees', 'Final_Fees']
            
            categorical_transformer = OneHotEncoder(handle_unknown='ignore')
            numerical_transformer = StandardScaler()
            
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numerical_transformer, numerical_features),
                    ('cat', categorical_transformer, categorical_features)
                ])
            
            # Model Selection
            model_choice = st.selectbox("Choose a Classification Model", ["Random Forest", "Gradient Boosting", "Logistic Regression", "SVM", "XGBoost"])
            model_dict = {
                "Random Forest": RandomForestClassifier(),
                "Gradient Boosting": GradientBoostingClassifier(),
                "Logistic Regression": LogisticRegression(),
                "SVM": SVC(),
                "XGBoost": XGBClassifier()
            }
            
            model = model_dict[model_choice]
            
            # Train-Test Split
            X = df[numerical_features + categorical_features]
            y = df['Negotiation_Outcome']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            pipeline = Pipeline([
                ('preprocessor', preprocessor),
                ('classifier', model)
            ])
            
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)
            
            # Display Metrics
            st.subheader("Model Performance")
            st.text(classification_report(y_test, y_pred))
            
            # Confusion Matrix
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            st.pyplot(fig)
    
    except Exception as e:
        st.error(f"Error processing the uploaded file: {e}")
