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
from transformers import pipeline
from gensim import corpora, models
import nltk
from nltk.tokenize import word_tokenize
import time

# Load spaCy model (ensure it's pre-installed in requirements.txt)
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    st.warning("spaCy model 'en_core_web_sm' is not available. The app will proceed without NLP analysis.")
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

        text_analysis_method = st.selectbox("Choose a Text Analysis Method", ["spaCy", "TF-IDF", "BERT", "LDA", "VADER", "Flair"])

        def extract_reasons(text):
            if not text.strip():
                return "No relevant text"
            try:
                if text_analysis_method == "spaCy" and nlp:
                    doc = nlp(text)
                    reasons = [ent.text for ent in doc.ents if ent.label_ in ['LAW', 'ORG', 'MONEY', 'GPE']]
                    reasons += [token.text for token in doc if token.dep_ in ['dobj', 'pobj', 'attr', 'nsubj']]
                elif text_analysis_method == "TF-IDF":
                    vectorizer = TfidfVectorizer(stop_words='english')
                    tfidf_matrix = vectorizer.fit_transform([text])
                    feature_names = vectorizer.get_feature_names_out()
                    scores = tfidf_matrix.toarray().flatten()
                    top_n_words = [feature_names[i] for i in scores.argsort()[-5:]]
                    reasons = ", ".join(top_n_words)
                elif text_analysis_method == "BERT":
                    summarizer = pipeline("summarization")
                    reasons = summarizer(text, max_length=50, min_length=25, do_sample=False)[0]['summary_text']
                elif text_analysis_method == "LDA":
                    tokens = word_tokenize(text.lower())
                    dictionary = corpora.Dictionary([tokens])
                    corpus = [dictionary.doc2bow(tokens)]
                    lda_model = models.LdaModel(corpus, num_topics=2, id2word=dictionary, passes=10)
                    topics = lda_model.show_topics(num_words=5, formatted=False)
                    reasons = ", ".join([word for topic in topics for word, _ in topic[1]])
                else:
                    reasons = "NLP method not supported"
            except Exception as e:
                st.error(f"Error in text processing: {e}")
                reasons = "Text analysis failed"
            return reasons
        
        df['Deviation_Reasons'] = df['Combined_Communication'].apply(extract_reasons)

        # Step 7: Classification Model with Hyperparameter Tuning
        st.header("Step 3: Predicting Negotiation Outcome")
        model_choices = {
            "RandomForest": RandomForestClassifier(),
            "GradientBoosting": GradientBoostingClassifier(),
            "XGBoost": XGBClassifier()
        }
        param_grids = {
            "RandomForest": {'n_estimators': [100, 200], 'max_depth': [None, 10]},
            "GradientBoosting": {'n_estimators': [100, 200], 'learning_rate': [0.01, 0.1]},
            "XGBoost": {'n_estimators': [100, 200], 'learning_rate': [0.01, 0.1], 'max_depth': [3, 6]}
        }
        selected_model_name = st.selectbox("Select a Model for Classification", list(model_choices.keys()))
        selected_model = model_choices[selected_model_name]
        param_grid = param_grids[selected_model_name]
        grid_search = GridSearchCV(selected_model, param_grid, cv=5)
        pipeline = Pipeline([('preprocessor', preprocessor), ('model', grid_search)])
        pipeline.fit(X_train, y_train)
        st.subheader("Best Model Parameters")
        st.write(grid_search.best_params_)
    except Exception as e:
        st.error(f"Error processing the uploaded file: {e}")
