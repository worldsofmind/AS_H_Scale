import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import pipeline
from gensim import corpora, models
import nltk
from nltk.tokenize import word_tokenize
import time
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MultiLabelBinarizer

# Load Hugging Face NER model with optimized settings
try:
    ner_pipeline = pipeline("ner", model="dslim/bert-base-NER", tokenizer="dslim/bert-base-NER", aggregation_strategy="simple")
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
except Exception as e:
    st.error(f"Error loading transformer models: {e}")
    ner_pipeline, summarizer = None, None

# Streamlit App Header
st.title("Honoria Scale Negotiation Analysis App")

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
        df.fillna("No Communication", inplace=True)

        # Display cleaned dataset
        st.subheader("Cleaned Dataset Preview")
        st.dataframe(df)

        # Allow user to download the cleaned dataset
        cleaned_csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Cleaned Dataset", cleaned_csv, "cleaned_dataset.csv", "text/csv")

        # Step 5: Extract Reasons for Deviation from H Scale
        st.header("Step 2: Analyze Reasons for Deviation")
        df['Combined_Communication'] = df['Billing_Communication'] + " " + df['Billing_Communication_Part2']

        text_analysis_method = st.selectbox("Choose a Text Analysis Method", [
            "Hugging Face NER - Uses a transformer model to extract named entities.",
            "TF-IDF - Identifies important words based on their frequency in the text.",
            "LDA - Identifies topics from the text using Latent Dirichlet Allocation.",
            "BART Summarization - Uses BART to summarize key points."
        ])

        def extract_reasons(text):
            if not isinstance(text, str) or not text.strip():
                return []
            try:
                if "Hugging Face NER" in text_analysis_method and ner_pipeline:
                    entities = ner_pipeline(text)
                    reasons = list(set([ent['word'] for ent in entities]))
                elif "TF-IDF" in text_analysis_method:
                    vectorizer = TfidfVectorizer(stop_words='english')
                    tfidf_matrix = vectorizer.fit_transform([text])
                    feature_names = vectorizer.get_feature_names_out()
                    scores = tfidf_matrix.toarray().flatten()
                    top_n_words = [feature_names[i] for i in scores.argsort()[-5:]]
                    reasons = top_n_words
                elif "LDA" in text_analysis_method:
                    tokens = word_tokenize(text.lower())
                    dictionary = corpora.Dictionary([tokens])
                    corpus = [dictionary.doc2bow(tokens)]
                    lda_model = models.LdaModel(corpus, num_topics=2, id2word=dictionary, passes=10)
                    topics = lda_model.show_topics(num_words=5, formatted=False)
                    reasons = list(set([word for topic in topics for word, _ in topic[1]]))
                elif "BART Summarization" in text_analysis_method and summarizer:
                    reasons = [summarizer(text, max_length=50, min_length=25, do_sample=False)[0]['summary_text']]
                else:
                    reasons = []
            except Exception as e:
                st.error(f"Error in text processing: {e}")
                reasons = []
            return reasons
        
        df['Deviation_Reasons'] = df['Combined_Communication'].apply(extract_reasons)

        # One-Hot Encoding of Extracted Reasons for Correlation Analysis
        mlb = MultiLabelBinarizer()
        reasons_encoded = mlb.fit_transform(df['Deviation_Reasons'])
        reasons_df = pd.DataFrame(reasons_encoded, columns=mlb.classes_)
        df = pd.concat([df, reasons_df], axis=1)

        # Correlation Calculation
        numeric_df = df.select_dtypes(include=[np.number])  # Select only numeric columns
        correlation_matrix = numeric_df.corr()
        
        negotiation_correlation = correlation_matrix.get('Negotiation_Rounds', None)
        if negotiation_correlation is not None:
            negotiation_correlation = negotiation_correlation.drop(['Negotiation_Rounds'])
            negotiation_correlation = negotiation_correlation[negotiation_correlation.abs() > 0.1]  # Filter correlations

        # Display Extracted Reasons
        st.subheader("Identified Reasons for Deviation")
        st.dataframe(df[['Case_Reference', 'Assigned_Solicitor', 'Deviation_Reasons']])

        # Display Correlation Analysis
        st.subheader("Correlation Between Reasons and Negotiation Frequency")
        if negotiation_correlation is not None and not negotiation_correlation.empty:
            st.dataframe(negotiation_correlation.sort_values(ascending=False))
        else:
            st.warning("No significant correlation data available.")

        # Plot Correlation Heatmap
        st.subheader("Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Error processing the uploaded file: {e}")
