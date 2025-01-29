import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import LatentDirichletAllocation
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

# Streamlit App Header
st.title("Honoria Scale Negotiation Prediction App")

# Step 1: Upload Dataset
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

        # Step 6: Proceed to Feature Engineering
        if st.button("Proceed to Feature Engineering"):
            st.session_state['dataset_cleaned'] = True
            st.session_state['df_cleaned'] = df.copy()

    except Exception as e:
        st.error(f"Error processing the uploaded file: {e}")

# Step 2: Feature Engineering (Only if dataset is cleaned)
if 'dataset_cleaned' in st.session_state and 'df_cleaned' in st.session_state:
    df = st.session_state['df_cleaned']
    st.header("Step 2: Feature Engineering")

    # Sentiment Analysis using multiple methods
    vader_analyzer = SentimentIntensityAnalyzer()
    flair_analyzer = TextClassifier.load('sentiment')

    df['TextBlob_Sentiment'] = df['Billing_Communication'].apply(lambda x: TextBlob(x).sentiment.polarity)
    df['VADER_Sentiment'] = df['Billing_Communication'].apply(lambda x: vader_analyzer.polarity_scores(x)['compound'])
    df['Flair_Sentiment'] = df['Billing_Communication'].apply(lambda x: flair_analyzer.predict(Sentence(x)) or 0)

    # Step 7: Extract Features from Billing Communication using BERT embeddings
    model = SentenceTransformer('all-MiniLM-L6-v2')
    df['BERT_Embeddings'] = df['Billing_Communication'].apply(lambda x: model.encode(x))
    bert_embeddings = np.vstack(df['BERT_Embeddings'].values)

    # Step 8: Apply LDA Topic Modeling on BERT embeddings with Non-Negativity Constraint
    scaler = MinMaxScaler(feature_range=(0, 1))
    bert_embeddings_scaled = scaler.fit_transform(bert_embeddings)

    lda_model = LatentDirichletAllocation(n_components=5, random_state=42)
    lda_topics = lda_model.fit_transform(bert_embeddings_scaled)

    for i in range(5):
        df[f'Topic_{i}'] = lda_topics[:, i]
    df.drop(columns=['BERT_Embeddings'], inplace=True)

    # Step 9: Encoding & Scaling
    categorical_features = ['Assigned_Solicitor']
    numerical_features = ['Negotiation_Rounds', 'Initial_Fees', 'Final_Fees', 'Fee_Reduction_Percentage', 'TextBlob_Sentiment', 'VADER_Sentiment', 'Flair_Sentiment'] + [f'Topic_{i}' for i in range(5)]

    categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse=False)
    numerical_transformer = StandardScaler()

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    df_transformed = preprocessor.fit_transform(df)

    # Step 10: Safe Train-Test Split for Small Datasets
    if len(df) >= 5:
        X_train, X_test, y_train, y_test = train_test_split(df_transformed, df['Negotiation_Outcome'], test_size=0.2, random_state=42, stratify=df['Negotiation_Outcome'])
    else:
        st.warning("Not enough samples for a proper train-test split. Using all data for training.")
        X_train, y_train = df_transformed, df['Negotiation_Outcome']
        X_test, y_test = X_train, y_train  # Use the same data for evaluation

    # Step 11: Model Selection and Hyperparameter Tuning
    model_choice = st.selectbox("Choose a model", ["Random Forest", "Gradient Boosting", "Logistic Regression", "SVM", "XGBoost"])
    param_grid = {
        'Random Forest': {'n_estimators': [50, 100, 200], 'max_depth': [3, 5, 7]},
        'Gradient Boosting': {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 0.2]},
        'XGBoost': {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 0.2], 'max_depth': [3, 5, 7]}
    }
    
    model = eval(model_choice.replace(" ", "_"))()
    grid_search = GridSearchCV(model, param_grid.get(model_choice, {}), cv=3)
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    st.write(f"Best Hyperparameters for {model_choice}: {grid_search.best_params_}")
