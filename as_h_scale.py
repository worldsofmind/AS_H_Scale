import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import shap
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Streamlit App Header
st.title("Honoria Scale Negotiation Prediction App")

# Step 1: Upload Dataset
st.header("Step 1: Upload and Clean Dataset")
uploaded_file = st.file_uploader("Upload Excel file", type=["xlsx"])
if uploaded_file:
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

# Step 2: Feature Engineering (Only if dataset is cleaned)
if 'dataset_cleaned' in st.session_state:
    st.header("Step 2: Feature Engineering")

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
    numerical_features = ['Negotiation_Rounds', 'Initial_Fees', 'Final_Fees', 'Fee_Reduction_Percentage'] + [f'Topic_{i}' for i in range(5)]

    categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
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

    # Step 11: Model Training
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Logistic Regression': LogisticRegression(max_iter=5000, multi_class='ovr', solver='lbfgs', C=0.1),
        'SVM': SVC(kernel='linear', probability=True)
    }

    for name, model in models.items():
        model.fit(X_train, y_train)
    
    # Step 12: Model Evaluation
    st.subheader("Model Performance")
    for name, model in models.items():
        y_pred = model.predict(X_test)
        st.write(f"**{name}** Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        st.text(classification_report(y_test, y_pred))

        # Display Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title(f"Confusion Matrix - {name}")
        st.pyplot(fig)

