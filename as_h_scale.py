import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, StratifiedKFold
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

    # Step 6: Feature Engineering
    df['Fee_Reduction_Percentage'] = ((df['Initial_Fees'] - df['Final_Fees']) / df['Initial_Fees']) * 100

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

    # Step 9: Encoding & Scaling (Removed PCA)
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

    # Step 10: Splitting Data using StratifiedKFold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    train_index, test_index = next(skf.split(df_transformed, df['Negotiation_Outcome']))
    X_train, X_test = df_transformed[train_index], df_transformed[test_index]
    y_train, y_test = df['Negotiation_Outcome'][train_index], df['Negotiation_Outcome'][test_index]

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

    # Step 13: Interactive Prediction
    st.subheader("Make a Prediction")
    user_input = {feature: st.number_input(feature, value=0.0) for feature in numerical_features}
    solicitor_input = st.text_input("Assigned Solicitor", "")

    if st.button("Predict Outcome"):
        user_data = np.array([list(user_input.values()) + [solicitor_input]])
        user_data_transformed = preprocessor.transform(user_data)
        prediction = models['Random Forest'].predict(user_data_transformed)[0]
        predicted_label = label_encoder.inverse_transform([prediction])[0]
        st.write(f"Predicted Negotiation Outcome: **{predicted_label}**")
