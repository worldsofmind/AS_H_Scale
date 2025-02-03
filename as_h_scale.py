import streamlit as st
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer, util
from collections import Counter
import torch

# Load semantic similarity model
@st.cache_data
def load_model():
    return SentenceTransformer('paraphrase-MiniLM-L6-v2')

model = load_model()

# Function to clean and preprocess the text
def clean_text(text):
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text

# Function to extract key phrases using TF-IDF
def extract_key_phrases(text):
    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(2, 3))
    tfidf_matrix = vectorizer.fit_transform([text])
    feature_names = vectorizer.get_feature_names_out()
    tfidf_scores = tfidf_matrix.toarray().flatten()
    
    top_n_indices = tfidf_scores.argsort()[-10:][::-1]
    key_phrases = [feature_names[i] for i in top_n_indices]
    
    return key_phrases

# Function for semantic similarity analysis
def semantic_similarity_analysis(text, reference_phrases):
    text_embedding = model.encode(text, convert_to_tensor=True)
    reference_embeddings = model.encode(reference_phrases, convert_to_tensor=True)
    
    similarities = util.pytorch_cos_sim(text_embedding, reference_embeddings)[0]
    similar_phrases = []
    
    for i, score in enumerate(similarities):
        if score > 0.5:  # Adjust threshold as needed
            similar_phrases.append(f"Similar to: '{reference_phrases[i]}' (Score: {score.item():.2f})")
    
    return similar_phrases

# Function to analyze the legal text
def analyze_text(text):
    analysis = {
        "Key Reasons Identified": [],
        "Key Phrases": [],
        "Semantic Similarity Insights": []
    }
    
    cleaned_text = clean_text(text)
    
    # Extract key phrases
    key_phrases = extract_key_phrases(cleaned_text)
    analysis["Key Phrases"] = key_phrases
    
    # Reference phrases for semantic similarity
    reference_phrases = [
        "significant workload",
        "case complexity",
        "high volume of communications",
        "urgency and procedural challenges",
        "client-related difficulties",
        "administrative burden",
        "legal arguments",
        "document preparation",
        "court hearings",
        "difficult negotiations"
    ]
    
    similar_insights = semantic_similarity_analysis(cleaned_text, reference_phrases)
    analysis["Semantic Similarity Insights"] = similar_insights
    
    # Identify reasons based on keywords and semantic insights
    if any(re.search(r'\bcomplex|intricate|challenging\b', cleaned_text, re.I)):
        analysis["Key Reasons Identified"].append("Complexity of the Case")
    if any(re.search(r'\bworkload|hours spent|extensive\b', cleaned_text, re.I)):
        analysis["Key Reasons Identified"].append("Significant Workload and Time Commitment")
    if any(re.search(r'\bcommunication|emails|correspondence\b', cleaned_text, re.I)):
        analysis["Key Reasons Identified"].append("High Volume of Communications")
    if any(re.search(r'\burgency|urgent|last minute\b', cleaned_text, re.I)):
        analysis["Key Reasons Identified"].append("Urgency and Procedural Challenges")
    if any(re.search(r'\bdifficult client|uncooperative|challenging parties\b', cleaned_text, re.I)):
        analysis["Key Reasons Identified"].append("Client-Related Difficulties")
    
    return analysis

# Streamlit App Interface
st.title("Legal Bill Deviation Analysis Tool")

uploaded_file = st.file_uploader("Upload a legal document (text file preferred):", type=["txt"])

if uploaded_file:
    try:
        # Try decoding with UTF-8
        legal_text = uploaded_file.read().decode("utf-8")
    except UnicodeDecodeError:
        try:
            # Fallback to ISO-8859-1 if UTF-8 fails
            legal_text = uploaded_file.read().decode("ISO-8859-1")
        except Exception as e:
            st.error(f"Unable to read the file: {e}")
            legal_text = None

    if legal_text:
        st.subheader("Uploaded Text Preview:")
        st.text_area("Legal Text", legal_text, height=300)
        
        if st.button("Analyze Text"):
            with st.spinner("Analyzing the legal document..."):
                analysis_result = analyze_text(legal_text)
            
            st.success("Analysis Complete!")
            
            st.subheader("Key Reasons Identified")
            for reason in analysis_result["Key Reasons Identified"]:
                st.markdown(f"- **{reason}**")
            
            st.subheader("Key Phrases")
            for phrase in analysis_result["Key Phrases"]:
                st.markdown(f"- {phrase}")
            
            st.subheader("Semantic Similarity Insights")
            for insight in analysis_result["Semantic Similarity Insights"]:
                st.markdown(f"- {insight}")
else:
    st.info("Please upload a legal document to begin the analysis.")
