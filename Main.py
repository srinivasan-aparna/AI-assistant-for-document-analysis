import streamlit as st
import requests
import numpy as np
import PyPDF2
import faiss
import pickle
import os
from typing import List, Dict, Any


DEEPSEEK_API_KEY = "**"

#  session state
if 'session_id' not in st.session_state:
    st.session_state.session_id = str(np.random.randint(0, 1000000))
    st.session_state.processed = False

# Paths with session ID to prevent conflicts
SESSION_ID = st.session_state.session_id
FAISS_INDEX_PATH = f"faiss_index_{SESSION_ID}.faiss"
DOCUMENT_STORE_PATH = f"document_store_{SESSION_ID}.pkl"

# Function to save FAISS index and documents
def save_data(faiss_index, documents):
    faiss.write_index(faiss_index, FAISS_INDEX_PATH)
    with open(DOCUMENT_STORE_PATH, 'wb') as f:
        pickle.dump(documents, f)

# Function to load FAISS index and documents
def load_data():
    if os.path.exists(FAISS_INDEX_PATH) and os.path.exists(DOCUMENT_STORE_PATH):
        faiss_index = faiss.read_index(FAISS_INDEX_PATH)
        with open(DOCUMENT_STORE_PATH, 'rb') as f:
            documents = pickle.load(f)
        return faiss_index, documents
    return None, None

# Function to extract text from an uploaded PDF
def extract_text_from_pdf(uploaded_file):
    pdf_reader = PyPDF2.PdfReader(uploaded_file)
    text = ""
    for page in pdf_reader.pages:
        extracted_text = page.extract_text()
        if extracted_text:
            text += extracted_text + "\n"
    return text.strip()

# Function to split text into chunks
def split_text(text, chunk_size=500, overlap=100):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

# Function to get embeddings from DeepSeek
def get_deepseek_embeddings(texts):
    url = "https://api.deepseek.com/v1/embeddings"
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {"input": texts}

    response = requests.post(url, json=data, headers=headers)
    response_json = response.json()

    if "data" in response_json:
        return np.array([item["embedding"] for item in response_json["data"]])
    else:
        raise ValueError(f"Error fetching embeddings: {response_json}")

# Function to process and store PDF with FAISS
def process_and_store_pdf(pdf_text):
    try:
        chunks = split_text(pdf_text)
        
        if not chunks:
            return False
        
        vectors = get_deepseek_embeddings(chunks)
        vectors = vectors.astype('float32') 
        
        # Create and build FAISS index
        dimension = vectors.shape[1]
        faiss_index = faiss.IndexFlatL2(dimension)  
        faiss_index.add(vectors)
        
        save_data(faiss_index, chunks)
        st.session_state.processed = True
        return True
    except Exception as e:
        st.error(f"Error processing PDF: {str(e)}")
        return False

# Function to retrieve relevant text chunks
def retrieve_relevant_text(query, faiss_index, documents, k=3):
    query_vector = get_deepseek_embeddings([query])
    query_vector = query_vector.astype('float32')
    
    D, I = faiss_index.search(query_vector, k)
    results = []
    
    for i, idx in enumerate(I[0]):
        if idx >= 0 and idx < len(documents):
            results.append({
                "text": documents[idx],
                "score": float(D[0][i])
            })
    
    # Sort by score (lower is better for L2 distance)
    results.sort(key=lambda x: x["score"])
    
    return results

# Function to interact with DeepSeek Chat API
def deepseek_chat(query, context):
    url = "https://api.deepseek.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": "You are an AI assistant that answers questions based on the given context."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
        ],
        "max_tokens": 500,
        "temperature": 0.2
    }

    response = requests.post(url, json=data, headers=headers)
    response_json = response.json()

    if "choices" in response_json:
        return response_json["choices"][0]["message"]["content"].strip()
    else:
        raise ValueError(f"Error fetching chat response: {response_json}")

# Streamlit UI 
st.sidebar.image("Logo.png", use_column_width=True)
st.sidebar.title("Upload and Settings")

# Upload PDF file
uploaded_file = st.sidebar.file_uploader("Upload a PDF file", type=["pdf"])
pdf_text = ""

if uploaded_file:
    pdf_text = extract_text_from_pdf(uploaded_file)
    st.sidebar.success("PDF uploaded successfully!")

# Process PDF button
if uploaded_file is not None and pdf_text:
    if st.sidebar.button("Process PDF"):
        with st.spinner("Processing PDF... This may take a moment."):
            success = process_and_store_pdf(pdf_text)
            if success:
                st.sidebar.success("PDF processed successfully!")
            else:
                st.sidebar.error("Failed to process PDF.")

# Main content area
st.title("Customer Specification Review System")

# Load FAISS index and documents if available
faiss_index, documents = load_data()

# User question input
question = st.text_input("Ask a question based on the document:")

if st.button("Get Answer"):
    if not pdf_text:
        st.warning("Please upload a PDF first.")
    else:
        context = ""
        
        # If we have a FAISS index, use it to find relevant chunks
        if faiss_index is not None and documents is not None:
            with st.spinner("Searching for relevant information..."):
                relevant_chunks = retrieve_relevant_text(question, faiss_index, documents)
                context = "\n\n".join([chunk["text"] for chunk in relevant_chunks])
        else:
            # Fall back to using all text if FAISS isn't available
            context = pdf_text
        
        with st.spinner("Generating answer..."):
            answer = deepseek_chat(question, context)
            st.write("###  Answer:")
            st.write(answer)
            
            # Show sources if we used FAISS
            if faiss_index is not None and documents is not None and relevant_chunks:
                if st.checkbox("Show source passages"):
                    st.markdown("### Relevant Passages")
                    for i, chunk in enumerate(relevant_chunks):
                        with st.expander(f"Passage {i+1} (Distance: {chunk['score']:.2f})"):
                            st.write(chunk["text"])

# Cleanup function to remove temporary files
def cleanup():
    for path in [FAISS_INDEX_PATH, DOCUMENT_STORE_PATH]:
        if os.path.exists(path):
            try:
                os.remove(path)
            except:
                pass

# Register cleanup to execute when the app closes
import atexit
atexit.register(cleanup)