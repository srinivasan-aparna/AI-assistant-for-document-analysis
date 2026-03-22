# AI-assistant-for-document-analysis
The AI Assistant designed to streamline the analysis of large-scale engineering and technical documents. It enables users to upload PDF specifications and interact with them through natural language queries, significantly reducing manual review effort while maintaining traceability and accuracy.

This system leverages advanced LLM pipelines, embedding-based retrieval, and efficient vector search to deliver context-aware, explainable responses.

# 🚀 Key Contributions
* 📂 PDF Ingestion & Processing:
Extracts and processes text from uploaded technical documents.
* 🔍 Semantic Search with FAISS:
Uses embedding-based similarity search to retrieve the most relevant document sections.
* 🧩 Retrieval-Augmented Generation (RAG):
 Combines retrieved context with LLM reasoning for accurate, grounded responses.
* 🧠 Context-Aware Question Answering:
 Enables multi-step reasoning over engineering specifications.
* 📊 Explainability & Traceability:
 Displays source passages used to generate answers.
* ⚡ Session-Based Isolation:
 Ensures independent processing using unique session IDs.

# 🏗️ System Architecture
1. Document Processing Pipeline
* PDF upload via Streamlit UI
* Text extraction using PyPDF2
* Chunking with overlap for contextual continuity
2. Embedding Layer
*Embeddings generated via DeepSeek API
* Converts text chunks into dense vector representations
3. Vector Indexing
* FAISS (IndexFlatL2) used for efficient similarity search
* Stores embeddings for fast retrieval
4. Retrieval Layer
* Top-K relevant chunks retrieved based on query similarity
5. Generation Layer
*DeepSeek Chat API generates answers using:
* User query
* Retrieved contextual passages

# 🖥️ Tech Stack
* Frontend: Streamlit
* LLM & Embeddings: DeepSeek API
* Vector Database: FAISS
* Backend: Python
* Document Processing: PyPDF2
* Data Handling: NumPy, Pickle

# 📊 Impact
* 📉 Reduced manual engineering review time by 40%
* 📄 Automated analysis of over 2,000 pages of technical specifications
* 🎯 Improved response accuracy through optimized retrieval and prompt tuning
* 🔍 Enhanced explainability with source-backed answers
