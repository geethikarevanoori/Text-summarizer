import streamlit as st
import pdfplumber
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
import os
import requests
import time
from langchain.docstore.document import Document

# Configuration
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
QA_MODEL = "deepset/bert-base-cased-squad2"
SUMMARIZATION_MODEL = "facebook/bart-large-cnn"

# Initialize session state
if 'hf_token' not in st.session_state:
    st.session_state.hf_token = ""
if 'processed_docs' not in st.session_state:
    st.session_state.processed_docs = []
if 'faiss_index' not in st.session_state:
    st.session_state.faiss_index = None

# PDF Text Extraction
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        try:
            with pdfplumber.open(pdf) as pdf_reader:
                for page in pdf_reader.pages:
                    text += page.extract_text() or ""
            text += "\n\n"
        except Exception as e:
            st.error(f"Error reading {pdf.name}: {e}")
    return text.strip()

# Split text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return text_splitter.split_text(text)

# Create FAISS vector store (in-memory only)
def create_vector_store(text_chunks):
    try:
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        documents = [Document(page_content=chunk) for chunk in text_chunks]
        vector_store = FAISS.from_documents(documents, embeddings)
        st.session_state.faiss_index = vector_store
        st.session_state.processed_docs = text_chunks
        return True
    except Exception as e:
        st.error(f"Error creating vector store: {e}")
        return False

# Query HuggingFace Inference API with retry logic
def query_hf_api(model, payload, max_retries=3):
    if not st.session_state.hf_token:
        return None, "API token not provided"

    API_URL = f"https://api-inference.huggingface.co/models/{model}"
    headers = {"Authorization": f"Bearer {st.session_state.hf_token}"}

    for attempt in range(max_retries):
        try:
            response = requests.post(API_URL, headers=headers, json=payload)
            
            if response.status_code == 200:
                return response.json(), None
            elif response.status_code == 401:
                return None, "Invalid API token - please check your HuggingFace token"
            elif response.status_code == 503:
                time.sleep(5 * (attempt + 1))  # Exponential backoff
                continue
            else:
                return None, f"API error: {response.text}"
                
        except Exception as e:
            return None, f"Connection error: {str(e)}"

    return None, "Max retries reached - please try again later"

# Handle user queries
def user_input(user_question):
    if not st.session_state.faiss_index:
        st.error("Please process PDFs first")
        return

    try:
        docs = st.session_state.faiss_index.similarity_search(user_question, k=3)
        
        if not docs:
            st.warning("No relevant information found in the uploaded PDFs.")
            return

        context = " ".join([doc.page_content for doc in docs])

        # Display Summary
        summary_payload = {
            "inputs": context,
            "parameters": {"max_length": 150, "min_length": 50}
        }
        
        summary_result, error = query_hf_api(SUMMARIZATION_MODEL, summary_payload)
        if error:
            st.error(f"Summary error: {error}")
        elif summary_result:
            st.write("### 📋 Summary:")
            st.info(summary_result[0]['summary_text'])

        # Display AI Reply
        qa_payload = {
            "inputs": {
                "question": user_question,
                "context": context
            }
        }
        
        qa_result, error = query_hf_api(QA_MODEL, qa_payload)
        if error:
            st.error(f"QA error: {error}")
        elif qa_result:
            st.write("### 🤖 AI Reply:")
            st.success(qa_result['answer'])

    except Exception as e:
        st.error(f"Error processing your query: {e}")

# Streamlit UI
def main():
    st.set_page_config("Secure PDF Chatbot", layout='wide', page_icon="🔒")
    st.header("Secure PDF Chat Agent 🤖🔒")
    
    st.warning("Note: This app processes documents entirely in memory and doesn't save any files to disk for security.")

    with st.sidebar:
        st.title("🔑 API Configuration")
        hf_token = st.text_input(
            "HuggingFace API Token:", 
            type="password",
            help="Get your token from huggingface.co/settings/tokens"
        )
        
        if hf_token:
            st.session_state.hf_token = hf_token
            st.success("API token configured")
        
        st.title("📄 Document Upload")
        pdf_docs = st.file_uploader(
            "Upload PDF documents:", 
            accept_multiple_files=True,
            type=["pdf"]
        )
        
        if st.button("Process Documents"):
            if pdf_docs:
                with st.spinner("Processing documents securely..."):
                    raw_text = get_pdf_text(pdf_docs)
                    if raw_text.strip():
                        text_chunks = get_text_chunks(raw_text)
                        if create_vector_store(text_chunks):
                            st.success("Documents processed securely in memory!")
                    else:
                        st.error("Could not extract text from documents")
            else:
                st.warning("Please upload at least one document")

        st.markdown("---")
        st.markdown("""
        **Security Notes:**
        - No files are saved to disk
        - All processing happens in memory
        - Your API token is only kept for the current session
        """)

    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        user_input(user_question)

if __name__ == "__main__":
    main()