# Multi-PDF Chat Agent 🤖📚

A Streamlit-based application that processes multiple PDFs, creates embeddings using FAISS, and provides question answering and summarization using BERT and Pegasus models.

## Features

- 📄 **PDF Processing**: Extract text from multiple PDFs simultaneously
- 🧠 **AI Summarization**: Generate concise summaries using Google's Pegasus model
- ❓ **Question Answering**: Get precise answers using BERT-large model
- 🔍 **Semantic Search**: FAISS vector store for efficient document retrieval
- 🎨 **User-Friendly UI**: Clean Streamlit interface with responsive design

## Prerequisites

- Python 3.10+
- Docker (optional for containerized deployment)

## Installation

### Local Development

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/text-summarizer.git
   cd text-summarizer
   Install dependencies:
   
pip install -r requirements.txt
Download models (takes 10-30 minutes):

python download_models.py

Docker Deployment

Build the Docker image:


docker build -t pdf-chatbot .
Run the container:


docker run -p 8501:8501 -v $(pwd)/faiss_index:/app/faiss_index pdf-chatbot
Usage
Launch the application:


streamlit run revanoori.py
In the web interface:

Upload PDF files in the sidebar

Click "Submit & Process" to create the knowledge base

Ask questions in the main input field

View summaries and answers in real-time

File Structure
text-summarizer/
├── models/               # Pre-downloaded model files
│   ├── bert-qa/          # BERT question-answering model
│   └── pegasus/          # Pegasus summarization model
├── faiss_index/          # Vector store directory (auto-created)
├── revanoori.py          # Main application code
├── download_models.py    # Model download script
├── Dockerfile            # Container configuration
└── requirements.txt      # Python dependencies
