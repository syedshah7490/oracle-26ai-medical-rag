# Oracle 26ai Medical RAG Chatbot

A Retrieval-Augmented Generation chatbot that answers questions from WHO medical documents using Oracle AI Database 26ai vector search, Ollama embeddings and Google Gemini.

## Knowledge Base

The project supports documents covering:

- Anaemia
- Diabetes
- Hypertension

The WHO PDF files are not included in this repository. Download them from official WHO sources and place them inside the `data/` directory.

## Features

- PDF text extraction
- OCR fallback with Tesseract
- Document chunking
- 768-dimensional embeddings using `nomic-embed-text`
- Oracle 26ai vector storage and similarity search
- Gemini-generated grounded answers
- Source-aware responses
- Interactive Streamlit interface

## Technology Stack

- Python 3.11
- Oracle AI Database 26ai
- Oracle AI Vector Search
- Ollama
- `nomic-embed-text`
- Google Gemini API
- Streamlit
- PyPDF and Tesseract OCR

## Environment Variables

Copy `.env.example` to `.env` and provide your private values:

```env
ORACLE_DB_USER=your_oracle_username
ORACLE_DB_PASSWORD=}
ORACLE_DB_DSN=localhost:1521/FREEPDB1
GEMINI_API_KEY=your_gemini_api_key
