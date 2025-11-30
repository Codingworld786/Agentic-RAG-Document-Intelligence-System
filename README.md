# Agentic RAG Chatbot

A **smart personal document assistant** that lets you upload PDFs, DOCX, and TXT files — then ask questions in natural language.  
It automatically decides whether to search your documents (RAG) or answer directly using the LLM — no unnecessary retrieval, zero hallucinations on your data.

### Features
- Upload any document (PDF, DOCX, TXT) via web interface
- Intelligent routing using **LangGraph** (document-based vs general knowledge)
- Real-time incremental indexing with **FAISS** + SentenceTransformers
- Streaming responses with source tracking
- Fully persistent — your documents and knowledge base survive restarts
- One-command deployment with **Docker**

### Tech Stack
- LangChain • LangGraph • FAISS • SentenceTransformers
- OpenAI GPT-4.1 (or any LLM)
- Gradio (beautiful dark UI)
- Docker + docker-compose

### Quick Start (Recommended)
```bash
git clone https://github.com/Codingworld786/Agentic-RAG-Document-Intelligence-System.git
cd Agentic-RAG-Document-Intelligence-System

# Add your OpenAI key
echo "OPENAI_API_KEY=your-key-here" > .env

docker compose up --build
