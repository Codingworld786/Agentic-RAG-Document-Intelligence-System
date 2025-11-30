# app_gradio.py
import gradio as gr
from src.search import RAGSearch
from langchain_core.messages import HumanMessage
from src.prompt import prompt_llm

from src.agents import ask

# Load your RAG system (loads FAISS + LLM)
rag = RAGSearch(llm_model="gpt-4.1")   # or "gpt-4.1" if you want

def respond(message, history):
    yield "Thinking..."
    answer = ask(message)
    yield answer



# Beautiful dark interface
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# RAG Document Chatbot")
    gr.Markdown("Ask anything about your PDFs — powered by GPT-4o + FAISS")
    
    gr.ChatInterface(
        fn=respond,
        title="Your Personal Document Assistant",
        description="Trained on your research papers",
        examples=[
            "What is the attention mechanism?",
            "Explain random forest vs gradient boosting",
            "Summarize the Transformer paper"
        ]
    )

# Launch — SIMPLE AND WORKING
demo.launch(
    server_name="127.0.0.1",
    server_port=7860,
    share=False
)