# app_gradio.py
import gradio as gr
from src.agents import ask, rag_search
from pathlib import Path


def respond(message, history):
    """Handle user messages and return conversation history"""
    if not message.strip():
        return history
    
    try:
        answer = ask(message)
        # history = history + [[message, answer]]
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": answer})
    except Exception as e:
        # history = history + [[message, f"Error: {str(e)}"]]
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": f"Error: {str(e)}"})
    
    return history


def handle_upload(file_paths):
    """
    When files are uploaded:
    1. Index them
    2. Update the global rag_search embeddings
    """
    if file_paths is None or len(file_paths) == 0:
        return "No files uploaded."
    
    results = []
    for file_path in file_paths:
        file_path = Path(file_path)
        
        if not file_path.exists():
            results.append(f"✗ {file_path.name}: File not found")
            continue
        
        try:
            rag_search.index_file(str(file_path))
            results.append(f"✓ {file_path.name}")
        except Exception as e:
            results.append(f"✗ {file_path.name}: {str(e)}")
    
    return "\n".join(results)


# === Gradio UI (Compatible with older versions) ===
with gr.Blocks() as demo:
    gr.Markdown("# Agentic RAG Document Assistant")
    gr.Markdown("Ask anything • Upload new PDFs/TXTs • Smart routing (RAG or direct)")

    chatbot = gr.Chatbot()
    # chatbot = gr.Chatbot(type="messages")
    
    msg = gr.Textbox(
        placeholder="Type your question here...",
        label="Question"
    )
    
    with gr.Row():
        submit_btn = gr.Button("Submit")
        clear_btn = gr.Button("Clear")
    
    # File upload section
    gr.Markdown("---")
    gr.Markdown("### 📁 Upload Documents to Knowledge Base")
    
    upload = gr.File(
        label="Select File (PDF, TXT, DOCX, MD)",
        file_types=[".pdf", ".txt", ".docx", ".md"],
        file_count="multiple"
    )
    upload_status = gr.Textbox(
        label="Status",
        interactive=False
    )
    
    # Wire up events
    upload.upload(
        fn=handle_upload,
        inputs=upload,
        outputs=upload_status
    )
    
    def clear_all():
        return "", []
    
    submit_btn.click(
        fn=respond,
        inputs=[msg, chatbot],
        outputs=chatbot
    )
    
    msg.submit(
        fn=respond,
        inputs=[msg, chatbot],
        outputs=chatbot
    )
    
    clear_btn.click(
        fn=clear_all,
        inputs=None,
        outputs=[msg, chatbot]
    )


# Launch
if __name__ == "__main__":
    print("Starting Gradio app...")
    print(f"Gradio version: {gr.__version__}")
    
    try:
        demo.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=True,  # Use share=True to bypass localhost issues
            show_error=True
        )
    except Exception as e:
        print(f"\n⚠️  Error: {e}")
        print("\nTrying alternative launch method...\n")
        demo.launch(share=True)