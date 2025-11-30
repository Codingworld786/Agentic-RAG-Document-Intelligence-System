from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

load_dotenv()

def get_llm(model: str = "gpt-4.1", temperature: float = 0.2):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in .env file")
    
    return ChatOpenAI(
        model=model,
        temperature=temperature,
        max_tokens=1024,
    )