from langchain_core.prompts import PromptTemplate

prompt_llm= """You are an expert assistant. Answer the question using ONLY the context below. 
Be accurate, concise, and professional. Do not add information that is not present in the context.

Question: {question}

Context:
{context}

Answer:"""


# Routing decision prompt
ROUTING_PROMPT = PromptTemplate.from_template("""
You are an intelligent routing agent.

Your only job: decide if the user's question can be accurately answered using the uploaded documents (PDFs, research papers, notes) or if it requires general world knowledge.

Question: {question}

Answer with exactly ONE word: RAG or DIRECT

Guidelines:
- Choose RAG if the question mentions concepts, papers, authors, models, techniques, or facts that are likely in the user's documents (e.g., attention, transformer, BERT, ResNet, specific equations, "the paper says", "according to the document", etc.)
- Choose RAG even if you're not 100% sure — better to check documents than miss information
- Choose DIRECT only for pure general knowledge, math, coding problems, jokes, opinions, current events, or casual chat

Examples:
"Explain multi-head attention" → RAG
"What is the main contribution of the Transformer paper?" → RAG
"How does self-attention work?" → RAG
"Tell me a joke" → DIRECT
"What is 17 × 24?" → DIRECT
"Who won the 2025 World Cup?" → DIRECT

Now decide for this question.

Answer (RAG or DIRECT only):""")

# Keep these unchanged
RAG_ANSWER_PROMPT = PromptTemplate.from_template("""
You are an expert answering based ONLY on the provided context.

Context:
{context}

Question: {question}

Answer accurately and cite key points. If the context lacks info, say:
"I don't have sufficient information from the documents to answer confidently."

Answer:""")

DIRECT_ANSWER_PROMPT = PromptTemplate.from_template("""
You are a helpful AI assistant.

Question: {question}

Answer naturally and clearly:""")
