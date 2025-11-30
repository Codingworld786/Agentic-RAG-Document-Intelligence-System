# src/agents.py
from typing import Literal
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage
import textwrap

from src.search import RAGSearch
from src.llm import get_llm
from src.models import AgentState
from src.prompt import ROUTING_PROMPT, RAG_ANSWER_PROMPT, DIRECT_ANSWER_PROMPT
from src.logger import logger, router_logger, retrieval_logger, answer_logger

# Initialize
rag_search = RAGSearch(llm_model="gpt-4.1")
llm = get_llm(model="gpt-4.1")
retrieved_chunks = []

def decide_route(state: AgentState) -> AgentState:
    global retrieved_chunks
    retrieved_chunks = []

    prompt = ROUTING_PROMPT.format(question=state["question"])
    response = llm.invoke([HumanMessage(content=prompt)])
    raw = response.content.strip()
    route = "rag" if "RAG" in raw.upper() else "direct"

    router_logger.info(f"Question: {state['question']}")
    router_logger.info(f"LLM Routing Response: {raw}")
    router_logger.info(f"DECISION → {'RAG (Documents)' if route == 'rag' else 'DIRECT (General Knowledge)'}")
    
    return {**state, "route": route}

def retrieve_context(state: AgentState) -> AgentState:
    global retrieved_chunks
    context = rag_search.get_context(state["question"], top_k=6)
    
    # Get structured results
    structured = rag_search._get_structured_context(state["question"], top_k=6)
    retrieved_chunks = structured

    retrieval_logger.info(f"Retrieved {len(retrieved_chunks)} chunks for: {state['question']}")
    for i, chunk in enumerate(retrieved_chunks, 1):
        src = chunk.get("source", "Unknown")
        score = chunk.get("score", 0.0)
        snippet = chunk["text"].replace("\n", " ")[:200] + "..."
        retrieval_logger.info(f"  [{i}] Score: {score:.3f} | Source: {src}")
        retrieval_logger.debug(f"      → {snippet}")

    return {**state, "context": context, "retrieved_chunks": retrieved_chunks}

def generate_answer(state: AgentState) -> AgentState:
    question = state["question"]
    route = state["route"]

    if route == "rag" and retrieved_chunks:
        prompt = RAG_ANSWER_PROMPT.format(context=state["context"], question=question)
        source_info = f"{len(retrieved_chunks)} document(s)"
    else:
        prompt = DIRECT_ANSWER_PROMPT.format(question=question)
        source_info = "General knowledge"

    response = ""
    for chunk in llm.stream([HumanMessage(content=prompt)]):
        response += chunk.content
    answer = response.strip()

    answer_logger.info(f"Answer generated via: {source_info.upper()}")
    answer_logger.info(f"Final Answer:\n{textwrap.fill(answer, 100)}")

    logger.info("═" * 80)  # Visual separator in log

    return {**state, "answer": answer}

# Routing
def choose_path(state: AgentState) -> Literal["retrieve", "generate"]:
    return "retrieve" if state["route"] == "rag" else "generate"

# Build graph (same as before)
workflow = StateGraph(AgentState)
workflow.add_node("decide", decide_route)
workflow.add_node("retrieve", retrieve_context)
workflow.add_node("generate", generate_answer)
workflow.set_entry_point("decide")
workflow.add_conditional_edges("decide", choose_path, {"retrieve": "retrieve", "generate": "generate"})
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", END)

agentic_rag = workflow.compile()

def ask(question: str) -> str:
    initial_state = {
        "question": question,
        "context": "", "answer": "", "route": "direct", "retrieved_chunks": []
    }
    result = agentic_rag.invoke(initial_state)
    return result["answer"]

# Test
if __name__ == "__main__":
    logger.info("Agentic RAG System Started")
    print("Agentic RAG Ready! Logs → logs/agentic_rag_*.log")
    print("Type 'quit' to exit\n" + "═" * 80)

    while True:
        q = input("\nQuestion: ").strip()
        if q.lower() in ["quit", "exit", "bye"]:
            logger.info("Session ended by user")
            break
        if not q:
            continue
        ask(q)