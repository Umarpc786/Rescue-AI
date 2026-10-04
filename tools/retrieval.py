from rag.faiss_store import FAISSKnowledgeBase

kb = FAISSKnowledgeBase()

def search_emergency_knowledge(query: str) -> list:
    """Retrieve relevant protocol documents from FAISS index."""
    return kb.search_protocol(query, k=2)
