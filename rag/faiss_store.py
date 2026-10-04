import os
import json
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

class FAISSKnowledgeBase:
    def __init__(self, json_path: str = "rag/documents/emergency_protocols.json"):
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.json_path = json_path
        self.vector_store = None
        self._initialize_kb()

    def _initialize_kb(self):
        if not os.path.exists(self.json_path):
            docs = [Document(page_content="Standard emergency protocol: Contact primary responders, secure area, render assistance.")]
        else:
            with open(self.json_path, "r") as f:
                data = json.load(f)
            docs = [
                Document(
                    page_content=item["content"],
                    metadata={"id": item["id"], "title": item["title"], "category": item["category"]}
                ) for item in data
            ]
        self.vector_store = FAISS.from_documents(docs, self.embeddings)

    def search_protocol(self, query: str, k: int = 2):
        if not self.vector_store:
            return []
        results = self.vector_store.similarity_search(query, k=k)
        return [{"content": doc.page_content, "metadata": doc.metadata} for doc in results]
