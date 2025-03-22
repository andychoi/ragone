# app/main.py

import os
import requests
from dotenv import load_dotenv
from fastapi import FastAPI
from langgraph.graph import StateGraph
from langchain_qdrant import QdrantVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
from sentence_transformers import SentenceTransformer, util
from pydantic import BaseModel
from typing import List

# Load environment variables from .env
load_dotenv()

# Configurations
QDRANT_HOST = os.getenv("QDRANT_HOST", "http://localhost:6333")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4")
AZURE_OPENAI_REGION = os.getenv("AZURE_OPENAI_REGION")
COLLECTION_NAME = "documents"

# Initialize clients
qdrant_client = QdrantClient(QDRANT_HOST)
embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")

# Ensure the collection exists in Qdrant
if COLLECTION_NAME not in [c.name for c in qdrant_client.get_collections().collections]:
    qdrant_client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=768,  # BAAI/bge-base-en-v1.5
            distance=Distance.COSINE
        )
    )

# Initialize Qdrant vector store
vector_store = QdrantVectorStore(
    client=qdrant_client,
    collection_name=COLLECTION_NAME,
    embedding=embedding_model
)

# Load reranker model
reranker = SentenceTransformer("BAAI/bge-reranker-base")

# FastAPI app
app = FastAPI()

# Define RAG State
class RAGState(BaseModel):
    query: str
    retrieved_docs: List[dict] = []
    reranked_docs: List[dict] = []
    final_answer: str = ""

# Define LangGraph
graph = StateGraph(RAGState)

# Define RAG nodes
def retrieve_documents(state: RAGState) -> RAGState:
    docs = vector_store.similarity_search(state.query, k=5)
    return RAGState(
        query=state.query,
        retrieved_docs=[{"text": doc.page_content} for doc in docs]
    )

def rerank_documents(state: RAGState) -> RAGState:
    query_embedding = reranker.encode(state.query, convert_to_tensor=True)

    reranked_docs = sorted(
        state.retrieved_docs,
        key=lambda doc: util.cos_sim(
            query_embedding,
            reranker.encode(doc["text"], convert_to_tensor=True)
        ),
        reverse=True
    )

    return RAGState(
        query=state.query,
        reranked_docs=reranked_docs
    )

def generate_response(state: RAGState) -> RAGState:
    context = "\n\n".join([doc["text"] for doc in state.reranked_docs])
    prompt = f"Context:\n{context}\nQuestion: {state.query}"

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {AZURE_OPENAI_API_KEY}",
        "api-key": AZURE_OPENAI_API_KEY,
        "azure-region": AZURE_OPENAI_REGION
    }

    response = requests.post(
        f"{AZURE_OPENAI_ENDPOINT}/openai/deployments/{AZURE_OPENAI_DEPLOYMENT_NAME}/chat/completions?api-version=2023-03-15-preview",
        headers=headers,
        json={
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 1024,
            "temperature": 0.7
        }
    )

    if response.status_code == 200:
        answer = response.json().get("choices", [{}])[0].get("message", {}).get("content", "No response generated.")
    else:
        answer = f"Error: {response.status_code}, {response.text}"

    return RAGState(
        query=state.query,
        final_answer=answer
    )

# Add named nodes
graph.add_node("retrieve_documents", retrieve_documents)
graph.add_node("rerank_documents", rerank_documents)
graph.add_node("generate_response", generate_response)

# Wire the flow using names
graph.add_edge("retrieve_documents", "rerank_documents")
graph.add_edge("rerank_documents", "generate_response")

# Set entry and exit points using node names
graph.set_entry_point("retrieve_documents")
graph.set_finish_point("generate_response")  # ✅ Correct method