import os
import requests
from dotenv import load_dotenv
from fastapi import FastAPI
from langgraph.graph import StateGraph
# from langchain_qdrant import Qdrant
from langchain_qdrant import QdrantVectorStore
# from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer, util
from pydantic import BaseModel
from typing import List

# Load environment variables from .env
load_dotenv()

# Configurations from .env
QDRANT_HOST = os.getenv("QDRANT_HOST", "http://localhost:6333")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4")
AZURE_OPENAI_REGION = os.getenv("AZURE_OPENAI_REGION")  # ✅ New: Set Azure Region

COLLECTION_NAME = "documents"

# Initialize Qdrant
qdrant_client = QdrantClient(QDRANT_HOST)
# embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en")
embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")

# vector_store = Qdrant(
vector_store = QdrantVectorStore(
    client=qdrant_client,
    collection_name=COLLECTION_NAME,
    embeddings=embedding_model
)

# Load BAAI/bge-reranker-base for reranking
# reranker = SentenceTransformer("BAAI/bge-reranker-base")
reranker = SentenceTransformer("BAAI/bge-reranker-base-v1.5")

app = FastAPI()

# Define RAGState
class RAGState(BaseModel):
    query: str
    retrieved_docs: List[dict] = []
    reranked_docs: List[dict] = []
    final_answer: str = ""

graph = StateGraph(RAGState)

@graph.add_node()
def retrieve_documents(state: RAGState) -> RAGState:
    """Retrieve top 5 similar documents from Qdrant."""
    docs = vector_store.similarity_search(state.query, k=5)
    return RAGState(query=state.query, retrieved_docs=[{"text": doc.page_content} for doc in docs])

@graph.add_node()
def rerank_documents(state: RAGState) -> RAGState:
    """Use BAAI/bge-reranker-base for reranking retrieved documents."""
    query_embedding = reranker.encode(state.query, convert_to_tensor=True)

    reranked_docs = sorted(
        state.retrieved_docs,
        key=lambda doc: util.cos_sim(query_embedding, reranker.encode(doc["text"], convert_to_tensor=True)),
        reverse=True
    )

    return RAGState(query=state.query, reranked_docs=reranked_docs)

@graph.add_node()
def generate_response(state: RAGState) -> RAGState:
    """Generate final answer using Azure OpenAI GPT-4."""
    context = "\n\n".join([doc["text"] for doc in state.reranked_docs])
    prompt = f"Context:\n{context}\nQuestion: {state.query}"

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {AZURE_OPENAI_API_KEY}",
        "api-key": AZURE_OPENAI_API_KEY,  # Required for Azure OpenAI
        "azure-region": AZURE_OPENAI_REGION  # ✅ New: Include Azure Region
    }

    response = requests.post(
        f"{AZURE_OPENAI_ENDPOINT}/openai/deployments/{AZURE_OPENAI_DEPLOYMENT_NAME}/chat/completions?api-version=2023-03-15-preview",
        headers=headers,
        json={
            "messages": [{"role": "system", "content": "You are a helpful AI assistant."},
                         {"role": "user", "content": prompt}],
            "max_tokens": 1024,
            "temperature": 0.7
        }
    )

    if response.status_code == 200:
        answer = response.json().get("choices", [{}])[0].get("message", {}).get("content", "No response generated.")
    else:
        answer = f"Error: {response.status_code}, {response.text}"

    return RAGState(query=state.query, final_answer=answer)

# Define Execution Flow
graph.add_edge(retrieve_documents, rerank_documents)
graph.add_edge(rerank_documents, generate_response)
graph.set_entry_point(retrieve_documents)
graph.set_exit_point(generate_response)

rag_pipeline = graph.compile()

@app.get("/query")
def query_rag(q: str):
    response = rag_pipeline.invoke(RAGState(query=q))
    return {"answer": response.final_answer}