import os
import requests
from fastapi import FastAPI
from langgraph.graph import StateGraph
from langchain_qdrant import Qdrant
from langchain_huggingface import HuggingFaceEmbeddings  # ✅ Fixed import
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer, util

# Configurations
QDRANT_HOST = os.getenv("QDRANT_HOST", "http://localhost:6333")
GEMMA_SERVER_URL = os.getenv("GEMMA_SERVER_URL", "http://localhost:11434")  # Local Gemma 3
COLLECTION_NAME = "documents"

# Initialize Qdrant (disable version check to suppress warnings)
qdrant_client = QdrantClient(QDRANT_HOST)
embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en")  # ✅ Fixed embedding import
vector_store = Qdrant(qdrant_client, collection_name=COLLECTION_NAME, embeddings=embedding_model)

# Load BAAI/bge-reranker-base for reranking
reranker = SentenceTransformer("BAAI/bge-reranker-base")

app = FastAPI()

class RAGState:
    query: str
    retrieved_docs: list
    reranked_docs: list
    final_answer: str

graph = StateGraph(RAGState)

@graph.add_node()
def retrieve_documents(state: RAGState):
    """Retrieve top 5 similar documents from Qdrant."""
    docs = vector_store.similarity_search(state.query, k=5)
    return RAGState(query=state.query, retrieved_docs=docs)

@graph.add_node()
def rerank_documents(state: RAGState):
    """Use BAAI/bge-reranker-base for reranking retrieved documents."""
    query_embedding = reranker.encode(state.query, convert_to_tensor=True)
    
    reranked_docs = sorted(
        state.retrieved_docs,
        key=lambda doc: util.cos_sim(query_embedding, reranker.encode(doc.page_content, convert_to_tensor=True)),
        reverse=True
    )

    return RAGState(query=state.query, reranked_docs=reranked_docs)

@graph.add_node()
def generate_response(state: RAGState):
    """Generate final answer using locally running Gemma 3."""
    context = "\n\n".join([doc.page_content for doc in state.reranked_docs])
    prompt = f"Context:\n{context}\nQuestion: {state.query}"
    
    # Call local Gemma 3 for response generation
    response = requests.post(
        f"{GEMMA_SERVER_URL}/api/generate",
        json={"prompt": prompt, "temperature": 0.7, "max_tokens": 1024}
    )

    if response.status_code == 200:
        answer = response.json().get("response", "No response generated.")
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