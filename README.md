# ragone
RAG One

⸻

📌 Directory Structure

rag-service/
│── docker-compose.yml
│── Dockerfile
│── app/
│   │── main.py          # RAG pipeline
│   │── ingestion.py     # Document ingestion
│   │── requirements.txt # Dependencies


⸻

📌 Step 6: Running the Service

1️⃣ Start the service:

docker-compose up --build

2️⃣ Ingest a document:

docker exec -it rag_pipeline python ingestion.py

3️⃣ Query the RAG service:

curl "http://localhost:8000/query?q=What is AI?"


⸻

🔹 Why Use BAAI/bge-reranker-base for Reranking?

✅ More Accurate Reranking → Uses cross-encoder ranking instead of simple vector similarity.
✅ Better Query-Document Matching → Focuses on semantic relationships, not just cosine similarity.
✅ Improves Final Response Quality → Ensures the best documents are selected before response generation.

⸻


📌 Summary
	•	Document Extraction → Docling
	•	Vector Search → Qdrant
	•	Embedding Model → DeepSeek
	•	Reranking → ✅ BAAI/bge-reranker-base
	•	Response Generation → DeepSeek LLM
	•	API Interface → FastAPI
	•	Graph-based Execution → LangGraph

