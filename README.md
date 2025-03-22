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

pip install sentence-transformers
python3 -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-reranker-base')"

Caches it in: ~/.cache/huggingface/hub/models--BAAI--bge-reranker-base

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


⸻

URL	Description
/	Home page with helpful links
/health	Health check endpoint
/query?q=...	Your RAG endpoint
/query-form	Simple HTML form for asking questions
/docs	Swagger UI
/redoc	ReDoc docs


⸻

1️⃣ Rebuild and Restart Containers

docker-compose down --rmi all --volumes --remove-orphans
docker-compose build --no-cache
docker-compose up -d

2️⃣ Test It

docker logs rag-service -f
curl "http://localhost:8000/query?q=What is Azure OpenAI?"



⸻
ß
✅ Final Setup Overview

Component	Model	Tool
Embedding	BAAI/bge-base-en-v1.5	HuggingFaceEmbeddings
Reranker	BAAI/bge-reranker-v2-m3	CrossEncoder
Response Generator	Azure OpenAI GPT-4	API


