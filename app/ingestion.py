import os
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from langchain.embeddings import HuggingFaceEmbeddings
from docling.document_converter import DocumentConverter
from uuid import uuid4

# Configurations
QDRANT_HOST = os.getenv("QDRANT_HOST", "http://localhost:6333")
COLLECTION_NAME = "documents"

# Initialize services
qdrant_client = QdrantClient(QDRANT_HOST)
embedding_model = HuggingFaceEmbeddings(model_name="deepseek-ai/deepseek-embedding")

def ingest_document(file_path):
    converter = DocumentConverter()
    result = converter.convert(file_path)
    text_data = result.document.export_to_markdown()

    vector = embedding_model.embed_query(text_data)
    doc_id = str(uuid4())

    qdrant_client.upsert(
        collection_name=COLLECTION_NAME,
        points=[PointStruct(id=doc_id, vector=vector, payload={"text": text_data})]
    )
    print(f"✅ Document {doc_id} stored in Qdrant.")

# Example usage
if __name__ == "__main__":
    ingest_document("example.pdf")