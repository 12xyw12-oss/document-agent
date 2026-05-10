import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "your-api-key-here")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
LLM_MODEL = "gpt-3.5-turbo-1106"
EMBEDDING_MODEL = "text-embedding-3-small"

VECTOR_DB_PATH = "./data/vector_db"
COLLECTION_NAME = "enterprise_docs"

KNOWLEDGE_GRAPH_PATH = "./data/knowledge_graph"

DOCUMENT_DIR = "./data/documents"
SUPPORTED_FORMATS = [".pdf", ".docx", ".pptx", ".txt", ".md", ".csv", ".xlsx"]

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

os.makedirs(DOCUMENT_DIR, exist_ok=True)
os.makedirs(VECTOR_DB_PATH, exist_ok=True)
os.makedirs(KNOWLEDGE_GRAPH_PATH, exist_ok=True)