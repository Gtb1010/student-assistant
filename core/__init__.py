# core/__init__.py
from core.auth           import login, register, can_upload, is_admin
from core.chunker        import TextChunker
from core.document_parser import DocumentParser
from core.embedder       import EmbeddingManager
from core.vector_store   import VectorStore
from core.retriever      import CustomRetriever, Chunk
from core.llm_client     import LLMClient
from core.chat_history   import FileChatMessageHistory
from core.quiz_generator import QuizGenerator