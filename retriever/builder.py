from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from config.settings import settings
from collections import defaultdict
import logging
import uuid

logger = logging.getLogger(__name__)


def score_documents(retriever: EnsembleRetriever, question: str) -> dict:
    """Compute a 0-1 relevance score per retrieved passage, keyed by page_content.

    EnsembleRetriever's weighted reciprocal rank fusion computes exactly this score
    internally to rank results, but discards it before returning Documents (see
    langchain_classic.retrievers.ensemble.EnsembleRetriever.weighted_reciprocal_rank).
    This recomputes the same score from the same retrievers/weights/c so citations
    can show a real, derived confidence instead of a fabricated one.
    """
    doc_lists = [r.invoke(question) for r in retriever.retrievers]
    rrf_score = defaultdict(float)
    for doc_list, weight in zip(doc_lists, retriever.weights):
        for rank, doc in enumerate(doc_list, start=1):
            rrf_score[doc.page_content] += weight / (rank + retriever.c)
    if not rrf_score:
        return {}
    max_score = max(rrf_score.values())
    return {content: (score / max_score) for content, score in rrf_score.items()}

class RetrieverBuilder:
    def __init__(self):
        """Initialize the retriever builder with a local embedding model (free, runs on CPU, no API key)."""
        self.embeddings = HuggingFaceEmbeddings(
            model_name=settings.EMBEDDING_MODEL,
            encode_kwargs={"normalize_embeddings": True},
        )
        logger.info(f"RetrieverBuilder using local embedding model '{settings.EMBEDDING_MODEL}'.")

    def build_hybrid_retriever(self, docs, collection_name: str = None):
        """Build a hybrid retriever using BM25 and vector-based retrieval.

        Uses a fresh, uniquely-named in-memory Chroma collection per call so that
        concurrent users on a shared deployment never see each other's documents.
        """
        try:
            # Create Chroma vector store in its own collection (no shared persist_directory -
            # multiple visitors hitting the same deployed app must not share document state).
            vector_store = Chroma.from_documents(
                documents=docs,
                embedding=self.embeddings,
                collection_name=collection_name or f"session-{uuid.uuid4().hex}",
            )
            logger.info("Vector store created successfully.")

            # Create BM25 retriever
            bm25 = BM25Retriever.from_documents(docs)
            logger.info("BM25 retriever created successfully.")

            # Create vector-based retriever
            vector_retriever = vector_store.as_retriever(search_kwargs={"k": settings.VECTOR_SEARCH_K})
            logger.info("Vector retriever created successfully.")

            # Combine retrievers into a hybrid retriever
            hybrid_retriever = EnsembleRetriever(
                retrievers=[bm25, vector_retriever],
                weights=settings.HYBRID_RETRIEVER_WEIGHTS
            )
            logger.info("Hybrid retriever created successfully.")
            return hybrid_retriever
        except Exception as e:
            logger.error(f"Failed to build hybrid retriever: {e}")
            raise
