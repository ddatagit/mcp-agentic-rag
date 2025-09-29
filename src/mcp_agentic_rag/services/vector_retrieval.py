"""Vector retrieval service for MCP Agentic RAG system."""

from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from qdrant_client import models
from qdrant_client import QdrantClient
import time
from typing import List, Dict, Any, Optional
from pydantic import ValidationError

from ..models.vector_match import VectorMatch


# FAQ text data
faq_text = """Question 1: What is the first step before building a machine learning model?
Answer 1: Understand the problem, define the objective, and identify the right metrics for evaluation.

Question 2: How important is data cleaning in ML?
Answer 2: Extremely important. Clean data improves model performance and reduces the chance of misleading results.

Question 3: Should I normalize or standardize my data?
Answer 3: Yes, especially for models sensitive to feature scales like SVMs, KNN, and neural networks.

Question 4: When should I use feature engineering?
Answer 4: Always consider it. Well-crafted features often yield better results than complex models.

Question 5: How to handle missing values?
Answer 5: Use imputation techniques like mean/median imputation, or model-based imputation depending on the context.

Question 6: Should I balance my dataset for classification tasks?
Answer 6: Yes, especially if the classes are imbalanced. Techniques include resampling, SMOTE, and class-weighting.

Question 7: How do I select features for my model?
Answer 7: Use domain knowledge, correlation analysis, or techniques like Recursive Feature Elimination or SHAP values.

Question 8: Is it good to use all features available?
Answer 8: Not always. Irrelevant or redundant features can reduce performance and increase overfitting.

Question 9: How do I avoid overfitting?
Answer 9: Use techniques like cross-validation, regularization, pruning (for trees), and dropout (for neural nets).

Question 10: Why is cross-validation important?
Answer 10: It provides a more reliable estimate of model performance by reducing bias from a single train-test split.

Question 11: What's a good train-test split ratio?
Answer 11: Common ratios are 80/20 or 70/30, but use cross-validation for more robust evaluation.

Question 12: Should I tune hyperparameters?
Answer 12: Yes. Use grid search, random search, or Bayesian optimization to improve model performance.

Question 13: What's the difference between training and validation sets?
Answer 13: Training set trains the model, validation set tunes hyperparameters, and test set evaluates final performance.

Question 14: How do I know if my model is underfitting?
Answer 14: It performs poorly on both training and test sets, indicating it hasn't learned patterns well.

Question 15: What are signs of overfitting?
Answer 15: High accuracy on training data but poor generalization to test or validation data.

Question 16: Is ensemble modeling useful?
Answer 16: Yes. Ensembles like Random Forests or Gradient Boosting often outperform individual models.

Question 17: When should I use deep learning?
Answer 17: Use it when you have large datasets, complex patterns, or tasks like image and text processing.

Question 18: What is data leakage and how to avoid it?
Answer 18: Data leakage is using future or target-related information during training. Avoid by carefully splitting and preprocessing.

Question 19: How do I measure model performance?
Answer 19: Choose appropriate metrics: accuracy, precision, recall, F1, ROC-AUC for classification; RMSE, MAE for regression.

Question 20: Why is model interpretability important?
Answer 20: It builds trust, helps debug, and ensures compliance—especially important in high-stakes domains like healthcare."""

new_faq_text = [i.replace("\n", " ") for i in faq_text.split("\n\n")]


def batch_iterate(lst: List, batch_size: int):
    """Iterate over list in batches."""
    for i in range(0, len(lst), batch_size):
        yield lst[i : i + batch_size]


class EmbedData:
    """Embedding service for text data."""

    def __init__(self,
                 embed_model_name: str = "nomic-ai/nomic-embed-text-v1.5",
                 batch_size: int = 32):

        self.embed_model_name = embed_model_name
        self.embed_model = self._load_embed_model()
        self.batch_size = batch_size
        self.embeddings = []

    def _load_embed_model(self):
        """Load the embedding model."""
        embed_model = SentenceTransformer(
            self.embed_model_name,
            trust_remote_code=True,
            cache_folder='./hf_cache'
        )
        return embed_model

    def generate_embedding(self, context: List[str]):
        """Generate embeddings for a batch of contexts."""
        return self.embed_model.encode(context, batch_size=self.batch_size, convert_to_tensor=False)

    def embed(self, contexts: List[str]):
        """Embed all contexts in batches."""
        self.contexts = contexts

        for batch_context in tqdm(batch_iterate(contexts, self.batch_size),
                                  total=len(contexts)//self.batch_size,
                                  desc="Embedding data in batches"):

            batch_embeddings = self.generate_embedding(batch_context)
            self.embeddings.extend(batch_embeddings)


class QdrantVDB:
    """Qdrant vector database service."""

    def __init__(self, collection_name: str, vector_dim: int = 768, batch_size: int = 512):
        self.collection_name = collection_name
        self.batch_size = batch_size
        self.vector_dim = vector_dim
        self.define_client()

    def define_client(self):
        """Initialize Qdrant client."""
        self.client = QdrantClient(
            url="http://localhost:6333",
            prefer_grpc=True
        )

    def create_collection(self):
        """Create collection if it doesn't exist."""
        if not self.client.collection_exists(collection_name=self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=self.vector_dim,
                    distance=models.Distance.DOT,
                    on_disk=True
                ),
                optimizers_config=models.OptimizersConfigDiff(
                    default_segment_number=5,
                    indexing_threshold=0
                )
            )

    def ingest_data(self, embeddata: EmbedData):
        """Ingest data into the vector database."""
        for batch_context, batch_embeddings in tqdm(
            zip(batch_iterate(embeddata.contexts, self.batch_size),
                batch_iterate(embeddata.embeddings, self.batch_size)),
            total=len(embeddata.contexts)//self.batch_size,
            desc="Ingesting in batches"
        ):
            self.client.upload_collection(
                collection_name=self.collection_name,
                vectors=batch_embeddings,
                payload=[{"context": context} for context in batch_context]
            )

        self.client.update_collection(
            collection_name=self.collection_name,
            optimizer_config=models.OptimizersConfigDiff(indexing_threshold=20000)
        )


class Retriever:
    """Vector retrieval service."""

    def __init__(self, vector_db: QdrantVDB, embeddata: EmbedData):
        self.vector_db = vector_db
        self.embeddata = embeddata

    def search(self, query: str) -> str:
        """
        Original search method that returns formatted text.

        Maintained for backward compatibility.
        """
        query_embedding = self.embeddata.embed_model.get_query_embedding(query)

        # select the top 3 results
        result = self.vector_db.client.search(
            collection_name=self.vector_db.collection_name,
            query_vector=query_embedding,
            search_params=models.SearchParams(
                quantization=models.QuantizationSearchParams(
                    ignore=True,
                    rescore=True,
                    oversampling=2.0,
                )
            ),
            limit=3,
            timeout=1000,
        )

        context = [dict(data) for data in result]
        combined_prompt = []

        for entry in context[:3]:
            context = entry["payload"]["context"]
            combined_prompt.append(context)

        final_output = "\n\n---\n\n".join(combined_prompt)
        return final_output

    def search_with_confidence(self, query: str, limit: int = 5, min_confidence: float = 0.0) -> Dict[str, Any]:
        """
        Enhanced search method that returns results with confidence scores and metadata.

        Args:
            query: Search query text
            limit: Maximum number of results to return
            min_confidence: Minimum confidence threshold

        Returns:
            Dictionary with search results and metadata
        """
        start_time = time.time()

        try:
            query_embedding = self.embeddata.embed_model.get_query_embedding(query)

            # Perform vector search
            result = self.vector_db.client.search(
                collection_name=self.vector_db.collection_name,
                query_vector=query_embedding,
                search_params=models.SearchParams(
                    quantization=models.QuantizationSearchParams(
                        ignore=True,
                        rescore=True,
                        oversampling=2.0,
                    )
                ),
                limit=limit,
                timeout=1000,
            )

            # Process results with confidence scores
            processed_results = []
            total_found = len(result)

            for i, entry in enumerate(result):
                # Qdrant returns similarity scores, normalize to confidence
                raw_score = entry.score
                confidence = max(0.0, min(1.0, raw_score))

                # Filter by minimum confidence
                if confidence >= min_confidence:
                    processed_results.append({
                        "content": entry.payload["context"],
                        "confidence": confidence,
                        "source_document": f"faq_document_{i+1}",
                        "metadata": {
                            "raw_score": raw_score,
                            "position": i,
                            "total_results": total_found
                        }
                    })

            search_time = time.time() - start_time

            # Calculate average confidence
            average_confidence = 0.0
            if processed_results:
                average_confidence = sum(r["confidence"] for r in processed_results) / len(processed_results)

            return {
                "results": processed_results,
                "total_found": total_found,
                "search_time_seconds": search_time,
                "average_confidence": average_confidence
            }

        except Exception as e:
            search_time = time.time() - start_time
            raise Exception(f"Vector search failed: {str(e)}")

    def validate_vector_input(self, input_data: Dict[str, Any]):
        """Validate input for vector search tool."""
        if not input_data.get("query"):
            raise ValueError("Query is required")

        query = input_data["query"]
        if len(query) == 0 or len(query) > 1000:
            raise ValueError(f"Query length {len(query)} not in range 1-1000")

        if "limit" in input_data:
            limit = input_data["limit"]
            if not isinstance(limit, int) or limit < 1 or limit > 20:
                raise ValueError("limit must be between 1 and 20")

        if "min_confidence" in input_data:
            min_confidence = input_data["min_confidence"]
            if not isinstance(min_confidence, (int, float)) or min_confidence < 0 or min_confidence > 1:
                raise ValueError("min_confidence must be between 0.0 and 1.0")


class VectorRetrievalService:
    """
    High-level vector retrieval service with enhanced functionality.

    This service provides a clean interface for vector search operations
    while maintaining backward compatibility with existing code.
    """

    def __init__(self, collection_name: str = "ml_faq_collection"):
        self.collection_name = collection_name
        self.embed_data = EmbedData()
        self.vector_db = QdrantVDB(collection_name)
        self.retriever = Retriever(self.vector_db, self.embed_data)

    def search(self, query: str) -> str:
        """Simple search interface for backward compatibility."""
        return self.retriever.search(query)

    def advanced_search(self, query: str, confidence_threshold: float = 0.7) -> List[VectorMatch]:
        """
        Advanced search that returns structured VectorMatch objects.

        Args:
            query: Search query text
            confidence_threshold: Minimum confidence for results

        Returns:
            List of VectorMatch objects
        """
        results = self.retriever.search_with_confidence(
            query=query,
            limit=10,
            min_confidence=confidence_threshold
        )

        vector_matches = []
        for i, result in enumerate(results["results"]):
            try:
                vector_match = VectorMatch(
                    document_id=f"ml_faq_{i+1:03d}",
                    content=result["content"],
                    score=result["confidence"],
                    source_document=result["source_document"],
                    metadata=result["metadata"]
                )
                vector_matches.append(vector_match)
            except Exception as e:
                # Log but don't fail for individual result errors
                print(f"Warning: Failed to create VectorMatch for result {i}: {e}")
                continue

        return vector_matches

    def get_search_stats(self, query: str) -> Dict[str, Any]:
        """Get search statistics without returning full results."""
        results = self.retriever.search_with_confidence(query=query, limit=20, min_confidence=0.0)

        return {
            "total_results": results["total_found"],
            "search_time": results["search_time_seconds"],
            "average_confidence": results["average_confidence"],
            "high_confidence_count": len([r for r in results["results"] if r["confidence"] >= 0.8]),
            "medium_confidence_count": len([r for r in results["results"] if 0.6 <= r["confidence"] < 0.8]),
            "low_confidence_count": len([r for r in results["results"] if r["confidence"] < 0.6])
        }