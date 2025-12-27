"""
Proof of Inference (PoI) Engine
Validates output consistency across redundant nodes via embedding similarity
"""
import logging
from typing import List, Dict, Any, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import time

from backend.config import settings
from backend.web3_client import web3_client

logger = logging.getLogger(__name__)


class PoIEngine:
    """
    Proof of Inference Engine
    
    Creates sessions with redundant nodes and validates consistency
    via embedding similarity calculations.
    """
    
    def __init__(self):
        # Load sentence transformer model for embeddings
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        logger.info("PoI Engine initialized with all-MiniLM-L6-v2 embedding model")
    
    def create_redundant_session(
        self,
        session_name: str,
        num_nodes: int = None,
        model: str = None
    ) -> Tuple[int, str]:
        """
        Create a Cortensor session with redundant nodes for PoI validation
        
        Args:
            session_name: Human-readable session name
            num_nodes: Number of redundant nodes (default: from settings)
            model: Model to use (default: general model from settings)
        
        Returns:
            Tuple of (session_id, transaction_hash)
        """
        if num_nodes is None:
            num_nodes = settings.poi_redundancy
        if model is None:
            model = settings.cortensor_model_general
        
        logger.info(f"Creating PoI session: {session_name} with {num_nodes} redundant nodes")
        
        # Create session with redundant parameter set
        session_id, tx_hash = web3_client.create_session(
            session_name=session_name,
            model=model,
            redundant=num_nodes,
            num_validators=0,  # No validators for PoI-only session
            task_timeout=120
        )
        
        return session_id, tx_hash
    
    def submit_and_validate(
        self,
        session_id: int,
        prompt: str,
        similarity_threshold: float = None
    ) -> Dict[str, Any]:
        """
        Submit task to redundant session and validate outputs via embedding similarity
        
        Args:
            session_id: Session ID with redundant nodes
            prompt: The prompt/input to process
            similarity_threshold: Cosine similarity threshold (default: from settings)
        
        Returns:
            Dictionary containing:
                - consensus_output: The agreed-upon output
                - similarity_score: Average similarity across all outputs
                - outputs: All node outputs
                - embeddings: All embeddings
                - similarity_matrix: Pairwise similarity matrix
                - passed: Whether consensus was reached
        """
        if similarity_threshold is None:
            similarity_threshold = settings.poi_similarity_threshold
        
        logger.info(f"PoI validation started for session {session_id}")
        
        # Step 1: Submit task to session
        task_id, tx_hash = web3_client.submit_task(session_id, prompt)
        logger.info(f"Task submitted: task_id={task_id}, tx={tx_hash}")
        
        # Step 2: Wait for all nodes to complete (poll for results)
        outputs = self._wait_for_all_results(session_id, task_id, timeout=120)
        
        if not outputs:
            raise RuntimeError("No outputs received from nodes")
        
        logger.info(f"Received {len(outputs)} outputs from nodes")
        
        # Step 3: Generate embeddings for all outputs
        output_texts = [result['output'] for result in outputs]
        embeddings = self.embedding_model.encode(output_texts, convert_to_numpy=True)
        
        # Step 4: Calculate pairwise cosine similarity
        similarity_matrix = cosine_similarity(embeddings)
        
        # Step 5: Calculate average similarity (excluding diagonal)
        n = len(similarity_matrix)
        if n > 1:
            # Sum all similarities and exclude diagonal (1.0s)
            total_similarity = np.sum(similarity_matrix) - n
            avg_similarity = total_similarity / (n * (n - 1))
        else:
            avg_similarity = 1.0
        
        # Step 6: Detect consensus cluster
        consensus_output, outliers = self._detect_consensus(
            output_texts,
            embeddings,
            similarity_matrix,
            similarity_threshold
        )
        
        # Step 7: Determine if validation passed
        passed = avg_similarity >= similarity_threshold
        
        result = {
            'session_id': session_id,
            'task_id': task_id,
            'consensus_output': consensus_output,
            'similarity_score': float(avg_similarity),
            'similarity_threshold': similarity_threshold,
            'passed': passed,
            'num_nodes': len(outputs),
            'outputs': output_texts,
            'embeddings': embeddings.tolist(),
            'similarity_matrix': similarity_matrix.tolist(),
            'outliers': outliers,
            'timestamp': time.time()
        }
        
        logger.info(f"PoI validation completed: similarity={avg_similarity:.3f}, passed={passed}")
        
        return result
    
    def _wait_for_all_results(
        self,
        session_id: int,
        task_id: int,
        timeout: int = 120,
        poll_interval: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Poll for task results until all redundant nodes have responded
        
        Args:
            session_id: Session ID
            task_id: Task ID
            timeout: Maximum wait time in seconds
            poll_interval: Seconds between polls
        
        Returns:
            List of result dictionaries from all nodes
        """
        start_time = time.time()
        
        while (time.time() - start_time) < timeout:
            try:
                results = web3_client.get_task_results(session_id, task_id)
                
                # Check if we have results from all nodes
                if results and len(results) > 0:
                    # Filter out empty results
                    valid_results = [r for r in results if r['output'] and r['output'].strip()]
                    
                    if valid_results:
                        logger.info(f"Received {len(valid_results)} valid results")
                        return valid_results
            
            except Exception as e:
                logger.warning(f"Error polling for results: {e}")
            
            time.sleep(poll_interval)
        
        raise TimeoutError(f"Timeout waiting for task results (session={session_id}, task={task_id})")
    
    def _detect_consensus(
        self,
        outputs: List[str],
        embeddings: np.ndarray,
        similarity_matrix: np.ndarray,
        threshold: float
    ) -> Tuple[str, List[int]]:
        """
        Detect consensus output and identify outliers
        
        Args:
            outputs: List of output strings
            embeddings: Array of embeddings
            similarity_matrix: Pairwise similarity matrix
            threshold: Similarity threshold
        
        Returns:
            Tuple of (consensus_output, outlier_indices)
        """
        n = len(outputs)
        
        if n == 1:
            return outputs[0], []
        
        # Calculate average similarity for each output to all others
        avg_similarities = []
        for i in range(n):
            # Average similarity of this output to all others (excluding self)
            similarities = [similarity_matrix[i][j] for j in range(n) if i != j]
            avg_sim = np.mean(similarities)
            avg_similarities.append(avg_sim)
        
        # Find the output with highest average similarity (most central)
        consensus_idx = np.argmax(avg_similarities)
        consensus_output = outputs[consensus_idx]
        
        # Identify outliers (outputs below threshold similarity to consensus)
        outliers = []
        for i in range(n):
            if i != consensus_idx and similarity_matrix[consensus_idx][i] < threshold:
                outliers.append(i)
        
        logger.info(f"Consensus detected: output {consensus_idx}, {len(outliers)} outliers")
        
        return consensus_output, outliers
    
    def calculate_poi_score(self, similarity_score: float) -> float:
        """
        Convert similarity score to PoI confidence score (0-1 scale)
        
        Args:
            similarity_score: Average cosine similarity
        
        Returns:
            PoI confidence score
        """
        # Similarity is already 0-1, but we can apply scaling if needed
        # For now, direct passthrough
        return max(0.0, min(1.0, similarity_score))
