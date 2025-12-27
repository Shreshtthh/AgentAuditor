"""
Proof of Useful Work (PoUW) Engine
Quality scoring via validator nodes using rubric-based prompts
"""
import logging
import re
from typing import List, Dict, Any, Tuple
import numpy as np
import time
import json

from backend.config import settings
from backend.web3_client import web3_client

logger = logging.getLogger(__name__)


# Default validation rubric
DEFAULT_RUBRIC = {
    "accuracy": {
        "weight": 0.35,
        "prompt_template": """Rate the ACCURACY of this AI output on a scale of 1-10.

Task: {task}
Input: {input}
Output: {output}

Criteria:
- Factual correctness
- Adherence to instructions
- Absence of hallucinations

Score (1-10): """
    },
    "completeness": {
        "weight": 0.30,
        "prompt_template": """Rate the COMPLETENESS of this AI output on a scale of 1-10.

Task: {task}
Input: {input}
Output: {output}

Criteria:
- All requirements addressed
- Sufficient detail
- Nothing important missing

Score (1-10): """
    },
    "coherence": {
        "weight": 0.25,
        "prompt_template": """Rate the COHERENCE of this AI output on a scale of 1-10.

Task: {task}
Input: {input}
Output: {output}

Criteria:
- Logical flow
- Clear structure
- Well-organized

Score (1-10): """
    },
    "usefulness": {
        "weight": 0.10,
        "prompt_template": """Rate the USEFULNESS of this AI output on a scale of 1-10.

Task: {task}
Input: {input}
Output: {output}

Criteria:
- Practical value
- Actionability
- Relevance

Score (1-10): """
    }
}


class PoUWEngine:
    """
    Proof of Useful Work Engine
    
    Uses validator nodes to score output quality via rubric-based prompts.
    Each criterion is scored 1-10 by multiple validators.
    """
    
    def __init__(self):
        self.rubric = DEFAULT_RUBRIC
        logger.info(f"PoUW Engine initialized with {len(self.rubric)} criteria")
    
    def create_validation_session(
        self,
        session_name: str,
        num_validators: int = None,
        model: str = None
    ) -> Tuple[int, str]:
        """
        Create a Cortensor session with validator nodes for PoUW scoring
        
        Args:
            session_name: Human-readable session name
            num_validators: Number of validator nodes (default: from settings)
            model: Model to use (default: general model from settings)
        
        Returns:
            Tuple of (session_id, transaction_hash)
        """
        if num_validators is None:
            num_validators = settings.pouw_num_validators
        if model is None:
            model = settings.cortensor_model_general
        
        logger.info(f"Creating PoUW session: {session_name} with {num_validators} validators")
        
        # Create session with validator nodes
        session_id, tx_hash = web3_client.create_session(
            session_name=session_name,
            model=model,
            redundant=1,  # Only 1 execution needed (validators score the output)
            num_validators=num_validators,
            task_timeout=120
        )
        
        return session_id, tx_hash
    
    def validate_output(
        self,
        output_to_validate: str,
        task_description: str,
        task_input: str,
        custom_rubric: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Validate an output using validator nodes with rubric-based scoring
        
        Args:
            output_to_validate: The AI output to score
            task_description: Description of what the task was
            task_input: The original input/prompt
            custom_rubric: Optional custom rubric (uses default if None)
        
        Returns:
            Dictionary containing:
                - overall_score: Weighted average score (0-1)
                - criterion_scores: Scores per criterion
                - validator_scores: Raw scores from each validator
                - passed: Whether output meets quality threshold
        """
        rubric = custom_rubric if custom_rubric else self.rubric
        
        logger.info(f"PoUW validation started with {len(rubric)} criteria")
        
        # Step 1: Create validation session
        session_id, tx_hash = self.create_validation_session(
            session_name=f"PoUW-Validation-{int(time.time())}"
        )
        
        # Step 2: For each criterion, generate validation prompt and collect scores
        criterion_results = {}
        
        for criterion_name, criterion_config in rubric.items():
            logger.info(f"Validating criterion: {criterion_name}")
            
            # Generate validation prompt from template
            validation_prompt = criterion_config['prompt_template'].format(
                task=task_description,
                input=task_input,
                output=output_to_validate
            )
            
            # Submit validation task
            task_id, _ = web3_client.submit_task(session_id, validation_prompt)
            
            # Wait for validator scores
            validator_results = self._wait_for_validator_results(session_id, task_id, timeout=90)
            
            # Parse scores from validator outputs
            scores = self._parse_validator_scores(validator_results)
            
            # Calculate statistics
            criterion_results[criterion_name] = {
                'scores': scores,
                'mean': np.mean(scores),
                'median': np.median(scores),
                'std': np.std(scores),
                'weight': criterion_config['weight'],
                'weighted_score': np.mean(scores) * criterion_config['weight']
            }
            
            logger.info(f"{criterion_name}: mean={np.mean(scores):.2f}, std={np.std(scores):.2f}")
        
        # Step 3: Calculate overall weighted score
        overall_score_raw = sum(r['weighted_score'] for r in criterion_results.values())
        overall_score = overall_score_raw / 10.0  # Convert from 0-10 to 0-1 scale
        
        # Step 4: Determine if validation passed (threshold: 0.7)
        passed = overall_score >= 0.7
        
        result = {
            'session_id': session_id,
            'overall_score': float(overall_score),
            'overall_score_raw': float(overall_score_raw),
            'criterion_scores': criterion_results,
            'num_validators': settings.pouw_num_validators,
            'passed': passed,
            'threshold': 0.7,
            'timestamp': time.time()
        }
        
        logger.info(f"PoUW validation completed: overall_score={overall_score:.3f}, passed={passed}")
        
        # Deactivate session to free resources
        try:
            web3_client.deactivate_session(session_id)
        except Exception as e:
            logger.warning(f"Failed to deactivate session: {e}")
        
        return result
    
    def _wait_for_validator_results(
        self,
        session_id: int,
        task_id: int,
        timeout: int = 90,
        poll_interval: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Poll for validation results from validator nodes
        
        Args:
            session_id: Session ID
            task_id: Task ID
            timeout: Maximum wait time in seconds
            poll_interval: Seconds between polls
        
        Returns:
            List of validator result dictionaries
        """
        start_time = time.time()
        
        while (time.time() - start_time) < timeout:
            try:
                results = web3_client.get_task_results(session_id, task_id)
                
                if results and len(results) >= settings.pouw_num_validators:
                    logger.info(f"Received {len(results)} validator results")
                    return results
            
            except Exception as e:
                logger.warning(f"Error polling for validator results: {e}")
            
            time.sleep(poll_interval)
        
        raise TimeoutError(f"Timeout waiting for validator results (session={session_id}, task={task_id})")
    
    def _parse_validator_scores(self, validator_results: List[Dict[str, Any]]) -> List[float]:
        """
        Parse scores from validator outputs
        
        Validators should return a number 1-10. This method extracts that number.
        
        Args:
            validator_results: List of result dictionaries
        
        Returns:
            List of parsed scores (1-10)
        """
        scores = []
        
        for result in validator_results:
            output = result['output'].strip()
            
            # Try to extract a number from the output
            try:
                # Look for a number in the output (first occurrence)
                numbers = re.findall(r'\b([1-9]|10)\b', output)
                
                if numbers:
                    score = float(numbers[0])
                    # Ensure score is in valid range
                    score = max(1.0, min(10.0, score))
                    scores.append(score)
                else:
                    # Default to middle score if parsing fails
                    logger.warning(f"Failed to parse score from: {output[:50]}... Using default 5.0")
                    scores.append(5.0)
            
            except Exception as e:
                logger.error(f"Error parsing validator score: {e}")
                scores.append(5.0)
        
        return scores
    
    def calculate_pouw_score(self, overall_score: float) -> float:
        """
        Convert PoUW overall score to confidence score (already 0-1)
        
        Args:
            overall_score: Overall PoUW score
        
        Returns:
            PoUW confidence score
        """
        return max(0.0, min(1.0, overall_score))
    
    def set_custom_rubric(self, rubric: Dict[str, Any]):
        """
        Set a custom validation rubric
        
        Args:
            rubric: Dictionary of criteria with weights and prompt templates
        """
        # Validate rubric structure
        total_weight = sum(c.get('weight', 0) for c in rubric.values())
        if not (0.99 <= total_weight <= 1.01):
            raise ValueError(f"Rubric weights must sum to 1.0 (got {total_weight})")
        
        self.rubric = rubric
        logger.info(f"Custom rubric set with {len(rubric)} criteria")
