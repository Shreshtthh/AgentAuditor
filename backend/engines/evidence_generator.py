"""
Evidence Bundle Generator
Creates verifiable, immutable audit records and stores them on IPFS
"""
import json
import hashlib
import time
from typing import Dict, Any
from eth_account.messages import encode_defunct
import logging

from backend.config import settings
from backend.web3_client import web3_client

logger = logging.getLogger(__name__)


class EvidenceBundleGenerator:
    """
    Generates evidence bundles containing all audit data
    Signs bundles for verification and uploads to IPFS
    """
    
    def __init__(self):
        self.account = web3_client.account
        logger.info("Evidence Bundle Generator initialized")
    
    def generate_bundle(
        self,
        audit_id: str,
        agent_id: str,
        task_description: str,
        task_input: str,
        poi_result: Dict[str, Any],
        pouw_result: Dict[str, Any],
        final_confidence: float
    ) -> Dict[str, Any]:
        """
        Generate a complete evidence bundle
        
        Args:
            audit_id: Unique audit identifier
            agent_id: Agent being audited
            task_description: Description of the task
            task_input: Original input
            poi_result: PoI validation results
            pouw_result: PoUW validation results
            final_confidence: Final confidence score
        
        Returns:
            Complete evidence bundle dictionary
        """
        logger.info(f"Generating evidence bundle for audit {audit_id}")
        
        # Create bundle structure
        bundle = {
            "version": "1.0",
            "audit_id": audit_id,
            "agent_id": agent_id,
            "timestamp": int(time.time()),
            "task": {
                "description": task_description,
                "input": task_input,
                "category": "general"
            },
            "poi_validation": {
                "session_id": poi_result.get('session_id'),
                "task_id": poi_result.get('task_id'),
                "similarity_score": poi_result.get('similarity_score'),
                "threshold": poi_result.get('similarity_threshold'),
                "passed": poi_result.get('passed'),
                "num_nodes": poi_result.get('num_nodes'),
                "consensus_output": poi_result.get('consensus_output'),
                "outliers": poi_result.get('outliers', []),
                "similarity_matrix": poi_result.get('similarity_matrix')
            },
            "pouw_validation": {
                "session_id": pouw_result.get('session_id'),
                "overall_score": pouw_result.get('overall_score'),
                "overall_score_raw": pouw_result.get('overall_score_raw'),
                "passed": pouw_result.get('passed'),
                "num_validators": pouw_result.get('num_validators'),
                "criterion_scores": pouw_result.get('criterion_scores', {})
            },
            "final_assessment": {
                "confidence_score": final_confidence,
                "poi_weight": settings.pouw_confidence_weight_poi,
                "pouw_weight": settings.pouw_confidence_weight_pouw,
                "calculation": f"{settings.pouw_confidence_weight_poi} * {poi_result.get('similarity_score', 0):.3f} + {settings.pouw_confidence_weight_pouw} * {pouw_result.get('overall_score', 0):.3f}"
            },
            "metadata": {
                "auditor_address": self.account.address,
                "network": "arbitrum-sepolia",
                "cortensor_version": "v2"
            }
        }
        
        # Generate hash of bundle
        bundle_json = json.dumps(bundle, sort_keys=True)
        bundle_hash = hashlib.sha256(bundle_json.encode()).hexdigest()
        bundle['bundle_hash'] = bundle_hash
        
        # Sign the bundle
        signature = self._sign_bundle(bundle_hash)
        bundle['signature'] = signature
        
        logger.info(f"Bundle generated: hash={bundle_hash[:16]}...")
        
        return bundle
    
    def _sign_bundle(self, bundle_hash: str) -> str:
        """
        Sign the bundle hash with auditor's private key
        
        Args:
            bundle_hash: SHA-256 hash of the bundle
        
        Returns:
            Hex-encoded signature
        """
        # Create message to sign
        message = encode_defunct(text=bundle_hash)
        
        # Sign with private key
        signed_message = web3_client.w3.eth.account.sign_message(
            message,
            private_key=self.account.key
        )
        
        return signed_message.signature.hex()
    
    def verify_bundle(self, bundle: Dict[str, Any]) -> bool:
        """
        Verify the authenticity of an evidence bundle
        
        Args:
            bundle: Evidence bundle to verify
        
        Returns:
            True if bundle is valid and signature matches
        """
        try:
            # Extract signature and hash
            signature = bundle.get('signature')
            claimed_hash = bundle.get('bundle_hash')
            
            if not signature or not claimed_hash:
                return False
            
            # Recalculate hash
            bundle_copy = bundle.copy()
            bundle_copy.pop('signature', None)
            bundle_copy.pop('bundle_hash', None)
            
            bundle_json = json.dumps(bundle_copy, sort_keys=True)
            actual_hash = hashlib.sha256(bundle_json.encode()).hexdigest()
            
            # Verify hash matches
            if actual_hash != claimed_hash:
                logger.warning("Bundle hash mismatch")
                return False
            
            # Verify signature
            message = encode_defunct(text=claimed_hash)
            recovered_address = web3_client.w3.eth.account.recover_message(
                message,
                signature=signature
            )
            
            # Check if recovered address matches auditor
            expected_address = bundle.get('metadata', {}).get('auditor_address')
            
            if recovered_address.lower() != expected_address.lower():
                logger.warning("Signature verification failed")
                return False
            
            logger.info("Bundle verification successful")
            return True
        
        except Exception as e:
            logger.error(f"Bundle verification error: {e}")
            return False
    
    def calculate_final_confidence(
        self,
        poi_similarity: float,
        pouw_score: float
    ) -> float:
        """
        Calculate final confidence score as weighted average of PoI and PoUW
        
        Args:
            poi_similarity: PoI similarity score (0-1)
            pouw_score: PoUW overall score (0-1)
        
        Returns:
            Final confidence score (0-1)
        """
        confidence = (
            settings.pouw_confidence_weight_poi * poi_similarity +
            settings.pouw_confidence_weight_pouw * pouw_score
        )
        
        return max(0.0, min(1.0, confidence))
