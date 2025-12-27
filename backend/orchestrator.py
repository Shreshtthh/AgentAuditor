"""
Audit orchestrator - coordinates PoI and PoUW validation
"""
import logging
import uuid
from typing import Dict, Any, Optional
from datetime import datetime

from backend.engines.poi_engine import PoIEngine
from backend.engines.pouw_engine import PoUWEngine
from backend.engines.evidence_generator import EvidenceBundleGenerator
from backend.engines.ipfs_client import ipfs_client
from backend.database import get_db
from backend.models import Agent, Audit

logger = logging.getLogger(__name__)


class AuditOrchestrator:
    """
    Orchestrates the complete audit process:
    1. Run PoI validation (redundant nodes)
    2. Run PoUW validation (validators)
    3. Generate evidence bundle
    4. Upload to IPFS
    5. Update reputation database
    """
    
    def __init__(self):
        self.poi_engine = PoIEngine()
        self.pouw_engine = PoUWEngine()
        self.evidence_generator = EvidenceBundleGenerator()
        logger.info("Audit Orchestrator initialized")
    
    def execute_audit(
        self,
        agent_id: str,
        task_description: str,
        task_input: str,
        agent_name: Optional[str] = None,
        category: Optional[str] = "general"
    ) -> Dict[str, Any]:
        """
        Execute a complete audit with PoI and PoUW validation
        
        Args:
            agent_id: Unique identifier for the agent
            task_description: Description of what the agent should do
            task_input: The actual input/prompt for the agent
            agent_name: Human-readable agent name
            category: Task category
        
        Returns:
            Complete audit results with confidence score and evidence
        """
        audit_id = f"audit_{uuid.uuid4().hex[:12]}"
        logger.info(f"Starting audit {audit_id} for agent {agent_id}")
        
        # Create audit record in database
        with get_db() as db:
            # Ensure agent exists
            agent = db.query(Agent).filter(Agent.agent_id == agent_id).first()
            if not agent:
                agent = Agent(
                    agent_id=agent_id,
                    name=agent_name or agent_id,
                    category=category
                )
                db.add(agent)
                db.commit()
            
            # Create audit record
            audit = Audit(
                audit_id=audit_id,
                agent_id=agent_id,
                task_description=task_description,
                task_input=task_input,
                category=category,
                status='processing'
            )
            db.add(audit)
            db.commit()
        
        try:
            # Step 1: Run PoI validation (redundant inference)
            logger.info("Step 1: Running PoI validation...")
            poi_session_id, _ = self.poi_engine.create_redundant_session(
                session_name=f"PoI-{audit_id}"
            )
            
            poi_result = self.poi_engine.submit_and_validate(
                session_id=poi_session_id,
                prompt=task_input
            )
            
            # Get consensus output for PoUW validation
            consensus_output = poi_result['consensus_output']
            
            # Step 2: Run PoUW validation (validator scoring)
            logger.info("Step 2: Running PoUW validation...")
            pouw_result = self.pouw_engine.validate_output(
                output_to_validate=consensus_output,
                task_description=task_description,
                task_input=task_input
            )
            
            # Step 3: Calculate final confidence score
            logger.info("Step 3: Calculating final confidence...")
            final_confidence = self.evidence_generator.calculate_final_confidence(
                poi_similarity=poi_result['similarity_score'],
                pouw_score=pouw_result['overall_score']
            )
            
            # Step 4: Generate evidence bundle
            logger.info("Step 4: Generating evidence bundle...")
            evidence_bundle = self.evidence_generator.generate_bundle(
                audit_id=audit_id,
                agent_id=agent_id,
                task_description=task_description,
                task_input=task_input,
                poi_result=poi_result,
                pouw_result=pouw_result,
                final_confidence=final_confidence
            )
            
            # Step 5: Upload to IPFS
            logger.info("Step 5: Uploading to IPFS...")
            ipfs_hash = ipfs_client.upload_bundle(evidence_bundle)
            
            # Step 6: Update database
            logger.info("Step 6: Updating database...")
            with get_db() as db:
                audit = db.query(Audit).filter(Audit.audit_id == audit_id).first()
                audit.session_id_poi = poi_session_id
                audit.session_id_pouw = pouw_result['session_id']
                audit.confidence_score = final_confidence
                audit.poi_similarity = poi_result['similarity_score']
                audit.pouw_mean_score = pouw_result['overall_score']
                audit.consensus_output = consensus_output
                audit.evidence_bundle_ipfs_hash = ipfs_hash
                audit.evidence_bundle_json = evidence_bundle
                audit.status = 'completed'
                audit.completed_at = datetime.utcnow()
                
                # Update agent reputation
                agent = db.query(Agent).filter(Agent.agent_id == agent_id).first()
                agent.total_audits += 1
                agent.last_audit_at = datetime.utcnow()
                
                # Recalculate overall confidence (running average)
                if agent.overall_confidence == 0.0:
                    agent.overall_confidence = final_confidence
                else:
                    # Exponential moving average with alpha=0.3
                    alpha = 0.3
                    agent.overall_confidence = (
                        alpha * final_confidence + 
                        (1 - alpha) * agent.overall_confidence
                    )
                
                db.commit()
            
            logger.info(f"Audit {audit_id} completed successfully. Confidence: {final_confidence:.3f}")
            
            return {
                'audit_id': audit_id,
                'agent_id': agent_id,
                'status': 'completed',
                'confidence_score': final_confidence,
                'poi_similarity': poi_result['similarity_score'],
                'pouw_mean_score': pouw_result['overall_score'],
                'consensus_output': consensus_output,
                'ipfs_hash': ipfs_hash,
                'evidence_bundle': evidence_bundle,
                'poi_details': poi_result,
                'pouw_details': pouw_result
            }
        
        except Exception as e:
            logger.error(f"Audit {audit_id} failed: {e}")
            
            # Update audit status to failed
            with get_db() as db:
                audit = db.query(Audit).filter(Audit.audit_id == audit_id).first()
                if audit:
                    audit.status = 'failed'
                    db.commit()
            
            raise
    
    def get_audit_status(self, audit_id: str) -> Dict[str, Any]:
        """Get status of an audit"""
        with get_db() as db:
            audit = db.query(Audit).filter(Audit.audit_id == audit_id).first()
            
            if not audit:
                return {'error': 'Audit not found'}
            
            return {
                'audit_id': audit.audit_id,
                'agent_id': audit.agent_id,
                'status': audit.status,
                'confidence_score': audit.confidence_score,
                'poi_similarity': audit.poi_similarity,
                'pouw_mean_score': audit.pouw_mean_score,
                'ipfs_hash': audit.evidence_bundle_ipfs_hash,
                'created_at': audit.created_at.isoformat() if audit.created_at else None,
                'completed_at': audit.completed_at.isoformat() if audit.completed_at else None
            }
    
    def get_agent_reputation(self, agent_id: str) -> Dict[str, Any]:
        """Get agent reputation and audit history"""
        with get_db() as db:
            agent = db.query(Agent).filter(Agent.agent_id == agent_id).first()
            
            if not agent:
                return {'error': 'Agent not found'}
            
            # Get recent audits
            recent_audits = db.query(Audit).filter(
                Audit.agent_id == agent_id,
                Audit.status == 'completed'
            ).order_by(Audit.created_at.desc()).limit(10).all()
            
            return {
                'agent_id': agent.agent_id,
                'name': agent.name,
                'category': agent.category,
                'overall_confidence': agent.overall_confidence,
                'total_audits': agent.total_audits,
                'last_audit_at': agent.last_audit_at.isoformat() if agent.last_audit_at else None,
                'recent_audits': [
                    {
                        'audit_id': a.audit_id,
                        'confidence_score': a.confidence_score,
                        'created_at': a.created_at.isoformat()
                    }
                    for a in recent_audits
                ]
            }


# Global orchestrator instance
orchestrator = AuditOrchestrator()
