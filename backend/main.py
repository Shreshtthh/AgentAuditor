"""
FastAPI application - REST API for Cortensor Agent Auditor
"""
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List
import logging
from sqlalchemy.orm import Session

from backend.config import settings
from backend.database import init_db, get_db_session
from backend.orchestrator import orchestrator
from backend.models import Agent, Audit

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Cortensor Agent Auditor API",
    description="Trust & verification layer for AI agents using PoI and PoUW",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins.split(','),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request/Response Models
class AuditRequest(BaseModel):
    """Request model for creating an audit"""
    agent_id: str = Field(..., description="Unique identifier for the agent")
    agent_name: Optional[str] = Field(None, description="Human-readable agent name")
    task_description: str = Field(..., description="Description of the task")
    task_input: str = Field(..., description="The input/prompt for the agent")
    category: Optional[str] = Field("general", description="Task category")


class AuditResponse(BaseModel):
    """Response model for audit results"""
    audit_id: str
    agent_id: str
    status: str
    confidence_score: Optional[float] = None
    poi_similarity: Optional[float] = None
    pouw_mean_score: Optional[float] = None
    consensus_output: Optional[str] = None
    ipfs_hash: Optional[str] = None
    estimated_time: Optional[str] = None


class AgentReputationResponse(BaseModel):
    """Response model for agent reputation"""
    agent_id: str
    name: str
    category: Optional[str]
    overall_confidence: float
    total_audits: int
    last_audit_at: Optional[str]


# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize database on startup"""
    logger.info("Starting Cortensor Agent Auditor API...")
    init_db()
    logger.info("Database initialized")


# API Routes
@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "service": "Cortensor Agent Auditor",
        "version": "1.0.0",
        "status": "running"
    }


@app.post("/api/v1/audit", response_model=AuditResponse)
async def create_audit(request: AuditRequest):
    """
    Submit a new audit request
    
    This endpoint:
    1. Creates a PoI session with redundant nodes
    2. Creates a PoUW session with validators
    3. Generates evidence bundle
    4. Uploads to IPFS
    5. Returns confidence score and proof
    """
    try:
        logger.info(f"Received audit request for agent: {request.agent_id}")
        
        # Execute audit (synchronous for now, can be made async with Celery)
        result = orchestrator.execute_audit(
            agent_id=request.agent_id,
            task_description=request.task_description,
            task_input=request.task_input,
            agent_name=request.agent_name,
            category=request.category
        )
        
        return AuditResponse(
            audit_id=result['audit_id'],
            agent_id=result['agent_id'],
            status=result['status'],
            confidence_score=result['confidence_score'],
            poi_similarity=result['poi_similarity'],
            pouw_mean_score=result['pouw_mean_score'],
            consensus_output=result['consensus_output'],
            ipfs_hash=result['ipfs_hash']
        )
    
    except Exception as e:
        logger.error(f"Audit request failed: {e}")
        raise HTTPException(status_code=500, detail=f"Audit failed: {str(e)}")


@app.get("/api/v1/audit/{audit_id}")
async def get_audit(audit_id: str):
    """Get audit status and results"""
    try:
        result = orchestrator.get_audit_status(audit_id)
        
        if 'error' in result:
            raise HTTPException(status_code=404, detail=result['error'])
        
        return result
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get audit: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/agent/{agent_id}/reputation", response_model=AgentReputationResponse)
async def get_agent_reputation(agent_id: str):
    """Get agent reputation and audit history"""
    try:
        result = orchestrator.get_agent_reputation(agent_id)
        
        if 'error' in result:
            raise HTTPException(status_code=404, detail=result['error'])
        
        return result
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get agent reputation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/agents", response_model=List[AgentReputationResponse])
async def list_agents(
    limit: int = 50,
    offset: int = 0,
    category: Optional[str] = None,
    db: Session = Depends(get_db_session)
):
    """List all agents with reputation scores"""
    try:
        query = db.query(Agent)
        
        if category:
            query = query.filter(Agent.category == category)
        
        agents = query.order_by(Agent.overall_confidence.desc()).limit(limit).offset(offset).all()
        
        return [
            AgentReputationResponse(
                agent_id=agent.agent_id,
                name=agent.name,
                category=agent.category,
                overall_confidence=agent.overall_confidence,
                total_audits=agent.total_audits,
                last_audit_at=agent.last_audit_at.isoformat() if agent.last_audit_at else None
            )
            for agent in agents
        ]
    
    except Exception as e:
        logger.error(f"Failed to list agents: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/audits")
async def list_audits(
    limit: int = 50,
    offset: int = 0,
    agent_id: Optional[str] = None,
    status: Optional[str] = None,
    db: Session = Depends(get_db_session)
):
    """List recent audits"""
    try:
        query = db.query(Audit)
        
        if agent_id:
            query = query.filter(Audit.agent_id == agent_id)
        
        if status:
            query = query.filter(Audit.status == status)
        
        audits = query.order_by(Audit.created_at.desc()).limit(limit).offset(offset).all()
        
        return [
            {
                'audit_id': audit.audit_id,
                'agent_id': audit.agent_id,
                'task_description': audit.task_description,
                'confidence_score': audit.confidence_score,
                'status': audit.status,
                'ipfs_hash': audit.evidence_bundle_ipfs_hash,
                'created_at': audit.created_at.isoformat() if audit.created_at else None
            }
            for audit in audits
        ]
    
    except Exception as e:
        logger.error(f"Failed to list audits: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True
    )
