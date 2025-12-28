"""
FastAPI application - REST API for Cortensor Agent Auditor
"""
import logging
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List
from sqlalchemy.orm import Session

from backend.config import settings
from backend.database import init_db, get_db_session
from backend.orchestrator import orchestrator
from backend.models import Agent, Audit

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level, "INFO"),
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
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==================== ROOT ROUTES ====================
@app.get("/")
async def root():
    """Health check - root endpoint"""
    return {
        "service": "Cortensor Agent Auditor",
        "version": "1.0.0",
        "status": "running"
    }


@app.get("/health")
async def health():
    """Detailed health check"""
    return {
        "status": "healthy",
        "database": "connected",
        "cortensor_session": settings.cortensor_session_id
    }
# =====================================================


# Request/Response Models
class AuditRequest(BaseModel):
    agent_id: str = Field(..., description="Unique identifier for the agent")
    agent_name: Optional[str] = Field(None, description="Human-readable agent name")
    task_description: str = Field(..., description="Description of the task")
    task_input: str = Field(..., description="The input/prompt for the agent")
    category: Optional[str] = Field("general", description="Task category")


class AuditResponse(BaseModel):
    audit_id: str
    agent_id: str
    status: str
    confidence_score: Optional[float] = None
    poi_similarity: Optional[float] = None
    pouw_mean_score: Optional[float] = None
    ipfs_hash: Optional[str] = None
    error: Optional[str] = None


class AgentReputationResponse(BaseModel):
    agent_id: str
    agent_name: Optional[str] = None
    total_audits: int = 0
    passed_audits: int = 0
    average_confidence: float = 0.0
    reputation_score: float = 0.0


# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize on startup"""
    logger.info("Starting Cortensor Agent Auditor API...")
    
    # Initialize database
    init_db()
    logger.info("Database initialized")


# ==================== API ROUTES ====================
@app.post("/api/v1/audit", response_model=AuditResponse)
async def create_audit(request: AuditRequest):
    """Submit a new audit request"""
    logger.info(f"Audit request for agent: {request.agent_id}")
    
    try:
        result = await orchestrator.run_audit(
            agent_id=request.agent_id,
            agent_name=request.agent_name,
            task_description=request.task_description,
            task_input=request.task_input,
            category=request.category
        )
        
        return AuditResponse(
            audit_id=result.get("audit_id", ""),
            agent_id=request.agent_id,
            status=result.get("status", "processing"),
            confidence_score=result.get("confidence_score"),
            poi_similarity=result.get("poi_similarity"),
            pouw_mean_score=result.get("pouw_mean_score"),
            ipfs_hash=result.get("ipfs_hash")
        )
    except Exception as e:
        logger.error(f"Audit failed: {e}")
        return AuditResponse(
            audit_id="",
            agent_id=request.agent_id,
            status="failed",
            error=str(e)
        )


@app.get("/api/v1/audit/{audit_id}")
async def get_audit(audit_id: str):
    """Get audit status and results"""
    try:
        result = await orchestrator.get_audit_status(audit_id)
        return result
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/api/v1/agent/{agent_id}/reputation", response_model=AgentReputationResponse)
async def get_agent_reputation(agent_id: str):
    """Get reputation score for an agent"""
    try:
        result = await orchestrator.get_agent_reputation(agent_id)
        return AgentReputationResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/api/v1/agents", response_model=List[AgentReputationResponse])
async def list_agents(limit: int = 10, offset: int = 0):
    """List all agents with their reputation"""
    try:
        agents = await orchestrator.list_agents(limit=limit, offset=offset)
        return [AgentReputationResponse(**a) for a in agents]
    except Exception as e:
        logger.error(f"Failed to list agents: {e}")
        return []


@app.get("/api/v1/audits")
async def list_audits(limit: int = 10, offset: int = 0, status: Optional[str] = None):
    """List all audits"""
    try:
        audits = await orchestrator.list_audits(limit=limit, offset=offset, status=status)
        return audits
    except Exception as e:
        logger.error(f"Failed to list audits: {e}")
        return []
# =====================================================


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True
    )
