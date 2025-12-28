"""
Database models for Agent Auditor
"""
from datetime import datetime
from sqlalchemy import Column, String, Integer, Float, DateTime, Text
from backend.database import Base


class Agent(Base):
    """Agent model for tracking AI agents"""
    __tablename__ = "agents"
    
    id = Column(Integer, primary_key=True, index=True)
    agent_id = Column(String(255), unique=True, index=True, nullable=False)
    name = Column(String(255), nullable=True)
    description = Column(Text, nullable=True)
    category = Column(String(100), nullable=True)
    total_audits = Column(Integer, default=0)
    passed_audits = Column(Integer, default=0)
    average_confidence = Column(Float, default=0.0)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class Audit(Base):
    """Audit model for tracking audit results"""
    __tablename__ = "audits"
    
    id = Column(Integer, primary_key=True, index=True)
    audit_id = Column(String(255), unique=True, index=True, nullable=False)
    agent_id = Column(String(255), index=True, nullable=False)
    task_description = Column(Text, nullable=True)
    task_input = Column(Text, nullable=True)
    agent_output = Column(Text, nullable=True)
    poi_similarity = Column(Float, nullable=True)
    pouw_score = Column(Float, nullable=True)
    confidence_score = Column(Float, default=0.0)
    status = Column(String(50), default="pending")
    evidence_hash = Column(String(255), nullable=True)
    ipfs_cid = Column(String(255), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)
