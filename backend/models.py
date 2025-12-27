"""
Database models for Cortensor Agent Auditor
"""
from sqlalchemy import Column, Integer, String, Float, DateTime, Text, Boolean, ForeignKey, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime

Base = declarative_base()


class Agent(Base):
    """Agent registry with reputation tracking"""
    __tablename__ = 'agents'
    
    agent_id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    description = Column(Text)
    category = Column(String)  # 'code', 'content', 'reasoning', etc.
    overall_confidence = Column(Float, default=0.0)
    total_audits = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_audit_at = Column(DateTime)
    
    # Relationships
    audits = relationship("Audit", back_populates="agent", cascade="all, delete-orphan")
    reputation_history = relationship("ReputationHistory", back_populates="agent", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Agent(agent_id={self.agent_id}, confidence={self.overall_confidence})>"


class Audit(Base):
    """Individual audit records"""
    __tablename__ = 'audits'
    
    audit_id = Column(String, primary_key=True)
    agent_id = Column(String, ForeignKey('agents.agent_id'), nullable=False)
    
    # Task details
    task_description = Column(Text, nullable=False)
    task_input = Column(Text, nullable=False)
    category = Column(String)
    
    # Cortensor details
    session_id_poi = Column(Integer)
    session_id_pouw = Column(Integer)
    task_id = Column(Integer)
    
    # Results
    confidence_score = Column(Float)
    poi_similarity = Column(Float)
    pouw_mean_score = Column(Float)
    consensus_output = Column(Text)
    
    # Evidence
    evidence_bundle_ipfs_hash = Column(String)
    evidence_bundle_json = Column(JSON)
    
    # Status
    status = Column(String, default='pending')  # pending, processing, completed, failed
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime)
    
    # Relationships
    agent = relationship("Agent", back_populates="audits")
    
    def __repr__(self):
        return f"<Audit(audit_id={self.audit_id}, confidence={self.confidence_score})>"


class ReputationHistory(Base):
    """Historical reputation snapshots for trend analysis"""
    __tablename__ = 'reputation_history'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    agent_id = Column(String, ForeignKey('agents.agent_id'), nullable=False)
    
    confidence_score = Column(Float, nullable=False)
    total_audits = Column(Integer, nullable=False)
    trend = Column(String)  # 'improving', 'declining', 'stable'
    
    recorded_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    agent = relationship("Agent", back_populates="reputation_history")
    
    def __repr__(self):
        return f"<ReputationHistory(agent_id={self.agent_id}, score={self.confidence_score})>"


class ValidatorStats(Base):
    """Track validator performance and consistency"""
    __tablename__ = 'validator_stats'
    
    validator_address = Column(String, primary_key=True)
    total_validations = Column(Integer, default=0)
    average_score_given = Column(Float, default=0.0)
    consistency_score = Column(Float, default=0.0)  # How often they agree with consensus
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def __repr__(self):
        return f"<ValidatorStats(address={self.validator_address}, consistency={self.consistency_score})>"
