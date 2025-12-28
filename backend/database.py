"""
Database configuration and session management
"""
import os
from contextlib import contextmanager
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session

from backend.config import settings

# Create database directory if needed
db_path = settings.database_url.replace("sqlite:///", "")
if db_path.startswith("./"):
    db_dir = os.path.dirname(db_path)
    if db_dir and not os.path.exists(db_dir):
        os.makedirs(db_dir, exist_ok=True)

# Create engine
engine = create_engine(
    settings.database_url,
    connect_args={"check_same_thread": False} if "sqlite" in settings.database_url else {}
)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create base class for models
Base = declarative_base()


def init_db():
    """Initialize database tables"""
    # Import models to register them with Base
    from backend.models import Agent, Audit
    Base.metadata.create_all(bind=engine)
    print("Database tables created successfully")


@contextmanager
def get_db_session():
    """Get database session as context manager"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_db():
    """Dependency for FastAPI to get DB session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
