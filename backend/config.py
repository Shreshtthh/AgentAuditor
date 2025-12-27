"""
Configuration module for Cortensor Agent Auditor
Loads all environment variables and provides type-safe config access
"""
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""
    
    # Blockchain Configuration
    arbitrum_sepolia_rpc_url: str
    private_key: str
    session_v2_address: str
    session_queue_v2_address: str
    
    # Cortensor Configuration
    cortensor_model_general: str = "cts-llm-2"
    cortensor_model_reasoning: str = "cts-llm-14"
    
    # PoI Configuration
    poi_redundancy: int = 3
    poi_similarity_threshold: float = 0.85
    
    # PoUW Configuration
    pouw_num_validators: int = 5
    pouw_confidence_weight_poi: float = 0.5
    pouw_confidence_weight_pouw: float = 0.5
    
    # Database
    database_url: str
    redis_url: str = "redis://localhost:6379/0"
    
    # IPFS Configuration
    ipfs_provider: str = "pinata"
    pinata_api_key: Optional[str] = None
    pinata_secret_key: Optional[str] = None
    web3_storage_token: Optional[str] = None
    
    # API Configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    secret_key: str
    cors_origins: str = "http://localhost:3000,http://localhost:5173"
    
    # Logging
    log_level: str = "INFO"
    
    class Config:
        env_file = ".env"
        case_sensitive = False


# Global settings instance
settings = Settings()
