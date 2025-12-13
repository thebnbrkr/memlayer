"""
Memlayer FastAPI Application
=============================

Self-hosted memory layer service with:
- Custom salience configuration API
- Multi-tenant support
- OpenAI-compatible memory enhancement
"""

import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from memlayer.routes import salience_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events."""
    # Startup
    print("🚀 Memlayer API starting up...")
    print(f"   Storage path: {os.getenv('STORAGE_PATH', '/app/data')}")
    print(f"   Operation mode: {os.getenv('OPERATION_MODE', 'online')}")
    print(f"   Debug mode: {os.getenv('DEBUG_MODE', 'false')}")

    yield

    # Shutdown
    print("👋 Memlayer API shutting down...")


# Create FastAPI app
app = FastAPI(
    title="Memlayer API",
    description="Self-hosted memory layer service with flexible salience configuration",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware (configure for your domain in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Routes
# ============================================================================

# Include salience configuration routes
app.include_router(salience_router)


# ============================================================================
# Health Check
# ============================================================================

@app.get("/health")
async def health_check():
    """
    Main health check endpoint for Docker health checks.

    Returns service status and configuration info.
    """
    return {
        "status": "healthy",
        "service": "memlayer-api",
        "version": "1.0.0",
        "storage_path": os.getenv("STORAGE_PATH", "/app/data"),
        "operation_mode": os.getenv("OPERATION_MODE", "online"),
        "database_configured": bool(os.getenv("DATABASE_URL")),
        "openai_configured": bool(os.getenv("OPENAI_API_KEY")),
    }


@app.get("/")
async def root():
    """
    Root endpoint with API information.
    """
    return {
        "service": "Memlayer API",
        "version": "1.0.0",
        "description": "Self-hosted memory layer service",
        "docs": "/docs",
        "health": "/health",
        "endpoints": {
            "salience_config": "/api/config/salience",
        }
    }


# ============================================================================
# Example: Memory Enhancement Endpoint (Future)
# ============================================================================

# Uncomment when you want to add direct chat enhancement endpoints:
#
# from pydantic import BaseModel
# from typing import List, Dict, Any
# from memlayer import OpenAI as MemlayerClient
#
# class ChatRequest(BaseModel):
#     messages: List[Dict[str, str]]
#     user_id: str
#     tenant_id: str
#     model: str = "gpt-4o-mini"
#     salience_config_name: Optional[str] = None
#
# @app.post("/api/chat")
# async def enhanced_chat(request: ChatRequest):
#     """
#     Chat endpoint with memory enhancement.
#
#     Automatically stores salient facts and retrieves relevant memories.
#     """
#     client = MemlayerClient(
#         model=request.model,
#         user_id=request.user_id,
#         tenant_id=request.tenant_id,
#         storage_path=os.getenv("STORAGE_PATH", "/app/data"),
#         operation_mode=os.getenv("OPERATION_MODE", "online"),
#     )
#
#     response = client.chat(request.messages)
#
#     return {
#         "response": response,
#         "memories_retrieved": len(client.last_retrieved_memories or []),
#         "salience_logs": client.get_salience_logs(),
#     }
