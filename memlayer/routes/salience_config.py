"""
FastAPI routes for salience configuration management.

Provides CRUD operations and testing endpoints for custom salience configurations.
"""

from typing import List, Optional, Dict, Any
from fastapi import APIRouter, HTTPException, Query, Depends
from pydantic import BaseModel

from memlayer.config.salience import (
    TenantSalienceConfig,
    SalienceComputationLog,
)
from memlayer.services.salience_calculator import SalienceCalculator

# Create router
router = APIRouter(prefix="/api/config/salience", tags=["Salience Configuration"])


# ============================================================================
# In-memory storage (for demo/testing - replace with actual DB in production)
# ============================================================================
_configs_db: Dict[str, Dict[str, TenantSalienceConfig]] = {}
_computation_logs: List[SalienceComputationLog] = []


def _get_tenant_configs(tenant_id: str) -> Dict[str, TenantSalienceConfig]:
    """Get or create tenant config storage."""
    if tenant_id not in _configs_db:
        _configs_db[tenant_id] = {}
    return _configs_db[tenant_id]


# ============================================================================
# Request/Response Models
# ============================================================================


class CreateConfigResponse(BaseModel):
    """Response for creating a new salience config."""
    config_id: str
    config_name: str
    status: str


class UpdateConfigResponse(BaseModel):
    """Response for updating a salience config."""
    config_id: str
    config_name: str
    status: str


class DeleteConfigResponse(BaseModel):
    """Response for deleting a salience config."""
    config_name: str
    status: str


class TestConfigRequest(BaseModel):
    """Request to test a salience config against a sample fact."""
    fact: str


class TestConfigResponse(BaseModel):
    """Response from testing a salience config."""
    salience_score: float
    threshold: float
    decision: str  # "STORE" or "SKIP"
    component_scores: Dict[str, float]
    full_log: Dict[str, Any]
    computation_time_ms: float


class ListConfigsResponse(BaseModel):
    """Response listing all configs for a tenant."""
    configs: List[TenantSalienceConfig]
    total_count: int


class AuditLogEntry(BaseModel):
    """Simplified audit log entry for API response."""
    log_id: str
    fact_text: str
    config_name: str
    salience_score: float
    threshold: float
    decision: str
    reasoning: str
    created_at: str


class AuditLogResponse(BaseModel):
    """Response with audit log entries."""
    logs: List[AuditLogEntry]
    total_count: int


# ============================================================================
# CRUD Endpoints
# ============================================================================


@router.post("/", response_model=CreateConfigResponse, status_code=201)
async def create_salience_config(config: TenantSalienceConfig):
    """
    Create a new salience configuration.

    Validates that:
    - Component weights sum to 1.0 (tolerance: 0.99-1.01)
    - Config name is unique for tenant
    - All required fields are present

    Example:
    ```json
    {
      "tenant_id": "customer_123",
      "config_name": "my_custom_score_54",
      "components": [
        {
          "name": "novelty",
          "weight": 0.4,
          "scoring_function": "embedding_similarity",
          "scoring_config": {"target": "existing_memories"}
        },
        {
          "name": "emotional_impact",
          "weight": 0.6,
          "scoring_function": "llm_scored",
          "scoring_config": {"prompt": "Score emotional impact..."}
        }
      ],
      "threshold_config": {
        "strategy": "percentile",
        "percentile": 70
      },
      "decision_rules": []
    }
    ```
    """
    tenant_configs = _get_tenant_configs(config.tenant_id)

    # Check if config name already exists
    if config.config_name in tenant_configs:
        raise HTTPException(
            status_code=409,
            detail=f"Config '{config.config_name}' already exists for tenant '{config.tenant_id}'"
        )

    # Weights are already validated by Pydantic model
    # Store config
    tenant_configs[config.config_name] = config

    return CreateConfigResponse(
        config_id=config.config_id,
        config_name=config.config_name,
        status="created"
    )


@router.get("/", response_model=ListConfigsResponse)
async def list_salience_configs(
    tenant_id: str = Query(..., description="Tenant ID to list configs for"),
    active_only: bool = Query(False, description="Only return active configs")
):
    """
    List all salience configurations for a tenant.

    Query parameters:
    - tenant_id: The tenant to list configs for (required)
    - active_only: If true, only return configs where is_active=true
    """
    tenant_configs = _get_tenant_configs(tenant_id)

    configs = list(tenant_configs.values())

    if active_only:
        configs = [c for c in configs if c.is_active]

    return ListConfigsResponse(
        configs=configs,
        total_count=len(configs)
    )


@router.get("/{config_name}", response_model=TenantSalienceConfig)
async def get_salience_config(
    config_name: str,
    tenant_id: str = Query(..., description="Tenant ID")
):
    """
    Get a specific salience configuration.

    Returns the complete config including all components, thresholds, and decision rules.
    """
    tenant_configs = _get_tenant_configs(tenant_id)

    if config_name not in tenant_configs:
        raise HTTPException(
            status_code=404,
            detail=f"Config '{config_name}' not found for tenant '{tenant_id}'"
        )

    return tenant_configs[config_name]


@router.put("/{config_name}", response_model=UpdateConfigResponse)
async def update_salience_config(
    config_name: str,
    config: TenantSalienceConfig
):
    """
    Update an existing salience configuration.

    All fields are replaced with the new values.
    Config name in path must match config.config_name in body.
    """
    if config.config_name != config_name:
        raise HTTPException(
            status_code=400,
            detail=f"Config name mismatch: path has '{config_name}', body has '{config.config_name}'"
        )

    tenant_configs = _get_tenant_configs(config.tenant_id)

    if config_name not in tenant_configs:
        raise HTTPException(
            status_code=404,
            detail=f"Config '{config_name}' not found for tenant '{config.tenant_id}'"
        )

    # Update config
    from datetime import datetime
    config.updated_at = datetime.utcnow()
    tenant_configs[config_name] = config

    return UpdateConfigResponse(
        config_id=config.config_id,
        config_name=config.config_name,
        status="updated"
    )


@router.delete("/{config_name}", response_model=DeleteConfigResponse)
async def delete_salience_config(
    config_name: str,
    tenant_id: str = Query(..., description="Tenant ID")
):
    """
    Delete a salience configuration.

    This is a hard delete. Consider using is_active=false for soft deletes.
    """
    tenant_configs = _get_tenant_configs(tenant_id)

    if config_name not in tenant_configs:
        raise HTTPException(
            status_code=404,
            detail=f"Config '{config_name}' not found for tenant '{tenant_id}'"
        )

    del tenant_configs[config_name]

    return DeleteConfigResponse(
        config_name=config_name,
        status="deleted"
    )


# ============================================================================
# Testing & Audit Endpoints
# ============================================================================


@router.post("/{config_name}/test", response_model=TestConfigResponse)
async def test_salience_config(
    config_name: str,
    request: TestConfigRequest,
    tenant_id: str = Query(..., description="Tenant ID")
):
    """
    Test a salience configuration against a sample fact.

    This endpoint allows you to see exactly how a fact would be scored,
    including:
    - Individual component scores
    - Final weighted salience score
    - Computed threshold
    - Which decision rule matched (if any)
    - Final decision (STORE or SKIP)
    - Complete computation log for debugging

    Example request:
    ```json
    {
      "fact": "Alice works at TechCorp as a software engineer"
    }
    ```

    Example response:
    ```json
    {
      "salience_score": 0.625,
      "threshold": 0.55,
      "decision": "STORE",
      "component_scores": {
        "novelty": 0.85,
        "emotional_impact": 0.3,
        "technical_depth": 0.6
      },
      "full_log": {
        "computation_steps": [...],
        "reasoning": "Salience score 0.625 >= threshold 0.55"
      },
      "computation_time_ms": 45.2
    }
    ```
    """
    tenant_configs = _get_tenant_configs(tenant_id)

    if config_name not in tenant_configs:
        raise HTTPException(
            status_code=404,
            detail=f"Config '{config_name}' not found for tenant '{tenant_id}'"
        )

    config = tenant_configs[config_name]

    # Run salience calculation
    calculator = SalienceCalculator(
        config=config,
        fact=request.fact,
        tenant_id=tenant_id,
        fact_id=None,  # No fact ID for test
        embedding_model=None  # TODO: inject actual embedding model
    )

    salience_score, decision, full_log = calculator.compute_salience()

    return TestConfigResponse(
        salience_score=salience_score,
        threshold=calculator.threshold,
        decision=decision,
        component_scores=calculator.component_scores,
        full_log=full_log,
        computation_time_ms=full_log.get("computation_time_ms", 0.0)
    )


@router.get("/{config_name}/audit-log", response_model=AuditLogResponse)
async def get_audit_log(
    config_name: str,
    tenant_id: str = Query(..., description="Tenant ID"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of log entries to return")
):
    """
    Get audit trail of decisions made with this configuration.

    Returns recent salience computations showing what decisions were made and why.

    In production, this would query the salience_computation_log table.
    For now, returns in-memory logs.
    """
    tenant_configs = _get_tenant_configs(tenant_id)

    if config_name not in tenant_configs:
        raise HTTPException(
            status_code=404,
            detail=f"Config '{config_name}' not found for tenant '{tenant_id}'"
        )

    config = tenant_configs[config_name]

    # Filter logs for this config and tenant
    filtered_logs = [
        log for log in _computation_logs
        if log.tenant_id == tenant_id and log.config_id == config.config_id
    ]

    # Sort by created_at descending (most recent first)
    filtered_logs.sort(key=lambda x: x.created_at, reverse=True)

    # Limit results
    filtered_logs = filtered_logs[:limit]

    # Convert to simplified format
    audit_entries = [
        AuditLogEntry(
            log_id=log.log_id,
            fact_text=log.fact_text,
            config_name=config_name,
            salience_score=log.final_salience_score,
            threshold=log.threshold_value,
            decision=log.decision,
            reasoning=log.reasoning,
            created_at=log.created_at.isoformat()
        )
        for log in filtered_logs
    ]

    return AuditLogResponse(
        logs=audit_entries,
        total_count=len(audit_entries)
    )


# ============================================================================
# Health Check
# ============================================================================


@router.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "salience-config",
        "total_tenants": len(_configs_db),
        "total_configs": sum(len(configs) for configs in _configs_db.values()),
        "total_logs": len(_computation_logs)
    }
