"""
FastAPI routes for salience configuration management.

Provides REST endpoints for CRUD operations on salience configurations,
testing configurations, and viewing audit logs.
"""

from typing import Dict, List, Optional, Any
from datetime import datetime
from pydantic import BaseModel

from memlayer.config.salience import TenantSalienceConfig
from memlayer.services.salience_calculator import SalienceCalculator


# Response models
class ConfigCreatedResponse(BaseModel):
    """Response after creating a new configuration."""
    config_id: str
    status: str = "created"
    config_name: str


class ConfigListResponse(BaseModel):
    """Response listing all configurations for a tenant."""
    configs: List[TenantSalienceConfig]
    count: int


class ConfigTestRequest(BaseModel):
    """Request to test a configuration."""
    fact: str
    recent_salience_history: Optional[List[float]] = None


class ComponentScoreDetail(BaseModel):
    """Details of a single component score."""
    name: str
    weight: float
    score: float
    weighted_contribution: float


class ConfigTestResponse(BaseModel):
    """Response from testing a configuration."""
    salience_score: float
    threshold: float
    decision: str  # "STORE" or "SKIP"
    component_scores: Dict[str, float]
    component_details: List[ComponentScoreDetail]
    full_log: Dict[str, Any]


class AuditLogEntry(BaseModel):
    """Single entry in audit log."""
    id: str
    tenant_id: str
    fact_id: str
    config_id: str
    config_snapshot: Dict[str, Any]
    component_scores: Dict[str, float]
    final_salience_score: float
    threshold_value: float
    decision_rule_matched: Optional[str]
    decision: str  # "STORE" or "SKIP"
    reasoning: str
    created_at: datetime


class AuditLogResponse(BaseModel):
    """Response containing audit log entries."""
    logs: List[AuditLogEntry]
    count: int


class ErrorResponse(BaseModel):
    """Error response."""
    error: str
    details: Optional[Dict[str, Any]] = None


def create_salience_router(config_store: "SalienceConfigStore"):
    """
    Create a FastAPI router for salience configuration endpoints.

    Args:
        config_store: Storage backend for configurations

    Returns:
        APIRouter with all salience config endpoints

    This function should be called in your main FastAPI app setup:

        from fastapi import FastAPI
        from memlayer.routes.salience_config import create_salience_router
        from memlayer.storage.salience_config_store import SalienceConfigStore

        app = FastAPI()
        config_store = SalienceConfigStore()
        router = create_salience_router(config_store)
        app.include_router(router, prefix="/api/config")
    """
    try:
        from fastapi import APIRouter, HTTPException, Query
    except ImportError:
        raise ImportError("FastAPI is required for the salience config router. Install with: pip install fastapi")

    router = APIRouter()

    # POST /api/config/salience - Create new configuration
    @router.post("/salience", response_model=ConfigCreatedResponse)
    async def create_salience_config(config: TenantSalienceConfig):
        """
        Create a new salience configuration.

        The configuration can include any custom components, threshold strategy,
        and decision rules. The system validates only that weights sum to 1.0.

        Example request:
            {
                "tenant_id": "tenant-123",
                "config_name": "my_custom_score_54",
                "components": [
                    {
                        "name": "novelty",
                        "weight": 0.5,
                        "description": "Novelty score",
                        "scoring_function": "embedding_similarity",
                        "scoring_config": {}
                    },
                    {
                        "name": "importance",
                        "weight": 0.5,
                        "description": "Importance score",
                        "scoring_function": "llm_scored",
                        "scoring_config": {}
                    }
                ],
                "threshold_config": {
                    "strategy": "mean_multiplier",
                    "multiplier": 0.8
                },
                "is_active": true
            }
        """
        try:
            config.created_at = datetime.utcnow()
            config.updated_at = datetime.utcnow()

            config_id = await config_store.create_config(config)

            return ConfigCreatedResponse(
                config_id=config_id,
                status="created",
                config_name=config.config_name
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error creating configuration: {str(e)}")

    # GET /api/config/salience - List all configurations for tenant
    @router.get("/salience", response_model=ConfigListResponse)
    async def list_salience_configs(tenant_id: str = Query(...)):
        """
        List all salience configurations for a tenant.

        Returns all configurations created by the tenant, including inactive ones.
        """
        try:
            configs = await config_store.list_configs(tenant_id)
            return ConfigListResponse(configs=configs, count=len(configs))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error listing configurations: {str(e)}")

    # GET /api/config/salience/{config_name} - Get specific configuration
    @router.get("/salience/{config_name}", response_model=TenantSalienceConfig)
    async def get_salience_config(tenant_id: str = Query(...), config_name: str = None):
        """
        Get a specific salience configuration by name.

        Returns the complete configuration including all components, rules, and metadata.
        """
        try:
            config = await config_store.get_config(tenant_id, config_name)
            if not config:
                raise HTTPException(status_code=404, detail="Configuration not found")
            return config
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error retrieving configuration: {str(e)}")

    # PUT /api/config/salience/{config_name} - Update configuration
    @router.put("/salience/{config_name}")
    async def update_salience_config(config_name: str, config: TenantSalienceConfig, tenant_id: str = Query(...)):
        """
        Update an existing salience configuration.

        All fields can be modified. The system re-validates weight sums and types.
        """
        try:
            config.updated_at = datetime.utcnow()

            success = await config_store.update_config(tenant_id, config_name, config)
            if not success:
                raise HTTPException(status_code=404, detail="Configuration not found")

            return {"status": "updated", "config_name": config_name}
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error updating configuration: {str(e)}")

    # DELETE /api/config/salience/{config_name} - Delete configuration
    @router.delete("/salience/{config_name}")
    async def delete_salience_config(tenant_id: str = Query(...), config_name: str = None):
        """
        Delete a salience configuration.

        This is a hard delete - the configuration and its audit logs are removed.
        Inactive configurations can also be deleted.
        """
        try:
            success = await config_store.delete_config(tenant_id, config_name)
            if not success:
                raise HTTPException(status_code=404, detail="Configuration not found")

            return {"status": "deleted", "config_name": config_name}
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error deleting configuration: {str(e)}")

    # POST /api/config/salience/{config_name}/test - Test configuration
    @router.post("/salience/{config_name}/test", response_model=ConfigTestResponse)
    async def test_salience_config(
        config_name: str,
        request: ConfigTestRequest,
        tenant_id: str = Query(...)
    ):
        """
        Test a configuration against a sample fact.

        This endpoint computes the salience score using the specified configuration
        and returns detailed breakdown of component scores, threshold calculation,
        and final decision.

        Example request:
            {
                "fact": "I learned that photosynthesis uses light energy...",
                "recent_salience_history": [0.5, 0.6, 0.7, 0.8, 0.6]
            }

        Example response:
            {
                "salience_score": 0.75,
                "threshold": 0.65,
                "decision": "STORE",
                "component_scores": {
                    "novelty": 0.8,
                    "importance": 0.7,
                    "technical_depth": 0.6
                },
                "component_details": [...],
                "full_log": {...}
            }
        """
        try:
            # Get configuration
            config = await config_store.get_config(tenant_id, config_name)
            if not config:
                raise HTTPException(status_code=404, detail="Configuration not found")

            # Create calculator
            calculator = SalienceCalculator(
                config=config,
                fact=request.fact,
                tenant_id=tenant_id,
                recent_salience_history=request.recent_salience_history
            )

            # Compute salience
            salience_score, decision, component_scores = calculator.compute_salience()
            threshold = calculator.log.get("threshold", 0.5)

            # Build component details
            component_details = []
            for component in config.components:
                score = component_scores.get(component.name, 0.0)
                weighted_contribution = score * component.weight
                component_details.append(ComponentScoreDetail(
                    name=component.name,
                    weight=component.weight,
                    score=score,
                    weighted_contribution=weighted_contribution
                ))

            return ConfigTestResponse(
                salience_score=salience_score,
                threshold=threshold,
                decision=decision,
                component_scores=component_scores,
                component_details=component_details,
                full_log=calculator.get_log()
            )

        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error testing configuration: {str(e)}")

    # GET /api/config/salience/{config_name}/audit-log - Get audit log
    @router.get("/salience/{config_name}/audit-log", response_model=AuditLogResponse)
    async def get_salience_audit_log(
        config_name: str,
        tenant_id: str = Query(...),
        limit: int = Query(100, ge=1, le=1000)
    ):
        """
        Get audit log of decisions made with this configuration.

        Returns a list of recent decisions, showing which facts were stored/skipped
        and why (which decision rule matched, component scores, etc.).

        Query parameters:
            - limit: Maximum number of log entries to return (default 100, max 1000)
        """
        try:
            logs = await config_store.get_audit_logs(tenant_id, config_name, limit=limit)
            return AuditLogResponse(logs=logs, count=len(logs))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error retrieving audit log: {str(e)}")

    return router


# Placeholder for config store interface
class SalienceConfigStore:
    """
    Abstract interface for storing salience configurations.

    Implementations should support:
    - Storing configs with tenant/config_name uniqueness
    - Retrieving, updating, deleting configs
    - Storing audit logs of decisions made with configs

    See memlayer.storage.salience_config_store for concrete implementations.
    """

    async def create_config(self, config: TenantSalienceConfig) -> str:
        """Create a configuration, return config_id."""
        raise NotImplementedError

    async def list_configs(self, tenant_id: str) -> List[TenantSalienceConfig]:
        """List all configurations for a tenant."""
        raise NotImplementedError

    async def get_config(self, tenant_id: str, config_name: str) -> Optional[TenantSalienceConfig]:
        """Get a specific configuration."""
        raise NotImplementedError

    async def update_config(self, tenant_id: str, config_name: str, config: TenantSalienceConfig) -> bool:
        """Update a configuration. Return True if successful."""
        raise NotImplementedError

    async def delete_config(self, tenant_id: str, config_name: str) -> bool:
        """Delete a configuration. Return True if successful."""
        raise NotImplementedError

    async def get_audit_logs(self, tenant_id: str, config_name: str, limit: int = 100) -> List[AuditLogEntry]:
        """Get audit logs for a configuration."""
        raise NotImplementedError

    async def log_decision(
        self,
        tenant_id: str,
        fact_id: str,
        config_id: str,
        config_snapshot: Dict[str, Any],
        component_scores: Dict[str, float],
        final_salience_score: float,
        threshold_value: float,
        decision_rule_matched: Optional[str],
        decision: str,
        reasoning: str
    ) -> str:
        """Log a salience decision. Return log_id."""
        raise NotImplementedError
