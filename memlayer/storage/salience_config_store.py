"""
Storage backend for salience configurations.

Provides in-memory and file-based storage options for configurations.
For production use, extend with database backends (PostgreSQL, MongoDB, etc.).
"""

import json
import uuid
from typing import Dict, List, Optional, Any
from datetime import datetime
from pathlib import Path

from memlayer.config.salience import TenantSalienceConfig
from memlayer.routes.salience_config import AuditLogEntry


class InMemorySalienceConfigStore:
    """
    In-memory storage for salience configurations.

    Useful for testing and development. Configurations are lost on restart.
    For production, use a persistent backend like FileBasedConfigStore or DatabaseConfigStore.
    """

    def __init__(self):
        """Initialize the in-memory store."""
        # Structure: {tenant_id: {config_name: config}}
        self.configs: Dict[str, Dict[str, TenantSalienceConfig]] = {}
        # Structure: {tenant_id: {config_name: [audit_entries]}}
        self.audit_logs: Dict[str, Dict[str, List[AuditLogEntry]]] = {}

    async def create_config(self, config: TenantSalienceConfig) -> str:
        """Create a configuration, return config_id."""
        tenant_id = config.tenant_id
        config_name = config.config_name

        if tenant_id not in self.configs:
            self.configs[tenant_id] = {}

        if config_name in self.configs[tenant_id]:
            raise ValueError(f"Configuration '{config_name}' already exists for tenant '{tenant_id}'")

        config_id = str(uuid.uuid4())
        config.created_at = datetime.utcnow()
        config.updated_at = datetime.utcnow()

        self.configs[tenant_id][config_name] = config

        if tenant_id not in self.audit_logs:
            self.audit_logs[tenant_id] = {}
        self.audit_logs[tenant_id][config_name] = []

        return config_id

    async def list_configs(self, tenant_id: str) -> List[TenantSalienceConfig]:
        """List all configurations for a tenant."""
        if tenant_id not in self.configs:
            return []
        return list(self.configs[tenant_id].values())

    async def get_config(self, tenant_id: str, config_name: str) -> Optional[TenantSalienceConfig]:
        """Get a specific configuration."""
        if tenant_id not in self.configs:
            return None
        return self.configs[tenant_id].get(config_name)

    async def update_config(self, tenant_id: str, config_name: str, config: TenantSalienceConfig) -> bool:
        """Update a configuration. Return True if successful."""
        if tenant_id not in self.configs:
            return False
        if config_name not in self.configs[tenant_id]:
            return False

        config.updated_at = datetime.utcnow()
        self.configs[tenant_id][config_name] = config
        return True

    async def delete_config(self, tenant_id: str, config_name: str) -> bool:
        """Delete a configuration. Return True if successful."""
        if tenant_id not in self.configs:
            return False
        if config_name not in self.configs[tenant_id]:
            return False

        del self.configs[tenant_id][config_name]

        if tenant_id in self.audit_logs and config_name in self.audit_logs[tenant_id]:
            del self.audit_logs[tenant_id][config_name]

        return True

    async def get_audit_logs(self, tenant_id: str, config_name: str, limit: int = 100) -> List[AuditLogEntry]:
        """Get audit logs for a configuration."""
        if tenant_id not in self.audit_logs:
            return []
        if config_name not in self.audit_logs[tenant_id]:
            return []

        logs = self.audit_logs[tenant_id][config_name]
        return logs[-limit:]  # Return last N entries

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
        log_id = str(uuid.uuid4())

        # Find config name from stored configs
        config_name = None
        if tenant_id in self.configs:
            for cn, cfg in self.configs[tenant_id].items():
                if cfg.config_name == config_snapshot.get("config_name"):
                    config_name = cn
                    break

        if not config_name:
            config_name = config_snapshot.get("config_name", "unknown")

        entry = AuditLogEntry(
            id=log_id,
            tenant_id=tenant_id,
            fact_id=fact_id,
            config_id=config_id,
            config_snapshot=config_snapshot,
            component_scores=component_scores,
            final_salience_score=final_salience_score,
            threshold_value=threshold_value,
            decision_rule_matched=decision_rule_matched,
            decision=decision,
            reasoning=reasoning,
            created_at=datetime.utcnow()
        )

        if tenant_id not in self.audit_logs:
            self.audit_logs[tenant_id] = {}
        if config_name not in self.audit_logs[tenant_id]:
            self.audit_logs[tenant_id][config_name] = []

        self.audit_logs[tenant_id][config_name].append(entry)
        return log_id


class FileBasedSalienceConfigStore(InMemorySalienceConfigStore):
    """
    File-based storage for salience configurations.

    Stores configurations as JSON files in a directory structure.
    Extends InMemorySalienceConfigStore with persistence.

    Directory structure:
        ./salience_configs/
        ├── {tenant_id}/
        │   ├── {config_name}.json
        │   └── {config_name}.audit.json
    """

    def __init__(self, storage_path: str = "./salience_configs"):
        """Initialize the file-based store."""
        super().__init__()
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self._load_all_configs()

    def _load_all_configs(self):
        """Load all configurations from disk."""
        for tenant_dir in self.storage_path.iterdir():
            if not tenant_dir.is_dir():
                continue

            tenant_id = tenant_dir.name

            for config_file in tenant_dir.glob("*.json"):
                if config_file.name.endswith(".audit.json"):
                    continue

                try:
                    with open(config_file, "r") as f:
                        data = json.load(f)
                        config = TenantSalienceConfig(**data)
                        config_name = config_file.stem

                        if tenant_id not in self.configs:
                            self.configs[tenant_id] = {}
                        self.configs[tenant_id][config_name] = config

                        # Load audit logs
                        audit_file = tenant_dir / f"{config_name}.audit.json"
                        if audit_file.exists():
                            with open(audit_file, "r") as af:
                                audit_data = json.load(af)
                                if tenant_id not in self.audit_logs:
                                    self.audit_logs[tenant_id] = {}
                                self.audit_logs[tenant_id][config_name] = [
                                    AuditLogEntry(**entry) for entry in audit_data
                                ]
                except Exception as e:
                    print(f"Error loading config {config_file}: {e}")

    def _save_config(self, tenant_id: str, config_name: str, config: TenantSalienceConfig):
        """Save a configuration to disk."""
        tenant_dir = self.storage_path / tenant_id
        tenant_dir.mkdir(parents=True, exist_ok=True)

        config_file = tenant_dir / f"{config_name}.json"
        with open(config_file, "w") as f:
            json.dump(config.model_dump(), f, indent=2, default=str)

    def _save_audit_logs(self, tenant_id: str, config_name: str):
        """Save audit logs to disk."""
        tenant_dir = self.storage_path / tenant_id
        tenant_dir.mkdir(parents=True, exist_ok=True)

        audit_file = tenant_dir / f"{config_name}.audit.json"
        logs = self.audit_logs.get(tenant_id, {}).get(config_name, [])

        with open(audit_file, "w") as f:
            json.dump([log.model_dump() for log in logs], f, indent=2, default=str)

    async def create_config(self, config: TenantSalienceConfig) -> str:
        """Create a configuration and save to disk."""
        config_id = await super().create_config(config)
        self._save_config(config.tenant_id, config.config_name, config)
        return config_id

    async def update_config(self, tenant_id: str, config_name: str, config: TenantSalienceConfig) -> bool:
        """Update a configuration and save to disk."""
        success = await super().update_config(tenant_id, config_name, config)
        if success:
            self._save_config(tenant_id, config_name, config)
        return success

    async def delete_config(self, tenant_id: str, config_name: str) -> bool:
        """Delete a configuration and remove from disk."""
        success = await super().delete_config(tenant_id, config_name)
        if success:
            tenant_dir = self.storage_path / tenant_id
            config_file = tenant_dir / f"{config_name}.json"
            audit_file = tenant_dir / f"{config_name}.audit.json"

            if config_file.exists():
                config_file.unlink()
            if audit_file.exists():
                audit_file.unlink()

        return success

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
        """Log a decision and save to disk."""
        log_id = await super().log_decision(
            tenant_id, fact_id, config_id, config_snapshot,
            component_scores, final_salience_score, threshold_value,
            decision_rule_matched, decision, reasoning
        )

        config_name = config_snapshot.get("config_name", "unknown")
        self._save_audit_logs(tenant_id, config_name)

        return log_id
