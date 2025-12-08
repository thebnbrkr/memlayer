-- ============================================================================
-- Salience Configuration System - Database Schema
-- ============================================================================
--
-- This migration creates tables for the fully flexible salience configuration
-- system. Users can define custom components, thresholds, and decision rules
-- with ZERO restrictions.
--
-- Tables:
-- 1. tenant_salience_config - Stores user-defined salience configurations
-- 2. salience_computation_log - Audit trail of all salience decisions
--
-- ============================================================================

-- Create UUID extension if not exists (required for PostgreSQL)
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ============================================================================
-- Table: tenant_salience_config
-- ============================================================================
-- Stores custom salience configurations for each tenant.
--
-- Key features:
-- - JSONB columns for maximum flexibility (components, threshold_config, decision_rules)
-- - No hardcoded constraints on what components can be defined
-- - Unique constraint on (tenant_id, config_name) to prevent duplicates
-- - Full audit trail with created_at/updated_at timestamps
-- ============================================================================

CREATE TABLE IF NOT EXISTS tenant_salience_config (
    -- Primary key
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),

    -- Tenant information
    tenant_id UUID NOT NULL,

    -- Config identification
    config_name VARCHAR(255) NOT NULL,

    -- Configuration data (stored as JSONB for maximum flexibility)
    -- This allows users to define ANY components they want with ANY structure
    components JSONB NOT NULL,
    -- Example: [
    --   {
    --     "name": "novelty",
    --     "weight": 0.4,
    --     "description": "How new?",
    --     "scoring_function": "embedding_similarity",
    --     "scoring_config": {"target": "existing_memories"}
    --   },
    --   {
    --     "name": "emotional_impact",
    --     "weight": 0.6,
    --     "scoring_function": "llm_scored",
    --     "scoring_config": {"prompt": "Score emotional impact..."}
    --   }
    -- ]

    -- Threshold configuration (JSONB for flexibility)
    threshold_config JSONB NOT NULL,
    -- Example: {
    --   "strategy": "percentile",
    --   "recent_facts_window": 150,
    --   "percentile": 70
    -- }

    -- Decision rules (JSONB array)
    decision_rules JSONB NOT NULL DEFAULT '[]'::jsonb,
    -- Example: [
    --   {
    --     "rule_id": "rule_1",
    --     "name": "Always keep high emotional",
    --     "condition": "emotional_impact > 0.85",
    --     "action": "STORE",
    --     "priority": 1
    --   }
    -- ]

    -- Status and metadata
    is_active BOOLEAN NOT NULL DEFAULT TRUE,

    -- Audit fields
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    created_by_user_id UUID,

    -- Constraints
    CONSTRAINT unique_tenant_config_name UNIQUE (tenant_id, config_name)
);

-- Indexes for performance
CREATE INDEX idx_tenant_salience_config_tenant_id ON tenant_salience_config(tenant_id);
CREATE INDEX idx_tenant_salience_config_active ON tenant_salience_config(is_active) WHERE is_active = TRUE;
CREATE INDEX idx_tenant_salience_config_created_at ON tenant_salience_config(created_at);

-- JSONB GIN indexes for fast querying
CREATE INDEX idx_tenant_salience_config_components ON tenant_salience_config USING GIN (components);
CREATE INDEX idx_tenant_salience_config_threshold ON tenant_salience_config USING GIN (threshold_config);
CREATE INDEX idx_tenant_salience_config_rules ON tenant_salience_config USING GIN (decision_rules);

-- Update trigger for updated_at
CREATE OR REPLACE FUNCTION update_tenant_salience_config_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_update_tenant_salience_config_updated_at
    BEFORE UPDATE ON tenant_salience_config
    FOR EACH ROW
    EXECUTE FUNCTION update_tenant_salience_config_updated_at();


-- ============================================================================
-- Table: salience_computation_log
-- ============================================================================
-- Complete audit trail of every salience computation.
--
-- This table stores:
-- - Which config was used (with snapshot of config at that time)
-- - Individual component scores
-- - Final salience score and threshold
-- - Which decision rule matched (if any)
-- - Final decision (STORE or SKIP)
-- - Reasoning for the decision
--
-- This provides full transparency and allows for:
-- - Debugging salience decisions
-- - A/B testing different configs
-- - Analyzing which components are most predictive
-- - Compliance and audit requirements
-- ============================================================================

CREATE TABLE IF NOT EXISTS salience_computation_log (
    -- Primary key
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),

    -- Tenant and fact information
    tenant_id UUID NOT NULL,
    fact_id UUID,  -- Optional: ID of the fact being scored
    fact_text TEXT NOT NULL,  -- The actual fact text

    -- Config information
    config_id UUID NOT NULL,  -- References tenant_salience_config.id

    -- Snapshot of the config at time of computation (for historical accuracy)
    -- This ensures audit trail is preserved even if config is later modified
    config_snapshot JSONB NOT NULL,
    -- Example: {
    --   "config_name": "my_custom_score_54",
    --   "components": [...],
    --   "threshold_config": {...},
    --   "decision_rules": [...]
    -- }

    -- Computation results
    component_scores JSONB NOT NULL,
    -- Example: {
    --   "novelty": 0.85,
    --   "emotional_impact": 0.3,
    --   "technical_depth": 0.6
    -- }

    final_salience_score FLOAT NOT NULL,
    threshold_value FLOAT NOT NULL,

    -- Decision information
    decision_rule_matched VARCHAR(255),  -- Name of the matched rule, if any
    decision VARCHAR(10) NOT NULL CHECK (decision IN ('STORE', 'SKIP')),
    reasoning TEXT NOT NULL,

    -- Performance metrics
    computation_time_ms FLOAT,

    -- Audit
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,

    -- Indexes
    CONSTRAINT fk_salience_log_config FOREIGN KEY (config_id)
        REFERENCES tenant_salience_config(id) ON DELETE CASCADE
);

-- Indexes for querying and analytics
CREATE INDEX idx_salience_log_tenant_id ON salience_computation_log(tenant_id);
CREATE INDEX idx_salience_log_config_id ON salience_computation_log(config_id);
CREATE INDEX idx_salience_log_decision ON salience_computation_log(decision);
CREATE INDEX idx_salience_log_created_at ON salience_computation_log(created_at);
CREATE INDEX idx_salience_log_fact_id ON salience_computation_log(fact_id) WHERE fact_id IS NOT NULL;

-- JSONB indexes for analytics
CREATE INDEX idx_salience_log_component_scores ON salience_computation_log USING GIN (component_scores);
CREATE INDEX idx_salience_log_config_snapshot ON salience_computation_log USING GIN (config_snapshot);

-- Composite index for common query pattern (tenant + recent logs)
CREATE INDEX idx_salience_log_tenant_created ON salience_computation_log(tenant_id, created_at DESC);


-- ============================================================================
-- Example Queries
-- ============================================================================

-- Example 1: Get all active configs for a tenant
-- SELECT * FROM tenant_salience_config
-- WHERE tenant_id = 'customer_123' AND is_active = TRUE;

-- Example 2: Get recent salience decisions for a config
-- SELECT fact_text, final_salience_score, threshold_value, decision, reasoning
-- FROM salience_computation_log
-- WHERE config_id = 'cfg_abc123'
-- ORDER BY created_at DESC
-- LIMIT 100;

-- Example 3: Analyze component importance (which components correlate with STORE decisions)
-- SELECT
--   jsonb_object_keys(component_scores) as component_name,
--   AVG((component_scores->>jsonb_object_keys(component_scores))::float) as avg_score,
--   COUNT(*) as count
-- FROM salience_computation_log
-- WHERE decision = 'STORE' AND tenant_id = 'customer_123'
-- GROUP BY component_name
-- ORDER BY avg_score DESC;

-- Example 4: Get configs using a specific scoring function
-- SELECT config_name, components
-- FROM tenant_salience_config
-- WHERE components @> '[{"scoring_function": "llm_scored"}]'::jsonb;


-- ============================================================================
-- Comments for documentation
-- ============================================================================

COMMENT ON TABLE tenant_salience_config IS
'Stores fully customizable salience configurations. Users can define ANY components, ANY weights, ANY thresholds. The ONLY validation: weights must sum to 1.0.';

COMMENT ON COLUMN tenant_salience_config.components IS
'JSONB array of component definitions. Each component has: name, weight, description, scoring_function, scoring_config. Users have complete freedom to define custom components.';

COMMENT ON COLUMN tenant_salience_config.threshold_config IS
'JSONB object defining threshold calculation strategy. Supports: mean_multiplier, percentile, absolute, exponential_decay, dynamic_percentile, confidence_weighted.';

COMMENT ON COLUMN tenant_salience_config.decision_rules IS
'JSONB array of decision rules. Rules are boolean expressions that can override threshold-based decisions. Evaluated in priority order.';

COMMENT ON TABLE salience_computation_log IS
'Complete audit trail of salience computations. Stores config snapshot, component scores, final decision, and reasoning for full transparency.';

COMMENT ON COLUMN salience_computation_log.config_snapshot IS
'Snapshot of the complete config at time of computation. Preserves historical accuracy even if config is later modified or deleted.';

COMMENT ON COLUMN salience_computation_log.component_scores IS
'Individual scores for each component. Key is component name (user-defined), value is score (0.0-1.0).';


-- ============================================================================
-- Validation Functions (Optional - for extra safety)
-- ============================================================================

-- Function to validate that component weights sum to 1.0
CREATE OR REPLACE FUNCTION validate_component_weights(components_json JSONB)
RETURNS BOOLEAN AS $$
DECLARE
    total_weight FLOAT;
BEGIN
    -- Sum all weights
    SELECT SUM((comp->>'weight')::float)
    INTO total_weight
    FROM jsonb_array_elements(components_json) AS comp;

    -- Check if sum is approximately 1.0 (allowing 0.99-1.01 for rounding)
    RETURN (total_weight >= 0.99 AND total_weight <= 1.01);
END;
$$ LANGUAGE plpgsql IMMUTABLE;

COMMENT ON FUNCTION validate_component_weights IS
'Validates that component weights sum to 1.0 (with tolerance 0.99-1.01 for float rounding).';

-- Example usage:
-- SELECT validate_component_weights('[
--   {"name": "novelty", "weight": 0.4},
--   {"name": "importance", "weight": 0.6}
-- ]'::jsonb);  -- Returns TRUE

-- SELECT validate_component_weights('[
--   {"name": "novelty", "weight": 0.4},
--   {"name": "importance", "weight": 0.5}
-- ]'::jsonb);  -- Returns FALSE (sum is 0.9, not 1.0)
