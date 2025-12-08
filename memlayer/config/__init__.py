"""Configuration models and utilities for Memlayer."""

# Debug mode configuration (from original config.py)
DEBUG_MODE = False

def set_debug_mode(enabled: bool):
    """
    Enable or disable debug mode globally.

    When DEBUG_MODE is True:
    - Shows detailed trace events
    - Prints background consolidation progress
    - Shows salience check details
    - Displays search tier selection

    When DEBUG_MODE is False:
    - Only shows key initialization messages
    - Only shows final extraction results
    - Minimal output for production use

    Args:
        enabled (bool): True to enable debug mode, False to disable
    """
    global DEBUG_MODE
    DEBUG_MODE = enabled

def is_debug_mode() -> bool:
    """
    Check if debug mode is currently enabled.

    Returns:
        bool: True if debug mode is enabled, False otherwise
    """
    return DEBUG_MODE


# Salience configuration models
from .salience import (
    ScoringFunctionType,
    ThresholdStrategy,
    SalienceComponent,
    AdaptiveThresholdConfig,
    DecisionRule,
    TenantSalienceConfig,
)

__all__ = [
    # Debug utilities
    "DEBUG_MODE",
    "set_debug_mode",
    "is_debug_mode",
    # Salience models
    "ScoringFunctionType",
    "ThresholdStrategy",
    "SalienceComponent",
    "AdaptiveThresholdConfig",
    "DecisionRule",
    "TenantSalienceConfig",
]
