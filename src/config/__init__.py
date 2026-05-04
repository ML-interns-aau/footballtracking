"""Configuration management module.

This module provides centralized configuration management for the football
analytics pipeline.

Quick Start:
    from src.config import CONFIG
    
    # Get a configuration value
    conf_threshold = CONFIG.get('detection.confidence_threshold', 0.30)
    
    # Get nested config section
    detection_config = CONFIG.get_dict('detection')

Available exports:
    CONFIG: Global ConfigLoader instance
    ConfigLoader: The configuration loader class
"""

from .config_loader import ConfigLoader, CONFIG

__all__ = ['ConfigLoader', 'CONFIG']
