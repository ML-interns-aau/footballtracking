"""Configuration loader with validation and caching.

This module provides centralized configuration management for the football
analytics pipeline. It loads settings from YAML files with fallback defaults.

Usage:
    from src.config.config_loader import CONFIG
    
    # Get configuration value with dot notation
    conf_threshold = CONFIG.get('detection.confidence_threshold', 0.30)
    
    # Get nested configuration
    pitch_width = CONFIG.get('pitch.width_m', 105.0)

The loader uses a singleton pattern to cache configuration and avoid
reloading the file on every access.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

# Try to import yaml, fall back to defaults if not available
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False


class ConfigLoader:
    """Load and provide access to pipeline configuration.
    
    This class implements the singleton pattern to ensure configuration
    is loaded only once and cached for subsequent accesses.
    
    Attributes:
        _instance: Singleton instance
        _config: Cached configuration dictionary
        _config_path: Path to the loaded configuration file
    """
    
    _instance = None
    _config = None
    _config_path = None
    
    def __new__(cls):
        """Create or return the singleton instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def load(self, config_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
        """Load configuration from YAML file.
        
        If config_path is not provided, uses the default path:
        configs/pipeline.yaml
        
        Args:
            config_path: Path to YAML configuration file
            
        Returns:
            Dictionary containing configuration values
            
        Note:
            Configuration is cached after first load. Call reload() to refresh.
        """
        if self._config is not None and config_path is None:
            return self._config
        
        if config_path is None:
            # Try environment-specific config first
            env = os.getenv('FOOTBALL_ENV', 'default')
            base_path = Path("configs")
            
            if env != 'default':
                env_config = base_path / f"pipeline.{env}.yaml"
                if env_config.exists():
                    config_path = env_config
                else:
                    config_path = base_path / "pipeline.yaml"
            else:
                config_path = base_path / "pipeline.yaml"
        else:
            config_path = Path(config_path)
        
        self._config_path = str(config_path)
        
        # Try to load from file
        if YAML_AVAILABLE and config_path.exists():
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    self._config = yaml.safe_load(f)
                return self._config
            except Exception:
                # Fall back to defaults on any error
                pass
        
        # Use hardcoded defaults
        self._config = self._default_config()
        return self._config
    
    def reload(self, config_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
        """Force reload configuration from file.
        
        Use this method to refresh configuration without restarting the application.
        
        Args:
            config_path: Optional new config file path
            
        Returns:
            Fresh configuration dictionary
        """
        self._config = None
        return self.load(config_path)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value using dot notation.
        
        Access nested configuration values using dot notation:
        - 'detection.confidence_threshold'
        - 'pitch.width_m'
        - 'video.target_fps'
        
        Args:
            key: Dot-separated path to configuration value
            default: Value to return if key not found
            
        Returns:
            Configuration value or default if not found
            
        Examples:
            >>> CONFIG.get('detection.confidence_threshold', 0.30)
            0.30
            >>> CONFIG.get('pitch.width_m')
            105.0
        """
        config = self.load()
        keys = key.split('.')
        value = config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def get_dict(self, key: str) -> Dict[str, Any]:
        """Get a nested configuration section as a dictionary.
        
        Args:
            key: Section key (e.g., 'detection', 'tracking')
            
        Returns:
            Dictionary containing the section, or empty dict if not found
        """
        result = self.get(key, {})
        return result if isinstance(result, dict) else {}
    
    def _default_config(self) -> Dict[str, Any]:
        """Return hardcoded default configuration.
        
        These defaults match the previously hardcoded values in main.py
        and other pipeline files. Used when config file is missing or
        YAML is not available.
        
        Returns:
            Dictionary with all default configuration values
        """
        return {
            'model': {
                'path': 'yolov8m_fixed.pt',
                'device': 'auto',
            },
            'detection': {
                'confidence_threshold': 0.30,
                'iou_threshold': 0.40,
                'image_size': 1280,
            },
            'tracking': {
                'track_threshold': 0.20,
                'track_buffer': 60,
                'match_threshold': 0.80,
            },
            'ball': {
                'max_trail_length': 25,
                'max_missed_frames': 30,
            },
            'team_classification': {
                'n_teams': 2,
                'history_length': 15,
                'refit_interval': 150,
            },
            'pitch': {
                'width_m': 105.0,
                'height_m': 68.0,
            },
            'video': {
                'target_fps': 15,
                'resize_width': 1280,
                'codec_preference': 'avc1',
                'codec_fallback': 'mp4v',
            },
            'speed': {
                'window_size': 8,
                'min_frames_for_speed': 3,
            },
            'possession': {
                'max_distance_m': 2.0,
            },
            'heatmap': {
                'bins_x': 105,
                'bins_y': 68,
            },
            'calibration': {
                'default_src_points': [
                    [0, 540],
                    [960, 540],
                    [720, 162],
                    [240, 162],
                ],
                'default_dst_points': [
                    [0, 68],
                    [105, 68],
                    [105, 0],
                    [0, 0],
                ],
            },
            'classes': {
                'ball_id': 0,
                'goalkeeper_id': 1,
                'player_id': 2,
                'referee_id': 3,
                'labels': {
                    0: 'Ball',
                    1: 'GK',
                    2: 'Player',
                    3: 'Referee',
                },
            },
            'output': {
                'video_format': 'mp4',
                'csv_format': 'csv',
                'save_intermediate': False,
            },
            'performance': {
                'max_frames_default': 0,
                'frame_batch_size': 1,
                'memory_warning_mb': 2048,
            },
        }
    
    @property
    def config_path(self) -> Optional[str]:
        """Return the path of the loaded configuration file."""
        return self._config_path
    
    def is_loaded_from_file(self) -> bool:
        """Check if configuration was loaded from a file or using defaults."""
        return self._config_path is not None and Path(self._config_path).exists()
    
    def validate(self, raise_on_error: bool = True) -> tuple[bool, list[str]]:
        """Validate configuration values.
        
        Checks that all required configuration keys exist and have
        valid values (non-negative, non-zero where appropriate).
        
        Args:
            raise_on_error: If True, raises ValueError on validation failure.
                          If False, returns (is_valid, error_messages).
                          
        Returns:
            Tuple of (is_valid, list of error messages)
            
        Raises:
            ValueError: If raise_on_error=True and validation fails.
            
        Examples:
            >>> is_valid, errors = CONFIG.validate(raise_on_error=False)
            >>> if not is_valid:
            >>>     print(f"Config errors: {errors}")
        """
        errors = []
        
        # Required configuration keys
        required_checks = [
            ('detection.confidence_threshold', (0.0, 1.0), "float between 0 and 1"),
            ('detection.iou_threshold', (0.0, 1.0), "float between 0 and 1"),
            ('detection.image_size', (1, None), "positive integer"),
            ('tracking.track_threshold', (0.0, 1.0), "float between 0 and 1"),
            ('tracking.track_buffer', (1, None), "positive integer"),
            ('pitch.width_m', (1.0, None), "positive float"),
            ('pitch.height_m', (1.0, None), "positive float"),
            ('video.target_fps', (0, None), "non-negative integer (0 = use source)"),
            ('video.resize_width', (0, None), "non-negative integer (0 = no resize)"),
        ]
        
        for key, bounds, description in required_checks:
            value = self.get(key)
            
            if value is None:
                errors.append(f"Missing required configuration: {key}")
                continue
            
            # Check type and bounds
            min_val, max_val = bounds
            
            try:
                if min_val is not None and value < min_val:
                    errors.append(f"{key} = {value}, must be >= {min_val}")
                if max_val is not None and value > max_val:
                    errors.append(f"{key} = {value}, must be <= {max_val}")
            except TypeError:
                errors.append(f"{key} has invalid type: {type(value).__name__}, expected {description}")
        
        is_valid = len(errors) == 0
        
        if raise_on_error and not is_valid:
            raise ValueError(f"Configuration validation failed:\n" + "\n".join(f"  - {e}" for e in errors))
        
        return is_valid, errors
    
    def get_environment(self) -> str:
        """Return the current environment (dev, prod, or default)."""
        return os.getenv('FOOTBALL_ENV', 'default')
    
    def print_summary(self):
        """Print a summary of loaded configuration for debugging."""
        print("Configuration Summary")
        print("=" * 50)
        print(f"Environment: {self.get_environment()}")
        print(f"Config path: {self.config_path or 'Using defaults'}")
        print(f"From file: {self.is_loaded_from_file()}")
        
        # Validate and show status
        is_valid, errors = self.validate(raise_on_error=False)
        print(f"Validation: {'✓ PASS' if is_valid else '✗ FAIL'}")
        if errors:
            for error in errors:
                print(f"  ⚠ {error}")
        
        # Show key values
        print("\nKey Settings:")
        key_settings = [
            'detection.confidence_threshold',
            'detection.iou_threshold',
            'tracking.track_buffer',
            'pitch.width_m',
            'video.target_fps',
            'performance.max_frames_default',
        ]
        for key in key_settings:
            value = self.get(key)
            print(f"  {key}: {value}")
        print("=" * 50)


# Global singleton instance
# Use this for all configuration access:
#   from src.config.config_loader import CONFIG
#   value = CONFIG.get('key.subkey', default_value)
CONFIG = ConfigLoader()
