import os
from pathlib import Path
from typing import Any, Dict, Optional, Union
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False
class ConfigLoader:
    _instance = None
    _config = None
    _config_path = None
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    def load(self, config_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
        if self._config is not None and config_path is None:
            return self._config
        if config_path is None:
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
        if YAML_AVAILABLE and config_path.exists():
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    self._config = yaml.safe_load(f)
                return self._config
            except Exception:
                pass
        self._config = self._default_config()
        return self._config
    def reload(self, config_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
        self._config = None
        return self.load(config_path)
    def get(self, key: str, default: Any = None) -> Any:
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
        result = self.get(key, {})
        return result if isinstance(result, dict) else {}
    def _default_config(self) -> Dict[str, Any]:
        return {
            'model': {
                'path': 'yolo11m.pt',
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
        return self._config_path
    def is_loaded_from_file(self) -> bool:
        return self._config_path is not None and Path(self._config_path).exists()
    def validate(self, raise_on_error: bool = True) -> tuple[bool, list[str]]:
        errors = []
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
        return os.getenv('FOOTBALL_ENV', 'default')
    def print_summary(self):
        print("Configuration Summary")
        print("=" * 50)
        print(f"Environment: {self.get_environment()}")
        print(f"Config path: {self.config_path or 'Using defaults'}")
        print(f"From file: {self.is_loaded_from_file()}")
        is_valid, errors = self.validate(raise_on_error=False)
        print(f"Validation: {'✓ PASS' if is_valid else '✗ FAIL'}")
        if errors:
            for error in errors:
                print(f"  ⚠ {error}")
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
CONFIG = ConfigLoader()