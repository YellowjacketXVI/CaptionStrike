"""
I/O utilities for CaptionStrike

Handles project layout, file operations, caption management, and logging.
"""

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)


class ProjectLayout:
    """Manages CaptionStrike project directory structure."""
    
    def __init__(self, root_path: Path, project_name: str):
        """Initialize project layout.
        
        Args:
            root_path: Root directory for all projects
            project_name: Name of the specific project
        """
        self.root_path = Path(root_path)
        self.project_name = project_name
        self.project_path = self.root_path / project_name
        
        # Define directory structure
        self.raw_dir = self.project_path / "raw"
        self.processed_dir = self.project_path / "processed"
        self.meta_dir = self.project_path / "meta"
        self.thumbs_dir = self.processed_dir / "thumbs"
        
        # Media subdirectories
        self.raw_image_dir = self.raw_dir / "image"
        self.raw_video_dir = self.raw_dir / "video"
        self.raw_audio_dir = self.raw_dir / "audio"
        
        self.processed_image_dir = self.processed_dir / "image"
        self.processed_video_dir = self.processed_dir / "video"
        self.processed_audio_dir = self.processed_dir / "audio"
        
        # Special subdirectories
        self.crops_dir = self.processed_image_dir / "crops"
        
        # Key files
        self.project_config_file = self.meta_dir / "project.json"
        self.run_logs_file = self.meta_dir / "run_logs.jsonl"
    
    def create_directories(self) -> None:
        """Create all necessary project directories."""
        directories = [
            self.raw_image_dir,
            self.raw_video_dir,
            self.raw_audio_dir,
            self.processed_image_dir,
            self.processed_video_dir,
            self.processed_audio_dir,
            self.thumbs_dir,
            self.crops_dir,
            self.meta_dir
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Created project directories for '{self.project_name}'")
    
    def exists(self) -> bool:
        """Check if project exists."""
        return self.project_path.exists() and self.project_config_file.exists()
    
    def get_raw_files(self, media_type: Optional[str] = None) -> List[Path]:
        """Get list of raw files, optionally filtered by media type.
        
        Args:
            media_type: Optional filter ('image', 'video', 'audio')
            
        Returns:
            List of Path objects for raw files
        """
        files = []
        
        if media_type is None or media_type == 'image':
            files.extend(self.raw_image_dir.glob('**/*'))
        if media_type is None or media_type == 'video':
            files.extend(self.raw_video_dir.glob('**/*'))
        if media_type is None or media_type == 'audio':
            files.extend(self.raw_audio_dir.glob('**/*'))
        
        return [f for f in files if f.is_file()]
    
    def get_processed_files(self, media_type: Optional[str] = None) -> List[Path]:
        """Get list of processed files, optionally filtered by media type.
        
        Args:
            media_type: Optional filter ('image', 'video', 'audio')
            
        Returns:
            List of Path objects for processed files
        """
        files = []
        
        if media_type is None or media_type == 'image':
            files.extend(self.processed_image_dir.glob('*.png'))
        if media_type is None or media_type == 'video':
            files.extend(self.processed_video_dir.glob('*.mp4'))
        if media_type is None or media_type == 'audio':
            files.extend(self.processed_audio_dir.glob('*.mp3'))
        
        return files
    
    def get_thumbnails(self) -> List[Path]:
        """Get list of thumbnail files."""
        return list(self.thumbs_dir.glob('*.jpg'))


class ProjectConfig:
    """Manages project configuration."""
    
    DEFAULT_CONFIG = {
        "name": "",
        "created": "",
        "models": {
            "captioner": "microsoft/Florence-2-base",
            "reasoning": {
                "enabled": False,
                "model": "Qwen/Qwen2.5-VL-7B-Instruct"
            },
            "single_model_mode": False,
            "single_model": "openbmb/MiniCPM-V-2_6"
        },
        "action": {
            "method": "first_frame",
            "rewrite_with_llm": True
        },
        "isolation": {
            "faces": True,
            "sam_refine": False
        },
        "processing": {
            "image_format": "png",
            "video_format": "mp4",
            "audio_format": "mp3",
            "audio_bitrate": "192k",
            "thumbnail_size": [256, 256]
        }
    }
    
    def __init__(self, config_file: Path):
        """Initialize project configuration.
        
        Args:
            config_file: Path to project.json file
        """
        self.config_file = config_file
        self._config = self.DEFAULT_CONFIG.copy()
    
    def load(self) -> Dict[str, Any]:
        """Load configuration from file."""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    loaded_config = json.load(f)
                    # Merge with defaults to ensure all keys exist
                    self._config.update(loaded_config)
            except Exception as e:
                logger.error(f"Failed to load config from {self.config_file}: {e}")
        
        return self._config
    
    def save(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Save configuration to file.
        
        Args:
            config: Optional config dict to save, uses current config if None
        """
        if config is not None:
            self._config.update(config)
        
        try:
            self.config_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self._config, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to save config to {self.config_file}: {e}")
            raise
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key path (e.g., 'models.captioner')."""
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any) -> None:
        """Set configuration value by key path."""
        keys = key.split('.')
        config = self._config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value


class RunLogger:
    """Handles logging of processing runs."""
    
    def __init__(self, log_file: Path):
        """Initialize run logger.
        
        Args:
            log_file: Path to run_logs.jsonl file
        """
        self.log_file = log_file
    
    def log_item(self, item_data: Dict[str, Any]) -> None:
        """Log a processed item.
        
        Args:
            item_data: Dictionary containing item processing information
        """
        # Add timestamp
        item_data['timestamp'] = datetime.now().isoformat()
        
        try:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(item_data, ensure_ascii=False) + '\n')
        except Exception as e:
            logger.error(f"Failed to log item to {self.log_file}: {e}")
    
    def get_logs(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get processing logs.
        
        Args:
            limit: Optional limit on number of entries to return
            
        Returns:
            List of log entries
        """
        logs = []
        
        if not self.log_file.exists():
            return logs
        
        try:
            with open(self.log_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        logs.append(json.loads(line))
        except Exception as e:
            logger.error(f"Failed to read logs from {self.log_file}: {e}")
        
        if limit:
            logs = logs[-limit:]
        
        return logs


def write_caption_file(caption_file: Path, caption: str) -> None:
    """Write caption to text file.
    
    Args:
        caption_file: Path to caption .txt file
        caption: Caption text to write
    """
    try:
        caption_file.parent.mkdir(parents=True, exist_ok=True)
        with open(caption_file, 'w', encoding='utf-8') as f:
            f.write(caption)
    except Exception as e:
        logger.error(f"Failed to write caption to {caption_file}: {e}")
        raise


def read_caption_file(caption_file: Path) -> str:
    """Read caption from text file.
    
    Args:
        caption_file: Path to caption .txt file
        
    Returns:
        Caption text, empty string if file doesn't exist
    """
    if not caption_file.exists():
        return ""
    
    try:
        with open(caption_file, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except Exception as e:
        logger.error(f"Failed to read caption from {caption_file}: {e}")
        return ""


def copy_to_raw(source_file: Path, destination_dir: Path) -> Path:
    """Copy file to raw directory.
    
    Args:
        source_file: Source file path
        destination_dir: Destination directory
        
    Returns:
        Path to copied file
    """
    try:
        destination_dir.mkdir(parents=True, exist_ok=True)
        dest_file = destination_dir / source_file.name
        shutil.copy2(source_file, dest_file)
        return dest_file
    except Exception as e:
        logger.error(f"Failed to copy {source_file} to {destination_dir}: {e}")
        raise
