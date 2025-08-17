"""
Media processing utilities for CaptionStrike

Handles conversion, probing, and thumbnail generation for images, videos, and audio.
"""

import io
import subprocess
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import logging

import ffmpeg
import numpy as np
from PIL import Image
from pydub import AudioSegment

logger = logging.getLogger(__name__)


class MediaProcessor:
    """Handles media file processing and conversion."""
    
    # Supported file extensions
    IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tiff', '.gif'}
    VIDEO_EXTENSIONS = {'.mp4', '.mov', '.mkv', '.avi', '.wmv', '.flv', '.webm'}
    AUDIO_EXTENSIONS = {'.mp3', '.wav', '.m4a', '.flac', '.aac', '.ogg', '.wma'}
    
    @classmethod
    def get_media_type(cls, file_path: Path) -> Optional[str]:
        """Determine media type from file extension.
        
        Args:
            file_path: Path to media file
            
        Returns:
            str or None: 'image', 'video', 'audio', or None if unsupported
        """
        ext = file_path.suffix.lower()
        if ext in cls.IMAGE_EXTENSIONS:
            return 'image'
        elif ext in cls.VIDEO_EXTENSIONS:
            return 'video'
        elif ext in cls.AUDIO_EXTENSIONS:
            return 'audio'
        return None
    
    @staticmethod
    def probe_media(file_path: Path) -> Dict[str, Any]:
        """Probe media file for metadata.
        
        Args:
            file_path: Path to media file
            
        Returns:
            dict: Media metadata
        """
        try:
            probe = ffmpeg.probe(str(file_path))
            return probe
        except Exception as e:
            logger.error(f"Failed to probe {file_path}: {e}")
            return {}
    
    @staticmethod
    def convert_image_to_png(src_path: Path, dst_path: Path) -> Path:
        """Convert image to PNG format.
        
        Args:
            src_path: Source image path
            dst_path: Destination path (will be changed to .png)
            
        Returns:
            Path: Actual output path with .png extension
        """
        try:
            img = Image.open(src_path)
            # Convert to RGB if necessary (handles RGBA, P mode, etc.)
            if img.mode in ('RGBA', 'LA', 'P'):
                # Create white background for transparency
                background = Image.new('RGB', img.size, (255, 255, 255))
                if img.mode == 'P':
                    img = img.convert('RGBA')
                background.paste(img, mask=img.split()[-1] if img.mode in ('RGBA', 'LA') else None)
                img = background
            elif img.mode != 'RGB':
                img = img.convert('RGB')
            
            output_path = dst_path.with_suffix('.png')
            img.save(output_path, 'PNG', optimize=True)
            return output_path
        except Exception as e:
            logger.error(f"Failed to convert image {src_path} to PNG: {e}")
            raise
    
    @staticmethod
    def convert_video_to_mp4(src_path: Path, dst_path: Path) -> Path:
        """Convert video to MP4 format with H.264 encoding.
        
        Args:
            src_path: Source video path
            dst_path: Destination path (will be changed to .mp4)
            
        Returns:
            Path: Actual output path with .mp4 extension
        """
        try:
            output_path = dst_path.with_suffix('.mp4')
            (
                ffmpeg
                .input(str(src_path))
                .output(
                    str(output_path),
                    vcodec='libx264',
                    acodec='aac',
                    strict='-2',
                    movflags='faststart',
                    preset='medium',
                    crf=23
                )
                .overwrite_output()
                .run(quiet=True, capture_stdout=True)
            )
            return output_path
        except Exception as e:
            logger.error(f"Failed to convert video {src_path} to MP4: {e}")
            raise
    
    @staticmethod
    def convert_audio_to_mp3(src_path: Path, dst_path: Path, bitrate: str = "192k") -> Path:
        """Convert audio to MP3 format.
        
        Args:
            src_path: Source audio path
            dst_path: Destination path (will be changed to .mp3)
            bitrate: Audio bitrate (default: 192k)
            
        Returns:
            Path: Actual output path with .mp3 extension
        """
        try:
            audio = AudioSegment.from_file(src_path)
            output_path = dst_path.with_suffix('.mp3')
            audio.export(output_path, format="mp3", bitrate=bitrate)
            return output_path
        except Exception as e:
            logger.error(f"Failed to convert audio {src_path} to MP3: {e}")
            raise
    
    @staticmethod
    def extract_video_frame(video_path: Path, timestamp: float = 0.1) -> Image.Image:
        """Extract a frame from video at specified timestamp.
        
        Args:
            video_path: Path to video file
            timestamp: Time in seconds to extract frame (default: 0.1s)
            
        Returns:
            PIL.Image: Extracted frame
        """
        try:
            out, _ = (
                ffmpeg
                .input(str(video_path), ss=timestamp)
                .filter('scale', 640, -1)  # Scale to 640px width, maintain aspect ratio
                .output('pipe:', vframes=1, format='image2', vcodec='mjpeg')
                .run(capture_stdout=True, capture_stderr=True, quiet=True)
            )
            return Image.open(io.BytesIO(out))
        except Exception as e:
            logger.error(f"Failed to extract frame from {video_path}: {e}")
            raise
    
    @staticmethod
    def create_thumbnail(image: Image.Image, size: Tuple[int, int] = (256, 256)) -> Image.Image:
        """Create thumbnail from image.
        
        Args:
            image: Source PIL Image
            size: Thumbnail size (width, height)
            
        Returns:
            PIL.Image: Thumbnail image
        """
        thumb = image.copy()
        thumb.thumbnail(size, Image.Resampling.LANCZOS)
        return thumb
    
    @staticmethod
    def save_thumbnail(image: Image.Image, output_path: Path, quality: int = 85) -> Path:
        """Save thumbnail to file.
        
        Args:
            image: PIL Image to save
            output_path: Output path (will be changed to .jpg)
            quality: JPEG quality (1-100)
            
        Returns:
            Path: Actual output path with .jpg extension
        """
        try:
            thumb_path = output_path.with_suffix('.jpg')
            # Ensure RGB mode for JPEG
            if image.mode in ('RGBA', 'LA', 'P'):
                background = Image.new('RGB', image.size, (255, 255, 255))
                if image.mode == 'P':
                    image = image.convert('RGBA')
                background.paste(image, mask=image.split()[-1] if image.mode in ('RGBA', 'LA') else None)
                image = background
            elif image.mode != 'RGB':
                image = image.convert('RGB')
            
            image.save(thumb_path, 'JPEG', quality=quality, optimize=True)
            return thumb_path
        except Exception as e:
            logger.error(f"Failed to save thumbnail to {output_path}: {e}")
            raise
