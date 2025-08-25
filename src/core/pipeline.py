"""
Core Processing Pipeline for CaptionStrike

Orchestrates the complete media processing workflow including conversion,
captioning, tagging, audio processing, and person isolation.
"""

import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import traceback

from PIL import Image

from .io import ProjectLayout, ProjectConfig, RunLogger, write_caption_file, copy_to_raw
from .media import MediaProcessor
from .tokens import generate_token, add_token_to_filename, add_token_to_caption, safe_filename
from ..adapters.florence2_captioner import Florence2Captioner
from ..adapters.qwen_vl_reasoner import QwenVLReasoner
from ..adapters.diarizer import AudioDiarizer
from ..adapters.person_isolator import PersonIsolator

logger = logging.getLogger(__name__)


class ProcessingPipeline:
    """Main processing pipeline for CaptionStrike."""
    
    def __init__(self, models_dir: Path):
        """Initialize processing pipeline.
        
        Args:
            models_dir: Directory containing model files
        """
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize adapters (lazy loading)
        self.florence_captioner = None
        self.qwen_reasoner = None
        self.audio_diarizer = None
        self.person_isolator = None
        
        self.media_processor = MediaProcessor()
        
        logger.info(f"Initialized processing pipeline with models dir: {models_dir}")
    
    def _load_florence_captioner(self, config: ProjectConfig) -> Florence2Captioner:
        """Load Florence-2 captioner based on config."""
        if self.florence_captioner is None:
            model_name = config.get("models.captioner", "microsoft/Florence-2-base")
            self.florence_captioner = Florence2Captioner(model_name)
        return self.florence_captioner
    
    def _load_qwen_reasoner(self, config: ProjectConfig) -> Optional[QwenVLReasoner]:
        """Load Qwen2.5-VL reasoner if enabled."""
        if not config.get("models.reasoning.enabled", False):
            return None
        
        if self.qwen_reasoner is None:
            model_name = config.get("models.reasoning.model", "Qwen/Qwen2.5-VL-7B-Instruct")
            try:
                self.qwen_reasoner = QwenVLReasoner(model_name, cache_dir=self.models_dir)
            except Exception as e:
                logger.warning(f"Failed to load Qwen reasoner: {e}")
                return None
        
        return self.qwen_reasoner
    
    def _load_audio_diarizer(self) -> AudioDiarizer:
        """Load audio diarizer."""
        if self.audio_diarizer is None:
            self.audio_diarizer = AudioDiarizer()
        return self.audio_diarizer
    
    def _load_person_isolator(self) -> PersonIsolator:
        """Load person isolator."""
        if self.person_isolator is None:
            sam_checkpoint = self.models_dir / "sam_vit_h_4b8939.pth"
            self.person_isolator = PersonIsolator(
                sam_checkpoint=sam_checkpoint if sam_checkpoint.exists() else None
            )
        return self.person_isolator
    
    def process_project(self,
                       layout: ProjectLayout,
                       reference_voice_clip: Optional[Path] = None,
                       first_sound_ts: Optional[float] = None,
                       end_sound_ts: Optional[float] = None,
                       force_reprocess: bool = False) -> Dict[str, Any]:
        """Process all media in a project.
        
        Args:
            layout: Project layout manager
            reference_voice_clip: Optional reference voice for audio processing
            first_sound_ts: Optional start timestamp for audio reference
            end_sound_ts: Optional end timestamp for audio reference
            force_reprocess: Whether to reprocess existing files
            
        Returns:
            Dict with processing results
        """
        try:
            # Load project configuration
            config = ProjectConfig(layout.project_config_file)
            config.load()
            
            # Initialize run logger
            run_logger = RunLogger(layout.run_logs_file)
            
            # Get raw files to process
            raw_files = layout.get_raw_files()
            
            if not raw_files:
                return {
                    "success": True,
                    "message": "No files to process",
                    "processed_count": 0,
                    "errors": []
                }
            
            logger.info(f"Processing {len(raw_files)} files in project '{layout.project_name}'")
            
            processed_count = 0
            errors = []
            
            # Process each file
            for raw_file in raw_files:
                try:
                    result = self._process_single_file(
                        raw_file, layout, config, run_logger,
                        reference_voice_clip, first_sound_ts, end_sound_ts,
                        force_reprocess
                    )
                    
                    if result["success"]:
                        processed_count += 1
                    else:
                        errors.append(f"{raw_file.name}: {result.get('error', 'Unknown error')}")
                        
                except Exception as e:
                    error_msg = f"{raw_file.name}: {str(e)}"
                    errors.append(error_msg)
                    logger.error(f"Failed to process {raw_file}: {e}")
                    logger.debug(traceback.format_exc())
            
            # Generate thumbnails
            self._generate_thumbnails(layout)
            
            return {
                "success": True,
                "message": f"Processed {processed_count}/{len(raw_files)} files",
                "processed_count": processed_count,
                "total_files": len(raw_files),
                "errors": errors
            }
            
        except Exception as e:
            logger.error(f"Pipeline processing failed: {e}")
            return {
                "success": False,
                "message": f"Pipeline failed: {str(e)}",
                "processed_count": 0,
                "errors": [str(e)]
            }
    
    def _process_single_file(self,
                           raw_file: Path,
                           layout: ProjectLayout,
                           config: ProjectConfig,
                           run_logger: RunLogger,
                           reference_voice_clip: Optional[Path],
                           first_sound_ts: Optional[float],
                           end_sound_ts: Optional[float],
                           force_reprocess: bool) -> Dict[str, Any]:
        """Process a single media file.
        
        Args:
            raw_file: Path to raw media file
            layout: Project layout manager
            config: Project configuration
            run_logger: Run logger instance
            reference_voice_clip: Optional reference voice clip
            first_sound_ts: Optional audio start timestamp
            end_sound_ts: Optional audio end timestamp
            force_reprocess: Whether to force reprocessing
            
        Returns:
            Dict with processing result
        """
        try:
            # Determine media type
            media_type = self.media_processor.get_media_type(raw_file)
            if media_type is None:
                return {"success": False, "error": "Unsupported media type"}
            
            # Generate token and safe filename
            token = generate_token()
            safe_base = safe_filename(raw_file.stem)
            base_with_token = add_token_to_filename(safe_base, token)
            
            # Determine output paths
            if media_type == "image":
                output_dir = layout.processed_image_dir
                output_file = output_dir / f"{base_with_token}.png"
            elif media_type == "video":
                output_dir = layout.processed_video_dir
                output_file = output_dir / f"{base_with_token}.mp4"
            elif media_type == "audio":
                output_dir = layout.processed_audio_dir
                output_file = output_dir / f"{base_with_token}.mp3"
            
            # Check if already processed (unless force reprocess)
            if not force_reprocess and output_file.exists():
                logger.info(f"Skipping already processed file: {raw_file.name}")
                return {"success": True, "message": "Already processed", "skipped": True}
            
            # Ensure output directory exists
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Process based on media type
            if media_type == "image":
                result = self._process_image(raw_file, output_file, token, layout, config)
            elif media_type == "video":
                result = self._process_video(raw_file, output_file, token, layout, config)
            elif media_type == "audio":
                result = self._process_audio(
                    raw_file, output_file, token, layout, config,
                    reference_voice_clip, first_sound_ts, end_sound_ts
                )
            
            # Log processing result
            log_entry = {
                "type": media_type,
                "source": str(raw_file),
                "output": str(output_file),
                "token": token,
                "success": result["success"]
            }
            
            if not result["success"]:
                log_entry["error"] = result.get("error", "Unknown error")
            
            run_logger.log_item(log_entry)
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to process single file {raw_file}: {e}")
            return {"success": False, "error": str(e)}
    
    def _process_image(self,
                      raw_file: Path,
                      output_file: Path,
                      token: str,
                      layout: ProjectLayout,
                      config: ProjectConfig) -> Dict[str, Any]:
        """Process an image file."""
        try:
            # Convert to PNG
            converted_file = self.media_processor.convert_image_to_png(raw_file, output_file)
            
            # Load Florence-2 captioner
            florence = self._load_florence_captioner(config)
            
            # Generate caption and analysis (with optional system prompt)
            system_prompt = config.get("captioning.system_prompt", "")
            # Expose prompt to Florence via env var fallback used by adapter
            import os
            if system_prompt:
                os.environ["CAPTIONSTRIKE_SYSTEM_PROMPT"] = system_prompt
            analysis = florence.analyze_image_comprehensive(converted_file)
            caption = analysis["caption"]
            
            # Optional reasoning enhancement
            qwen = self._load_qwen_reasoner(config)
            if qwen is not None:
                try:
                    reasoning_result = qwen.refine_caption(caption, converted_file, analysis)
                    if reasoning_result["reasoning_success"]:
                        caption = reasoning_result["refined_caption"]
                except Exception as e:
                    logger.warning(f"Reasoning enhancement failed: {e}")
            
            # Add token to caption
            final_caption = add_token_to_caption(caption, token)
            
            # Write caption file
            caption_file = converted_file.with_suffix('.txt')
            write_caption_file(caption_file, final_caption)
            
            # Optional person isolation
            if config.get("isolation.faces", False):
                try:
                    isolator = self._load_person_isolator()
                    if isolator.is_available():
                        isolation_result = isolator.isolate_persons(
                            converted_file,
                            layout.processed_image_dir,
                            safe_filename(raw_file.stem),
                            use_sam=config.get("isolation.sam_refine", False)
                        )
                        logger.info(f"Person isolation: {isolation_result['message']}")
                except Exception as e:
                    logger.warning(f"Person isolation failed: {e}")
            
            return {
                "success": True,
                "output_file": converted_file,
                "caption": final_caption,
                "analysis": analysis
            }
            
        except Exception as e:
            logger.error(f"Failed to process image {raw_file}: {e}")
            return {"success": False, "error": str(e)}
    
    def _process_video(self,
                      raw_file: Path,
                      output_file: Path,
                      token: str,
                      layout: ProjectLayout,
                      config: ProjectConfig) -> Dict[str, Any]:
        """Process a video file."""
        try:
            # Convert to MP4
            converted_file = self.media_processor.convert_video_to_mp4(raw_file, output_file)
            
            # Load Florence-2 captioner
            florence = self._load_florence_captioner(config)
            
            # Analyze first frame and generate caption with action tag
            video_analysis = florence.caption_video_first_frame(converted_file)
            caption = video_analysis["caption"]
            action_tag = video_analysis["action_tag"]
            
            # Optional reasoning enhancement
            qwen = self._load_qwen_reasoner(config)
            if qwen is not None:
                try:
                    # Extract first frame for reasoning
                    frame = self.media_processor.extract_video_frame(converted_file)
                    reasoning_result = qwen.refine_caption(caption, frame)
                    if reasoning_result["reasoning_success"]:
                        # Preserve action tag in refined caption
                        refined = reasoning_result["refined_caption"]
                        if action_tag not in refined:
                            caption = f"{refined} [{action_tag}]"
                        else:
                            caption = refined
                except Exception as e:
                    logger.warning(f"Video reasoning enhancement failed: {e}")
            
            # Add token to caption
            final_caption = add_token_to_caption(caption, token)
            
            # Write caption file
            caption_file = converted_file.with_suffix('.txt')
            write_caption_file(caption_file, final_caption)
            
            return {
                "success": True,
                "output_file": converted_file,
                "caption": final_caption,
                "action_tag": action_tag,
                "analysis": video_analysis
            }
            
        except Exception as e:
            logger.error(f"Failed to process video {raw_file}: {e}")
            return {"success": False, "error": str(e)}
    
    def _process_audio(self,
                      raw_file: Path,
                      output_file: Path,
                      token: str,
                      layout: ProjectLayout,
                      config: ProjectConfig,
                      reference_voice_clip: Optional[Path],
                      first_sound_ts: Optional[float],
                      end_sound_ts: Optional[float]) -> Dict[str, Any]:
        """Process an audio file."""
        try:
            # Convert to MP3
            converted_file = self.media_processor.convert_audio_to_mp3(raw_file, output_file)
            
            # Load audio diarizer
            diarizer = self._load_audio_diarizer()
            
            # Process audio with diarization and stitching
            reference_window = None
            if first_sound_ts is not None and end_sound_ts is not None:
                reference_window = (first_sound_ts, end_sound_ts)
            
            audio_result = diarizer.process_audio_complete(
                converted_file,
                layout.processed_audio_dir,
                safe_filename(raw_file.stem),
                reference_voice_clip,
                reference_window
            )
            
            # Add token to summary text
            summary_text = audio_result["summary_text"]
            final_caption = add_token_to_caption(summary_text, token)
            
            # Write caption file
            caption_file = converted_file.with_suffix('.txt')
            write_caption_file(caption_file, final_caption)
            
            return {
                "success": audio_result["success"],
                "output_file": converted_file,
                "stitched_file": audio_result.get("stitched_audio_path"),
                "caption": final_caption,
                "audio_analysis": audio_result
            }
            
        except Exception as e:
            logger.error(f"Failed to process audio {raw_file}: {e}")
            return {"success": False, "error": str(e)}
    
    def _generate_thumbnails(self, layout: ProjectLayout) -> None:
        """Generate thumbnails for processed media."""
        try:
            # Generate thumbnails for images
            for image_file in layout.get_processed_files("image"):
                try:
                    img = self.media_processor.create_thumbnail(
                        Image.open(image_file), (256, 256)
                    )
                    thumb_path = layout.thumbs_dir / f"{image_file.stem}.jpg"
                    self.media_processor.save_thumbnail(img, thumb_path)
                except Exception as e:
                    logger.warning(f"Failed to create thumbnail for {image_file}: {e}")
            
            # Generate thumbnails for videos (first frame)
            for video_file in layout.get_processed_files("video"):
                try:
                    frame = self.media_processor.extract_video_frame(video_file)
                    thumb = self.media_processor.create_thumbnail(frame, (256, 256))
                    thumb_path = layout.thumbs_dir / f"{video_file.stem}.jpg"
                    self.media_processor.save_thumbnail(thumb, thumb_path)
                except Exception as e:
                    logger.warning(f"Failed to create video thumbnail for {video_file}: {e}")
            
            logger.info("Thumbnail generation completed")
            
        except Exception as e:
            logger.error(f"Thumbnail generation failed: {e}")
    
    def add_files_to_project(self,
                           layout: ProjectLayout,
                           file_paths: List[Path]) -> Dict[str, Any]:
        """Add files to project raw directory.
        
        Args:
            layout: Project layout manager
            file_paths: List of file paths to add
            
        Returns:
            Dict with results
        """
        try:
            added_files = []
            errors = []

            for file_path in file_paths:
                try:
                    # Handle ZIP archives: extract and treat contents as uploads
                    if file_path.suffix.lower() == ".zip":
                        try:
                            import zipfile
                            with zipfile.ZipFile(file_path, 'r') as zf:
                                for member in zf.infolist():
                                    if member.is_dir():
                                        continue
                                    # Extract to a temp staging area inside the project
                                    staging_dir = layout.project_path / "_staging"
                                    staging_dir.mkdir(parents=True, exist_ok=True)
                                    extracted_path = zf.extract(member, path=staging_dir)
                                    extracted_path = Path(extracted_path)
                                    media_type = self.media_processor.get_media_type(extracted_path)
                                    if media_type is None:
                                        errors.append(f"{member.filename}: Unsupported media type in zip")
                                        continue
                                    if media_type == "image":
                                        dest_dir = layout.raw_image_dir
                                    elif media_type == "video":
                                        dest_dir = layout.raw_video_dir
                                    elif media_type == "audio":
                                        dest_dir = layout.raw_audio_dir
                                    copied_file = copy_to_raw(extracted_path, dest_dir)
                                    added_files.append(copied_file)
                        except Exception as ze:
                            errors.append(f"{file_path.name}: Failed to extract zip - {ze}")
                        continue

                    media_type = self.media_processor.get_media_type(file_path)
                    if media_type is None:
                        errors.append(f"{file_path.name}: Unsupported media type")
                        continue

                    # Determine destination directory
                    if media_type == "image":
                        dest_dir = layout.raw_image_dir
                    elif media_type == "video":
                        dest_dir = layout.raw_video_dir
                    elif media_type == "audio":
                        dest_dir = layout.raw_audio_dir

                    # Copy file to raw directory
                    copied_file = copy_to_raw(file_path, dest_dir)
                    added_files.append(copied_file)

                except Exception as e:
                    errors.append(f"{file_path.name}: {str(e)}")
            
            return {
                "success": True,
                "added_count": len(added_files),
                "added_files": added_files,
                "errors": errors
            }
            
        except Exception as e:
            logger.error(f"Failed to add files to project: {e}")
            return {
                "success": False,
                "added_count": 0,
                "added_files": [],
                "errors": [str(e)]
            }
