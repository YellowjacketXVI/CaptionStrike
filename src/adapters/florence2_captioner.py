"""
Florence-2 Captioner Adapter for CaptionStrike

Provides image captioning, tagging, and grounding using Microsoft's Florence-2 model.
Supports both base and large variants with configurable prompts.
"""

import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
import warnings

import torch
import numpy as np
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM

logger = logging.getLogger(__name__)

# Suppress some warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")


class Florence2Captioner:
    """Florence-2 model adapter for image captioning and analysis."""
    
    # Available model variants
    MODELS = {
        "base": "microsoft/Florence-2-base",
        "large": "microsoft/Florence-2-large"
    }
    
    # Florence-2 task prompts
    TASKS = {
        "caption": "<CAPTION>",
        "detailed_caption": "<DETAILED_CAPTION>",
        "more_detailed_caption": "<MORE_DETAILED_CAPTION>",
        "od": "<OD>",  # Object detection
        "dense_region_caption": "<DENSE_REGION_CAPTION>",
        "region_proposal": "<REGION_PROPOSAL>",
        "ocr": "<OCR>",
        "ocr_with_region": "<OCR_WITH_REGION>"
    }
    
    def __init__(self, model_name: str = "base", device: Optional[str] = None, torch_dtype: Optional[torch.dtype] = None):
        """Initialize Florence-2 captioner.
        
        Args:
            model_name: Model variant ('base' or 'large') or full model path
            device: Device to run model on (auto-detected if None)
            torch_dtype: Torch data type (auto-selected if None)
        """
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.torch_dtype = torch_dtype or (torch.float16 if torch.cuda.is_available() else torch.float32)
        
        # Resolve model path
        if model_name in self.MODELS:
            self.model_path = self.MODELS[model_name]
        else:
            self.model_path = model_name
        
        self.model = None
        self.processor = None
        self._loaded = False
        
        logger.info(f"Initialized Florence-2 captioner with model: {self.model_path}")
    
    def load_model(self) -> None:
        """Load the Florence-2 model and processor."""
        if self._loaded:
            return
        
        try:
            logger.info(f"Loading Florence-2 model: {self.model_path}")
            
            # Load processor
            self.processor = AutoProcessor.from_pretrained(
                self.model_path, 
                trust_remote_code=True
            )
            
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=self.torch_dtype,
                trust_remote_code=True
            ).to(self.device)
            
            self._loaded = True
            logger.info(f"Successfully loaded Florence-2 model on {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to load Florence-2 model: {e}")
            raise
    
    def _run_task(self, image: Image.Image, task: str, text_input: Optional[str] = None, system_prompt: str = "") -> Dict[str, Any]:
        """Run a Florence-2 task on an image.

        Args:
            image: PIL Image
            task: Task prompt (e.g., "<CAPTION>")
            text_input: Optional text input for the task
            system_prompt: Optional system prompt to influence model behavior

        Returns:
            Dict containing task results
        """
        if not self._loaded:
            self.load_model()

        try:
            # Prepare inputs
            base = task
            if text_input:
                base = base + text_input
            if system_prompt:
                prompt = f"{system_prompt}\n\n{base}"
            else:
                prompt = base
            
            inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(self.device)
            
            # Generate
            with torch.no_grad():
                generated_ids = self.model.generate(
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs["pixel_values"],
                    max_new_tokens=1024,
                    num_beams=3,
                    do_sample=False
                )
            
            # Decode results
            generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
            parsed_answer = self.processor.post_process_generation(
                generated_text, 
                task=task, 
                image_size=(image.width, image.height)
            )
            
            return parsed_answer
            
        except Exception as e:
            logger.error(f"Failed to run Florence-2 task {task}: {e}")
            return {}
    
    def caption_image(self, image: Union[Image.Image, Path, str], detailed: bool = False) -> Dict[str, Any]:
        """Generate caption for an image.
        
        Args:
            image: PIL Image, file path, or path string
            detailed: Whether to use detailed captioning
            
        Returns:
            Dict with 'caption' key and other metadata
        """
        # Load image if path provided
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert('RGB')
        elif not isinstance(image, Image.Image):
            raise ValueError("Image must be PIL Image or file path")
        
        # Choose task based on detail level
        task = self.TASKS["detailed_caption"] if detailed else self.TASKS["caption"]

        try:
            # Optional system prompt support via environment variable fallback
            import os
            system_prompt = os.environ.get("CAPTIONSTRIKE_SYSTEM_PROMPT", "")
            result = self._run_task(image, task, system_prompt=system_prompt)

            # Extract caption from result
            caption = ""
            if task in result:
                caption = result[task]
            
            return {
                "caption": caption,
                "task": task,
                "raw_result": result
            }
            
        except Exception as e:
            logger.error(f"Failed to caption image: {e}")
            return {
                "caption": "Failed to generate caption",
                "task": task,
                "error": str(e)
            }
    
    def detect_objects(self, image: Union[Image.Image, Path, str]) -> Dict[str, Any]:
        """Detect objects in an image.
        
        Args:
            image: PIL Image, file path, or path string
            
        Returns:
            Dict with detected objects and bounding boxes
        """
        # Load image if path provided
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert('RGB')
        
        try:
            result = self._run_task(image, self.TASKS["od"])
            
            objects = []
            if self.TASKS["od"] in result:
                od_result = result[self.TASKS["od"]]
                if "bboxes" in od_result and "labels" in od_result:
                    for bbox, label in zip(od_result["bboxes"], od_result["labels"]):
                        objects.append({
                            "label": label,
                            "bbox": bbox,
                            "confidence": 1.0  # Florence-2 doesn't provide confidence scores
                        })
            
            return {
                "objects": objects,
                "count": len(objects),
                "raw_result": result
            }
            
        except Exception as e:
            logger.error(f"Failed to detect objects: {e}")
            return {
                "objects": [],
                "count": 0,
                "error": str(e)
            }
    
    def analyze_image_comprehensive(self, image: Union[Image.Image, Path, str]) -> Dict[str, Any]:
        """Perform comprehensive image analysis including captioning and object detection.
        
        Args:
            image: PIL Image, file path, or path string
            
        Returns:
            Dict with caption, objects, and tags
        """
        # Load image if path provided
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert('RGB')
        
        # Get caption
        caption_result = self.caption_image(image, detailed=True)

        # Get objects
        objects_result = self.detect_objects(image)
        
        # Extract tags from objects
        tags = list(set([obj["label"] for obj in objects_result["objects"]]))
        
        return {
            "caption": caption_result["caption"],
            "objects": objects_result["objects"],
            "tags": tags,
            "object_count": objects_result["count"],
            "analysis_success": not ("error" in caption_result or "error" in objects_result)
        }
    
    def caption_video_first_frame(self, video_path: Path) -> Dict[str, Any]:
        """Caption a video using its first frame and infer action tags.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Dict with caption and action tag
        """
        try:
            # Import here to avoid circular imports
            from ..core.media import MediaProcessor
            
            # Extract first frame
            frame = MediaProcessor.extract_video_frame(video_path, timestamp=0.1)
            
            # Analyze frame
            analysis = self.analyze_image_comprehensive(frame)
            
            # Infer action from detected objects
            action_tag = self._infer_action_from_objects(analysis["objects"])
            
            # Enhance caption for video context
            base_caption = analysis["caption"]
            video_caption = self._enhance_caption_for_video(base_caption, action_tag)
            
            return {
                "caption": video_caption,
                "action_tag": action_tag,
                "frame_analysis": analysis,
                "video_path": str(video_path)
            }
            
        except Exception as e:
            logger.error(f"Failed to caption video {video_path}: {e}")
            return {
                "caption": "A video showing various subjects and actions",
                "action_tag": "ACTION:generic",
                "error": str(e)
            }
    
    def _infer_action_from_objects(self, objects: List[Dict[str, Any]]) -> str:
        """Infer action tag from detected objects.
        
        Args:
            objects: List of detected objects
            
        Returns:
            Action tag string
        """
        # Simple heuristic-based action inference
        labels = [obj["label"].lower() for obj in objects]
        
        # Action mappings based on common objects
        action_mappings = {
            "person": "ACTION:person_activity",
            "car": "ACTION:driving",
            "bicycle": "ACTION:cycling",
            "dog": "ACTION:animal_activity",
            "cat": "ACTION:animal_activity",
            "food": "ACTION:eating",
            "book": "ACTION:reading",
            "laptop": "ACTION:computing",
            "phone": "ACTION:communication",
            "ball": "ACTION:sports",
            "guitar": "ACTION:music",
            "camera": "ACTION:photography"
        }
        
        # Find best matching action
        for label in labels:
            for key, action in action_mappings.items():
                if key in label:
                    return action
        
        # Default action
        return "ACTION:generic"
    
    def _enhance_caption_for_video(self, caption: str, action_tag: str) -> str:
        """Enhance image caption for video context.
        
        Args:
            caption: Original image caption
            action_tag: Inferred action tag
            
        Returns:
            Enhanced video caption
        """
        # Add video context words
        video_words = ["video", "footage", "clip", "recording"]
        
        # Check if caption already mentions video context
        caption_lower = caption.lower()
        has_video_context = any(word in caption_lower for word in video_words)
        
        if not has_video_context:
            # Prepend video context
            if caption.startswith("A "):
                caption = "A video showing " + caption[2:]
            elif caption.startswith("An "):
                caption = "A video showing " + caption[3:]
            else:
                caption = f"A video showing {caption.lower()}"
        
        return caption
    
    def is_available(self) -> bool:
        """Check if Florence-2 model is available."""
        try:
            from transformers import AutoProcessor
            AutoProcessor.from_pretrained(self.model_path, trust_remote_code=True)
            return True
        except Exception:
            return False
