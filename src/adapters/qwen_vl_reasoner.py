"""
Qwen2.5-VL Reasoner Adapter for CaptionStrike

Provides enhanced reasoning and caption refinement using Qwen2.5-VL-7B-Instruct.
Used as an optional enhancement layer over Florence-2 base captions.
"""

import logging
import subprocess
import sys
from pathlib import Path
from typing import Dict, Any, Optional, Union
import warnings

import torch
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor

logger = logging.getLogger(__name__)

# Suppress some warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")


def download_qwen_model(model_name: str, cache_dir: Path) -> None:
    """Ensure Qwen model files are present in cache_dir."""
    try:
        Qwen2VLForConditionalGeneration.from_pretrained(
            model_name, cache_dir=cache_dir, local_files_only=True
        )
        AutoProcessor.from_pretrained(
            model_name, cache_dir=cache_dir, local_files_only=True
        )
        AutoTokenizer.from_pretrained(
            model_name, cache_dir=cache_dir, local_files_only=True
        )
        logger.info("Qwen model already cached")
    except Exception:
        logger.info("Downloading Qwen model files...")
        Qwen2VLForConditionalGeneration.from_pretrained(
            model_name, cache_dir=cache_dir
        )
        AutoProcessor.from_pretrained(model_name, cache_dir=cache_dir)
        AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        logger.info("Qwen model download complete")


class QwenVLReasoner:
    """Qwen2.5-VL model adapter for enhanced caption reasoning."""

    # Available model variants
    MODELS = {
        "7b": "Qwen/Qwen2.5-VL-7B-Instruct",
        "3b": "Qwen/Qwen2.5-VL-3B-Instruct",
        "2b": "Qwen/Qwen2.5-VL-2B-Instruct"
    }

    def __init__(
        self,
        model_name: str = "7b",
        device: Optional[str] = None,
        torch_dtype: Optional[torch.dtype] = None,
        cache_dir: Optional[Path] = None,
    ) -> None:
        """Initialize Qwen2.5-VL reasoner.

        Args:
            model_name: Model variant ('7b', '3b', '2b') or full model path
            device: Device to run model on (auto-detected if None)
            torch_dtype: Torch data type (auto-selected if None)
            cache_dir: Directory to cache model files
        """
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.torch_dtype = torch_dtype or (torch.float16 if torch.cuda.is_available() else torch.float32)
        self.cache_dir = Path(cache_dir) if cache_dir else None

        # Resolve model path
        if model_name in self.MODELS:
            self.model_path = self.MODELS[model_name]
        else:
            self.model_path = model_name

        self.model = None
        self.processor = None
        self.tokenizer = None
        self._process_vision_info = None
        self._loaded = False

        logger.info(f"Initialized Qwen2.5-VL reasoner with model: {self.model_path}")

    def load_model(self) -> None:
        """Load the Qwen2.5-VL model and processor."""
        if self._loaded:
            return

        try:
            # lazily import vision utils and install if missing
            if self._process_vision_info is None:
                try:
                    from qwen_vl_utils import process_vision_info
                except ImportError:
                    logger.warning("qwen_vl_utils not found, attempting install...")
                    subprocess.run([sys.executable, "-m", "pip", "install", "qwen-vl-utils"], check=False)
                    from qwen_vl_utils import process_vision_info  # type: ignore
                self._process_vision_info = process_vision_info

            # ensure model files downloaded
            if self.cache_dir:
                download_qwen_model(self.model_path, self.cache_dir)

            logger.info(f"Loading Qwen2.5-VL model: {self.model_path}")

            # Load model
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                self.model_path,
                torch_dtype=self.torch_dtype,
                attn_implementation="flash_attention_2" if torch.cuda.is_available() else "eager",
                device_map="auto" if torch.cuda.is_available() else None,
                cache_dir=self.cache_dir,
            )

            # Load processor and tokenizer
            self.processor = AutoProcessor.from_pretrained(self.model_path, cache_dir=self.cache_dir)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, cache_dir=self.cache_dir)

            if not torch.cuda.is_available():
                self.model = self.model.to(self.device)

            self._loaded = True
            logger.info(f"Successfully loaded Qwen2.5-VL model on {self.device}")

        except Exception as e:
            logger.error(f"Failed to load Qwen2.5-VL model: {e}")
            raise
    
    def refine_caption(self, 
                      original_caption: str, 
                      image: Union[Image.Image, Path, str],
                      context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Refine an existing caption using visual reasoning.
        
        Args:
            original_caption: Original caption from Florence-2
            image: PIL Image, file path, or path string
            context: Optional context information (objects, tags, etc.)
            
        Returns:
            Dict with refined caption and reasoning metadata
        """
        if not self._loaded:
            self.load_model()
        
        # Load image if path provided
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert('RGB')
        elif not isinstance(image, Image.Image):
            raise ValueError("Image must be PIL Image or file path")
        
        try:
            # Construct reasoning prompt
            prompt = self._build_refinement_prompt(original_caption, context)
            
            # Prepare messages for Qwen2.5-VL
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt}
                    ]
                }
            ]
            
            # Process inputs
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            image_inputs, video_inputs = self._process_vision_info(messages)
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt"
            )
            inputs = inputs.to(self.device)
            
            # Generate response
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9
                )
            
            # Decode response
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            
            refined_caption = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
            
            # Clean up the response
            refined_caption = self._clean_response(refined_caption)
            
            return {
                "refined_caption": refined_caption,
                "original_caption": original_caption,
                "improvement_detected": len(refined_caption) > len(original_caption) * 0.8,
                "reasoning_success": True
            }
            
        except Exception as e:
            logger.error(f"Failed to refine caption: {e}")
            return {
                "refined_caption": original_caption,  # Fallback to original
                "original_caption": original_caption,
                "improvement_detected": False,
                "reasoning_success": False,
                "error": str(e)
            }
    
    def analyze_image_detailed(self, image: Union[Image.Image, Path, str]) -> Dict[str, Any]:
        """Perform detailed image analysis with reasoning.
        
        Args:
            image: PIL Image, file path, or path string
            
        Returns:
            Dict with detailed analysis
        """
        if not self._loaded:
            self.load_model()
        
        # Load image if path provided
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert('RGB')
        
        try:
            prompt = """Analyze this image in detail. Provide:
1. A comprehensive one-sentence description focusing on the main subject, setting, lighting, and mood
2. Key objects and their relationships
3. Visual style and composition notes

Keep the description concise but informative, suitable for training data."""
            
            # Prepare messages
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt}
                    ]
                }
            ]
            
            # Process and generate
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            image_inputs, video_inputs = self._process_vision_info(messages)
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt"
            )
            inputs = inputs.to(self.device)
            
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9
                )
            
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            
            analysis = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
            
            # Extract main caption from analysis
            main_caption = self._extract_main_caption(analysis)
            
            return {
                "caption": main_caption,
                "detailed_analysis": analysis,
                "analysis_success": True
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze image: {e}")
            return {
                "caption": "A detailed image requiring further analysis",
                "detailed_analysis": "",
                "analysis_success": False,
                "error": str(e)
            }
    
    def _build_refinement_prompt(self, original_caption: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Build prompt for caption refinement.
        
        Args:
            original_caption: Original caption to refine
            context: Optional context information
            
        Returns:
            Refinement prompt string
        """
        prompt = f"""I have an initial caption for this image: "{original_caption}"

Please refine this caption to be more descriptive and accurate while keeping it as a single, concise sentence. Focus on:
- Main subject and their appearance/pose
- Setting and environment details
- Lighting conditions and mood
- Visual style and composition

"""
        
        if context and "objects" in context:
            objects = [obj.get("label", "") for obj in context["objects"]]
            if objects:
                prompt += f"Detected objects include: {', '.join(objects[:5])}\n"
        
        prompt += "Provide only the refined caption, nothing else."
        
        return prompt
    
    def _clean_response(self, response: str) -> str:
        """Clean up model response to extract just the caption.
        
        Args:
            response: Raw model response
            
        Returns:
            Cleaned caption string
        """
        # Remove common prefixes and suffixes
        response = response.strip()
        
        # Remove quotes if present
        if response.startswith('"') and response.endswith('"'):
            response = response[1:-1]
        
        # Remove "Caption:" prefix if present
        if response.lower().startswith("caption:"):
            response = response[8:].strip()
        
        # Take only the first sentence if multiple sentences
        sentences = response.split('. ')
        if len(sentences) > 1:
            response = sentences[0] + '.'
        
        return response.strip()
    
    def _extract_main_caption(self, analysis: str) -> str:
        """Extract main caption from detailed analysis.
        
        Args:
            analysis: Full analysis text
            
        Returns:
            Main caption string
        """
        # Look for numbered points and extract the first one
        lines = analysis.split('\n')
        
        for line in lines:
            line = line.strip()
            if line.startswith('1.') or line.startswith('1)'):
                caption = line[2:].strip()
                return self._clean_response(caption)
        
        # Fallback: take first sentence
        sentences = analysis.split('. ')
        if sentences:
            return self._clean_response(sentences[0] + '.')
        
        return analysis[:200] + "..." if len(analysis) > 200 else analysis
    
    def is_available(self) -> bool:
        """Check if Qwen2.5-VL model is available."""
        try:
            from transformers import AutoProcessor
            AutoProcessor.from_pretrained(self.model_path)
            return True
        except Exception:
            return False
