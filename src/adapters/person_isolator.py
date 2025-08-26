"""
Person Isolation Adapter for CaptionStrike

Provides face detection and person isolation using InsightFace and optional SAM.
Creates cropped images focused on detected persons for dataset creation.
"""

import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
import warnings

import cv2
import numpy as np
from PIL import Image
import torch

logger = logging.getLogger(__name__)

# InsightFace for face detection
try:
    from insightface.app import FaceAnalysis
    INSIGHTFACE_AVAILABLE = True
except ImportError:
    INSIGHTFACE_AVAILABLE = False
    logger.warning("InsightFace not available. Person isolation will be disabled.")

# Optional SAM for segmentation refinement
try:
    from segment_anything import sam_model_registry, SamPredictor
    SAM_AVAILABLE = True
except ImportError:
    SAM_AVAILABLE = False


class PersonIsolator:
    """Person isolation using face detection and optional segmentation."""
    
    def __init__(self, 
                 face_model: str = "buffalo_l",
                 sam_model_type: str = "vit_h",
                 sam_checkpoint: Optional[Path] = None,
                 device: Optional[str] = None):
        """Initialize person isolator.
        
        Args:
            face_model: InsightFace model name
            sam_model_type: SAM model type ('vit_h', 'vit_l', 'vit_b')
            sam_checkpoint: Path to SAM checkpoint file
            device: Device to run models on
        """
        self.face_model = face_model
        self.sam_model_type = sam_model_type
        self.sam_checkpoint = sam_checkpoint
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        self.face_app = None
        self.sam_predictor = None
        self._face_loaded = False
        self._sam_loaded = False
        
        logger.info(f"Initialized person isolator with face model: {face_model}")
    
    def load_face_model(self) -> None:
        """Load InsightFace model for face detection."""
        if self._face_loaded or not INSIGHTFACE_AVAILABLE:
            return
        
        try:
            logger.info(f"Loading InsightFace model: {self.face_model}")
            
            self.face_app = FaceAnalysis(name=self.face_model)
            self.face_app.prepare(ctx_id=0 if torch.cuda.is_available() else -1, det_size=(640, 640))
            
            self._face_loaded = True
            logger.info("Successfully loaded InsightFace model")
            
        except Exception as e:
            logger.error(f"Failed to load InsightFace model: {e}")
            raise
    
    def load_sam_model(self) -> None:
        """Load SAM model for segmentation refinement."""
        if self._sam_loaded or not SAM_AVAILABLE or self.sam_checkpoint is None:
            return
        
        try:
            logger.info(f"Loading SAM model: {self.sam_model_type}")
            
            sam = sam_model_registry[self.sam_model_type](checkpoint=str(self.sam_checkpoint))
            sam.to(device=self.device)
            
            self.sam_predictor = SamPredictor(sam)
            self._sam_loaded = True
            
            logger.info("Successfully loaded SAM model")
            
        except Exception as e:
            logger.error(f"Failed to load SAM model: {e}")
            # Don't raise - SAM is optional
    
    def detect_faces(self, image: Union[Image.Image, np.ndarray, Path, str]) -> List[Dict[str, Any]]:
        """Detect faces in an image.
        
        Args:
            image: PIL Image, numpy array, or file path
            
        Returns:
            List of face detection results
        """
        if not self._face_loaded:
            self.load_face_model()
        
        if not INSIGHTFACE_AVAILABLE:
            logger.warning("InsightFace not available, returning empty face list")
            return []
        
        # Convert to OpenCV format
        if isinstance(image, (str, Path)):
            cv_image = cv2.imread(str(image))
        elif isinstance(image, Image.Image):
            cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        elif isinstance(image, np.ndarray):
            cv_image = image.copy()
        else:
            raise ValueError("Unsupported image format")
        
        try:
            faces = self.face_app.get(cv_image)
            
            face_results = []
            for i, face in enumerate(faces):
                # Extract face information
                bbox = face.bbox.astype(int)
                landmarks = face.kps.astype(int) if hasattr(face, 'kps') else None
                
                face_data = {
                    "face_id": i,
                    "bbox": bbox.tolist(),  # [x1, y1, x2, y2]
                    "confidence": float(face.det_score) if hasattr(face, 'det_score') else 1.0,
                    "landmarks": landmarks.tolist() if landmarks is not None else None,
                    "age": int(face.age) if hasattr(face, 'age') else None,
                    "gender": face.sex if hasattr(face, 'sex') else None
                }
                
                face_results.append(face_data)
            
            logger.info(f"Detected {len(face_results)} faces")
            return face_results
            
        except Exception as e:
            logger.error(f"Failed to detect faces: {e}")
            return []
    
    def crop_faces(
        self,
        image: Union[Image.Image, np.ndarray, Path, str],
        output_dir: Path,
        base_name: str,
        padding_ratio: float = 0.3,
        faces: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Path]:
        """Crop detected faces from image.
        
        Args:
            image: Source image
            output_dir: Directory to save crops
            base_name: Base name for crop files
            padding_ratio: Padding around face bbox (0.3 = 30% padding)
            
        Returns:
            List of paths to saved face crops
        """
        # Load image
        if isinstance(image, (str, Path)):
            pil_image = Image.open(image).convert('RGB')
            image_path = Path(image)
        elif isinstance(image, Image.Image):
            pil_image = image.convert('RGB')
            image_path = None
        else:
            # Convert numpy array to PIL
            if isinstance(image, np.ndarray):
                pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            else:
                raise ValueError("Unsupported image format")
            image_path = None
        
        # Detect faces if not provided
        if faces is None:
            faces = self.detect_faces(pil_image)

        if not faces:
            logger.info("No faces detected for cropping")
            return []
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        crop_paths = []
        img_width, img_height = pil_image.size
        
        for face in faces:
            try:
                # Get face bounding box
                x1, y1, x2, y2 = face["bbox"]
                
                # Add padding
                face_width = x2 - x1
                face_height = y2 - y1
                padding_x = int(face_width * padding_ratio)
                padding_y = int(face_height * padding_ratio)
                
                # Calculate crop bounds with padding
                crop_x1 = max(0, x1 - padding_x)
                crop_y1 = max(0, y1 - padding_y)
                crop_x2 = min(img_width, x2 + padding_x)
                crop_y2 = min(img_height, y2 + padding_y)
                
                # Crop face
                face_crop = pil_image.crop((crop_x1, crop_y1, crop_x2, crop_y2))
                
                # Save crop
                crop_filename = f"{base_name}__face_{face['face_id']:02d}.png"
                crop_path = output_dir / crop_filename
                face_crop.save(crop_path, 'PNG', optimize=True)
                
                crop_paths.append(crop_path)
                logger.info(f"Saved face crop: {crop_path}")
                
            except Exception as e:
                logger.error(f"Failed to crop face {face['face_id']}: {e}")
                continue
        
        return crop_paths
    
    def segment_person(self, 
                      image: Union[Image.Image, np.ndarray, Path, str],
                      face_bbox: List[int]) -> Optional[np.ndarray]:
        """Segment person using SAM based on face detection.
        
        Args:
            image: Source image
            face_bbox: Face bounding box [x1, y1, x2, y2]
            
        Returns:
            Binary mask array or None if SAM not available
        """
        if not self._sam_loaded:
            self.load_sam_model()
        
        if not SAM_AVAILABLE or self.sam_predictor is None:
            logger.warning("SAM not available for person segmentation")
            return None
        
        try:
            # Convert to OpenCV format
            if isinstance(image, (str, Path)):
                cv_image = cv2.imread(str(image))
            elif isinstance(image, Image.Image):
                cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            elif isinstance(image, np.ndarray):
                cv_image = image.copy()
            else:
                raise ValueError("Unsupported image format")
            
            # Set image for SAM
            self.sam_predictor.set_image(cv_image)
            
            # Use face center as prompt point
            x1, y1, x2, y2 = face_bbox
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            
            # Generate mask
            masks, scores, logits = self.sam_predictor.predict(
                point_coords=np.array([[center_x, center_y]]),
                point_labels=np.array([1]),
                multimask_output=True
            )
            
            # Choose best mask
            best_mask = masks[np.argmax(scores)]
            
            return best_mask
            
        except Exception as e:
            logger.error(f"Failed to segment person: {e}")
            return None
    
    def isolate_persons(self,
                       image: Union[Image.Image, Path, str],
                       output_dir: Path,
                       base_name: str,
                       use_sam: bool = False,
                       save_original: bool = True) -> Dict[str, Any]:
        """Complete person isolation pipeline.
        
        Args:
            image: Source image
            output_dir: Directory for output files
            base_name: Base name for output files
            use_sam: Whether to use SAM for segmentation refinement
            save_original: Whether to save original crops alongside segmented versions
            
        Returns:
            Dict with isolation results
        """
        try:
            # Load image
            if isinstance(image, (str, Path)):
                pil_image = Image.open(image).convert('RGB')
            else:
                pil_image = image.convert('RGB')
            
            # Detect faces
            faces = self.detect_faces(pil_image)
            
            if not faces:
                return {
                    "success": False,
                    "face_count": 0,
                    "crop_paths": [],
                    "message": "No faces detected"
                }
            
            # Create crops directory
            crops_dir = output_dir / "crops"
            crops_dir.mkdir(parents=True, exist_ok=True)
            
            # Crop faces
            crop_paths = self.crop_faces(pil_image, crops_dir, base_name, faces=faces)
            
            # Optional SAM segmentation
            segmented_paths = []
            if use_sam and SAM_AVAILABLE:
                for i, face in enumerate(faces):
                    mask = self.segment_person(pil_image, face["bbox"])
                    if mask is not None:
                        # Apply mask to create segmented version
                        img_array = np.array(pil_image)
                        segmented = img_array.copy()
                        segmented[~mask] = [255, 255, 255]  # White background
                        
                        # Crop to face region with padding
                        x1, y1, x2, y2 = face["bbox"]
                        padding = 50
                        crop_x1 = max(0, x1 - padding)
                        crop_y1 = max(0, y1 - padding)
                        crop_x2 = min(img_array.shape[1], x2 + padding)
                        crop_y2 = min(img_array.shape[0], y2 + padding)
                        
                        segmented_crop = segmented[crop_y1:crop_y2, crop_x1:crop_x2]
                        segmented_pil = Image.fromarray(segmented_crop)
                        
                        # Save segmented crop
                        seg_filename = f"{base_name}__face_{i:02d}_segmented.png"
                        seg_path = crops_dir / seg_filename
                        segmented_pil.save(seg_path, 'PNG', optimize=True)
                        segmented_paths.append(seg_path)
            
            return {
                "success": True,
                "face_count": len(faces),
                "crop_paths": crop_paths,
                "segmented_paths": segmented_paths,
                "faces_data": faces,
                "message": f"Successfully isolated {len(faces)} person(s)"
            }
            
        except Exception as e:
            logger.error(f"Failed to isolate persons: {e}")
            return {
                "success": False,
                "face_count": 0,
                "crop_paths": [],
                "segmented_paths": [],
                "faces_data": [],
                "message": f"Error: {str(e)}"
            }
    
    def is_available(self) -> bool:
        """Check if person isolation is available."""
        return INSIGHTFACE_AVAILABLE
    
    def sam_available(self) -> bool:
        """Check if SAM segmentation is available."""
        return SAM_AVAILABLE and self.sam_checkpoint is not None
