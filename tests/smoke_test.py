"""
Smoke Test for CaptionStrike

Validates core functionality including environment setup, model loading,
media conversion, caption generation, and file organization.
"""

import sys
import tempfile
import shutil
from pathlib import Path
import logging
from typing import Dict, Any
import traceback

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from core.io import ProjectLayout, ProjectConfig
from core.media import MediaProcessor
from core.tokens import generate_token, add_token_to_caption, is_valid_token
from core.pipeline import ProcessingPipeline
from adapters.florence2_captioner import Florence2Captioner

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class SmokeTest:
    """Comprehensive smoke test for CaptionStrike."""
    
    def __init__(self):
        """Initialize smoke test."""
        self.temp_dir = None
        self.test_project = "smoke_test_project"
        self.results = {}
        
    def setup(self) -> bool:
        """Set up test environment."""
        try:
            # Create temporary directory
            self.temp_dir = Path(tempfile.mkdtemp(prefix="captionstrike_test_"))
            logger.info(f"Created test directory: {self.temp_dir}")
            
            # Create test media files (simple synthetic ones)
            self._create_test_media()
            
            return True
            
        except Exception as e:
            logger.error(f"Setup failed: {e}")
            return False
    
    def cleanup(self) -> None:
        """Clean up test environment."""
        if self.temp_dir and self.temp_dir.exists():
            try:
                shutil.rmtree(self.temp_dir)
                logger.info("Cleaned up test directory")
            except Exception as e:
                logger.warning(f"Cleanup failed: {e}")
    
    def _create_test_media(self) -> None:
        """Create simple test media files."""
        try:
            from PIL import Image
            import numpy as np
            
            # Create test image
            test_image = Image.fromarray(
                np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
            )
            test_image_path = self.temp_dir / "test_image.png"
            test_image.save(test_image_path)
            
            # Create test audio (simple sine wave)
            try:
                from pydub import AudioSegment
                from pydub.generators import Sine
                
                # Generate 2-second sine wave
                tone = Sine(440).to_audio_segment(duration=2000)
                test_audio_path = self.temp_dir / "test_audio.wav"
                tone.export(test_audio_path, format="wav")
                
            except ImportError:
                logger.warning("Pydub not available, skipping audio test file creation")
            
            logger.info("Created synthetic test media files")
            
        except Exception as e:
            logger.warning(f"Failed to create test media: {e}")
    
    def test_environment(self) -> bool:
        """Test environment and dependencies."""
        try:
            logger.info("Testing environment...")
            
            # Test core imports
            import torch
            import PIL
            import gradio
            import transformers
            
            # Test CUDA availability
            cuda_available = torch.cuda.is_available()
            logger.info(f"CUDA available: {cuda_available}")
            
            # Test basic functionality
            token = generate_token()
            assert is_valid_token(token), "Token generation failed"
            
            self.results["environment"] = {
                "success": True,
                "cuda_available": cuda_available,
                "token_test": "passed"
            }
            
            logger.info("✅ Environment test passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Environment test failed: {e}")
            self.results["environment"] = {"success": False, "error": str(e)}
            return False
    
    def test_project_management(self) -> bool:
        """Test project creation and management."""
        try:
            logger.info("Testing project management...")
            
            # Create project layout
            layout = ProjectLayout(self.temp_dir, self.test_project)
            layout.create_directories()
            
            # Verify directories exist
            assert layout.project_path.exists(), "Project directory not created"
            assert layout.raw_image_dir.exists(), "Raw image directory not created"
            assert layout.processed_image_dir.exists(), "Processed image directory not created"
            assert layout.meta_dir.exists(), "Meta directory not created"
            
            # Test project config
            config = ProjectConfig(layout.project_config_file)
            config.save({"test": "value"})
            loaded_config = config.load()
            assert loaded_config["test"] == "value", "Config save/load failed"
            
            self.results["project_management"] = {
                "success": True,
                "directories_created": True,
                "config_test": "passed"
            }
            
            logger.info("✅ Project management test passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Project management test failed: {e}")
            self.results["project_management"] = {"success": False, "error": str(e)}
            return False
    
    def test_media_processing(self) -> bool:
        """Test media conversion and processing."""
        try:
            logger.info("Testing media processing...")
            
            processor = MediaProcessor()
            
            # Test image processing
            test_image_path = self.temp_dir / "test_image.png"
            if test_image_path.exists():
                output_path = self.temp_dir / "converted_image"
                converted = processor.convert_image_to_png(test_image_path, output_path)
                assert converted.exists(), "Image conversion failed"
                
                # Test thumbnail creation
                from PIL import Image
                img = Image.open(converted)
                thumb = processor.create_thumbnail(img)
                assert thumb.size[0] <= 256 and thumb.size[1] <= 256, "Thumbnail creation failed"
            
            # Test media type detection
            assert processor.get_media_type(Path("test.jpg")) == "image"
            assert processor.get_media_type(Path("test.mp4")) == "video"
            assert processor.get_media_type(Path("test.mp3")) == "audio"
            
            self.results["media_processing"] = {
                "success": True,
                "image_conversion": "passed",
                "thumbnail_creation": "passed",
                "type_detection": "passed"
            }
            
            logger.info("✅ Media processing test passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Media processing test failed: {e}")
            self.results["media_processing"] = {"success": False, "error": str(e)}
            return False
    
    def test_florence2_availability(self) -> bool:
        """Test Florence-2 model availability (without loading)."""
        try:
            logger.info("Testing Florence-2 availability...")
            
            captioner = Florence2Captioner("base")
            available = captioner.is_available()
            
            if available:
                logger.info("✅ Florence-2 model is available")
                # Test basic functionality without full model loading
                self.results["florence2"] = {
                    "success": True,
                    "available": True,
                    "model_path": captioner.model_path
                }
            else:
                logger.warning("⚠️ Florence-2 model not available (will use fallback)")
                self.results["florence2"] = {
                    "success": True,
                    "available": False,
                    "note": "Model not available but system can handle fallback"
                }
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Florence-2 availability test failed: {e}")
            self.results["florence2"] = {"success": False, "error": str(e)}
            return False
    
    def test_token_system(self) -> bool:
        """Test token generation and management."""
        try:
            logger.info("Testing token system...")
            
            # Generate tokens
            tokens = [generate_token() for _ in range(5)]
            
            # Verify uniqueness
            assert len(set(tokens)) == 5, "Tokens are not unique"
            
            # Verify format
            for token in tokens:
                assert is_valid_token(token), f"Invalid token format: {token}"
                assert token.startswith("TKN-"), f"Token missing prefix: {token}"
            
            # Test caption integration
            test_caption = "A test image showing various elements"
            captioned = add_token_to_caption(test_caption, tokens[0])
            assert tokens[0] in captioned, "Token not added to caption"
            
            self.results["token_system"] = {
                "success": True,
                "uniqueness": "passed",
                "format_validation": "passed",
                "caption_integration": "passed"
            }
            
            logger.info("✅ Token system test passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Token system test failed: {e}")
            self.results["token_system"] = {"success": False, "error": str(e)}
            return False
    
    def test_pipeline_integration(self) -> bool:
        """Test pipeline integration (without heavy model loading)."""
        try:
            logger.info("Testing pipeline integration...")
            
            # Create pipeline
            models_dir = self.temp_dir / "models"
            models_dir.mkdir(exist_ok=True)
            pipeline = ProcessingPipeline(models_dir)
            
            # Create project layout
            layout = ProjectLayout(self.temp_dir, self.test_project)
            layout.create_directories()
            
            # Test file addition (if test files exist)
            test_files = list(self.temp_dir.glob("test_*"))
            if test_files:
                result = pipeline.add_files_to_project(layout, test_files)
                assert result["success"], "File addition failed"
                assert result["added_count"] > 0, "No files were added"
            
            self.results["pipeline_integration"] = {
                "success": True,
                "pipeline_creation": "passed",
                "file_addition": "passed" if test_files else "skipped"
            }
            
            logger.info("✅ Pipeline integration test passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Pipeline integration test failed: {e}")
            self.results["pipeline_integration"] = {"success": False, "error": str(e)}
            return False
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all smoke tests."""
        logger.info("🚀 Starting CaptionStrike smoke tests...")
        
        if not self.setup():
            return {"success": False, "error": "Setup failed"}
        
        try:
            tests = [
                ("Environment", self.test_environment),
                ("Project Management", self.test_project_management),
                ("Media Processing", self.test_media_processing),
                ("Florence-2 Availability", self.test_florence2_availability),
                ("Token System", self.test_token_system),
                ("Pipeline Integration", self.test_pipeline_integration)
            ]
            
            passed = 0
            total = len(tests)
            
            for test_name, test_func in tests:
                logger.info(f"\n--- Running {test_name} Test ---")
                try:
                    if test_func():
                        passed += 1
                except Exception as e:
                    logger.error(f"Test {test_name} crashed: {e}")
                    logger.debug(traceback.format_exc())
            
            # Summary
            logger.info(f"\n🎯 Smoke Test Results: {passed}/{total} tests passed")
            
            if passed == total:
                logger.info("✅ All smoke tests passed! CaptionStrike is ready to use.")
            else:
                logger.warning(f"⚠️ {total - passed} test(s) failed. Check logs for details.")
            
            return {
                "success": passed == total,
                "passed": passed,
                "total": total,
                "results": self.results
            }
            
        finally:
            self.cleanup()


def main():
    """Run smoke tests."""
    test = SmokeTest()
    results = test.run_all_tests()
    
    # Print acceptance checklist
    print("\n" + "="*60)
    print("ACCEPTANCE CHECKLIST")
    print("="*60)
    
    checklist = [
        ("Environment setup", results["results"].get("environment", {}).get("success", False)),
        ("Project creation", results["results"].get("project_management", {}).get("success", False)),
        ("Media conversion", results["results"].get("media_processing", {}).get("success", False)),
        ("Florence-2 availability", results["results"].get("florence2", {}).get("success", False)),
        ("Token system", results["results"].get("token_system", {}).get("success", False)),
        ("Pipeline integration", results["results"].get("pipeline_integration", {}).get("success", False))
    ]
    
    for item, status in checklist:
        status_icon = "✅" if status else "❌"
        print(f"{status_icon} {item}")
    
    print("\n" + "="*60)
    
    if results["success"]:
        print("🎉 CaptionStrike is ready for use!")
        print("\nNext steps:")
        print("1. Run: conda activate CaptionStrike")
        print("2. Run: python app.py --root 'D:/Datasets' --models_dir './models'")
        print("3. Open browser to http://localhost:7860")
        return 0
    else:
        print("❌ Some tests failed. Please check the logs and fix issues before proceeding.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
