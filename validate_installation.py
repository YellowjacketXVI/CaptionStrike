#!/usr/bin/env python3
"""
CaptionStrike Installation Validator

Quick validation script to check if CaptionStrike is properly installed
and ready to use. Run this before starting the main application.
"""

import sys
import importlib
from pathlib import Path


def check_python_version():
    """Check Python version compatibility."""
    version = sys.version_info
    if version.major != 3 or version.minor < 10:
        print(f"❌ Python {version.major}.{version.minor} detected. Python 3.10+ required.")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} - OK")
    return True


def check_dependencies():
    """Check required dependencies."""
    required_packages = [
        ("torch", "PyTorch"),
        ("torchvision", "TorchVision"),
        ("transformers", "Transformers"),
        ("gradio", "Gradio"),
        ("PIL", "Pillow"),
        ("numpy", "NumPy"),
        ("cv2", "OpenCV"),
        ("pydub", "PyDub"),
        ("ulid", "ULID"),
        ("tqdm", "TQDM")
    ]
    
    optional_packages = [
        ("pyannote.audio", "PyAnnote Audio"),
        ("faster_whisper", "Faster Whisper"),
        ("insightface", "InsightFace")
    ]
    
    print("\n📦 Checking required dependencies...")
    missing_required = []
    
    for package, name in required_packages:
        try:
            importlib.import_module(package)
            print(f"✅ {name} - OK")
        except ImportError:
            print(f"❌ {name} - MISSING")
            missing_required.append(name)
    
    print("\n📦 Checking optional dependencies...")
    missing_optional = []
    
    for package, name in optional_packages:
        try:
            importlib.import_module(package)
            print(f"✅ {name} - OK")
        except ImportError:
            print(f"⚠️ {name} - MISSING (optional)")
            missing_optional.append(name)
    
    if missing_required:
        print(f"\n❌ Missing required packages: {', '.join(missing_required)}")
        print("Install with: conda env create -f environment.yml")
        return False
    
    if missing_optional:
        print(f"\n⚠️ Missing optional packages: {', '.join(missing_optional)}")
        print("Some features may be limited.")
    
    return True


def check_cuda():
    """Check CUDA availability."""
    try:
        import torch
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0) if device_count > 0 else "Unknown"
            print(f"✅ CUDA available - {device_count} device(s), Primary: {device_name}")
            return True
        else:
            print("⚠️ CUDA not available - will use CPU (slower)")
            return True
    except Exception as e:
        print(f"❌ CUDA check failed: {e}")
        return False


def check_file_structure():
    """Check project file structure."""
    required_dirs = [
        "src",
        "src/core",
        "src/adapters",
        "src/ui",
        "tests"
    ]
    
    required_files = [
        "app.py",
        "environment.yml",
        "README.md",
        "src/core/pipeline.py",
        "src/adapters/florence2_captioner.py",
        "src/ui/app.py",
        "tests/test_smoke.py"
    ]
    
    print("\n📁 Checking file structure...")
    
    missing_dirs = []
    for dir_path in required_dirs:
        if not Path(dir_path).exists():
            missing_dirs.append(dir_path)
        else:
            print(f"✅ {dir_path}/ - OK")
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
        else:
            print(f"✅ {file_path} - OK")
    
    if missing_dirs or missing_files:
        print(f"\n❌ Missing directories: {missing_dirs}")
        print(f"❌ Missing files: {missing_files}")
        return False
    
    return True


def check_models_access():
    """Check if we can access model repositories."""
    try:
        from transformers import AutoProcessor
        
        print("\n🤖 Checking model access...")
        
        # Test Florence-2 access (without downloading)
        try:
            # This will check if the model exists without downloading
            model_info = AutoProcessor.from_pretrained(
                "microsoft/Florence-2-base", 
                trust_remote_code=True,
                use_fast=False,
                local_files_only=False
            )
            print("✅ Florence-2 model accessible")
        except Exception as e:
            print(f"⚠️ Florence-2 model access issue: {e}")
            print("   Model will be downloaded on first use")
        
        return True
        
    except Exception as e:
        print(f"❌ Model access check failed: {e}")
        return False


def run_basic_functionality_test():
    """Run basic functionality test."""
    try:
        print("\n🧪 Running basic functionality test...")
        
        # Add src to path
        sys.path.insert(0, str(Path(__file__).parent / "src"))
        
        # Test token generation
        from src.core.tokens import generate_token, is_valid_token
        token = generate_token()
        assert is_valid_token(token), "Token validation failed"
        print("✅ Token generation - OK")
        
        # Test media processor
        from src.core.media import MediaProcessor
        processor = MediaProcessor()
        assert processor.get_media_type(Path("test.jpg")) == "image"
        print("✅ Media type detection - OK")
        
        # Test project layout
        from src.core.io import ProjectLayout
        import tempfile
        with tempfile.TemporaryDirectory() as temp_dir:
            layout = ProjectLayout(Path(temp_dir), "test_project")
            layout.create_directories()
            assert layout.project_path.exists()
        print("✅ Project management - OK")
        
        return True
        
    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        return False


def main():
    """Run all validation checks."""
    print("🎯 CaptionStrike Installation Validator")
    print("=" * 50)
    
    checks = [
        ("Python Version", check_python_version),
        ("Dependencies", check_dependencies),
        ("CUDA Support", check_cuda),
        ("File Structure", check_file_structure),
        ("Model Access", check_models_access),
        ("Basic Functionality", run_basic_functionality_test)
    ]
    
    passed = 0
    total = len(checks)
    
    for check_name, check_func in checks:
        print(f"\n🔍 {check_name}...")
        try:
            if check_func():
                passed += 1
            else:
                print(f"❌ {check_name} check failed")
        except Exception as e:
            print(f"❌ {check_name} check crashed: {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 Validation Results: {passed}/{total} checks passed")
    
    if passed == total:
        print("\n🎉 CaptionStrike is ready to use!")
        print("\n🚀 Next steps:")
        print("   1. Run: python app.py --root 'D:/Datasets' --models_dir './models'")
        print("   2. Open browser to: http://localhost:7860")
        print("   3. Create a project and start processing media!")
        print("\n💡 For detailed testing, run: pytest")
        return 0
    else:
        print(f"\n❌ {total - passed} validation check(s) failed")
        print("\n🔧 Troubleshooting:")
        print("   1. Ensure conda environment is activated:")
        print("      conda activate CaptionStrike")
        print("   2. Reinstall dependencies if needed:")
        print("      conda env create -f environment.yml --force")
        print("   3. Check internet connection for model downloads")
        print("   4. Verify file permissions in project directory")
        return 1


if __name__ == "__main__":
    sys.exit(main())
