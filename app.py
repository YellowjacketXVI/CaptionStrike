#!/usr/bin/env python3
"""
CaptionStrike - Local Dataset Builder

Main application launcher for the CaptionStrike dataset creation tool.
Provides a Gradio web interface for AI-powered media captioning and organization.

Usage:
    python app.py --root "D:/Datasets" --models_dir "./models"
    python app.py --root "/path/to/datasets" --models_dir "/path/to/models" --port 7860

Features:
    - Florence-2 powered image captioning and tagging
    - Optional Qwen2.5-VL reasoning enhancement
    - Audio speaker diarization and isolation
    - Person detection and isolation
    - Automatic format conversion (PNG/MP4/MP3)
    - ULID-based unique token system
    - Web-based project management interface
"""

import sys
import argparse
import logging
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.ui.app import CaptionStrikeUI
from src.adapters.qwen_vl_reasoner import download_qwen_model


def setup_logging(verbose: bool = False) -> None:
    """Set up logging configuration.
    
    Args:
        verbose: Enable verbose logging
    """
    level = logging.DEBUG if verbose else logging.INFO
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('captionstrike.log', mode='a')
        ]
    )
    
    # Reduce noise from some libraries
    logging.getLogger('transformers').setLevel(logging.WARNING)
    logging.getLogger('torch').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)


def validate_paths(root_dir: str, models_dir: str) -> tuple[Path, Path]:
    """Validate and create necessary directories.
    
    Args:
        root_dir: Root directory for datasets
        models_dir: Directory for model files
        
    Returns:
        Tuple of validated Path objects
        
    Raises:
        ValueError: If paths are invalid
    """
    root_path = Path(root_dir).resolve()
    models_path = Path(models_dir).resolve()
    
    # Create directories if they don't exist
    try:
        root_path.mkdir(parents=True, exist_ok=True)
        models_path.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        raise ValueError(f"Failed to create directories: {e}")
    
    # Verify write permissions
    if not root_path.is_dir() or not models_path.is_dir():
        raise ValueError("Specified paths are not directories")
    
    return root_path, models_path


def print_startup_info(root_dir: Path, models_dir: Path, port: int) -> None:
    """Print startup information and instructions.
    
    Args:
        root_dir: Root directory for datasets
        models_dir: Directory for model files
        port: Server port
    """
    print("\n" + "="*60)
    print("🎯 CaptionStrike - Local Dataset Builder")
    print("="*60)
    print(f"📁 Datasets root: {root_dir}")
    print(f"🤖 Models directory: {models_dir}")
    print(f"🌐 Server port: {port}")
    print("\n🚀 Starting web interface...")
    print(f"   Open your browser to: http://localhost:{port}")
    print("\n💡 Quick Start:")
    print("   1. Create a new project")
    print("   2. Drag & drop your media files")
    print("   3. Configure processing options")
    print("   4. Click 'RUN PIPELINE'")
    print("   5. Review and edit captions in the gallery")
    print("\n📚 Documentation: See README.md for detailed instructions")
    print("="*60 + "\n")


def check_dependencies() -> bool:
    """Check if required dependencies are available.
    
    Returns:
        True if all critical dependencies are available
    """
    missing_deps = []
    
    try:
        import torch
    except ImportError:
        missing_deps.append("torch")
    
    try:
        import transformers
    except ImportError:
        missing_deps.append("transformers")
    
    try:
        import gradio
    except ImportError:
        missing_deps.append("gradio")
    
    try:
        import PIL
    except ImportError:
        missing_deps.append("pillow")
    
    if missing_deps:
        print(f"❌ Missing required dependencies: {', '.join(missing_deps)}")
        print("Please install them using:")
        print("   conda env create -f environment.yml")
        print("   conda activate CaptionStrike")
        return False
    
    return True


def main() -> int:
    """Main application entry point.
    
    Returns:
        Exit code (0 for success, 1 for error)
    """
    parser = argparse.ArgumentParser(
        description="CaptionStrike - Local Dataset Builder",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python app.py --root "D:/Datasets" --models_dir "./models"
    python app.py --root "/home/user/datasets" --models_dir "/home/user/models" --port 8080
    python app.py --root "./data" --models_dir "./models" --verbose

For more information, see README.md
        """
    )
    
    parser.add_argument(
        "--root",
        type=str,
        required=True,
        help="Root directory for dataset projects (e.g., 'D:/Datasets')"
    )
    
    parser.add_argument(
        "--models_dir",
        type=str,
        default="./models",
        help="Directory for model files (default: './models')"
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port for web interface (default: 7860)"
    )
    
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host for web interface (default: 127.0.0.1)"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create public Gradio link (use with caution)"
    )

    parser.add_argument(
        "--prefetch-qwen",
        action="store_true",
        help="Download Qwen2.5-VL model files to --models_dir and exit",
    )
    
    args = parser.parse_args()
    
    # Set up logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    try:
        # Check dependencies
        if not check_dependencies():
            return 1
        
        # Validate paths
        root_dir, models_dir = validate_paths(args.root, args.models_dir)
        
        if args.prefetch_qwen:
            download_qwen_model("Qwen/Qwen2.5-VL-7B-Instruct", models_dir)
            return 0

        # Print startup info
        print_startup_info(root_dir, models_dir, args.port)

        # Initialize UI
        logger.info("Initializing CaptionStrike UI...")
        ui = CaptionStrikeUI(root_dir, models_dir)

        # Build interface
        logger.info("Building Gradio interface...")
        interface = ui.build_interface()

        # Launch application
        logger.info(f"Launching web interface on {args.host}:{args.port}")
        interface.launch(
            server_name=args.host,
            server_port=args.port,
            share=args.share,
            show_error=True,
            quiet=not args.verbose
        )

        return 0
        
    except KeyboardInterrupt:
        logger.info("Application stopped by user")
        return 0
        
    except Exception as e:
        logger.error(f"Application failed to start: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        
        print(f"\n❌ Failed to start CaptionStrike: {e}")
        print("\n🔧 Troubleshooting:")
        print("   1. Check that all dependencies are installed:")
        print("      conda env create -f environment.yml")
        print("      conda activate CaptionStrike")
        print("   2. Verify directory permissions for --root and --models_dir")
        print("   3. Run with --verbose for detailed error information")
        print("   4. Check the log file: captionstrike.log")
        
        return 1


if __name__ == "__main__":
    sys.exit(main())
