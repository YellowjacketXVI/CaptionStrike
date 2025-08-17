# CaptionStrike — Local Dataset Builder

A local-first application for creating training datasets using **Florence-2** for automatic image captioning, tagging, and dataset organization. Build high-quality datasets with drag-and-drop simplicity and AI-powered automation.

## ✨ Features

- **🎯 Florence-2 Integration**: Primary perception model for captioning, tagging, and grounding
- **🧠 Optional Reasoning**: Qwen2.5-VL-7B for enhanced caption refinement
- **🎵 Audio Processing**: Speaker diarization and isolation using pyannote.audio
- **👤 Person Isolation**: Face detection with InsightFace + optional SAM refinement
- **🖼️ Smart Conversion**: Auto-convert to standard formats (PNG/MP4/MP3)
- **🏷️ ULID Tokens**: Unique, sortable identifiers for all processed media
- **🌐 Web Interface**: Gradio-based UI with drag-drop and inline editing
- **📊 Progress Tracking**: Comprehensive logging and project management

## 🚀 Quick Start (Windows PowerShell)

### 1. Environment Setup

```powershell
# Navigate to CaptionStrike directory
cd D:\Dropbox\SandBox\CaptionStrike

# Create conda environment
conda env create -f environment.yml

# Activate environment
conda activate CaptionStirke
```

### 2. Launch Application

```powershell
# Start the local web interface
python app.py --root "D:\Datasets" --models_dir ".\models"

# Or specify custom paths
python app.py --root "C:\Your\Dataset\Path" --models_dir "C:\Your\Models\Path" --port 7860
```

### 3. Using the Interface

1. **Create Project**: Enter a project name and click "Create Project"
2. **Add Media**: Drag and drop images, videos, or audio files
3. **Configure Options**:
   - Toggle person isolation (face crops)
   - Provide reference voice clip for audio processing
   - Set audio timestamp ranges
4. **Run Pipeline**: Click "RUN pipeline" to process all media
5. **Review Results**: Browse thumbnails and edit captions inline
6. **Export**: Find processed files in `<root>\<project>\processed\`

### 4. Windows-Specific Setup Tips

```powershell
# If you encounter path issues, use full Windows paths:
python app.py --root "C:\Users\YourName\Documents\Datasets" --models_dir "C:\Users\YourName\Documents\Models"

# To check if conda environment is active:
conda info --envs

# To verify Python and dependencies:
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

## 📁 Project Structure

```
<root>\
└── <project_name>\
    ├── raw\                    # Original uploaded files
    │   ├── image\
    │   ├── video\
    │   └── audio\
    ├── processed\              # Converted & captioned files
    │   ├── image\              # PNG files with captions
    │   ├── video\              # MP4 files with action tags
    │   ├── audio\              # MP3 files with transcripts
    │   └── thumbs\             # 256px thumbnails for UI
    └── meta\
        ├── project.json        # Configuration & model settings
        └── run_logs.jsonl      # Processing history
```

## 🔧 Configuration

Edit `<project>\meta\project.json` to customize:

```json
{
  "models": {
    "captioner": "microsoft/Florence-2-base",
    "reasoning": {
      "enabled": false,
      "model": "Qwen/Qwen2.5-VL-7B-Instruct"
    }
  },
  "action": {
    "method": "first_frame",
    "rewrite_with_llm": true
  },
  "isolation": {
    "faces": true,
    "sam_refine": false
  }
}
```

## 🎯 Model Options

### Primary Captioning (Florence-2)
- `microsoft/Florence-2-base` (default, faster)
- `microsoft/Florence-2-large` (more detailed)

### Optional Reasoning Enhancement
- `Qwen/Qwen2.5-VL-7B-Instruct` (detailed analysis)
- Enable via `reasoning.enabled: true` in project config

### Single Model Alternative
- `openbmb/MiniCPM-V-2_6` (all-in-one option)
- Enable via `single_model_mode: true`

## 🛠️ System Requirements

### Minimum
- **OS**: Windows 10/11, Linux, macOS
- **RAM**: 8GB (16GB recommended)
- **Storage**: 10GB free space
- **Python**: 3.10+

### Recommended
- **GPU**: NVIDIA GPU with 6GB+ VRAM (CUDA support)
- **RAM**: 16GB+ for large models
- **Storage**: SSD for faster processing

### Dependencies
- PyTorch 2.2+
- Transformers 4.42+
- Gradio 4.44+
- FFmpeg (auto-installed via conda)

## 📋 File Format Support

### Input Formats
- **Images**: PNG, JPG, JPEG, WebP, BMP, TIFF, GIF
- **Videos**: MP4, MOV, MKV, AVI, WMV, FLV, WebM
- **Audio**: MP3, WAV, M4A, FLAC, AAC, OGG, WMA

### Output Formats
- **Images**: PNG (RGB, optimized)
- **Videos**: MP4 (H.264, AAC, faststart)
- **Audio**: MP3 (192kbps)

## 🔍 Processing Pipeline

1. **Media Ingestion**: Copy originals to `raw/` folders
2. **Format Conversion**: Convert to standard formats
3. **AI Analysis**:
   - Images: Florence-2 captioning + object detection
   - Videos: First-frame analysis + action tag inference
   - Audio: Speaker diarization + transcript generation
4. **Optional Enhancement**: Qwen2.5-VL reasoning refinement
5. **Token Assignment**: Append unique ULID tokens
6. **Thumbnail Generation**: Create 256px previews
7. **Logging**: Record all processing steps

## 🎨 Caption Format

All captions follow this format:
```
A detailed description of the subject, setting, lighting, and mood [TKN-01HQXYZ123ABC456DEF789]
```

Video captions include action tags:
```
A video showing a person walking in a park with natural lighting [ACTION:person_activity] [TKN-01HQXYZ123ABC456DEF789]
```

## 🧪 Testing

Run the smoke test to verify installation:

```powershell
python tests\smoke_test.py
```

This will test:
- ✅ Environment setup
- ✅ Model loading
- ✅ Media conversion
- ✅ Caption generation
- ✅ Token assignment
- ✅ File organization

## 🔧 Troubleshooting

### Model Download Issues
```powershell
# Pre-download models manually
python -c "from transformers import AutoProcessor; AutoProcessor.from_pretrained('microsoft/Florence-2-base', trust_remote_code=True)"
```

### CUDA/GPU Issues
```powershell
# Check CUDA availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### FFmpeg Issues
```powershell
# Verify FFmpeg installation
ffmpeg -version
```

### Windows Path Issues
```powershell
# If you get path errors, try using raw strings or forward slashes:
python app.py --root "D:/Datasets" --models_dir "./models"

# Or escape backslashes:
python app.py --root "D:\\Datasets" --models_dir ".\\models"
```

## 📚 Advanced Usage

### Batch Processing
Process multiple projects programmatically:

```python
from src.core.pipeline import Pipeline
from src.core.io import ProjectLayout

# Initialize pipeline
pipeline = Pipeline(models_dir=r".\models")

# Process project (use raw strings for Windows paths)
layout = ProjectLayout(r"D:\Datasets", "my_project")
pipeline.process_project(layout)
```

### Custom Model Integration
Add new model adapters in `src\adapters\`:

```python
class CustomCaptioner:
    def caption_image(self, image):
        # Your custom implementation
        return {"caption": "Custom caption"}
```

## 🤝 Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m 'Add amazing feature'`
4. Push to branch: `git push origin feature/amazing-feature`
5. Open Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Microsoft** for Florence-2 model
- **Alibaba** for Qwen2.5-VL model
- **PyAnnote** team for audio diarization
- **InsightFace** team for face detection
- **Gradio** team for the web interface framework

---

## Adjustments from AugmentInstructions.txt

This implementation enhances the original scaffold with:

- **Florence-2 Integration**: Replaced placeholder captioning with actual HuggingFace Florence-2 models
- **Modular Architecture**: Proper adapter pattern for different AI models  
- **Enhanced Configuration**: Comprehensive project.json with model selection options
- **Better Error Handling**: Graceful fallbacks when models aren't available
- **Comprehensive Testing**: Full smoke test suite and acceptance validation
- **Professional Documentation**: Complete setup guide and troubleshooting section

The core functionality remains true to the original vision while providing a production-ready implementation with proper error handling and extensibility.
