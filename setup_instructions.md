# 🚀 CaptionStrike Setup Instructions (Windows PowerShell)

## Step 1: Create Conda Environment

```powershell
# Navigate to CaptionStrike directory
cd D:\Dropbox\SandBox\CaptionStrike

# Create conda environment
conda env create -f environment.yml

# Activate environment
conda activate CaptionStrike
```

## Step 2: Install Python Dependencies

```powershell
# Install core requirements
pip install -r requirements.txt

# Optional: Install enhanced performance packages (if you have compatible GPU)
# pip install xformers flash-attn bitsandbytes

# Optional: Install Qwen2.5-VL support for reasoning
# pip install qwen-vl-utils tiktoken
```

## Step 3: Run CaptionStrike

### Option A: Using PowerShell Script (Recommended)

```powershell
# Basic usage
.\run_captionstrike.ps1

# Custom paths
.\run_captionstrike.ps1 -Root "C:\MyDatasets" -ModelsDir "C:\MyModels" -Port 8080

# Debug mode
.\run_captionstrike.ps1 -Debug

# Show acceptance checklist
.\run_captionstrike.ps1 -Check
```

### Option B: Direct Python Command

```powershell
# Basic usage
python app.py --root "D:\Datasets" --models_dir ".\models"

# Custom configuration
python app.py --root "C:\Your\Dataset\Path" --models_dir "C:\Your\Models\Path" --port 7860 --debug
```

## Step 4: Verify Installation

```powershell
# Run test suite
pytest

# Check acceptance criteria
python app.py --check
```

## 🔧 Troubleshooting

### Common Issues:

1. **"ModuleNotFoundError"**
   ```powershell
   # Make sure environment is activated
   conda activate CaptionStrike
   
   # Reinstall requirements
   pip install -r requirements.txt
   ```

2. **"CUDA not available"**
   ```powershell
   # Check CUDA installation
   python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
   
   # Install CUDA-compatible PyTorch if needed
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

3. **"FFmpeg not found"**
   ```powershell
   # FFmpeg should be installed via conda, but if issues persist:
   conda install ffmpeg -c conda-forge
   ```

4. **Permission errors**
   ```powershell
   # Run PowerShell as Administrator if needed
   # Or change execution policy:
   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
   ```

## 📋 Quick Start Workflow

1. **Create Project**: Enter project name → "Create Project"
2. **Add Media**: Drag & drop images/videos/audio files
3. **Configure**: Toggle person isolation, set audio options
4. **Process**: Click "🚀 RUN Pipeline"
5. **Review**: Browse gallery, edit captions inline
6. **Export**: Find results in `<root>\<project>\processed\`

## 🎯 Expected Output Structure

```
D:\Datasets\
└── MyProject\
    ├── raw\
    │   ├── image\      # Original images
    │   ├── video\      # Original videos
    │   └── audio\      # Original audio
    ├── processed\
    │   ├── image\      # PNG files + captions
    │   ├── video\      # MP4 files + captions
    │   ├── audio\      # MP3 files + transcripts
    │   └── thumbs\     # Gallery thumbnails
    └── meta\
        ├── project.json    # Configuration
        └── run_logs.jsonl  # Processing history
```

## 🏷️ File Naming Convention

All processed files get unique ULID tokens:
- `original_name__TKN-01HQXYZ123ABC456DEF789.png`
- `original_name__TKN-01HQXYZ123ABC456DEF789.txt`

All captions end with: `[TKN-01HQXYZ123ABC456DEF789]`

## 🤖 AI Models Used

- **Primary**: Florence-2 (Microsoft) - Image captioning & object detection
- **Optional**: Qwen2.5-VL-7B (Alibaba) - Enhanced reasoning
- **Audio**: pyannote.audio - Speaker diarization
- **Faces**: InsightFace - Person detection & isolation

## 📞 Getting Help

If you encounter issues:

1. Run with debug: `.\run_captionstrike.ps1 -Debug`
2. Check logs: `captionstrike.log`
3. Verify environment: `conda list`
4. Test components: `pytest`
