import pytest
from pathlib import Path
from PIL import Image
import numpy as np
import sys

# Add repository root to path for module imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.io import ProjectLayout, ProjectConfig
from src.core.media import MediaProcessor
from src.core.tokens import generate_token, add_token_to_caption, is_valid_token
from src.adapters.florence2_captioner import Florence2Captioner


@pytest.fixture
def temp_dir(tmp_path):
    return tmp_path


@pytest.fixture
def test_media(temp_dir):
    image = Image.fromarray(np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8))
    image_path = temp_dir / "test_image.png"
    image.save(image_path)
    audio_path = None
    try:
        from pydub import AudioSegment
        from pydub.generators import Sine

        tone = Sine(440).to_audio_segment(duration=2000)
        audio_path = temp_dir / "test_audio.wav"
        tone.export(audio_path, format="wav")
    except Exception:
        pass
    return {"image": image_path, "audio": audio_path}


def test_environment():
    import torch, PIL, gradio, transformers  # noqa: F401

    token = generate_token()
    assert is_valid_token(token)


def test_project_management(temp_dir):
    layout = ProjectLayout(temp_dir, "smoke_test_project")
    layout.create_directories()
    assert layout.project_path.exists()
    config = ProjectConfig(layout.project_config_file)
    config.save({"test": "value"})
    assert config.load()["test"] == "value"


def test_media_processing(temp_dir, test_media):
    processor = MediaProcessor()
    converted = processor.convert_image_to_png(test_media["image"], temp_dir / "converted_image")
    assert converted.exists()
    with Image.open(converted) as img:
        thumb = processor.create_thumbnail(img)
    assert max(thumb.size) <= 256
    assert processor.get_media_type(Path("test.jpg")) == "image"
    assert processor.get_media_type(Path("test.mp4")) == "video"
    assert processor.get_media_type(Path("test.mp3")) == "audio"


def test_florence2_availability():
    captioner = Florence2Captioner("base")
    assert isinstance(captioner.is_available(), bool)


def test_token_system():
    tokens = [generate_token() for _ in range(5)]
    assert len(set(tokens)) == 5
    for token in tokens:
        assert is_valid_token(token)
        assert token.startswith("TKN-")
    captioned = add_token_to_caption("A test image", tokens[0])
    assert tokens[0] in captioned


def test_pipeline_integration(temp_dir, test_media):
    try:
        from src.core.pipeline import ProcessingPipeline
    except Exception:
        pytest.skip("ProcessingPipeline dependencies not available")

    pipeline = ProcessingPipeline(temp_dir / "models")
    layout = ProjectLayout(temp_dir, "smoke_test_project")
    layout.create_directories()
    test_files = [p for p in [test_media["image"], test_media["audio"]] if p and p.exists()]
    if test_files:
        result = pipeline.add_files_to_project(layout, test_files)
        assert result["success"]
        assert result["added_count"] == len(test_files)
