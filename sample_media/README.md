# Sample Media for CaptionStrike Testing

This directory contains sample media files for testing CaptionStrike functionality.

## Files Included

### Images
- `sample_image.jpg` - A test image for caption generation and person isolation testing
- `test_photo.png` - Another test image with different characteristics

### Videos  
- `sample_video.mp4` - A short test video for first-frame analysis and action tagging
- `test_clip.mov` - Additional video sample for format conversion testing

### Audio
- `sample_audio.wav` - Test audio file for diarization and speaker isolation
- `voice_sample.mp3` - Reference voice clip for audio processing

## Usage

1. **Manual Testing**: Drag and drop these files into CaptionStrike projects to test functionality
2. **Automated Testing**: The smoke test (`tests/smoke_test.py`) uses these files automatically
3. **Development**: Use these as reference examples when developing new features

## Creating Your Own Test Media

### Images
- Use diverse subjects, lighting conditions, and compositions
- Include images with multiple people for person isolation testing
- Vary image formats and sizes

### Videos
- Keep test videos short (5-30 seconds) for faster processing
- Include clear actions and subjects for action tagging
- Test different video formats and codecs

### Audio
- Include multi-speaker conversations for diarization testing
- Provide clear reference clips for speaker identification
- Test various audio formats and quality levels

## Notes

- All sample files should be small in size for quick testing
- Files are used for functional testing, not quality assessment
- Replace with your own content as needed for specific use cases
