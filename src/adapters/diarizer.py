"""
Audio Diarization Adapter for CaptionStrike

Provides speaker diarization and audio stitching using pyannote.audio.
Isolates single speakers from multi-speaker audio for dataset creation.
"""

import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
import warnings

import torch
import numpy as np
from pydub import AudioSegment
from pyannote.audio import Pipeline
from pyannote.core import Annotation, Segment
import faster_whisper

logger = logging.getLogger(__name__)

# Suppress some warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning, module="pyannote")


class AudioDiarizer:
    """Audio diarization and speaker isolation using pyannote.audio."""
    
    def __init__(self, 
                 diarization_model: str = "pyannote/speaker-diarization-3.1",
                 whisper_model: str = "base",
                 device: Optional[str] = None):
        """Initialize audio diarizer.
        
        Args:
            diarization_model: Pyannote diarization model name
            whisper_model: Faster-whisper model size ('tiny', 'base', 'small', 'medium', 'large')
            device: Device to run models on (auto-detected if None)
        """
        self.diarization_model = diarization_model
        self.whisper_model = whisper_model
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        self.pipeline = None
        self.whisper = None
        self._loaded = False
        
        logger.info(f"Initialized audio diarizer with model: {diarization_model}")
    
    def load_models(self) -> None:
        """Load diarization and transcription models."""
        if self._loaded:
            return
        
        try:
            logger.info("Loading pyannote diarization pipeline...")
            
            # Load diarization pipeline
            self.pipeline = Pipeline.from_pretrained(
                self.diarization_model,
                use_auth_token=None  # Set HF token if needed for some models
            )
            
            # Load Whisper for transcription
            logger.info(f"Loading Whisper model: {self.whisper_model}")
            self.whisper = faster_whisper.WhisperModel(
                self.whisper_model,
                device=self.device,
                compute_type="float16" if torch.cuda.is_available() else "float32"
            )
            
            self._loaded = True
            logger.info("Successfully loaded audio processing models")
            
        except Exception as e:
            logger.error(f"Failed to load audio models: {e}")
            raise
    
    def diarize_audio(self, audio_path: Path) -> Annotation:
        """Perform speaker diarization on audio file.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Pyannote Annotation object with speaker segments
        """
        if not self._loaded:
            self.load_models()
        
        try:
            logger.info(f"Diarizing audio: {audio_path}")
            diarization = self.pipeline(str(audio_path))
            return diarization
            
        except Exception as e:
            logger.error(f"Failed to diarize audio {audio_path}: {e}")
            raise
    
    def extract_speaker_segments(self, 
                                audio_path: Path,
                                target_speaker: Optional[str] = None,
                                reference_clip: Optional[Path] = None,
                                reference_window: Optional[Tuple[float, float]] = None) -> Dict[str, Any]:
        """Extract segments for a specific speaker.
        
        Args:
            audio_path: Path to audio file
            target_speaker: Specific speaker label to extract (if known)
            reference_clip: Path to reference audio clip for speaker identification
            reference_window: (start_time, end_time) window for speaker identification
            
        Returns:
            Dict with speaker segments and metadata
        """
        if not self._loaded:
            self.load_models()
        
        try:
            # Perform diarization
            diarization = self.diarize_audio(audio_path)
            
            # Load full audio
            full_audio = AudioSegment.from_file(audio_path)
            
            # Determine target speaker
            if target_speaker is None:
                target_speaker = self._identify_target_speaker(
                    diarization, reference_clip, reference_window
                )
            
            # Extract segments for target speaker
            segments = []
            total_duration = 0.0
            
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                if speaker == target_speaker:
                    start_ms = int(turn.start * 1000)
                    end_ms = int(turn.end * 1000)
                    duration = turn.end - turn.start
                    
                    segments.append({
                        "start": turn.start,
                        "end": turn.end,
                        "duration": duration,
                        "start_ms": start_ms,
                        "end_ms": end_ms,
                        "speaker": speaker
                    })
                    
                    total_duration += duration
            
            return {
                "target_speaker": target_speaker,
                "segments": segments,
                "segment_count": len(segments),
                "total_duration": total_duration,
                "full_duration": len(full_audio) / 1000.0,
                "coverage_ratio": total_duration / (len(full_audio) / 1000.0)
            }
            
        except Exception as e:
            logger.error(f"Failed to extract speaker segments: {e}")
            raise
    
    def stitch_speaker_audio(self, 
                           audio_path: Path,
                           segments: List[Dict[str, Any]],
                           output_path: Path,
                           crossfade_ms: int = 50) -> Path:
        """Stitch together audio segments for a single speaker.
        
        Args:
            audio_path: Path to source audio file
            segments: List of segment dictionaries
            output_path: Path for output stitched audio
            crossfade_ms: Crossfade duration between segments
            
        Returns:
            Path to stitched audio file
        """
        try:
            # Load source audio
            source_audio = AudioSegment.from_file(audio_path)
            
            # Extract and stitch segments
            stitched = AudioSegment.silent(duration=0)
            
            for i, segment in enumerate(segments):
                # Extract segment
                segment_audio = source_audio[segment["start_ms"]:segment["end_ms"]]
                
                # Add crossfade between segments (except first)
                if i > 0 and crossfade_ms > 0:
                    stitched = stitched.append(segment_audio, crossfade=crossfade_ms)
                else:
                    stitched += segment_audio
            
            # Export stitched audio
            output_path.parent.mkdir(parents=True, exist_ok=True)
            stitched.export(output_path, format="mp3", bitrate="192k")
            
            logger.info(f"Stitched {len(segments)} segments to {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Failed to stitch audio segments: {e}")
            raise
    
    def transcribe_audio(self, audio_path: Path, language: Optional[str] = None) -> Dict[str, Any]:
        """Transcribe audio using Whisper.
        
        Args:
            audio_path: Path to audio file
            language: Optional language code (auto-detected if None)
            
        Returns:
            Dict with transcription and metadata
        """
        if not self._loaded:
            self.load_models()
        
        try:
            logger.info(f"Transcribing audio: {audio_path}")
            
            segments, info = self.whisper.transcribe(
                str(audio_path),
                language=language,
                beam_size=5,
                best_of=5,
                temperature=0.0
            )
            
            # Collect transcription
            full_text = ""
            segment_list = []
            
            for segment in segments:
                full_text += segment.text + " "
                segment_list.append({
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment.text.strip()
                })
            
            return {
                "text": full_text.strip(),
                "segments": segment_list,
                "language": info.language,
                "language_probability": info.language_probability,
                "duration": info.duration
            }
            
        except Exception as e:
            logger.error(f"Failed to transcribe audio {audio_path}: {e}")
            return {
                "text": "",
                "segments": [],
                "language": "unknown",
                "language_probability": 0.0,
                "duration": 0.0,
                "error": str(e)
            }
    
    def process_audio_complete(self,
                             audio_path: Path,
                             output_dir: Path,
                             base_name: str,
                             reference_clip: Optional[Path] = None,
                             reference_window: Optional[Tuple[float, float]] = None) -> Dict[str, Any]:
        """Complete audio processing pipeline: diarize, extract, stitch, transcribe.
        
        Args:
            audio_path: Path to source audio file
            output_dir: Directory for output files
            base_name: Base name for output files
            reference_clip: Optional reference clip for speaker identification
            reference_window: Optional time window for speaker identification
            
        Returns:
            Dict with processing results and file paths
        """
        try:
            # Extract speaker segments
            speaker_data = self.extract_speaker_segments(
                audio_path, None, reference_clip, reference_window
            )
            
            # Stitch speaker audio
            stitched_path = output_dir / f"{base_name}__single_speaker.mp3"
            self.stitch_speaker_audio(
                audio_path, speaker_data["segments"], stitched_path
            )
            
            # Transcribe stitched audio
            transcription = self.transcribe_audio(stitched_path)
            
            # Create summary text
            summary_lines = [
                f"Single-speaker audio extracted from {audio_path.name}",
                f"Target speaker: {speaker_data['target_speaker']}",
                f"Segments used: {speaker_data['segment_count']}",
                f"Total duration: {speaker_data['total_duration']:.2f}s",
                f"Coverage: {speaker_data['coverage_ratio']:.1%}",
                "",
                "Transcript:",
                transcription["text"]
            ]
            
            summary_text = "\n".join(summary_lines)
            
            return {
                "stitched_audio_path": stitched_path,
                "summary_text": summary_text,
                "speaker_data": speaker_data,
                "transcription": transcription,
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Failed to process audio completely: {e}")
            return {
                "stitched_audio_path": None,
                "summary_text": f"Failed to process audio: {str(e)}",
                "speaker_data": {},
                "transcription": {},
                "success": False,
                "error": str(e)
            }
    
    def _identify_target_speaker(self,
                               diarization: Annotation,
                               reference_clip: Optional[Path] = None,
                               reference_window: Optional[Tuple[float, float]] = None) -> str:
        """Identify target speaker from diarization.
        
        Args:
            diarization: Pyannote diarization annotation
            reference_clip: Optional reference audio clip
            reference_window: Optional time window for identification
            
        Returns:
            Speaker label string
        """
        # Strategy 1: Use reference window if provided
        if reference_window is not None:
            start_time, end_time = reference_window
            window_segment = Segment(start_time, end_time)
            
            # Find speaker with most overlap in reference window
            best_speaker = None
            best_overlap = 0.0
            
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                overlap_duration = max(0.0, min(window_segment.end, turn.end) - max(window_segment.start, turn.start))
                if overlap_duration > best_overlap:
                    best_overlap = overlap_duration
                    best_speaker = speaker
            
            if best_speaker is not None:
                return best_speaker
        
        # Strategy 2: Choose speaker with longest total duration
        speaker_durations = {}
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            duration = turn.end - turn.start
            speaker_durations[speaker] = speaker_durations.get(speaker, 0.0) + duration
        
        if speaker_durations:
            return max(speaker_durations, key=speaker_durations.get)
        
        # Fallback
        return "SPEAKER_00"
    
    def is_available(self) -> bool:
        """Check if audio processing models are available."""
        try:
            from pyannote.audio import Pipeline
            Pipeline.from_pretrained(self.diarization_model)
            return True
        except Exception:
            return False
