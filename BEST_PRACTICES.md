# 📘 Project Best Practices

## 1. Project Purpose
CaptionStrike is a local-first dataset builder that converts user-supplied media (images, videos, audio) into standardized formats and generates high-quality captions, tags, and summaries using vision-language models. It exposes a Gradio-based web UI for project management, media ingestion, processing, and inline caption editing. The domain centers on dataset creation and organization for ML training.

## 2. Project Structure
- Root
  - app.py — CLI entry point; launches Gradio UI, handles logging, argument parsing, dependency checks.
  - requirements.txt, environment.yml — Python and Conda dependencies.
  - run_captionstrike.ps1, launch_captionstrike.bat — convenience scripts.
  - validate_installation.py, test_environment.ps1 — validation utilities.
  - README.md — setup and usage documentation.
  - tests/ — pytest-based smoke tests.
  - sample_media/ — example media (if provided).
- src/
  - ui/app.py — Gradio Blocks UI (CaptionStrikeUI): project CRUD, file upload, pipeline execution, gallery, caption editor, settings.
  - core/
    - io.py — ProjectLayout (folder structure and file discovery), ProjectConfig (JSON config w/ nested key access), RunLogger (JSONL logs), caption file I/O.
    - media.py — MediaProcessor (ffmpeg/PIL/pydub for conversions, probing, frame extraction, thumbnails).
    - pipeline.py — ProcessingPipeline (orchestrates conversion, captioning, diarization, person isolation, reasoning, thumbnails, logging; lazy model load).
    - tokens.py — ULID token utilities; add to filenames and captions.
  - adapters/
    - florence2_captioner.py — Florence-2 adapter (captioning, OD, analysis, video first-frame caption).
    - qwen_vl_reasoner.py — Qwen2.5-VL adapter for primary captioning and optional reasoning/refinement; model prefetch.
    - diarizer.py — PyAnnote and optional Faster-Whisper-based diarization, speaker isolation, stitching, transcription.
    - person_isolator.py — InsightFace face detection; optional SAM segmentation for refined crops.

Conventions and separation of concerns:
- Adapters isolate external model-specific integration behind simple methods (adapter pattern) and lazy-load heavy models.
- Core modules are framework-agnostic infrastructure (pipeline, media ops, config, tokens, logging).
- UI triggers pipeline operations and persists user settings to per-project config (meta/project.json).
- Projects are isolated under <root>/<project> with raw/, processed/, meta/ subfolders; thumbnails under processed/thumbs/.

Entry points and configuration:
- CLI: python app.py --root <datasets_root> --models_dir <models> [--port] [--host] [--verbose] [--share] [--prefetch-qwen] [--system_prompt].
- Per-project config: meta/project.json (see ProjectConfig.DEFAULT_CONFIG for defaults and key paths).
- Logging: captionstrike.log (root) + per-run JSONL: meta/run_logs.jsonl.

## 3. Test Strategy
- Framework: pytest (tests/test_smoke.py).
- Philosophy: lightweight smoke coverage that avoids heavy downloads/inference; verify environment, conversions, tokens, and basic integration points.
- Structure:
  - tests/__init__.py for package discovery.
  - tests/test_smoke.py uses tmp_path fixtures, PIL image generation, optional pydub audio.
- Conventions and guidelines:
  - Name tests test_*.py and functions test_*.
  - Prefer unit tests for pure utility code (tokens, io, media conversions) with deterministic inputs.
  - Avoid actual large-model inference in CI; use feature flags/is_available checks and pytest.skip when heavyweight deps are missing.
  - For pipeline tests, limit to add_files_to_project and non-ML paths; gate model-dependent flows behind mocks or availability checks.
  - Use tmp_path for filesystem isolation. Do not read/write outside temp dirs or project structure.
  - If expanding, add integration tests that run a tiny end-to-end flow with minimal inputs and disabled heavy features; assert side effects (files created, captions written, logs appended).
  - Keep tests parallel-safe (no shared global state), especially if enabling max_workers.

## 4. Code Style
- Language: Python 3.10+.
- Typing: Prefer type hints for function signatures and key variables; return concrete Dict[str, Any] shapes consistently containing success, error, and payload fields.
- Naming:
  - snake_case for functions/variables/files; PascalCase for classes.
  - Descriptive names for adapters and pipeline methods; avoid abbreviations in public APIs.
- Imports and paths:
  - Use pathlib.Path for filesystem operations; avoid os.path joins.
  - Keep src on sys.path in entrypoints only (app.py). Prefer relative imports within src/.
- Error handling:
  - Catch exceptions at boundaries (UI events, pipeline steps, adapters); return dicts with success=False and error message; log with logger.
  - Never let heavy adapter errors crash the entire pipeline; fallback with reasonable defaults where possible.
- Logging:
  - Use module-level logger = logging.getLogger(__name__).
  - App config sets global logging, reduces noise from libraries.
  - Log warnings for optional-feature failures; info for normal operations; debug for verbose details.
- I/O and formats:
  - Standardize outputs: images .png, videos .mp4 (H.264/AAC, faststart), audio .mp3 (192k).
  - Thumbnails are .jpg (RGB) with quality and optimization.
- Prompts:
  - Build prompts using _build_agentic_prompt(base, system_prompt, context_diary, media_type); keep single-responsibility and reuse.
- Concurrency:
  - processing.max_workers config controls ThreadPoolExecutor; be thread-safe when logging or writing shared resources (use locks where needed).

## 5. Common Patterns
- Adapter pattern for model integrations (Florence, Qwen, diarization, person isolation) with lazy model loading and is_available checks.
- Orchestrator pattern in ProcessingPipeline: single entry to process files, choose adapters, route per-media-type logic, write outputs, append logs, generate thumbnails.
- Configuration access via nested dotted keys (ProjectConfig.get/set) merged with defaults.
- Token management using ULID: append to filenames via __TKN-*, and to captions as [TKN-...]; ensure idempotent via removal before append.
- Streaming UI updates using Python generators (yield) for progress logs in Gradio.
- Media utilities wrap ffmpeg, PIL, and pydub into safe conversions and frame extraction.

## 6. Do's and Don'ts
- ✅ Do
  - Validate and create project directories with ProjectLayout; never write outside project scope.
  - Use write_caption_file/read_caption_file for caption persistence.
  - Use safe_filename before constructing derived names; always append ULID tokens via tokens.add_token_* helpers.
  - Handle optional dependencies gracefully (InsightFace, SAM, Faster-Whisper, transformers models); check availability and degrade with warnings.
  - Centralize new settings in ProjectConfig.DEFAULT_CONFIG and access via dotted keys.
  - Keep heavy model loading lazy and cached on first use; reuse adapters.
  - Log structured run results to meta/run_logs.jsonl using RunLogger.
  - Keep UI actions pure-orchestrators; route actual processing to ProcessingPipeline.
  - Use Path for all file system paths and ensure parent dirs exist before writes.

- ❌ Don’t
  - Don’t hardcode absolute paths or OS-specific separators; avoid string path concatenation.
  - Don’t crash the pipeline for a single-file failure; record error and continue.
  - Don’t write large model downloads or temp files into raw/processed; store caches under --models_dir when possible.
  - Don’t run heavyweight inference in unit tests; avoid remote downloads in CI by default.
  - Don’t change processed file extensions or caption token format; downstream tooling expects current conventions.
  - Don’t bypass ProjectConfig when adding UI-controlled settings.

## 7. Tools & Dependencies
- Key libraries
  - gradio — Web UI and components.
  - torch, transformers — Model runtime and VLM backends (Qwen2.5-VL, Florence-2).
  - pillow (PIL) — Image manipulation.
  - ffmpeg-python — Video processing (conversion, frame extraction).
  - pydub — Audio conversion and stitching.
  - numpy — Array utilities.
  - ulid — ULID token generation.
  - pyannote.audio — Speaker diarization (optional, may require HF token for some models).
  - faster-whisper — Fast Whisper transcription (optional).
  - insightface — Face detection (optional); segment-anything (optional) for segmentation refinement.

- Setup
  - Create environment: conda env create -f environment.yml; conda activate CaptionStrike
  - Run app: python app.py --root <datasets_root> --models_dir <models> [--port 7860]
  - Optional: prefetch Qwen: python app.py --root <root> --models_dir <models> --prefetch-qwen
  - Verify: pytest (smoke tests), ffmpeg -version, torch CUDA checks

## 8. Other Notes
- Extending adapters
  - Add new adapter classes under src/adapters; follow existing method shapes (e.g., caption_image, generate_caption, refine_caption, is_available) and lazy-load models.
  - Wire selection via ProjectConfig (e.g., models.captioner) and update the UI dropdown in src/ui/app.py.
- UI wiring
  - Persist model and prompt settings via save_model_settings; reload on project change using load_model_settings; store long-form notes in meta/context.txt.
- Prompts and context
  - Use system_prompt and context diary to build agentic prompts via _build_agentic_prompt; keep media-type-specific base prompts in config.
- Filesystem invariants
  - Processed outputs live exclusively under processed/{image,video,audio}; thumbnails always under processed/thumbs with .jpg extensions.
- Logging & errors
  - All pipeline operations should log structured results and keep run_logs.jsonl append-only.
- Concurrency
  - When enabling processing.max_workers > 1, ensure writes to shared resources are protected (e.g., logger, RunLogger); avoid mutating shared adapters unless thread-safe.
- For LLM-generated code
  - Maintain return dict schema with success, error, and payload keys; preserve tokens in captions and filenames; avoid altering caption suffix format; maintain consistent naming and typing patterns.
