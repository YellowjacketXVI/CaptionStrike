"""
Gradio UI for CaptionStrike

Provides a web-based interface for project management, file upload,
processing control, and dataset review with inline caption editing.
"""

import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import json
from datetime import datetime

import gradio as gr
from PIL import Image

from ..core.io import ProjectLayout, ProjectConfig, read_caption_file, write_caption_file
from ..core.pipeline import ProcessingPipeline

logger = logging.getLogger(__name__)


class CaptionStrikeUI:
    """Main UI class for CaptionStrike application."""

    def __init__(self, root_dir: Path, models_dir: Path):
        """Initialize CaptionStrike UI.

        Args:
            root_dir: Root directory for projects
            models_dir: Directory containing model files
        """
        self.root_dir = Path(root_dir)
        self.models_dir = Path(models_dir)
        self.pipeline = ProcessingPipeline(models_dir)

        # Ensure directories exist
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Initialized CaptionStrike UI - Root: {root_dir}, Models: {models_dir}")

    def list_projects(self) -> List[str]:
        """Get list of existing projects."""
        try:
            projects = []
            for item in self.root_dir.iterdir():
                if item.is_dir():
                    layout = ProjectLayout(self.root_dir, item.name)
                    if layout.exists():
                        projects.append(item.name)
            return sorted(projects)
        except Exception as e:
            logger.error(f"Failed to list projects: {e}")
            return []

    def create_project(self, project_name: str) -> Tuple[gr.Dropdown, str]:
        """Create a new project.

        Args:
            project_name: Name for the new project

        Returns:
            Tuple of (updated dropdown, status message)
        """
        try:
            if not project_name or not project_name.strip():
                return gr.Dropdown(), "Please enter a project name"

            project_name = project_name.strip()

            # Check if project already exists
            layout = ProjectLayout(self.root_dir, project_name)
            if layout.exists():
                return gr.Dropdown(), f"Project '{project_name}' already exists"

            # Create project structure
            layout.create_directories()

            # Create default configuration
            config = ProjectConfig(layout.project_config_file)
            default_config = config.DEFAULT_CONFIG.copy()
            default_config.update({
                "name": project_name,
                "created": datetime.now().isoformat()
            })
            config.save(default_config)

            # Update dropdown
            projects = self.list_projects()

            return (
                gr.Dropdown(choices=projects, value=project_name),
                f"✅ Created project '{project_name}'"
            )

        except Exception as e:
            logger.error(f"Failed to create project: {e}")
            return gr.Dropdown(), f"❌ Failed to create project: {str(e)}"

    def add_files_to_project(self, project_name: str, files: List[str]) -> str:
        """Add files to a project.

        Args:
            project_name: Name of the project
            files: List of file paths to add

        Returns:
            Status message
        """
        try:
            if not project_name:
                return "❌ Please select a project first"

            if not files:
                return "❌ No files provided"

            layout = ProjectLayout(self.root_dir, project_name)
            if not layout.exists():
                return f"❌ Project '{project_name}' does not exist"

            # Convert file paths to Path objects
            file_paths = [Path(f) for f in files]

            # Add files using pipeline
            result = self.pipeline.add_files_to_project(layout, file_paths)

            if result["success"]:
                message = f"✅ Added {result['added_count']} file(s) to project"
                if result["errors"]:
                    message += f"\n⚠️ {len(result['errors'])} error(s):\n" + "\n".join(result["errors"][:3])
                return message
            else:
                return f"❌ Failed to add files: {result['errors'][0] if result['errors'] else 'Unknown error'}"

        except Exception as e:
            logger.error(f"Failed to add files: {e}")
            return f"❌ Error adding files: {str(e)}"

    def run_processing(self,
                      project_name: str,
                      use_person_isolation: bool,
                      reference_voice_clip: str,
                      first_sound_ts: Optional[float],
                      end_sound_ts: Optional[float],
                      force_reprocess: bool = False,
                      system_prompt: str = ""):
        """Run the processing pipeline on a project with streaming updates."""
        try:
            if not project_name:
                yield "❌ Please select a project first"
                return

            layout = ProjectLayout(self.root_dir, project_name)
            if not layout.exists():
                yield f"❌ Project '{project_name}' does not exist"
                return

            # Update project configuration
            config = ProjectConfig(layout.project_config_file)
            config.load()
            config.set("isolation.faces", use_person_isolation)
            if system_prompt is not None:
                config.set("captioning.system_prompt", system_prompt.strip())
            config.save()

            # Prepare audio processing parameters
            ref_clip = Path(reference_voice_clip) if reference_voice_clip and reference_voice_clip.strip() else None

            # Get files to process for progress tracking
            raw_files = layout.get_raw_files()
            total_files = len(raw_files)

            if total_files == 0:
                yield "❌ No files to process"
                return

            yield f"🚀 Starting processing for project '{project_name}'\n📁 Found {total_files} files to process..."

            # Run processing pipeline with streaming updates
            logger.info(f"Starting processing for project '{project_name}'")

            processed_count = 0
            errors = []

            for i, raw_file in enumerate(raw_files, 1):
                try:
                    yield f"🔄 Processing {i}/{total_files}: {raw_file.name}..."

                    # Process single file (simplified version of pipeline logic)
                    result = self.pipeline._process_single_file(
                        raw_file, layout, config, None,
                        ref_clip, first_sound_ts, end_sound_ts, force_reprocess
                    )

                    if result["success"]:
                        processed_count += 1
                        yield f"✅ Completed {i}/{total_files}: {raw_file.name}"
                    else:
                        error_msg = f"{raw_file.name}: {result.get('error', 'Unknown error')}"
                        errors.append(error_msg)
                        yield f"❌ Failed {i}/{total_files}: {error_msg}"

                except Exception as e:
                    error_msg = f"{raw_file.name}: {str(e)}"
                    errors.append(error_msg)
                    yield f"❌ Error {i}/{total_files}: {error_msg}"

            # Final summary
            if processed_count > 0:
                message = f"✅ Processing complete! Successfully processed {processed_count}/{total_files} files."
                if errors:
                    message += f"\n⚠️ {len(errors)} error(s) occurred:\n" + "\n".join(errors[:3])
                    if len(errors) > 3:
                        message += f"\n... and {len(errors) - 3} more errors"
                yield message
            else:
                yield f"❌ Processing failed: No files were successfully processed"

        except Exception as e:
            logger.error(f"Processing failed: {e}")
            yield f"❌ Processing error: {str(e)}"

    def load_project_gallery(self, project_name: str, page: int = 1, items_per_page: int = 20) -> Tuple[gr.Gallery, str, int, int]:
        """Load project gallery with pagination.

        Args:
            project_name: Name of the project
            page: Current page number (1-based)
            items_per_page: Number of items per page

        Returns:
            Tuple of (gallery component, status message, current_page, total_pages)
        """
        try:
            if not project_name:
                return gr.Gallery(value=[]), "Please select a project", 1, 1

            layout = ProjectLayout(self.root_dir, project_name)
            if not layout.exists():
                return gr.Gallery(value=[]), f"Project '{project_name}' does not exist", 1, 1

            # Get thumbnails
            thumbnails = layout.get_thumbnails()

            if not thumbnails:
                # If no thumbnails, try to show raw images
                raw_images = layout.get_raw_files("image")
                all_items = [(str(img_path), img_path.stem) for img_path in raw_images]
                status_prefix = "raw images (run processing to generate thumbnails)"
            else:
                # Create gallery items from thumbnails
                all_items = [(str(thumb_path), thumb_path.stem) for thumb_path in sorted(thumbnails)]
                status_prefix = "processed items"

            if not all_items:
                return gr.Gallery(value=[]), "No images found in project", 1, 1

            # Calculate pagination
            total_items = len(all_items)
            total_pages = max(1, (total_items + items_per_page - 1) // items_per_page)
            page = max(1, min(page, total_pages))  # Clamp page to valid range

            # Get items for current page
            start_idx = (page - 1) * items_per_page
            end_idx = min(start_idx + items_per_page, total_items)
            page_items = all_items[start_idx:end_idx]

            status = f"Page {page}/{total_pages} - Showing {len(page_items)} of {total_items} {status_prefix}"

            return (
                gr.Gallery(value=page_items),
                status,
                page,
                total_pages
            )

        except Exception as e:
            logger.error(f"Failed to load gallery: {e}")
            return gr.Gallery(value=[]), f"Error loading gallery: {str(e)}", 1, 1

    def navigate_gallery(self, project_name: str, current_page: int, direction: str) -> Tuple[gr.Gallery, str, int, int]:
        """Navigate gallery pages."""
        if direction == "prev":
            new_page = max(1, current_page - 1)
        elif direction == "next":
            new_page = current_page + 1
        else:
            new_page = current_page

        return self.load_project_gallery(project_name, new_page)

    def load_caption_for_editing(self, project_name: str, selected_image: str) -> str:
        """Load caption for the selected image.

        Args:
            project_name: Name of the project
            selected_image: Path to selected image

        Returns:
            Caption text for editing
        """
        try:
            if not project_name or not selected_image:
                return ""

            layout = ProjectLayout(self.root_dir, project_name)

            # Find corresponding caption file
            selected_path = Path(selected_image)

            # Look for caption file in processed directories
            for media_type in ["image", "video", "audio"]:
                processed_dir = getattr(layout, f"processed_{media_type}_dir")
                for processed_file in processed_dir.glob(f"{selected_path.stem}.*"):
                    caption_file = processed_file.with_suffix('.txt')
                    if caption_file.exists():
                        return read_caption_file(caption_file)

            return "Caption not found"

        except Exception as e:
            logger.error(f"Failed to load caption: {e}")
            return f"Error loading caption: {str(e)}"

    def save_edited_caption(self, project_name: str, selected_image: str, new_caption: str) -> str:
        """Save edited caption.

        Args:
            project_name: Name of the project
            selected_image: Path to selected image
            new_caption: New caption text

        Returns:
            Status message
        """
        try:
            if not project_name or not selected_image:
                return "❌ No project or image selected"

            layout = ProjectLayout(self.root_dir, project_name)
            selected_path = Path(selected_image)

            # Find corresponding caption file
            for media_type in ["image", "video", "audio"]:
                processed_dir = getattr(layout, f"processed_{media_type}_dir")
                for processed_file in processed_dir.glob(f"{selected_path.stem}.*"):
                    caption_file = processed_file.with_suffix('.txt')
                    if caption_file.exists():
                        write_caption_file(caption_file, new_caption)
                        return "✅ Caption saved successfully"

            return "❌ Caption file not found"

        except Exception as e:
            logger.error(f"Failed to save caption: {e}")
            return f"❌ Error saving caption: {str(e)}"

    def get_project_stats(self, project_name: str) -> str:
        """Get project statistics.

        Args:
            project_name: Name of the project

        Returns:
            Statistics string
        """
        try:
            if not project_name:
                return "No project selected"

            layout = ProjectLayout(self.root_dir, project_name)
            if not layout.exists():
                return "Project does not exist"

            # Count files
            raw_files = layout.get_raw_files()
            processed_files = layout.get_processed_files()
            thumbnails = layout.get_thumbnails()

            raw_by_type = {
                "image": len(layout.get_raw_files("image")),
                "video": len(layout.get_raw_files("video")),
                "audio": len(layout.get_raw_files("audio"))
            }

            processed_by_type = {
                "image": len(layout.get_processed_files("image")),
                "video": len(layout.get_processed_files("video")),
                "audio": len(layout.get_processed_files("audio"))
            }

            stats = [
                f"📁 Project: {project_name}",
                f"📥 Raw files: {len(raw_files)} (🖼️{raw_by_type['image']} 🎬{raw_by_type['video']} 🎵{raw_by_type['audio']})",
                f"✅ Processed: {len(processed_files)} (🖼️{processed_by_type['image']} 🎬{processed_by_type['video']} 🎵{processed_by_type['audio']})",
                f"🖼️ Thumbnails: {len(thumbnails)}"
            ]

            return "\n".join(stats)

        except Exception as e:
            logger.error(f"Failed to get project stats: {e}")
            return f"Error getting stats: {str(e)}"


        def load_context_diary(self, project_name: str) -> str:
            """Load context/diary text from meta/context.txt for the project."""
            try:
                if not project_name:
                    return ""
                layout = ProjectLayout(self.root_dir, project_name)
                context_file = layout.meta_dir / "context.txt"
                if context_file.exists():
                    text = context_file.read_text(encoding="utf-8")
                    logger.debug(f"Loaded context diary for project '{project_name}', {len(text)} chars")
                    return text
                return ""
            except Exception as e:
                logger.error(f"Failed to load context diary: {e}")
                return ""

        def save_context_diary(self, project_name: str, context_text: str) -> str:
            """Save context/diary text to meta/context.txt for the project."""
            try:
                if not project_name:
                    return "❌ Please select a project first"
                layout = ProjectLayout(self.root_dir, project_name)
                layout.meta_dir.mkdir(parents=True, exist_ok=True)
                context_file = layout.meta_dir / "context.txt"
                (layout.meta_dir / "context.txt").write_text(context_text or "", encoding="utf-8")
                logger.info(f"Saved context diary for project '{project_name}' to {context_file}")
                return "✅ Context/Diary saved"
            except Exception as e:
                logger.error(f"Failed to save context diary: {e}")
                return f"❌ Error saving context: {str(e)}"

        def get_file_counts(self, project_name: Optional[str]) -> str:
            """Return counts of raw files by type for a project as a Markdown summary."""
            try:
                if not project_name:
                    return "No project selected"
                layout = ProjectLayout(self.root_dir, project_name)
                if not layout.exists():
                    return f"Project '{project_name}' does not exist"
                img = len(layout.get_raw_files("image"))
                vid = len(layout.get_raw_files("video"))
                aud = len(layout.get_raw_files("audio"))
                total = img + vid + aud
                msg = f"Raw files ready: {total} (🖼️ {img} / 🎬 {vid} / 🎵 {aud})"
                logger.debug(f"File counts for '{project_name}': {msg}")
                return msg
            except Exception as e:
                logger.error(f"Failed to compute file counts: {e}")
                return "Unable to compute file counts"

        def is_ready_to_run(self, project_name: Optional[str]) -> bool:
            """Project is ready to run if selected and has at least one raw file."""
            try:
                if not project_name:
                    return False
                layout = ProjectLayout(self.root_dir, project_name)
                if not layout.exists():
                    return False
                total = len(layout.get_raw_files("image")) + len(layout.get_raw_files("video")) + len(layout.get_raw_files("audio"))
                ready = total > 0
                logger.debug(f"Run readiness for '{project_name}': {ready} (total raw={total})")
                return ready
            except Exception as e:
                logger.error(f"Failed to compute run readiness: {e}")
                return False

        def compute_run_button_state(self, project_name: Optional[str]) -> gr.Update:
            """Return a Gradio update to enable/disable the Run button."""
            return gr.update(interactive=self.is_ready_to_run(project_name))

        def load_model_settings(self, project_name: str) -> Tuple[str, bool, str, str, str, str, str, str]:
            """Load model settings and prompts for a project.
            Returns: (captioner, reasoning_enabled, reasoning_model, system_prompt, context_diary, image_prompt, video_prompt, audio_prompt)
            """
            try:
                if not project_name:
                    return ("", False, "", "", "", "", "", "")
                layout = ProjectLayout(self.root_dir, project_name)
                config = ProjectConfig(layout.project_config_file)
                config.load()
                captioner = config.get("models.captioner", "Qwen/Qwen2.5-VL-7B-Instruct")
                reasoning_enabled = bool(config.get("models.reasoning.enabled", False))
                reasoning_model = config.get("models.reasoning.model", "Qwen/Qwen2.5-VL-7B-Instruct")
                system_prompt = config.get("captioning.system_prompt", "")
                context_diary = self.load_context_diary(project_name)
                image_prompt = config.get("captioning.image_prompt", "")
                video_prompt = config.get("captioning.video_prompt", "")
                audio_prompt = config.get("captioning.audio_prompt", "")
                return (captioner, reasoning_enabled, reasoning_model, system_prompt, context_diary, image_prompt, video_prompt, audio_prompt)
            except Exception as e:
                logger.error(f"Failed to load model settings: {e}")
                return ("", False, "", "", "", "", "", "")

        def save_model_settings(self,
                                project_name: str,
                                captioner: str,
                                reasoning_enabled: bool,
                                reasoning_model: str,
                                system_prompt: str,
                                image_prompt: str,
                                video_prompt: str,
                                audio_prompt: str) -> str:
            """Persist model selections and all prompts to project.json."""
            try:
                if not project_name:
                    return "❌ Please select a project first"
                layout = ProjectLayout(self.root_dir, project_name)
                config = ProjectConfig(layout.project_config_file)
                config.load()
                config.set("models.captioner", captioner)
                config.set("models.reasoning.enabled", bool(reasoning_enabled))
                config.set("models.reasoning.model", reasoning_model)
                config.set("captioning.system_prompt", (system_prompt or "").strip())
                config.set("captioning.image_prompt", (image_prompt or "").strip())
                config.set("captioning.video_prompt", (video_prompt or "").strip())
                config.set("captioning.audio_prompt", (audio_prompt or "").strip())
                config.save()
                logger.info(f"Saved model settings and agentic prompts for '{project_name}': captioner={captioner}")
                return "✅ Model settings and prompts saved"
            except Exception as e:
                logger.error(f"Failed to save model settings: {e}")
                return f"❌ Error saving model settings: {str(e)}"

        def get_run_logs_path(self, project_name: str) -> Tuple[str, str]:
            """Return path to run_logs.jsonl for download (as two outputs)."""
            try:
                if not project_name:
                    return ("", "")
                layout = ProjectLayout(self.root_dir, project_name)
                path = str(layout.run_logs_file)
                return (path, path)
            except Exception as e:
                logger.error(f"Failed to resolve run logs path: {e}")
                return ("", "")

        def get_error_summary(self, project_name: str) -> str:
            """Get a summary of recent errors from run logs."""
            try:
                if not project_name:
                    return "No project selected"
                layout = ProjectLayout(self.root_dir, project_name)
                if not layout.run_logs_file.exists():
                    return "No processing logs found"

                # Read recent log entries
                with open(layout.run_logs_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()

                # Get last 10 entries and check for errors
                recent_lines = lines[-10:] if len(lines) > 10 else lines
                errors = []

                for line in recent_lines:
                    try:
                        entry = json.loads(line.strip())
                        if not entry.get("success", True) and "error" in entry:
                            errors.append(f"• {entry.get('source', 'Unknown')}: {entry['error']}")
                    except json.JSONDecodeError:
                        continue

                if errors:
                    return f"Recent errors ({len(errors)}):\n" + "\n".join(errors[-5:])
                else:
                    return "✅ No recent errors found"

            except Exception as e:
                logger.error(f"Failed to get error summary: {e}")
                return f"Error reading logs: {str(e)}"

    def build_interface(self) -> gr.Blocks:
        """Build the Gradio interface.

        Returns:
            Gradio Blocks interface
        """
        with gr.Blocks(
            title="CaptionStrike - Local Dataset Builder",
            theme=gr.themes.Soft(),
            css="""
            .project-stats {
                font-family: monospace;
                background: #f8f9fa;
                padding: 10px;
                border-radius: 5px;
            }
            """
        ) as interface:

            with gr.Tabs():
                with gr.TabItem("Setup"):
                    gr.Markdown("""
                    # 🎯 CaptionStrike — Local Dataset Builder

                    Create high-quality training datasets with AI-powered captioning using **Florence-2** (default) and optional **Qwen2.5-VL** reasoning.
                    """)


            with gr.Row():
                with gr.Column(scale=1):
                    # Project Management
                    gr.Markdown("## 📁 Project Management")

                    project_dropdown = gr.Dropdown(
                        choices=self.list_projects(),
                        label="Select Project",
                        interactive=True,
                        value=None
                    )

                    with gr.Row():
                        new_project_name = gr.Textbox(
                            label="New Project Name",
                            placeholder="Enter project name...",
                            scale=3
                        )
                        create_btn = gr.Button("Create Project", scale=1, variant="primary")

                    project_stats = gr.Markdown(
                        "No project selected",
                        elem_classes=["project-stats"]
                    )

                    # File Upload
                    gr.Markdown("## 📤 Add Files")

                    file_upload = gr.Files(
                        file_count="multiple",
                        file_types=["image", "video", "audio", ".zip"],
                        label="Drop files here or click to browse"
                    )

                    add_files_btn = gr.Button("Add to Project", variant="secondary")
                    add_status = gr.Textbox(label="Status", interactive=False)

                    # Wizard Step Summary
                    counts_box = gr.Markdown("", elem_classes=["project-stats"])

                with gr.Column(scale=2):
                    # Processing Options (Essential)
                    gr.Markdown("## ⚙️ Processing Options")

                    with gr.Row():
                        use_isolation = gr.Checkbox(
                            label="🧑 Person Isolation (face crops)",
                            value=False
                        )
                        force_reprocess = gr.Checkbox(
                            label="🔄 Force Reprocess",
                            value=False
                        )

                    with gr.Accordion("Advanced Settings", open=False):
                        with gr.TabItem("Captioning"):
                            system_prompt = gr.Textbox(
                                label="System prompt (optional)",
                                placeholder="Provide a system prompt to guide captioning...",
                                lines=2
                            )

                            # Agentic prompts per media type
                            gr.Markdown("### 🎯 Agentic Prompts per Media Type")
                            image_prompt = gr.Textbox(
                                label="Image Prompt",
                                placeholder="Describe the subject, setting, lighting, and mood...",
                                lines=2
                            )
                            video_prompt = gr.Textbox(
                                label="Video Prompt",
                                placeholder="Describe the action and context shown in this video frame...",
                                lines=2
                            )
                            audio_prompt = gr.Textbox(
                                label="Audio Prompt",
                                placeholder="Summarize the key points and context from this conversation...",
                                lines=2
                            )

                            # Context/Diary
                            context_diary = gr.Textbox(
                                label="Project Context / Diary",
                                placeholder="Notes, goals, constraints. Will be saved to meta/context.txt and appended to prompts.",
                                lines=4
                            )
                            with gr.Row():
                                save_context_btn = gr.Button("Save Context", variant="secondary")
                                context_save_status = gr.Textbox(label="Context Status", interactive=False)

                        with gr.TabItem("Models"):
                            captioner_model = gr.Dropdown(
                                label="Captioner Model",
                                choices=[
                                    "Qwen/Qwen2.5-VL-7B-Instruct",
                                    "Qwen/Qwen2.5-VL-3B-Instruct",
                                    "Qwen/Qwen2.5-VL-2B-Instruct",
                                    "microsoft/Florence-2-base",
                                    "microsoft/Florence-2-large"
                                ],
                                value="Qwen/Qwen2.5-VL-7B-Instruct"
                            )
                            reasoning_enabled = gr.Checkbox(
                                label="Enable Qwen Reasoning",
                                value=False
                            )
                            reasoning_model = gr.Dropdown(
                                label="Qwen Model",
                                choices=[
                                    "Qwen/Qwen2.5-VL-7B-Instruct",
                                    "Qwen/Qwen2.5-VL-3B-Instruct",
                                    "Qwen/Qwen2.5-VL-2B-Instruct"
                                ],
                                value="Qwen/Qwen2.5-VL-7B-Instruct"
                            )
                            save_models_btn = gr.Button("Save Model Settings", variant="secondary")
                            save_models_status = gr.Textbox(label="Model Settings Status", interactive=False)

                        with gr.TabItem("Audio"):
                            ref_voice_clip = gr.Textbox(
                                label="Reference voice clip path (optional)",
                                placeholder="Path to reference .wav/.mp3 file..."
                            )
                            with gr.Row():
                                first_ts = gr.Number(label="Start timestamp (s)", value=None, precision=1)
                                end_ts = gr.Number(label="End timestamp (s)", value=None, precision=1)

                    # Run Processing
                    run_btn = gr.Button(
                        "🚀 RUN PIPELINE",
                        variant="primary",
                        size="lg",
                        interactive=False
                    )

                    run_status = gr.Textbox(
                        label="Processing Status / Logs",
                        interactive=False,
                        lines=6
                    )

            with gr.TabItem("Review"):
                with gr.Row():
                    with gr.Column(scale=2):
                        # Gallery and Editing
                        gr.Markdown("## 🖼️ Dataset Gallery")
                        with gr.Row():
                            load_gallery_btn = gr.Button("🔄 Load/Refresh Gallery")
                            gallery_status = gr.Textbox(
                                label="Gallery Status",
                                interactive=False,
                                scale=2
                            )

                        # Pagination controls
                        with gr.Row():
                            prev_page_btn = gr.Button("◀ Previous", size="sm")
                            current_page = gr.Number(label="Page", value=1, precision=0, minimum=1)
                            total_pages = gr.Number(label="Total", value=1, precision=0, interactive=False)
                            next_page_btn = gr.Button("Next ▶", size="sm")

                        gallery = gr.Gallery(
                            label="Processed Media",
                            show_label=True,
                            elem_id="main-gallery",
                            columns=4,
                            rows=3,
                            height="500px",
                            allow_preview=True
                        )

                    with gr.Column(scale=1):
                        # Caption Editing
                        gr.Markdown("## ✏️ Caption Editor")
                        caption_editor = gr.Textbox(
                            label="Caption",
                            lines=6,
                            placeholder="Select an item from the gallery to edit its caption..."
                        )
                        with gr.Row():
                            save_caption_btn = gr.Button("💾 Save Caption", variant="secondary")
                            caption_save_status = gr.Textbox(
                                label="Save Status",
                                interactive=False,
                                scale=2
                            )
                        # Error summary and logs
                        gr.Markdown("### 🔍 Error Summary")
                        error_summary = gr.Textbox(
                            label="Recent Errors",
                            lines=4,
                            interactive=False,
                            placeholder="No errors to display"
                        )
                        refresh_errors_btn = gr.Button("🔄 Refresh Errors", size="sm")

                        logs_path = gr.Textbox(label="Run Logs Path", interactive=False)
                        download_logs = gr.File(label="Download run_logs.jsonl", interactive=False)


            # Event Handlers

            # Project creation
            create_btn.click(
                fn=self.create_project,
                inputs=[new_project_name],
                outputs=[project_dropdown, add_status]
            ).then(
                fn=self.get_project_stats,
                inputs=[project_dropdown],
                outputs=[project_stats]
            ).then(
                fn=self.compute_run_button_state,
                inputs=[project_dropdown],
                outputs=[run_btn]
            )

            # Project selection updates stats
            project_dropdown.change(
                fn=self.get_project_stats,
                inputs=[project_dropdown],
                outputs=[project_stats]
            ).then(
                fn=self.compute_run_button_state,
                inputs=[project_dropdown],
                outputs=[run_btn]
            ).then(
                fn=self.load_model_settings,
                inputs=[project_dropdown],
                outputs=[captioner_model, reasoning_enabled, reasoning_model, system_prompt, context_diary, image_prompt, video_prompt, audio_prompt]
            ).then(
                fn=self.get_run_logs_path,
                inputs=[project_dropdown],
                outputs=[logs_path, download_logs]
            ).then(
                fn=self.load_context_diary,
                inputs=[project_dropdown],
                outputs=[context_diary]
            )

            # File upload
            add_files_btn.click(
                fn=self.add_files_to_project,
                inputs=[project_dropdown, file_upload],
                outputs=[add_status]
            ).then(
                fn=self.get_project_stats,
                inputs=[project_dropdown],
                outputs=[project_stats]
            ).then(
                fn=self.compute_run_button_state,
                inputs=[project_dropdown],
                outputs=[run_btn]
            ).then(
                fn=self.get_file_counts,
                inputs=[project_dropdown],
                outputs=[counts_box]
            )

            # Processing pipeline
            run_btn.click(
                fn=self.run_processing,
                inputs=[
                    project_dropdown,
                    use_isolation,
                    ref_voice_clip,
                    first_ts,
                    end_ts,
                    force_reprocess,
                    system_prompt
                ],
                outputs=[run_status]
            ).then(
                fn=self.get_project_stats,
                inputs=[project_dropdown],
                outputs=[project_stats]
            ).then(
                fn=self.load_project_gallery,
                inputs=[project_dropdown, gr.State(1)],
                outputs=[gallery, gallery_status, current_page, total_pages]
            )

            # Gallery loading
            load_gallery_btn.click(
                fn=self.load_project_gallery,
                inputs=[project_dropdown, current_page],
                outputs=[gallery, gallery_status, current_page, total_pages]
            )

            # Pagination controls
            prev_page_btn.click(
                fn=self.navigate_gallery,
                inputs=[project_dropdown, current_page, gr.State("prev")],
                outputs=[gallery, gallery_status, current_page, total_pages]
            )

            next_page_btn.click(
                fn=self.navigate_gallery,
                inputs=[project_dropdown, current_page, gr.State("next")],
                outputs=[gallery, gallery_status, current_page, total_pages]
            )

            current_page.change(
                fn=self.load_project_gallery,
                inputs=[project_dropdown, current_page],
                outputs=[gallery, gallery_status, current_page, total_pages]
            )

            # Gallery selection for caption editing
            gallery.select(
                fn=self.load_caption_for_editing,
                inputs=[project_dropdown, gallery],
                outputs=[caption_editor]
            )

            # Caption saving
            save_caption_btn.click(
                fn=self.save_edited_caption,
                inputs=[project_dropdown, gallery, caption_editor],
                outputs=[caption_save_status]
            )

            # Auto-refresh stats after processing
            # Save model settings and context
            save_models_btn.click(
                fn=self.save_model_settings,
                inputs=[project_dropdown, captioner_model, reasoning_enabled, reasoning_model, system_prompt, image_prompt, video_prompt, audio_prompt],
                outputs=[save_models_status]
            )
            save_context_btn.click(
                fn=self.save_context_diary,
                inputs=[project_dropdown, context_diary],
                outputs=[context_save_status]
            )

            # Enable/disable run button based on readiness
            project_dropdown.change(
                fn=self.compute_run_button_state,
                inputs=[project_dropdown],
                outputs=[run_btn]
            )

            # Error summary refresh
            refresh_errors_btn.click(
                fn=self.get_error_summary,
                inputs=[project_dropdown],
                outputs=[error_summary]
            )

            # Auto-load error summary when project changes
            project_dropdown.change(
                fn=self.get_error_summary,
                inputs=[project_dropdown],
                outputs=[error_summary]
            )

        return interface
