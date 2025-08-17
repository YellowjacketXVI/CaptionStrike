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
                      force_reprocess: bool = False) -> str:
        """Run the processing pipeline on a project.
        
        Args:
            project_name: Name of the project
            use_person_isolation: Whether to enable person isolation
            reference_voice_clip: Path to reference voice clip
            first_sound_ts: Start timestamp for audio reference
            end_sound_ts: End timestamp for audio reference
            force_reprocess: Whether to force reprocessing
            
        Returns:
            Status message
        """
        try:
            if not project_name:
                return "❌ Please select a project first"
            
            layout = ProjectLayout(self.root_dir, project_name)
            if not layout.exists():
                return f"❌ Project '{project_name}' does not exist"
            
            # Update project configuration
            config = ProjectConfig(layout.project_config_file)
            config.load()
            config.set("isolation.faces", use_person_isolation)
            config.save()
            
            # Prepare audio processing parameters
            ref_clip = Path(reference_voice_clip) if reference_voice_clip and reference_voice_clip.strip() else None
            
            # Run processing pipeline
            logger.info(f"Starting processing for project '{project_name}'")
            result = self.pipeline.process_project(
                layout=layout,
                reference_voice_clip=ref_clip,
                first_sound_ts=first_sound_ts,
                end_sound_ts=end_sound_ts,
                force_reprocess=force_reprocess
            )
            
            if result["success"]:
                message = f"✅ {result['message']}"
                if result["errors"]:
                    message += f"\n⚠️ {len(result['errors'])} error(s):\n" + "\n".join(result["errors"][:3])
                return message
            else:
                return f"❌ Processing failed: {result['message']}"
            
        except Exception as e:
            logger.error(f"Processing failed: {e}")
            return f"❌ Processing error: {str(e)}"
    
    def load_project_gallery(self, project_name: str) -> Tuple[gr.Gallery, str]:
        """Load project gallery with thumbnails.
        
        Args:
            project_name: Name of the project
            
        Returns:
            Tuple of (gallery component, status message)
        """
        try:
            if not project_name:
                return gr.Gallery(value=[]), "Please select a project"
            
            layout = ProjectLayout(self.root_dir, project_name)
            if not layout.exists():
                return gr.Gallery(value=[]), f"Project '{project_name}' does not exist"
            
            # Get thumbnails
            thumbnails = layout.get_thumbnails()
            
            if not thumbnails:
                # If no thumbnails, try to show raw images
                raw_images = layout.get_raw_files("image")
                gallery_items = []
                for img_path in raw_images[:20]:  # Limit to 20 for performance
                    try:
                        # Create a simple thumbnail
                        img = Image.open(img_path)
                        img.thumbnail((256, 256))
                        gallery_items.append((str(img_path), img_path.stem))
                    except Exception:
                        continue
                
                if gallery_items:
                    return gr.Gallery(value=gallery_items), f"Showing {len(gallery_items)} raw images (run processing to generate thumbnails)"
                else:
                    return gr.Gallery(value=[]), "No images found in project"
            
            # Create gallery items from thumbnails
            gallery_items = []
            for thumb_path in sorted(thumbnails):
                # Find corresponding processed file
                processed_name = thumb_path.stem
                gallery_items.append((str(thumb_path), processed_name))
            
            return (
                gr.Gallery(value=gallery_items),
                f"Loaded {len(gallery_items)} processed items"
            )
            
        except Exception as e:
            logger.error(f"Failed to load gallery: {e}")
            return gr.Gallery(value=[]), f"Error loading gallery: {str(e)}"
    
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

            gr.Markdown("""
            # 🎯 CaptionStrike — Local Dataset Builder

            Create high-quality training datasets with AI-powered captioning using **Florence-2** and optional **Qwen2.5-VL** enhancement.
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
                        file_types=["image", "video", "audio"],
                        label="Drop files here or click to browse"
                    )

                    add_files_btn = gr.Button("Add to Project", variant="secondary")
                    add_status = gr.Textbox(label="Status", interactive=False)

                    # Processing Options
                    gr.Markdown("## ⚙️ Processing Options")

                    with gr.Group():
                        use_isolation = gr.Checkbox(
                            label="🧑 Enable person isolation (face crops)",
                            value=False
                        )

                        force_reprocess = gr.Checkbox(
                            label="🔄 Force reprocess existing files",
                            value=False
                        )

                    # Audio Processing Options
                    with gr.Group():
                        gr.Markdown("### 🎵 Audio Processing")

                        ref_voice_clip = gr.Textbox(
                            label="Reference voice clip path (optional)",
                            placeholder="Path to reference .wav/.mp3 file..."
                        )

                        with gr.Row():
                            first_ts = gr.Number(
                                label="Start timestamp (s)",
                                value=None,
                                precision=1
                            )
                            end_ts = gr.Number(
                                label="End timestamp (s)",
                                value=None,
                                precision=1
                            )

                    # Run Processing
                    run_btn = gr.Button(
                        "🚀 RUN PIPELINE",
                        variant="primary",
                        size="lg"
                    )

                    run_status = gr.Textbox(
                        label="Processing Status",
                        interactive=False,
                        lines=3
                    )

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

                    gallery = gr.Gallery(
                        label="Processed Media",
                        show_label=True,
                        elem_id="main-gallery",
                        columns=4,
                        rows=3,
                        height="400px",
                        allow_preview=True
                    )

                    # Caption Editing
                    gr.Markdown("## ✏️ Caption Editor")

                    caption_editor = gr.Textbox(
                        label="Caption",
                        lines=3,
                        placeholder="Select an item from the gallery to edit its caption..."
                    )

                    with gr.Row():
                        save_caption_btn = gr.Button("💾 Save Caption", variant="secondary")
                        caption_save_status = gr.Textbox(
                            label="Save Status",
                            interactive=False,
                            scale=2
                        )

            # Event Handlers

            # Project creation
            create_btn.click(
                fn=self.create_project,
                inputs=[new_project_name],
                outputs=[project_dropdown, add_status]
            )

            # Project selection updates stats
            project_dropdown.change(
                fn=self.get_project_stats,
                inputs=[project_dropdown],
                outputs=[project_stats]
            )

            # File upload
            add_files_btn.click(
                fn=self.add_files_to_project,
                inputs=[project_dropdown, file_upload],
                outputs=[add_status]
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
                    force_reprocess
                ],
                outputs=[run_status]
            )

            # Gallery loading
            load_gallery_btn.click(
                fn=self.load_project_gallery,
                inputs=[project_dropdown],
                outputs=[gallery, gallery_status]
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
            run_btn.click(
                fn=self.get_project_stats,
                inputs=[project_dropdown],
                outputs=[project_stats]
            ).then(
                fn=self.load_project_gallery,
                inputs=[project_dropdown],
                outputs=[gallery, gallery_status]
            )

        return interface
