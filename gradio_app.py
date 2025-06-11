import gradio as gr
import time
from src.generator import ImageGenerator
import random
from typing import List, Tuple
import logging
import torch
import os

# Custom CSS for modern dark theme
CUSTOM_CSS = """
.gradio-container {
    background: linear-gradient(135deg, #1a1a1a 0%, #2d2d2d 100%);
    color: #ffffff;
}
.container {
    max-width: 1200px;
    margin: auto;
    padding: 20px;
}
.prompt-box {
    background: rgba(255, 255, 255, 0.1);
    border-radius: 10px;
    padding: 20px;
    margin-bottom: 20px;
}
.generate-btn {
    background: linear-gradient(45deg, #FF6B6B, #4ECDC4);
    border: none;
    border-radius: 25px;
    padding: 15px 30px;
    color: white;
    font-weight: bold;
    transition: all 0.3s ease;
}
.generate-btn:hover {
    transform: translateY(-2px);
    box-shadow: 0 5px 15px rgba(0,0,0,0.3);
}
.gallery {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
    gap: 20px;
    padding: 20px;
}
.example-prompt {
    background: rgba(255, 255, 255, 0.05);
    padding: 10px;
    border-radius: 5px;
    cursor: pointer;
    transition: all 0.3s ease;
}
.example-prompt:hover {
    background: rgba(255, 255, 255, 0.1);
}
.status-indicator {
    padding: 10px;
    border-radius: 5px;
    margin: 10px 0;
    text-align: center;
    font-weight: bold;
}
.status-loading {
    background: rgba(255, 193, 7, 0.2);
    color: #ffc107;
}
.status-ready {
    background: rgba(76, 175, 80, 0.2);
    color: #4caf50;
}
.status-error {
    background: rgba(244, 67, 54, 0.2);
    color: #f44336;
}
"""

# Example prompts
EXAMPLE_PROMPTS = [
    "A serene landscape with mountains and a lake at sunset, digital art",
    "Cyberpunk cityscape with neon lights and flying cars",
    "Magical forest with glowing mushrooms and fairy lights",
    "Abstract geometric patterns in vibrant colors",
    "Portrait of a futuristic robot with expressive eyes"
]

# Style presets
STYLE_PRESETS = {
    "Photorealistic": "photorealistic, highly detailed, 8k uhd",
    "Anime": "anime style, vibrant colors, detailed illustration",
    "Digital Art": "digital art, trending on artstation, highly detailed",
    "Abstract": "abstract art, modern, contemporary, artistic"
}

class ImaGenInterface:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.generator = None
        self.model_status = "loading"
        self.initialize_generator()

    def initialize_generator(self):
        """Initialize the image generator with proper error handling"""
        try:
            self.model_status = "loading"
            self.generator = ImageGenerator()
            
            # Try to load model
            if not self.generator.load_model():
                self.model_status = "error"
                raise gr.Error("Model files not found. Please run model_downloader.py first to download the model.")
            
            self.model_status = "ready"
            self.logger.info("Image generator initialized successfully")
            
        except Exception as e:
            self.model_status = "error"
            self.logger.error(f"Error initializing generator: {str(e)}")
            raise gr.Error(f"Failed to initialize image generator: {str(e)}")

    def _check_internet_connection(self) -> bool:
        """Check if we have internet connectivity"""
        try:
            import socket
            socket.create_connection(("8.8.8.8", 53), timeout=3)
            return True
        except OSError:
            return False

    def get_status_html(self):
        """Get HTML for status indicator"""
        status_colors = {
            "loading": ("status-loading", "⏳ Loading Model..."),
            "ready": ("status-ready", "✅ Model Ready"),
            "error": ("status-error", "❌ Model Error")
        }
        color_class, message = status_colors.get(self.model_status, ("status-error", "Unknown Status"))
        return f'<div class="status-indicator {color_class}">{message}</div>'

    def map_quality_to_steps(self, quality: int) -> int:
        """Map quality (1-10) to number of steps (2-15)"""
        return max(2, min(15, int(2 + (quality - 1) * 1.5)))

    def generate_images(
        self,
        prompt: str,
        quality: int,
        size: str,
        guidance_scale: float,
        seed: int,
        batch_count: int
    ) -> Tuple[List[Tuple[str, str]], str]:
        """Generate images with progress updates"""
        try:
            # Ensure generator is initialized
            if self.generator is None or self.generator.model is None:
                self.initialize_generator()
            
            if not prompt or not prompt.strip():
                raise gr.Error("Please enter a prompt")
            
            # Map quality to steps
            steps = self.map_quality_to_steps(quality)
            
            # Parse size
            width = height = int(size.split('x')[0])
            
            # Set seed if provided
            if seed != -1:
                torch.manual_seed(seed)
            
            # Generate batch of images
            images = []
            for i in range(batch_count):
                # Add style preset if present
                if any(style in prompt.lower() for style in STYLE_PRESETS.keys()):
                    style = next(style for style in STYLE_PRESETS.keys() if style.lower() in prompt.lower())
                    prompt = f"{prompt}, {STYLE_PRESETS[style]}"
                
                try:
                    image = self.generator.generate_image(
                        prompt=prompt,
                        steps=steps,
                        guidance_scale=guidance_scale,
                        width=width,
                        height=height
                    )
                    
                    if image:
                        # Save image with timestamp
                        timestamp = int(time.time())
                        filename = f"output/generated_{timestamp}_{i}.png"
                        image.save(filename)
                        images.append((filename, f"Image {i+1}"))
                    else:
                        raise Exception("Failed to generate image")
                except Exception as e:
                    self.logger.error(f"Error generating image {i+1}: {str(e)}")
                    raise gr.Error(f"Failed to generate image {i+1}: {str(e)}")
            
            return images, self.get_status_html()
            
        except Exception as e:
            self.model_status = "error"
            self.logger.error(f"Error generating images: {str(e)}")
            raise gr.Error(f"Failed to generate images: {str(e)}")

    def create_interface(self):
        """Create the Gradio interface"""
        with gr.Blocks(css=CUSTOM_CSS) as interface:
            gr.Markdown("# 🎨 ImaGen - AI Image Generator")
            
            # Status indicator
            status_html = gr.HTML(self.get_status_html())
            
            with gr.Row():
                with gr.Column(scale=2):
                    # Main prompt input
                    prompt = gr.Textbox(
                        label="Prompt",
                        placeholder="Describe the digital art you want to create...",
                        lines=3
                    )
                    
                    # Quality and size controls
                    with gr.Row():
                        quality = gr.Slider(
                            minimum=1,
                            maximum=10,
                            value=5,
                            step=1,
                            label="Quality (1=Fast, 10=Best)"
                        )
                        size = gr.Dropdown(
                            choices=["256x256", "512x512", "768x768"],
                            value="512x512",
                            label="Image Size"
                        )
                    
                    # Advanced settings
                    with gr.Accordion("Advanced Settings", open=False):
                        guidance_scale = gr.Slider(
                            minimum=0.5,
                            maximum=2.0,
                            value=1.0,
                            step=0.1,
                            label="Guidance Scale"
                        )
                        seed = gr.Number(
                            value=-1,
                            label="Seed (-1 for random)"
                        )
                        batch_count = gr.Slider(
                            minimum=1,
                            maximum=4,
                            value=1,
                            step=1,
                            label="Number of Images"
                        )
                    
                    # Style presets
                    with gr.Row():
                        for style in STYLE_PRESETS:
                            style_btn = gr.Button(style)
                            style_btn.click(
                                fn=lambda s=style, p=prompt: f"{p}, {STYLE_PRESETS[s]}" if p else STYLE_PRESETS[s],
                                inputs=[prompt],
                                outputs=[prompt]
                            )
                    
                    # Generate button
                    generate_btn = gr.Button("✨ Generate", variant="primary")
                    
                    # Output gallery
                    gallery = gr.Gallery(
                        label="Generated Images",
                        show_label=True,
                        elem_id="gallery",
                        columns=2,
                        rows=2,
                        object_fit="contain",
                        height="auto"
                    )
                    
                    # Example prompts
                    gr.Examples(
                        examples=EXAMPLE_PROMPTS,
                        inputs=prompt,
                        label="Example Prompts"
                    )
                    
                    # Connect generate button
                    generate_btn.click(
                        fn=self.generate_images,
                        inputs=[
                            prompt,
                            quality,
                            size,
                            guidance_scale,
                            seed,
                            batch_count
                        ],
                        outputs=[gallery, status_html]
                    )
            
            return interface

def main():
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('debug.log'),
            logging.StreamHandler()
        ]
    )
    
    # Create output directory if it doesn't exist
    os.makedirs("output", exist_ok=True)
    
    # Create and launch interface
    interface = ImaGenInterface()
    app = interface.create_interface()
    app.launch(share=True)

if __name__ == "__main__":
    main() 