import torch
import logging
from pathlib import Path
from diffusers import DiffusionPipeline, LatentConsistencyModelPipeline
from PIL import Image
import os
import sys
import gradio as gr

class ImageGenerator:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ImageGenerator, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        # Setup logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # Create file handler
        file_handler = logging.FileHandler('debug.log')
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        
        self.logger.info("ImageGenerator initialized")
        
        # Initialize paths
        self.base_model_path = Path("models/lcm_dreamshaper")
        self.model_path = self.base_model_path / "model_files"
        
        # Set device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.logger.debug(f"Device: {self.device}")
        
        # Initialize model as None
        self.model = None
        
        self._initialized = True
    
    def _verify_model_files(self) -> bool:
        """Verify if model files exist in the local directory"""
        required_files = [
            "model_index.json",
            "scheduler/scheduler_config.json",
            "text_encoder/config.json",
            "tokenizer/vocab.json",
            "unet/config.json",
            "vae/config.json"
        ]
        
        self.logger.debug("Verifying model files...")
        for file in required_files:
            exists = (self.model_path / file).exists()
            self.logger.debug(f"Checking {file}: {'✓' if exists else '✗'}")
            if not exists:
                self.logger.error(f"Missing required file: {file}")
                return False
        return True
    
    def load_model(self) -> bool:
        """Load the model from local files"""
        try:
            if not self._verify_model_files():
                self.logger.error("Model files not found. Please run model_downloader.py first.")
                return False
            
            self.logger.info("Loading model from local files...")
            self.model = LatentConsistencyModelPipeline.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                local_files_only=True
            )
            
            if self.device == "cuda":
                self.model = self.model.to("cuda")
            
            # LCM doesn't need eval() mode
            self.logger.info("Model loaded successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            self.logger.error("Full traceback:", exc_info=True)
            return False
    
    def generate_image(
        self,
        prompt: str,
        steps: int = 4,  # LCM is fast, we can use fewer steps
        guidance_scale: float = 1.0,  # Optimized for LCM
        width: int = 512,
        height: int = 512
    ) -> Image.Image:
        """Generate an image from a prompt"""
        try:
            if self.model is None:
                self.logger.error("Model not loaded")
                return None
            
            self.logger.info(f"Generating image with prompt: {prompt}")
            self.logger.debug(f"Parameters: steps={steps}, guidance_scale={guidance_scale}, size={width}x{height}")
            
            # Generate image
            with torch.no_grad():
                image = self.model(
                    prompt=prompt,
                    num_inference_steps=steps,
                    guidance_scale=guidance_scale,
                    width=width,
                    height=height
                ).images[0]
            
            self.logger.info("Image generated successfully")
            return image
            
        except Exception as e:
            self.logger.error(f"Error generating image: {str(e)}")
            self.logger.error("Full traceback:", exc_info=True)
            return None
    
    def cleanup(self):
        """Clean up resources"""
        try:
            if self.model is not None:
                self.model = None
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                self.logger.info("Model resources cleaned up")
        except Exception as e:
            self.logger.error(f"Error during cleanup: {str(e)}")
            self.logger.error("Full traceback:", exc_info=True)

    def __del__(self):
        """Destructor to ensure proper cleanup."""
        self.cleanup() 