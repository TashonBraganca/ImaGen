import torch
import time
import logging
from diffusers import DiffusionPipeline
from pathlib import Path
from PIL import Image
import os
from utils import setup_logging, ensure_directory

def test_lcm_dreamshaper():
    # Setup logging
    logger = setup_logging()
    logger.info("Starting LCM Dreamshaper model test")

    # Ensure output directory exists
    output_dir = Path("output")
    ensure_directory(output_dir)

    try:
        # Record start time for model loading
        load_start = time.time()
        
        # Initialize the pipeline
        logger.info("Loading LCM Dreamshaper model...")
        pipe = DiffusionPipeline.from_pretrained(
            "SimianLuo/LCM_Dreamshaper_v7",
            torch_dtype=torch.float32,  # Use float32 for CPU
            use_safetensors=True
        )
        
        # Move to CPU explicitly
        pipe = pipe.to("cpu")
        
        load_time = time.time() - load_start
        logger.info(f"Model loaded in {load_time:.2f} seconds")

        # Test generation
        prompt = "a beautiful sunset over mountains, digital art"
        logger.info(f"Generating image with prompt: {prompt}")
        
        # Record generation time
        gen_start = time.time()
        
        # Generate image
        image = pipe(
            prompt=prompt,
            num_inference_steps=4,  # LCM is fast, we can use fewer steps
            guidance_scale=1.0
        ).images[0]
        
        gen_time = time.time() - gen_start
        logger.info(f"Image generated in {gen_time:.2f} seconds")

        # Save the image
        output_path = output_dir / "test_generation.png"
        image.save(output_path)
        logger.info(f"Image saved to {output_path}")

        # Print summary
        print("\nTest Summary:")
        print(f"Model loading time: {load_time:.2f} seconds")
        print(f"Image generation time: {gen_time:.2f} seconds")
        print(f"Total time: {load_time + gen_time:.2f} seconds")
        print(f"Output saved to: {output_path}")

    except Exception as e:
        logger.error(f"Error during model test: {str(e)}")
        raise

if __name__ == "__main__":
    test_lcm_dreamshaper() 