import os
import torch
from pathlib import Path
from diffusers import DiffusionPipeline
import logging
import sys

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('model_download.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

def download_model():
    """Download the LCM Dreamshaper model and save it locally"""
    logger = setup_logging()
    
    # Define model paths
    base_path = Path("models/lcm_dreamshaper")
    model_path = base_path / "model_files"
    
    # Create directories if they don't exist
    model_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Model will be saved to: {model_path}")
    
    try:
        # Download model
        logger.info("Downloading LCM Dreamshaper model...")
        model = DiffusionPipeline.from_pretrained(
            "SimianLuo/LCM_Dreamshaper_v7",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            use_safetensors=True
        )
        
        # Save model locally
        logger.info("Saving model locally...")
        model.save_pretrained(model_path)
        
        logger.info("Model downloaded and saved successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Error downloading model: {str(e)}")
        logger.error("Full traceback:", exc_info=True)
        return False

if __name__ == "__main__":
    if download_model():
        print("\nLCM Dreamshaper model downloaded successfully!")
        print("You can now run the application and it will use the local model.")
    else:
        print("\nFailed to download model. Please check the logs for details.") 