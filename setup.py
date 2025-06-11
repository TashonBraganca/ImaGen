import os
import subprocess
import sys
import webbrowser
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_directories():
    """Create necessary directories"""
    directories = [
        "models/lcm_dreamshaper",
        "output",
        "src"
    ]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Created directory: {directory}")

def install_requirements():
    """Install required packages"""
    requirements = [
        "torch>=2.0.0",
        "diffusers>=0.33.1",
        "transformers>=4.36.0",
        "accelerate>=0.25.0",
        "gradio>=3.41.2",
        "pillow>=10.0.0"
    ]
    
    logger.info("Installing requirements...")
    for package in requirements:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            logger.info(f"Installed {package}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install {package}: {str(e)}")
            raise

def create_run_script():
    """Create a run.py script that launches the application"""
    run_script = """
import os
import webbrowser
import time
import subprocess
import sys
from pathlib import Path

def run_app():
    # Get the directory of run.py
    current_dir = Path(__file__).parent.absolute()
    
    # Change to the project directory
    os.chdir(current_dir)
    
    # Start the Gradio app
    process = subprocess.Popen([sys.executable, "gradio_app.py"])
    
    # Wait for the server to start
    time.sleep(3)
    
    # Open the browser
    webbrowser.open("http://127.0.0.1:7860")
    
    try:
        # Keep the script running
        process.wait()
    except KeyboardInterrupt:
        process.terminate()
        print("\\nApplication closed.")

if __name__ == "__main__":
    run_app()
"""
    
    with open("run.py", "w") as f:
        f.write(run_script)
    logger.info("Created run.py script")

def create_model_downloader():
    """Create the model downloader script"""
    downloader_script = """
import os
import logging
from huggingface_hub import snapshot_download
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def download_model():
    model_id = "SimianLuo/LCM_Dreamshaper_v7"
    local_dir = "models/lcm_dreamshaper"
    
    logger.info(f"Downloading model {model_id}...")
    try:
        snapshot_download(
            repo_id=model_id,
            local_dir=local_dir,
            local_dir_use_symlinks=False
        )
        logger.info("Model downloaded successfully!")
    except Exception as e:
        logger.error(f"Failed to download model: {str(e)}")
        raise

if __name__ == "__main__":
    download_model()
"""
    
    with open("model_downloader.py", "w") as f:
        f.write(downloader_script)
    logger.info("Created model_downloader.py script")

def main():
    """Main setup function"""
    try:
        logger.info("Starting setup...")
        
        # Create directories
        create_directories()
        
        # Install requirements
        install_requirements()
        
        # Create run script
        create_run_script()
        
        # Create model downloader
        create_model_downloader()
        
        # Download the model
        logger.info("Downloading model...")
        subprocess.check_call([sys.executable, "model_downloader.py"])
        
        logger.info("Setup completed successfully!")
        logger.info("You can now run the application using: python run.py")
        
    except Exception as e:
        logger.error(f"Setup failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 