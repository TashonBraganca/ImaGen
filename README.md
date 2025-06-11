# ImaGen - Local LCM Dreamshaper Image Generation

A powerful, offline-capable AI image generation application that uses the LCM Dreamshaper model to create high-quality digital art.

## Features

- 🎨 Generate high-quality images from text prompts
- 💻 Works completely offline
- 🚀 Fast generation with optimized parameters
- 🎯 Multiple style presets
- 🖼️ Batch image generation
- 🎮 User-friendly interface

## Quick Start

1. **Clone the repository**:
```bash
git clone https://github.com/yourusername/Imagen.git
cd Imagen
```

2. **Run the setup script**:
```bash
python setup.py
```
This will:
- Create necessary directories
- Install required packages
- Download the LCM Dreamshaper model
- Create a run script

3. **Launch the application**:
```bash
python run.py
```
This will:
- Start the Gradio server
- Open your default browser to the application
- Display the user interface

## Usage

1. **Enter a prompt** describing the image you want to generate
2. **Adjust settings**:
   - Quality (1-10)
   - Image size (256x256, 512x512, 768x768)
   - Guidance scale
   - Seed (for reproducible results)
   - Number of images to generate

3. **Click "Generate"** to create your images
4. **View results** in the gallery
5. **Save images** from the output directory

## Style Presets

- **Photorealistic**: Highly detailed, 8K UHD quality
- **Anime**: Vibrant colors, detailed illustration
- **Digital Art**: Trending on ArtStation style
- **Abstract**: Modern, contemporary artistic style

## Requirements

- Python 3.8 or higher
- 8GB RAM minimum (16GB recommended)
- 10GB free disk space for the model

## Troubleshooting

If you encounter any issues:

1. **Model download fails**:
   - Check your internet connection
   - Ensure you have enough disk space
   - Try running `python model_downloader.py` manually

2. **Application won't start**:
   - Verify all requirements are installed
   - Check if port 7860 is available
   - Ensure the model files are present

3. **Generation errors**:
   - Check the console for error messages
   - Verify your prompt is valid
   - Try reducing the batch size or image dimensions

## Sample Outputs

Click the links below to view sample output images:

- [Generated Image 1](output/generated_1749645124_0.png)
- [Generated Image 2](output/generated_1749645188_0.png)
- [Test Generation](output/test_generation.png)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [SimianLuo](https://github.com/SimianLuo) for the LCM Dreamshaper model
- [Gradio](https://gradio.app/) for the interface framework
- [Diffusers](https://github.com/huggingface/diffusers) library