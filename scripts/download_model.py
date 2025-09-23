#!/usr/bin/env python3
"""
Script to download Qwen2.5-Coder-7B-Instruct quantized model
"""

import os
import requests
import sys
from pathlib import Path
from tqdm import tqdm

import logging

logger = logging.getLogger("code_compass")

def download_file(url, filename):
    """Download file with progress bar"""
    logger.info(f"📥 Downloading {filename}...")
    logger.info(f"🔗 URL: {url}")
    
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    if total_size == 0:
        logger.info("❌ Could not determine file size")
        return False
    
    logger.info(f"📊 File size: {total_size / (1024*1024*1024):.2f} GB")
    
    with open(filename, 'wb') as file, tqdm(
        desc=filename,
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as progress_bar:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                file.write(chunk)
                progress_bar.update(len(chunk))
    
    logger.info(f"✅ Downloaded {filename} successfully!")
    return True

def main():
    """Main download function"""
    logger.info("🔍 Qwen2.5-Coder-7B-Instruct Model Downloader")
    logger.info("=" * 50)
    
    # Available quantization options
    models = {
        "Q4_K_M": {
            "url": "https://huggingface.co/bartowski/Qwen2.5-Coder-7B-Instruct-GGUF/resolve/main/Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf",
            "filename": "qwen2.5-coder-7b-instruct-q4_k_m.gguf",
            "size": "~4.5 GB",
            "description": "4-bit quantization, best balance of quality and size (RECOMMENDED)"
        },
        "Q5_K_M": {
            "url": "https://huggingface.co/bartowski/Qwen2.5-Coder-7B-Instruct-GGUF/resolve/main/Qwen2.5-Coder-7B-Instruct-Q5_K_M.gguf",
            "filename": "qwen2.5-coder-7b-instruct-q5_k_m.gguf",
            "size": "~5.5 GB",
            "description": "5-bit quantization, higher quality than Q4"
        },
        "Q6_K": {
            "url": "https://huggingface.co/bartowski/Qwen2.5-Coder-7B-Instruct-GGUF/resolve/main/Qwen2.5-Coder-7B-Instruct-Q6_K.gguf",
            "filename": "qwen2.5-coder-7b-instruct-q6_k.gguf",
            "size": "~6.5 GB",
            "description": "6-bit quantization, highest quality"
        },
        "Q8_0": {
            "url": "https://huggingface.co/bartowski/Qwen2.5-Coder-7B-Instruct-GGUF/resolve/main/Qwen2.5-Coder-7B-Instruct-Q8_0.gguf",
            "filename": "qwen2.5-coder-7b-instruct-q8_0.gguf",
            "size": "~7.5 GB",
            "description": "8-bit quantization, near full precision"
        }
    }
    
    logger.info("📋 Available model variants:")
    logger.info()
    for i, (key, info) in enumerate(models.items(), 1):
        marker = " ⭐ RECOMMENDED" if key == "Q4_K_M" else ""
        logger.info(f"{i}. {key}{marker}")
        logger.info(f"   Size: {info['size']}")
        logger.info(f"   Description: {info['description']}")
        logger.info()
    
    # Get user choice
    while True:
        try:
            choice = input("Enter your choice (1-4) or 'q' to quit: ").strip()
            
            if choice.lower() == 'q':
                logger.info("👋 Download cancelled.")
                return
            
            choice_num = int(choice)
            if 1 <= choice_num <= len(models):
                selected_key = list(models.keys())[choice_num - 1]
                selected_model = models[selected_key]
                break
            else:
                logger.info("❌ Invalid choice. Please enter 1-4.")
        except ValueError:
            logger.info("❌ Invalid input. Please enter a number 1-4 or 'q'.")
    
    logger.info(f"📦 Selected: {selected_key}")
    logger.info(f"📁 Filename: {selected_model['filename']}")
    logger.info(f"📊 Size: {selected_model['size']}")
    logger.info()
    
    # Check if file already exists
    if os.path.exists(selected_model['filename']):
        overwrite = input(f"⚠️  File {selected_model['filename']} already exists. Overwrite? (y/n): ")
        if overwrite.lower() != 'y':
            logger.info("👋 Download cancelled.")
            return
    
    # Create models directory if it doesn't exist
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    # Full path for the model
    model_path = models_dir / selected_model['filename']
    
    # Download the model
    try:
        success = download_file(selected_model['url'], str(model_path))
        
        if success:
            logger.info()
            logger.info("🎉 Download completed successfully!")
            logger.info(f"📁 Model saved to: {model_path}")
            logger.info()
            logger.info("🚀 To use the model:")
            logger.info("   1. Make sure the model path in llm_service.py points to this file")
            logger.info("   2. Run your main application: python main.py")
            logger.info("   3. Click 'Initialize LLM' in the web interface")
            logger.info()
            logger.info("💡 System Requirements:")
            logger.info("   - RAM: At least 8GB (16GB+ recommended)")
            logger.info("   - Storage: Ensure you have enough free space")
            logger.info("   - CPU: Modern multi-core processor recommended")
        else:
            logger.info("❌ Download failed!")
            return 1
            
    except KeyboardInterrupt:
        logger.info("\n🛑 Download interrupted by user")
        # Clean up partial file
        if os.path.exists(model_path):
            os.remove(model_path)
            logger.info(f"🗑️  Cleaned up partial file: {model_path}")
        return 1
    except Exception as e:
        logger.info(f"❌ Error during download: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())