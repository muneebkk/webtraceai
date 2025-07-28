# Hassan Hadi: Utility functions for image processing and validation
# Focus: Image processing, file validation, and helper functions

import io
from PIL import Image
from typing import Union
import os

def process_image(image_data: Union[bytes, io.BytesIO]) -> Image.Image:
    """
    Process uploaded image data
    Simplified version using PIL instead of OpenCV
    """
    try:
        if isinstance(image_data, bytes):
            image = Image.open(io.BytesIO(image_data))
        elif isinstance(image_data, io.BytesIO):
            image = Image.open(image_data)
        else:
            raise ValueError("Invalid image data type")
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        return image
        
    except Exception as e:
        # Return a default image if processing fails
        default_image = Image.new('RGB', (800, 600), color='white')
        return default_image

def validate_file(file) -> bool:
    """
    Validate uploaded file
    """
    try:
        # Check if file exists
        if not file:
            return False
        
        # Check file size (max 10MB)
        file.file.seek(0, 2)  # Seek to end
        file_size = file.file.tell()
        file.file.seek(0)  # Reset to beginning
        
        if file_size > 10 * 1024 * 1024:  # 10MB limit
            return False
        
        # Check file extension
        allowed_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.bmp'}
        file_extension = os.path.splitext(file.filename.lower())[1]
        
        if file_extension not in allowed_extensions:
            return False
        
        return True
        
    except Exception:
        return False

def save_image(image: Image.Image, filepath: str) -> bool:
    """
    Save image to file
    """
    try:
        image.save(filepath)
        return True
    except Exception:
        return False

def resize_image(image: Image.Image, max_width: int = 800, max_height: int = 600) -> Image.Image:
    """
    Resize image while maintaining aspect ratio
    """
    try:
        # Calculate new dimensions
        width, height = image.size
        ratio = min(max_width / width, max_height / height)
        
        if ratio < 1:
            new_width = int(width * ratio)
            new_height = int(height * ratio)
            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        return image
        
    except Exception:
        return image 