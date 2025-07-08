from PIL import Image
import io
from typing import Union
from fastapi import UploadFile

def process_image(image_data: Union[bytes, io.BytesIO]) -> Image.Image:
    """
    Process uploaded image data and return PIL Image object
    """
    try:
        if isinstance(image_data, bytes):
            image = Image.open(io.BytesIO(image_data))
        else:
            image = Image.open(image_data)
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize if too large (for performance)
        max_size = (1920, 1080)
        if image.size[0] > max_size[0] or image.size[1] > max_size[1]:
            image.thumbnail(max_size, Image.Resampling.LANCZOS)
        
        return image
        
    except Exception as e:
        raise ValueError(f"Failed to process image: {str(e)}")

def validate_file(file: UploadFile) -> bool:
    """
    Validate uploaded file format
    """
    allowed_types = ['image/jpeg', 'image/jpg', 'image/png']
    allowed_extensions = ['.jpg', '.jpeg', '.png']
    
    # Check MIME type
    if file.content_type not in allowed_types:
        return False
    
    # Check file extension
    if not any(file.filename.lower().endswith(ext) for ext in allowed_extensions):
        return False
    
    return True

def extract_text_from_html(html_content: str) -> str:
    """
    Extract visible text content from HTML
    """
    try:
        from bs4 import BeautifulSoup
        
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        # Get text
        text = soup.get_text()
        
        # Clean up whitespace
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)
        
        return text
        
    except Exception as e:
        print(f"Error extracting text from HTML: {e}")
        return ""

def format_probabilities(probabilities: dict) -> dict:
    """
    Format probability values as percentages
    """
    return {
        tool: round(prob * 100, 2) 
        for tool, prob in probabilities.items()
    }

def get_file_size_mb(file: UploadFile) -> float:
    """
    Get file size in MB
    """
    try:
        # Read file content to get size
        content = file.file.read()
        file.file.seek(0)  # Reset file pointer
        return len(content) / (1024 * 1024)
    except Exception:
        return 0.0 