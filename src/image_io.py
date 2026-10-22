"""
CAMMP-SFSE: Image I/O Module

Safe loading of SEM fractograph images (.tif, .png, .jpg, .bmp),
conversion to 8-bit grayscale (uint8), optional bottom scale-bar
cropping, and saving outputs.
"""

import os
import numpy as np
import cv2
from pathlib import Path


SUPPORTED_EXTENSIONS = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}


def load_image(path: str) -> np.ndarray:
    """Load an image file and return it as a numpy array.
    
    Parameters
    ----------
    path : str
        Path to the image file.
    
    Returns
    -------
    np.ndarray
        Loaded image (BGR or grayscale).
    
    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the file extension is not supported or the image cannot be read.
    """
    path = str(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Image file not found: {path}")
    
    ext = os.path.splitext(path)[1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported image format '{ext}'. "
            f"Supported: {sorted(SUPPORTED_EXTENSIONS)}"
        )
    
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Failed to read image: {path}")
    
    return img


def to_grayscale_uint8(img: np.ndarray) -> np.ndarray:
    """Convert an image to 8-bit grayscale (uint8, range 0-255).
    
    Handles:
    - Already grayscale uint8 images (returned as-is)
    - Multi-channel (BGR/BGRA) images (converted via luminance)
    - 16-bit or float images (rescaled to 0-255)
    
    Parameters
    ----------
    img : np.ndarray
        Input image array.
    
    Returns
    -------
    np.ndarray
        2D uint8 grayscale image.
    """
    # Handle multi-channel -> grayscale
    if img.ndim == 3:
        if img.shape[2] == 4:  # BGRA
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
        elif img.shape[2] == 3:  # BGR
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            img = img[:, :, 0]
    
    # Handle bit depth conversion
    if img.dtype == np.uint16:
        img = (img / 256).astype(np.uint8)
    elif img.dtype in (np.float32, np.float64):
        # Normalize float images to 0-255
        mn, mx = img.min(), img.max()
        if mx - mn > 0:
            img = ((img - mn) / (mx - mn) * 255).astype(np.uint8)
        else:
            img = np.zeros_like(img, dtype=np.uint8)
    elif img.dtype != np.uint8:
        img = img.astype(np.uint8)
    
    return img


def crop_banner(img: np.ndarray, pixels: int) -> np.ndarray:
    """Crop the bottom portion of an image to remove SEM info banners.
    
    Parameters
    ----------
    img : np.ndarray
        Input image (2D or 3D).
    pixels : int
        Number of pixel rows to remove from the bottom.
    
    Returns
    -------
    np.ndarray
        Cropped image.
    
    Raises
    ------
    ValueError
        If pixels >= image height.
    """
    if pixels <= 0:
        return img
    
    h = img.shape[0]
    if pixels >= h:
        raise ValueError(
            f"Cannot crop {pixels} pixels from an image with height {h}"
        )
    
    return img[:h - pixels]


def save_image(img: np.ndarray, path: str) -> str:
    """Save an image to disk, creating parent directories if needed.
    
    Parameters
    ----------
    img : np.ndarray
        Image to save.
    path : str
        Output file path.
    
    Returns
    -------
    str
        Absolute path to the saved file.
    """
    path = str(path)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    cv2.imwrite(path, img)
    return os.path.abspath(path)


def collect_image_paths(input_path: str) -> list:
    """Collect image file paths from a file or directory.
    
    Parameters
    ----------
    input_path : str
        Path to a single image file or a directory of images.
    
    Returns
    -------
    list of str
        Sorted list of image file paths.
    
    Raises
    ------
    FileNotFoundError
        If the path does not exist.
    ValueError
        If no supported images are found.
    """
    p = Path(input_path)
    
    if not p.exists():
        raise FileNotFoundError(f"Input path not found: {input_path}")
    
    if p.is_file():
        ext = p.suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported image format: {ext}")
        return [str(p)]
    
    if p.is_dir():
        paths = sorted(
            str(f) for f in p.iterdir()
            if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
        )
        if not paths:
            raise ValueError(f"No supported images found in: {input_path}")
        return paths
    
    raise FileNotFoundError(f"Input path is neither file nor directory: {input_path}")
