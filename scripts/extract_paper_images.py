"""
Helper script to extract real experimental SEM fractographs from hole_extract.pdf
and save clean grayscale micrographs into data/raw/.
"""

import os
import io
import numpy as np
from PIL import Image
from pypdf import PdfReader


def extract_paper_images(pdf_path: str = "hole_extract.pdf",
                         output_dir: str = "data/raw") -> list:
    """Extract clean SEM fractographs from hole_extract.pdf.
    
    Parameters
    ----------
    pdf_path : str
        Path to the reference PDF.
    output_dir : str
        Directory to save extracted micrographs.
        
    Returns
    -------
    list of str
        List of paths to the extracted images.
    """
    os.makedirs(output_dir, exist_ok=True)
    reader = PdfReader(pdf_path)
    
    # In Banerjee et al. (hole_extract.pdf), Page 9 (0-indexed 8) contains the full
    # gallery of experimental fractographs analyzed in the paper.
    # Images 3, 7, 11, 15, 22 are the primary distinct SEM micrographs.
    p9_images = reader.pages[8].images
    
    image_specs = [
        {"idx": 3, "name": "sem_real_1.png", "desc": "SEM Fractograph Sample 1"},
        {"idx": 7, "name": "sem_real_2.png", "desc": "SEM Fractograph Sample 2"},
        {"idx": 11, "name": "sem_real_3.png", "desc": "SEM Fractograph Sample 3"},
        {"idx": 15, "name": "sem_real_4.png", "desc": "SEM Fractograph Sample 4"},
        {"idx": 22, "name": "sem_real_5.png", "desc": "SEM Fractograph Sample 5"},
    ]
    
    saved_paths = []
    
    for spec in image_specs:
        idx = spec["idx"]
        out_name = spec["name"]
        raw_data = p9_images[idx].data
        
        pil_img = Image.open(io.BytesIO(raw_data)).convert("L")
        img_arr = np.array(pil_img)
        
        # Trim any trailing black border rows if present (some scans have 2 black rows)
        while img_arr.shape[0] > 0 and np.mean(img_arr[-1, :]) == 0:
            img_arr = img_arr[:-1, :]
            
        clean_img = Image.fromarray(img_arr)
        out_path = os.path.join(output_dir, out_name)
        clean_img.save(out_path)
        saved_paths.append(out_path)
        print(f"[Extracted] {spec['desc']} -> {out_path} ({clean_img.size[0]}x{clean_img.size[1]} px, mean={np.mean(img_arr):.1f})")
        
    return saved_paths


if __name__ == "__main__":
    extract_paper_images()
