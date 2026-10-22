"""
CAMMP-SFSE: Quantification Module

Compute global metrics (Void/Surface/Mixed Area Fractions, Void Number
Density) and per-region morphometrics (Area, Perimeter, Equivalent
Circular Diameter, Aspect Ratio, Circularity), exporting clean CSV
reports without NaN/Inf values.
"""

import numpy as np
import pandas as pd
from skimage.measure import regionprops
import os


def compute_region_morphometrics(label_map: np.ndarray,
                                 classifications: dict,
                                 void_fractions: dict,
                                 scale: float = 1.0) -> pd.DataFrame:
    """Compute per-region morphometric measurements.
    
    Parameters
    ----------
    label_map : np.ndarray
        2D int32 region label map.
    classifications : dict
        Region label -> class string.
    void_fractions : dict
        Region label -> void fraction.
    scale : float
        Spatial calibration (um/px). Default 1.0 for pixel units.
    
    Returns
    -------
    pd.DataFrame
        Per-region metrics with columns:
        Region_ID, Class, Void_Fraction, Area_px, Area_um2,
        Perimeter_px, Perimeter_um, ECD_px, ECD_um,
        Aspect_Ratio, Circularity
    """
    props = regionprops(label_map)
    
    records = []
    for rp in props:
        label = rp.label
        area_px = rp.area
        perimeter_px = rp.perimeter
        
        # Equivalent Circular Diameter: sqrt(4 * A / pi)
        ecd_px = np.sqrt(4.0 * area_px / np.pi)
        
        # Aspect Ratio from fitted ellipse
        major_len = rp.axis_major_length if hasattr(rp, "axis_major_length") else getattr(rp, "major_axis_length", 0.0)
        minor_len = rp.axis_minor_length if hasattr(rp, "axis_minor_length") else getattr(rp, "minor_axis_length", 0.0)
        if major_len > 0 and minor_len > 0:
            aspect_ratio = major_len / minor_len
        else:
            aspect_ratio = 1.0
        
        # Circularity: (4 * pi * A) / P^2
        if perimeter_px > 0:
            circularity = (4.0 * np.pi * area_px) / (perimeter_px ** 2)
        else:
            circularity = 0.0
        
        # Sanitize: clamp circularity to [0, 1] range
        circularity = float(np.clip(circularity, 0.0, 1.0))
        
        # Scale conversions
        area_um2 = area_px * (scale ** 2)
        perimeter_um = perimeter_px * scale
        ecd_um = ecd_px * scale
        
        cls = classifications.get(label, "Unknown")
        f_void = void_fractions.get(label, 0.0)
        
        records.append({
            "Region_ID": int(label),
            "Class": cls,
            "Void_Fraction": round(float(f_void), 4),
            "Area_px": int(area_px),
            "Area_um2": round(float(area_um2), 4),
            "Perimeter_px": round(float(perimeter_px), 2),
            "Perimeter_um": round(float(perimeter_um), 4),
            "ECD_px": round(float(ecd_px), 2),
            "ECD_um": round(float(ecd_um), 4),
            "Aspect_Ratio": round(float(aspect_ratio), 4),
            "Circularity": round(float(circularity), 4),
        })
    
    df = pd.DataFrame(records)
    
    # Sanitize: replace any remaining NaN/Inf with 0
    df = df.replace([np.inf, -np.inf], 0.0)
    df = df.fillna(0.0)
    
    return df


def compute_summary_metrics(df_regions: pd.DataFrame,
                            image_area_px: int,
                            scale: float = 1.0) -> pd.DataFrame:
    """Compute global fractographic summary metrics.
    
    Parameters
    ----------
    df_regions : pd.DataFrame
        Per-region metrics from compute_region_morphometrics.
    image_area_px : int
        Total image area in pixels (excluding boundaries).
    scale : float
        Spatial calibration (um/px).
    
    Returns
    -------
    pd.DataFrame
        Summary metrics DataFrame (single row).
    """
    total_area_px = int(image_area_px)
    total_area_um2 = total_area_px * (scale ** 2)
    
    # Class-wise area sums
    void_area = int(df_regions.loc[
        df_regions["Class"] == "Void", "Area_px"
    ].sum()) if len(df_regions) > 0 else 0
    
    surface_area = int(df_regions.loc[
        df_regions["Class"] == "Surface", "Area_px"
    ].sum()) if len(df_regions) > 0 else 0
    
    mixed_area = int(df_regions.loc[
        df_regions["Class"] == "Mixed", "Area_px"
    ].sum()) if len(df_regions) > 0 else 0
    
    classified_total = void_area + surface_area + mixed_area
    
    if classified_total > 0:
        af_void = void_area / classified_total
        af_surface = surface_area / classified_total
        af_mixed = mixed_area / classified_total
    else:
        af_void = af_surface = af_mixed = 0.0
    
    # Void counts and density
    n_void = int((df_regions["Class"] == "Void").sum()) if len(df_regions) > 0 else 0
    n_surface = int((df_regions["Class"] == "Surface").sum()) if len(df_regions) > 0 else 0
    n_mixed = int((df_regions["Class"] == "Mixed").sum()) if len(df_regions) > 0 else 0
    n_total = len(df_regions)
    
    if total_area_um2 > 0:
        void_density = n_void / total_area_um2
    else:
        void_density = 0.0
    
    summary = pd.DataFrame([{
        "Total_Regions": n_total,
        "Void_Count": n_void,
        "Surface_Count": n_surface,
        "Mixed_Count": n_mixed,
        "Image_Area_px": total_area_px,
        "Image_Area_um2": round(float(total_area_um2), 4),
        "AF_Void": round(float(af_void), 6),
        "AF_Surface": round(float(af_surface), 6),
        "AF_Mixed": round(float(af_mixed), 6),
        "Void_Number_Density_per_um2": round(float(void_density), 8),
        "Scale_um_per_px": round(float(scale), 6),
    }])
    
    # Sanitize
    summary = summary.replace([np.inf, -np.inf], 0.0)
    summary = summary.fillna(0.0)
    
    return summary


def export_csv(df: pd.DataFrame, path: str) -> str:
    """Export a DataFrame to CSV, creating directories if needed.
    
    Parameters
    ----------
    df : pd.DataFrame
        Data to export.
    path : str
        Output file path.
    
    Returns
    -------
    str
        Absolute path to the saved CSV.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    
    # Final sanitization pass
    df = df.replace([np.inf, -np.inf], 0.0)
    df = df.fillna(0.0)
    
    df.to_csv(path, index=False)
    return os.path.abspath(path)
