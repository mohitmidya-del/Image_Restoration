import cv2
import numpy as np
from src.preprocess import bgr_to_lab, lab_to_bgr


def apply_clahe(
    image: np.ndarray, clip_limit: float = 2.0, tile_grid_size: tuple[int, int] = (8, 8)
) -> np.ndarray:
    """
    Enhances contrast of faded old photographs using CLAHE
    (Contrast Limited Adaptive Histogram Equalization) on the L (Lightness) channel.
    """
    # Split BGR into Lightness (L) and Color (A, B) channels
    l_channel, a_channel, b_channel = bgr_to_lab(image)

    # Create CLAHE object and apply ONLY to the Lightness channel
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    enhanced_l = clahe.apply(l_channel)

    # Merge enhanced Lightness back with original colors
    return lab_to_bgr(enhanced_l, a_channel, b_channel)


def stretch_contrast(image: np.ndarray) -> np.ndarray:
    """
    Performs linear intensity normalization (histogram stretching)
    so the darkest pixel becomes 0 and the brightest becomes 255.
    """
    return cv2.normalize(image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)