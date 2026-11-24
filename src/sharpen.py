import cv2
import numpy as np


def unsharp_mask(
    image: np.ndarray,
    sigma: float = 1.0,
    strength: float = 1.2,
    threshold: int = 0,
) -> np.ndarray:
    """
    Sharpens an image using classical Unsharp Masking:
    Sharpened = Original + strength * (Original - Blurred)

    - sigma: Standard deviation for Gaussian blur (controls edge thickness).
    - strength: How strongly edges are boosted (1.0 - 1.5 is natural).
    - threshold: Minimum brightness difference required to sharpen (avoids sharpening flat noise).
    """
    # Create a softly blurred version of the image
    blurred = cv2.GaussianBlur(image, (0, 0), sigma)

    # Blend original and blurred: Original * (1 + strength) + Blurred * (-strength)
    sharpened = cv2.addWeighted(image, 1.0 + strength, blurred, -strength, 0)

    # Optional threshold check so smooth skin/sky areas don't get grainy
    if threshold > 0:
        low_contrast_mask = np.absolute(image.astype(np.int16) - blurred.astype(np.int16)) < threshold
        np.copyto(sharpened, image, where=low_contrast_mask)

    return sharpened


def detect_edges_laplacian(image: np.ndarray) -> np.ndarray:
    """
    Extracts edge details using the Laplacian operator (useful for intermediate
    edge visualization in Chapter 5 of the project report).
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    return cv2.convertScaleAbs(laplacian)