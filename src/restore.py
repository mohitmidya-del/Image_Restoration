import cv2
import numpy as np


def detect_scratches(
    image: np.ndarray, kernel_size: int = 9, threshold_val: int = 45
) -> np.ndarray:
    """
    Automatically detects thin bright scratches, creases, and dust lines
    using a Morphological Top-Hat transform followed by binary thresholding.
    Returns a binary mask (white = scratch to repair, black = healthy image).
    """
    # Convert to grayscale if a 3-channel color image is passed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Create a morphological structuring element (kernel)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    # Top-Hat = Original Image - Morphological Opening
    # Isolates bright thin structures (scratches) that are smaller than the kernel
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)

    # Convert the Top-Hat intensities into a clean binary mask (0 or 255)
    _, scratch_mask = cv2.threshold(tophat, threshold_val, 255, cv2.THRESH_BINARY)

    # Slightly dilate the mask so the edges of each scratch are fully covered
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    scratch_mask = cv2.dilate(scratch_mask, dilate_kernel, iterations=1)

    return scratch_mask


def inpaint_scratches(
    image: np.ndarray, scratch_mask: np.ndarray, inpaint_radius: int = 3
) -> np.ndarray:
    """
    Fills in the masked scratch areas using neighboring pixel textures
    via the Telea Fast Marching inpainting algorithm.
    """
    return cv2.inpaint(image, scratch_mask, inpaint_radius, cv2.INPAINT_TELEA)


def restore_damage(
    image: np.ndarray,
    kernel_size: int = 9,
    threshold_val: int = 45,
    inpaint_radius: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Complete structural restoration step:
    1. Detects scratches into a binary mask (saved for intermediate report visuals).
    2. Inpaints the detected scratches.
    Returns both (restored_image, scratch_mask).
    """
    mask = detect_scratches(
        image, kernel_size=kernel_size, threshold_val=threshold_val
    )
    restored = inpaint_scratches(image, mask, inpaint_radius=inpaint_radius)
    return restored, mask