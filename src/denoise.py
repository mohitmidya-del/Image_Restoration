import cv2
import numpy as np


def remove_salt_and_pepper(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """
    Applies a Median Filter to remove isolated white/black dust specks (salt-and-pepper noise).
    `kernel_size` must be an odd positive integer (e.g., 3 or 5).
    """
    if kernel_size % 2 == 0:
        kernel_size += 1
    return cv2.medianBlur(image, kernel_size)


def smooth_film_grain(
    image: np.ndarray, diameter: int = 7, sigma_color: int = 50, sigma_space: int = 50
) -> np.ndarray:
    """
    Applies a Bilateral Filter to smooth out film grain while keeping edges sharp.
    - diameter: Pixel neighborhood size.
    - sigma_color: How much color difference is allowed to be blurred together.
    - sigma_space: How far spatially pixels influence each other.
    """
    return cv2.bilateralFilter(image, diameter, sigma_color, sigma_space)


def denoise_image(
    image: np.ndarray, median_ksize: int = 3, bilateral_strength: int = 50
) -> np.ndarray:
    """
    Complete two-stage denoising function:
    1. Median filter eliminates sharp dust spots.
    2. Bilateral filter smooths grainy texture without blurring facial features.
    """
    step1_median = remove_salt_and_pepper(image, kernel_size=median_ksize)
    step2_bilateral = smooth_film_grain(
        step1_median,
        diameter=7,
        sigma_color=bilateral_strength,
        sigma_space=bilateral_strength,
    )
    return step2_bilateral