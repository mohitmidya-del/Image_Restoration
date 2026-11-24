import cv2
import numpy as np


def calculate_mse(reference_img: np.ndarray, processed_img: np.ndarray) -> float:
    """
    Calculates Mean Squared Error (MSE) between two images of the same size.
    Lower MSE means closer pixel-level similarity to the clean reference image.
    Formula: MSE = (1 / (M * N)) * sum((I1 - I2)^2)
    """
    # Ensure both images have identical dimensions
    if reference_img.shape != processed_img.shape:
        processed_img = cv2.resize(
            processed_img, (reference_img.shape[1], reference_img.shape[0])
        )

    # Convert to float64 before squaring to prevent uint8 (0-255) overflow
    diff = reference_img.astype(np.float64) - processed_img.astype(np.float64)
    mse = float(np.mean(diff ** 2))
    return round(mse, 4)


def calculate_psnr(reference_img: np.ndarray, processed_img: np.ndarray) -> float:
    """
    Calculates Peak Signal-to-Noise Ratio (PSNR) in decibels (dB).
    Higher PSNR indicates better restoration quality (typically 25 dB - 40 dB).
    Formula: PSNR = 10 * log10((MAX_I^2) / MSE), where MAX_I = 255.
    """
    mse = calculate_mse(reference_img, processed_img)
    if mse == 0:
        return float("inf")

    max_pixel = 255.0
    psnr = 10.0 * np.log10((max_pixel ** 2) / mse)
    return round(float(psnr), 2)