import time
import cv2
import numpy as np

from src.preprocess import resize_image, to_grayscale
from src.denoise import denoise_image
from src.restore import restore_damage
from src.enhance import apply_clahe
from src.sharpen import unsharp_mask, detect_edges_laplacian


def run_baseline_method(image: np.ndarray) -> tuple[np.ndarray, float]:
    """
    'Method 1' (Baseline) for Chapter 5 comparison table:
    Applies only basic Gaussian blur + standard Histogram Equalization.
    """
    start_time = time.perf_counter()

    resized = resize_image(image)
    blurred = cv2.GaussianBlur(resized, (5, 5), 0)
    ycrcb = cv2.cvtColor(blurred, cv2.COLOR_BGR2YCrCb)
    ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
    baseline_output = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)

    elapsed = round(time.perf_counter() - start_time, 4)
    return baseline_output, elapsed


def run_restoration_pipeline(
    image: np.ndarray,
    max_dim: int = 800,
    median_ksize: int = 3,
    bilateral_strength: int = 50,
    scratch_kernel: int = 9,
    scratch_thresh: int = 45,
    inpaint_radius: int = 3,
    clahe_clip: float = 2.0,
    sharpen_strength: float = 1.2,
) -> dict:
    """
    'Proposed Method' (Full 5-Stage Classical Restoration Pipeline).
    Returns a dictionary containing every intermediate image (for Chapter 5 report)
    and the total execution time in seconds.
    """
    start_time = time.perf_counter()

    # Stage 1: Preprocessing
    original = resize_image(image, max_dim=max_dim)
    grayscale = to_grayscale(original)

    # Stage 2: Noise Removal (Median + Bilateral)
    denoised = denoise_image(
        original,
        median_ksize=median_ksize,
        bilateral_strength=bilateral_strength,
    )

    # Stage 3: Scratch Detection & Inpainting
    inpainted, scratch_mask = restore_damage(
        denoised,
        kernel_size=scratch_kernel,
        threshold_val=scratch_thresh,
        inpaint_radius=inpaint_radius,
    )

    # Stage 4: Contrast Enhancement (CLAHE on LAB L-channel)
    enhanced = apply_clahe(inpainted, clip_limit=clahe_clip)

    # Stage 5: Edge Sharpening (Unsharp Masking)
    final_output = unsharp_mask(enhanced, sigma=1.0, strength=sharpen_strength)
    edges = detect_edges_laplacian(final_output)

    elapsed_time = round(time.perf_counter() - start_time, 4)

    return {
        "original": original,
        "grayscale": grayscale,
        "denoised": denoised,
        "scratch_mask": scratch_mask,
        "inpainted": inpainted,
        "enhanced": enhanced,
        "edges": edges,
        "final": final_output,
        "processing_time": elapsed_time,
    }