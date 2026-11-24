import cv2
import numpy as np


def load_image(image_path: str) -> np.ndarray:
    """
    Loads an image from disk using OpenCV.
    Note: OpenCV loads color images in BGR (Blue-Green-Red) order by default.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not load image at path: {image_path}")
    return image


def resize_image(image: np.ndarray, max_dim: int = 800) -> np.ndarray:
    """
    Resizes an image proportionally so its largest side equals `max_dim`.
    Prevents massive 4K scans from slowing down the filters while keeping aspect ratio.
    """
    height, width = image.shape[:2]

    # Only resize if the image is larger than max_dim
    if max(height, width) <= max_dim:
        return image

    if width > height:
        scale = max_dim / float(width)
    else:
        scale = max_dim / float(height)

    new_width = int(width * scale)
    new_height = int(height * scale)

    # INTER_AREA is the best interpolation method for shrinking images cleanly
    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    return resized


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Converts a BGR image into a single-channel Grayscale image."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def bgr_to_lab(image: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Converts a BGR image to LAB color space and splits it into 3 channels:
    - L: Lightness / Brightness (0 to 255)
    - A: Green-to-Red color component
    - B: Blue-to-Yellow color component
    """
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab_image)
    return l_channel, a_channel, b_channel


def lab_to_bgr(
    l_channel: np.ndarray, a_channel: np.ndarray, b_channel: np.ndarray
) -> np.ndarray:
    """Merges L, A, B channels back together and converts back to BGR."""
    merged_lab = cv2.merge([l_channel, a_channel, b_channel])
    return cv2.cvtColor(merged_lab, cv2.COLOR_LAB2BGR)