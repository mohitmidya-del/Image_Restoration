import cv2
import numpy as np
import streamlit as st
from PIL import Image

from src.pipeline import run_restoration_pipeline, run_baseline_method
from src.metrics import calculate_mse, calculate_psnr

st.set_page_config(page_title="Old Photograph Restoration System", layout="wide")
# Hide the top-right Deploy button and Streamlit menu bar
st.markdown(
    """
    <style>
    .stAppDeployButton, [data-testid="stAppDeployButton"] {
        display: none !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Old Photograph Restoration System")
st.markdown(
    "**Classical Image Processing Pipeline** (`PCC-IT-502` Mini Project) — "
    "Denoising • Morphological Scratch Inpainting • CLAHE Contrast Enhancement • Unsharp Masking"
)

# --- Sidebar Controls (Live Parameter Tuning for Viva Demo) ---
st.sidebar.header("Pipeline Parameters")

median_ksize = st.sidebar.slider("1. Median Filter Kernel (Dust Removal)", 1, 9, 3, step=2)
bilateral_strength = st.sidebar.slider("2. Bilateral Filter Strength (Grain)", 10, 120, 50, step=5)
scratch_kernel = st.sidebar.slider("3. Scratch Detection Kernel Size", 3, 21, 9, step=2)
scratch_thresh = st.sidebar.slider("4. Scratch Sensitivity Threshold", 15, 100, 45, step=5)
inpaint_radius = st.sidebar.slider("5. Inpainting Radius", 1, 10, 3, step=1)
clahe_clip = st.sidebar.slider("6. CLAHE Contrast Limit", 1.0, 5.0, 2.0, step=0.2)
sharpen_strength = st.sidebar.slider("7. Unsharp Mask Strength", 0.0, 2.5, 1.2, step=0.1)

st.sidebar.markdown("---")
uploaded_file = st.file_uploader(
    "Upload a Damaged Old Photograph",
    type=["jpg", "jpeg", "png", "bmp", "webp", "avif", "tiff"],
)
reference_file = st.file_uploader(
    "Optional: Upload Clean Ground-Truth Image (for exact MSE & PSNR)",
    type=["jpg", "jpeg", "png", "bmp", "webp", "avif", "tiff"],
)


def decode_uploaded_image(file_obj) -> np.ndarray:
    """Converts any uploaded image (JPG, PNG, WEBP, AVIF, BMP) into an OpenCV BGR array."""
    pil_image = Image.open(file_obj).convert("RGB")
    rgb_array = np.array(pil_image)
    return cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)


def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    """Converts OpenCV BGR image to RGB so browser displays true colors."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


if uploaded_file is not None:
    input_bgr = decode_uploaded_image(uploaded_file)

    # Run Proposed Pipeline & Baseline Method
    results = run_restoration_pipeline(
        input_bgr,
        median_ksize=median_ksize,
        bilateral_strength=bilateral_strength,
        scratch_kernel=scratch_kernel,
        scratch_thresh=scratch_thresh,
        inpaint_radius=inpaint_radius,
        clahe_clip=clahe_clip,
        sharpen_strength=sharpen_strength,
    )
    baseline_bgr, baseline_time = run_baseline_method(input_bgr)

    # --- Section 1: Side-by-Side Before & After ---
    st.subheader("1. Before vs. After Restoration")
    col1, col2 = st.columns(2)
    with col1:
        st.image(bgr_to_rgb(results["original"]), caption="Original Input Image", use_container_width=True)
    with col2:
        st.image(bgr_to_rgb(results["final"]), caption="Final Restored Output", use_container_width=True)

    # Download button for restored image
    _, encoded_img = cv2.imencode(".png", results["final"])
    st.download_button(
        label="Download Restored Image (PNG)",
        data=encoded_img.tobytes(),
        file_name="restored_photo.png",
        mime="image/png",
    )

    # --- Section 2: Intermediate Pipeline Stages (Required for Report & Viva) ---
    st.subheader("2. Intermediate Image Processing Stages")
    r1_c1, r1_c2, r1_c3 = st.columns(3)
    with r1_c1:
        st.image(results["grayscale"], caption="Stage 1: Grayscale Conversion", use_container_width=True)
    with r1_c2:
        st.image(bgr_to_rgb(results["denoised"]), caption="Stage 2: Denoised (Median + Bilateral)", use_container_width=True)
    with r1_c3:
        st.image(results["scratch_mask"], caption="Stage 3a: Morphological Scratch Mask", use_container_width=True)

    r2_c1, r2_c2, r2_c3 = st.columns(3)
    with r2_c1:
        st.image(bgr_to_rgb(results["inpainted"]), caption="Stage 3b: Telea Inpainted Image", use_container_width=True)
    with r2_c2:
        st.image(bgr_to_rgb(results["enhanced"]), caption="Stage 4: Contrast Enhanced (CLAHE)", use_container_width=True)
    with r2_c3:
        st.image(results["edges"], caption="Stage 5: Laplacian Edge Map", use_container_width=True)

    # --- Section 3: Quantitative Performance Evaluation ---
    st.subheader("3. Quantitative Evaluation (Method 1 vs. Proposed Method)")
    ref_img = decode_uploaded_image(reference_file) if reference_file is not None else results["original"]
    ref_resized = cv2.resize(ref_img, (results["original"].shape[1], results["original"].shape[0]))

    base_mse = calculate_mse(ref_resized, baseline_bgr)
    base_psnr = calculate_psnr(ref_resized, baseline_bgr)
    prop_mse = calculate_mse(ref_resized, results["final"])
    prop_psnr = calculate_psnr(ref_resized, results["final"])

    st.table(
        [
            {
                "Method": "Method 1 (Basic Gaussian + HistEq)",
                "MSE": base_mse,
                "PSNR (dB)": base_psnr,
                "Processing Time (s)": baseline_time,
            },
            {
                "Method": "Proposed Method (Full Restoration Pipeline)",
                "MSE": prop_mse,
                "PSNR (dB)": prop_psnr,
                "Processing Time (s)": results["processing_time"],
            },
        ]
    )
else:
    st.info("Upload an old or damaged photograph above to start the restoration pipeline.")