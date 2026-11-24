import cv2
import numpy as np
import streamlit as st
from PIL import Image

from src.pipeline import run_restoration_pipeline, run_baseline_method
from src.metrics import calculate_mse, calculate_psnr

st.set_page_config(page_title="Old Photograph Restoration System", layout="wide")

# --- Custom CSS: Hide Deploy Button, Add Image Borders, Hover Effects & Animations ---
st.markdown(
    """
    <style>
    /* Hide the top-right Deploy button */
    .stAppDeployButton, [data-testid="stAppDeployButton"] {
        display: none !important;
    }

    /* Make top header transparent so the grid & green glow extend seamlessly to the top */
    [data-testid="stHeader"] {
        background: transparent !important;
    }

    [data-testid="stAppViewContainer"] {
        background-color: #040812 !important;
        background-image:
            radial-gradient(circle 360px at 50% 15%, rgba(59, 130, 246, 0.22), transparent 75%),
            repeating-radial-gradient(circle at 50% 35%, transparent 0px, transparent 48px, rgba(59, 130, 246, 0.05) 49px, transparent 50px);
        background-attachment: fixed;
    }

    /* Darker translucent sidebar so it contrasts cleanly against the grid */
    [data-testid="stSidebar"] {
        background-color: rgba(11, 15, 25, 0.92) !important;
        border-right: 1px solid rgba(16, 185, 129, 0.15);
    }

    @keyframes fadeSlideUp {
        from { opacity: 0; transform: translateY(12px); }
        to   { opacity: 1; transform: translateY(0); }
    }

    [data-testid="stImage"] img {
        border: 2px solid rgba(16, 185, 129, 0.45);
        border-radius: 12px;
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.45);
        transition: transform 0.25s ease, border-color 0.25s ease, box-shadow 0.25s ease;
        animation: fadeSlideUp 0.5s ease-out;
    }

    [data-testid="stImage"] img:hover {
        transform: scale(1.015);
        border-color: #10B981;
        box-shadow: 0 0 18px rgba(16, 185, 129, 0.45);
    }

    .stButton > button, .stDownloadButton > button {
        border: 1px solid #10B981 !important;
        border-radius: 8px !important;
        transition: all 0.2s ease !important;
    }

    .stButton > button:hover, .stDownloadButton > button:hover {
        background-color: #10B981 !important;
        color: #06090F !important;
        font-weight: 600 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Session State for Resettable Sliders ---
DEFAULTS = {
    "median_ksize": 3,
    "bilateral_strength": 50,
    "scratch_kernel": 9,
    "scratch_thresh": 45,
    "inpaint_radius": 3,
    "clahe_clip": 2.0,
    "sharpen_strength": 1.2,
}

for key, val in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = val


def reset_parameters():
    for key, val in DEFAULTS.items():
        st.session_state[key] = val


# --- Header ---
st.title("Old Photograph Restoration System")
st.markdown(
    "**Classical Image Processing Pipeline** (`PCC-IT-502` Mini Project) — "
    "Denoising • Morphological Scratch Inpainting • CLAHE Contrast Enhancement • Unsharp Masking"
)

# --- Sidebar Controls ---
st.sidebar.header("Pipeline Parameters")
st.sidebar.button("↺ Reset Parameters", on_click=reset_parameters, use_container_width=True)
st.sidebar.markdown("---")

median_ksize = st.sidebar.slider("1. Median Filter Kernel (Dust Removal)", 1, 9, step=2, key="median_ksize")
bilateral_strength = st.sidebar.slider("2. Bilateral Filter Strength (Grain)", 10, 120, step=5, key="bilateral_strength")
scratch_kernel = st.sidebar.slider("3. Scratch Detection Kernel Size", 3, 21, step=2, key="scratch_kernel")
scratch_thresh = st.sidebar.slider("4. Scratch Sensitivity Threshold", 15, 100, step=5, key="scratch_thresh")
inpaint_radius = st.sidebar.slider("5. Inpainting Radius", 1, 10, step=1, key="inpaint_radius")
clahe_clip = st.sidebar.slider("6. CLAHE Contrast Limit", 1.0, 5.0, step=0.2, key="clahe_clip")
sharpen_strength = st.sidebar.slider("7. Unsharp Mask Strength", 0.0, 2.5, step=0.1, key="sharpen_strength")

# --- File Uploaders ---
uploaded_file = st.file_uploader(
    "Upload a Damaged Old Photograph",
    type=["jpg", "jpeg", "png", "bmp", "webp", "avif", "tiff"],
)
reference_file = st.file_uploader(
    "Optional: Upload Clean Ground-Truth Image (for exact MSE & PSNR)",
    type=["jpg", "jpeg", "png", "bmp", "webp", "avif", "tiff"],
)


def decode_uploaded_image(file_obj) -> np.ndarray:
    pil_image = Image.open(file_obj).convert("RGB")
    rgb_array = np.array(pil_image)
    return cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)


def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


if uploaded_file is not None:
    input_bgr = decode_uploaded_image(uploaded_file)

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

    tab1, tab2, tab3 = st.tabs(
        ["Before vs. After", "6 Intermediate Stages", "MSE & PSNR Evaluation"]
    )

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            st.image(bgr_to_rgb(results["original"]), caption="Original Input Image", use_container_width=True)
        with col2:
            st.image(bgr_to_rgb(results["final"]), caption="Final Restored Output", use_container_width=True)

        _, encoded_img = cv2.imencode(".png", results["final"])
        st.download_button(
            label="⬇ Download Restored Image (PNG)",
            data=encoded_img.tobytes(),
            file_name="restored_photo.png",
            mime="image/png",
        )

    with tab2:
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

    with tab3:
        ref_img = decode_uploaded_image(reference_file) if reference_file is not None else results["original"]
        ref_resized = cv2.resize(ref_img, (results["original"].shape[1], results["original"].shape[0]))

        base_mse = calculate_mse(ref_resized, baseline_bgr)
        base_psnr = calculate_psnr(ref_resized, baseline_bgr)
        prop_mse = calculate_mse(ref_resized, results["final"])
        prop_psnr = calculate_psnr(ref_resized, results["final"])

        st.dataframe(
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
            ],
            use_container_width=True,
        )
else:
    st.info("Upload an old or damaged photograph above to start the restoration pipeline.")