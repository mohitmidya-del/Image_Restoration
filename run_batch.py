import os
import glob
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.preprocess import load_image
from src.pipeline import run_restoration_pipeline, run_baseline_method
from src.metrics import calculate_mse, calculate_psnr


def create_synthetic_sample(save_path: str) -> None:
    """
    Creates a realistic sample damaged old photo inside dataset/original/
    if the folder is empty, so you can test the entire project immediately.
    """
    # Create a warm vintage portrait-style synthetic scene
    img = np.zeros((400, 500, 3), dtype=np.uint8)
    img[:] = (145, 165, 185)  # Faded sepia background
    cv2.rectangle(img, (60, 50), (440, 350), (120, 140, 160), -1)
    cv2.circle(img, (250, 175), 65, (150, 175, 200), -1)  # Head silhouette
    cv2.ellipse(img, (250, 320), (110, 75), 0, 180, 360, (110, 125, 145), -1)  # Shoulders

    # Save clean reference version in ground_truth
    gt_path = os.path.join("dataset", "ground_truth", os.path.basename(save_path))
    cv2.imwrite(gt_path, img)

    # Add synthetic salt-and-pepper dust + Gaussian film grain
    rng = np.random.default_rng(42)
    noise = rng.normal(0, 12, img.shape).astype(np.int16)
    damaged = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # Add white scratches and fold creases
    cv2.line(damaged, (90, 40), (210, 360), (245, 245, 245), 2)
    cv2.line(damaged, (320, 70), (390, 290), (250, 250, 250), 2)
    cv2.line(damaged, (140, 220), (410, 240), (240, 240, 240), 1)

    cv2.imwrite(save_path, damaged)


def save_comparison_grid(results: dict, save_path: str) -> None:
    """
    Saves a clean 2x3 figure showing Original, Intermediate Steps, and Final Output
    for direct inclusion in Chapter 5 of your Mini Project Report and PPT.
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    panels = [
        (cv2.cvtColor(results["original"], cv2.COLOR_BGR2RGB), "1. Original Damaged Input", None),
        (cv2.cvtColor(results["denoised"], cv2.COLOR_BGR2RGB), "2. Denoised (Median + Bilateral)", None),
        (results["scratch_mask"], "3. Morphological Scratch Mask", "gray"),
        (cv2.cvtColor(results["inpainted"], cv2.COLOR_BGR2RGB), "4. Inpainted (Telea Restoration)", None),
        (cv2.cvtColor(results["enhanced"], cv2.COLOR_BGR2RGB), "5. Contrast Enhanced (CLAHE)", None),
        (cv2.cvtColor(results["final"], cv2.COLOR_BGR2RGB), "6. Final Restored & Sharpened", None),
    ]

    for ax, (img, title, cmap) in zip(axes.flat, panels):
        ax.imshow(img, cmap=cmap)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close(fig)


def main():
    input_dir = os.path.join("dataset", "original")
    gt_dir = os.path.join("dataset", "ground_truth")
    output_dir = os.path.join("dataset", "results")
    os.makedirs(output_dir, exist_ok=True)

    # Find all images in dataset/original
    extensions = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
    image_paths = []
    for ext in extensions:
        image_paths.extend(glob.glob(os.path.join(input_dir, ext)))

    # If user hasn't added photos yet, create 1 synthetic test photo automatically
    if not image_paths:
        sample_path = os.path.join(input_dir, "sample_damaged_photo.png")
        create_synthetic_sample(sample_path)
        image_paths = [sample_path]
        print(f"[INFO] Created synthetic test image at: {sample_path}")

    metrics_rows = []

    for img_path in sorted(image_paths):
        filename = os.path.basename(img_path)
        name_no_ext = os.path.splitext(filename)[0]
        print(f"Processing: {filename}...")

        raw_img = load_image(img_path)
        results = run_restoration_pipeline(raw_img)
        baseline_img, baseline_time = run_baseline_method(raw_img)

        # Save final restored image and 6-panel report grid
        cv2.imwrite(os.path.join(output_dir, f"{name_no_ext}_restored.png"), results["final"])
        save_comparison_grid(results, os.path.join(output_dir, f"{name_no_ext}_grid.png"))

        # Check if a matching clean ground-truth image exists; otherwise compare to original
        gt_path = os.path.join(gt_dir, filename)
        if os.path.exists(gt_path):
            ref_img = cv2.resize(load_image(gt_path), (results["original"].shape[1], results["original"].shape[0]))
        else:
            ref_img = results["original"]

        metrics_rows.append(
            {
                "Image": filename,
                "Method 1 MSE": calculate_mse(ref_img, baseline_img),
                "Proposed MSE": calculate_mse(ref_img, results["final"]),
                "Method 1 PSNR (dB)": calculate_psnr(ref_img, baseline_img),
                "Proposed PSNR (dB)": calculate_psnr(ref_img, results["final"]),
                "Method 1 Time (s)": baseline_time,
                "Proposed Time (s)": results["processing_time"],
            }
        )

    df = pd.DataFrame(metrics_rows)
    csv_path = os.path.join(output_dir, "evaluation_metrics.csv")
    df.to_csv(csv_path, index=False)

    print("\n--- Evaluation Results Summary ---")
    print(df.to_string(index=False))
    print(f"\n[DONE] All comparison grids and metrics saved in: {output_dir}")


if __name__ == "__main__":
    main()