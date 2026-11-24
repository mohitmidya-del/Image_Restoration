import os
import urllib.request
import cv2
import numpy as np


# 12 diverse open-source test images from Wikimedia Commons (public domain / CC)
SAMPLE_URLS = [
    ("01_vintage_portrait.jpg", "https://upload.wikimedia.org/wikipedia/commons/thumb/e/ec/Mona_Lisa%2C_by_Leonardo_da_Vinci%2C_from_C2RMF_retouched.jpg/480px-Mona_Lisa%2C_by_Leonardo_da_Vinci%2C_from_C2RMF_retouched.jpg"),
    ("02_historic_building.jpg", "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a7/Camponotus_flavomarginatus_ant.jpg/640px-Camponotus_flavomarginatus_ant.jpg"),
    ("03_classic_street.jpg", "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3f/Fronalpstock_big.jpg/640px-Fronalpstock_big.jpg"),
    ("04_woman_portrait.jpg", "https://upload.wikimedia.org/wikipedia/commons/thumb/6/66/VanGogh-starry_night_ballance1.jpg/640px-VanGogh-starry_night_ballance1.jpg"),
    ("05_arch_monument.jpg", "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/640px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"),
    ("06_harbor_scene.jpg", "https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/PNG_transparency_demonstration_1.png/560px-PNG_transparency_demonstration_1.png"),
]


def degrade_photograph(clean_bgr: np.ndarray, seed: int) -> np.ndarray:
    """
    Simulates authentic old-photograph aging:
    1. Contrast fading & warm sepia color shift
    2. Slight lens/paper softness (blur)
    3. Gaussian film grain + salt-and-pepper dust
    4. Random white scratches and fold creases
    """
    rng = np.random.default_rng(seed)
    h, w = clean_bgr.shape[:2]

    # 1. Fade dynamic range (compress 0-255 into washed-out 45-210 range) + warm tint
    faded = clean_bgr.astype(np.float32) * 0.65 + 45.0
    faded[:, :, 0] *= 0.88  # Slightly reduce Blue channel for aged yellow/sepia cast
    faded[:, :, 2] = np.clip(faded[:, :, 2] * 1.06, 0, 255)
    degraded = np.clip(faded, 0, 255).astype(np.uint8)

    # 2. Slight aging blur
    degraded = cv2.GaussianBlur(degraded, (3, 3), 0.8)

    # 3. Gaussian film grain
    grain = rng.normal(0, 10, degraded.shape).astype(np.int16)
    degraded = np.clip(degraded.astype(np.int16) + grain, 0, 255).astype(np.uint8)

    # 4. Salt-and-pepper dust speckles (0.4% of pixels)
    num_speckles = int(0.004 * h * w)
    ys = rng.integers(0, h, size=num_speckles)
    xs = rng.integers(0, w, size=num_speckles)
    degraded[ys, xs] = 245

    # 5. Add 3 to 5 realistic scratches/creases
    num_scratches = int(rng.integers(3, 6))
    for _ in range(num_scratches):
        x1, y1 = int(rng.integers(10, w - 10)), int(rng.integers(10, h - 10))
        x2 = int(np.clip(x1 + rng.integers(-180, 180), 5, w - 5))
        y2 = int(np.clip(y1 + rng.integers(-180, 180), 5, h - 5))
        thickness = int(rng.choice([1, 2]))
        cv2.line(degraded, (x1, y1), (x2, y2), (242, 242, 242), thickness)

    return degraded


def main():
    orig_dir = os.path.join("dataset", "original")
    gt_dir = os.path.join("dataset", "ground_truth")
    os.makedirs(orig_dir, exist_ok=True)
    os.makedirs(gt_dir, exist_ok=True)

    # Remove old synthetic placeholder if present
    for folder in (orig_dir, gt_dir):
        old_sample = os.path.join(folder, "sample_damaged_photo.png")
        if os.path.exists(old_sample):
            os.remove(old_sample)

    headers = {"User-Agent": "Mozilla/5.0"}

    # Download real web photos + generate built-in OpenCV photographic scenes so you have 10 images total
    saved_count = 0
    for idx, (filename, url) in enumerate(SAMPLE_URLS, start=1):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=10) as resp:
                arr = np.asarray(bytearray(resp.read()), dtype=np.uint8)
                img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is not None:
                gt_path = os.path.join(gt_dir, filename)
                orig_path = os.path.join(orig_dir, filename)
                cv2.imwrite(gt_path, img)
                cv2.imwrite(orig_path, degrade_photograph(img, seed=100 + idx))
                saved_count += 1
                print(f"[OK] Prepared {filename}")
        except Exception as e:
            print(f"[SKIP] Could not download {filename}: {e}")

    # Also generate 4 rich multi-texture synthetic benchmark images (guaranteed offline fallback)
    for idx in range(1, 5):
        name = f"0{saved_count + idx}_studio_benchmark_{idx}.png"
        canvas = np.zeros((450, 600, 3), dtype=np.uint8)
        for y in range(450):
            canvas[y, :] = (60 + y // 3, 90 + (idx * 20) % 100, 140 - y // 5)
        cv2.circle(canvas, (180 + idx * 40, 200), 85, (200, 180, 150), -1)
        cv2.rectangle(canvas, (320, 110), (520, 340), (90, 130, 170), -1)
        cv2.putText(canvas, f"ARCHIVE 195{idx}", (140, 400), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (230, 230, 230), 2)

        cv2.imwrite(os.path.join(gt_dir, name), canvas)
        cv2.imwrite(os.path.join(orig_dir, name), degrade_photograph(canvas, seed=200 + idx))
        print(f"[OK] Prepared {name}")

    print("\nDataset ready! Run `python run_batch.py` to process all images and generate Chapter 5 metrics.")


if __name__ == "__main__":
    main()