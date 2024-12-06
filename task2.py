import cv2
import numpy as np
import glob
import os

frei_chen_masks = [
    np.array([[1, np.sqrt(2), 1], [0, 0, 0], [-1, -np.sqrt(2), -1]]),  # M1
    np.array([[1, 0, -1], [np.sqrt(2), 0, -np.sqrt(2)], [1, 0, -1]]),  # M2
    np.array([[0, 1, -np.sqrt(2)], [1, 0, -1], [-np.sqrt(2), -1, 0]]),  # M3
    np.array([[np.sqrt(2), -1, 0], [-1, 0, 1], [0, 1, np.sqrt(2)]]),  # M4
    np.array([[0, 1, 0], [1, 0, -1], [0, -1, 0]]),  # M5
    np.array([[-1, 0, 1], [0, 0, 0], [1, 0, -1]]),  # M6
    np.array([[1, -2, 1], [4, 0, -4], [1, -2, 1]]),  # M7
    np.array([[-2, 1, -2], [1, 4, 1], [-2, 1, -2]]),  # M8
    np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])  # M9
]

kirsch_masks = {
    "N": np.array([[-3, -3, 5], [-3, 0, 5], [-3, -3, 5]]),
    "W": np.array([[-3, -3, -3], [-3, 0, 5], [5, 5, 5]]),
    "S": np.array([[-3, -3, -3], [-3, 0, -3], [5, 5, 5]]),
    "E": np.array([[5, 5, 5], [-3, 0, -3], [-3, -3, -3]]),
    "NW": np.array([[5, -3, -3], [5, 0, -3], [5, -3, -3]]),
    "SW": np.array([[-3, -3, -3], [5, 0, -3], [5, 5, -3]]),
    "SE": np.array([[-3, 0, -3], [-3, 0, -3], [5, 5, 5]]),
    "NE": np.array([[-3, 0, 5], [-3, 0, 5], [5, 5, 5]])
}

nevatia_babu_masks = [
    np.array([[100, 100, 100, 100, 100], [100, 100, 100, 100, 100], [0, 0, 0, 0, 0], [-100, -100, -100, -100, -100],
              [-100, -100, -100, -100, -100]])  # 0°
]

output_dir = "output_edges"
os.makedirs(output_dir, exist_ok=True)

image_files = sorted(glob.glob("tumor*.png"))

if not image_files:
    print("No images found. Check the file names and directory.")
    exit()

for img_file in image_files:
    # Load image
    img = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Could not read {img_file}. Skipping.")
        continue

    # Apply Kirsch Masks
    kirsch_edges = np.zeros_like(img, dtype=np.float32)
    for mask in kirsch_masks.values():
        filtered = cv2.filter2D(img, cv2.CV_32F, mask)
        kirsch_edges = np.maximum(kirsch_edges, filtered)

    # Apply Frei and Chen Masks
    frei_chen_edges = np.zeros_like(img, dtype=np.float32)
    for mask in frei_chen_masks:
        filtered = cv2.filter2D(img, cv2.CV_32F, mask)
        frei_chen_edges = np.maximum(frei_chen_edges, filtered)

    # Apply Nevatia and Babu Masks
    nevatia_babu_edges = np.zeros_like(img, dtype=np.float32)
    for mask in nevatia_babu_masks:
        filtered = cv2.filter2D(img, cv2.CV_32F, mask)
        nevatia_babu_edges = np.maximum(nevatia_babu_edges, filtered)

    # Combine results
    combined_edges = kirsch_edges + frei_chen_edges + nevatia_babu_edges
    combined_edges = cv2.normalize(combined_edges, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Apply Gaussian and Median filtering for noise cleanup
    noise_cleaned = cv2.GaussianBlur(combined_edges, (3, 3), 0)
    noise_cleaned = cv2.medianBlur(noise_cleaned, 3)

    # Apply threshold to create binary mask
    _, binary_mask = cv2.threshold(noise_cleaned, 50, 255, cv2.THRESH_BINARY)

    # Save results
    base_name = os.path.basename(img_file).split('.')[0]
    cv2.imwrite(os.path.join(output_dir, f"{base_name}_combined.jpg"), combined_edges)
    cv2.imwrite(os.path.join(output_dir, f"{base_name}_cleaned.jpg"), noise_cleaned)
    cv2.imwrite(os.path.join(output_dir, f"{base_name}_binary.jpg"), binary_mask)

print(f"Edge detection with noise cleanup completed. Results saved in '{output_dir}'.")
