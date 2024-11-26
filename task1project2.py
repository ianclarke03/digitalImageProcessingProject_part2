import cv2
import numpy as np
import matplotlib.pyplot as plt

def reduce_contrast(image, factor=0.5):
    
    return np.clip(image * factor, 0, 255).astype(np.uint8)

def linear_stretching(image, bit_depth=8):
    
    min_val = np.min(image)
    max_val = np.max(image)
    max_intensity = (2 ** bit_depth) - 1
    
    stretched_image = (max_intensity * (image - min_val) / (max_val - min_val)).astype(np.uint8)
    return stretched_image

def piecewise_linear_stretching(image, t):
    # convert image to float for calculations
    image = image.astype(np.float32)
    min_val = np.min(image)
    max_val = np.max(image)
    
    # apply piecewise formula
    stretched_image = np.where(
        image <= t,
        (image - min_val) / (t - min_val) * t,
        ((image - t) / (max_val - t) * (max_val - t) + t)
    )
    # clip values and convert back to uint8
    return np.clip(stretched_image, 0, 255).astype(np.uint8)


def calculate_eme(image, k1, k2, c=0.01):
    h, w = image.shape
    block_h, block_w = h // k1, w // k2
    
    eme = 0
    for i in range(0, h, block_h):
        for j in range(0, w, block_w):
            block = image[i:i + block_h, j:j + block_w]
            Iw_max = np.max(block)
            Iw_min = np.min(block)
            eme += np.log((Iw_max + c) / (Iw_min + c))
    
    eme /= (k1 * k2)
    return eme

image = cv2.imread('/Users/berg/Desktop/project2dig/Te-meTr_0004.jpg', cv2.IMREAD_GRAYSCALE)

reduced_contrast_image = reduce_contrast(image)

linear_stretched_image = linear_stretching(reduced_contrast_image)

threshold = 128  # example threshold
piecewise_stretched_image = piecewise_linear_stretching(reduced_contrast_image, threshold)

k1, k2 = 4, 4  # number of blocks
eme_linear = calculate_eme(linear_stretched_image, k1, k2)
eme_piecewise = calculate_eme(piecewise_stretched_image, k1, k2)

hist_eq_image = cv2.equalizeHist(reduced_contrast_image)
eme_hist_eq = calculate_eme(hist_eq_image, k1, k2)

print(f"EME (Linear Stretching): {eme_linear}")
print(f"EME (Piecewise Linear Stretching): {eme_piecewise}")
print(f"EME (Histogram Equalization): {eme_hist_eq}")

plt.figure(figsize=(15, 10))
plt.subplot(2, 2, 1)
plt.title("Original Image")
plt.imshow(image, cmap='gray')
plt.axis('off')

plt.subplot(2, 2, 2)
plt.title("Reduced Contrast Image")
plt.imshow(reduced_contrast_image, cmap='gray')
plt.axis('off')

plt.subplot(2, 2, 3)
plt.title("Linear Stretching")
plt.imshow(linear_stretched_image, cmap='gray')
plt.axis('off')

plt.subplot(2, 2, 4)
plt.title("Piecewise Linear Stretching")
plt.imshow(piecewise_stretched_image, cmap='gray')
plt.axis('off')

plt.figure(figsize=(7, 5))
plt.title("Histogram Equalization")
plt.imshow(hist_eq_image, cmap='gray')
plt.axis('off')

plt.show()
