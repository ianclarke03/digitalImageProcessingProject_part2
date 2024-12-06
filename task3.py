import numpy as np
import cv2
import matplotlib.pyplot as plt

# List of 10 dummy images as placeholders
image_paths = [f"tumor{i}.png" for i in range(0, 10)]

# Function to embed an annotation into an image
def embed(original_image, annotation):
    binary_string = ''.join([format(ord(char), '08b') for char in annotation])

    # Extract all bit planes
    rows, cols = original_image.shape
    bit_planes = np.zeros((rows, cols, 8), dtype=np.uint8)

    for i in range(8):
        bit_planes[:, :, i] = (original_image >> i) & 1

    # Zero out the LSB (bit plane 0)
    bit_planes[:, :, 0] = 0

    # Embed binary_string into the LSB bit plane
    num_bits = len(binary_string)
    if num_bits > rows * cols:
        raise ValueError('The image is too small to hide the annotation.')

    for i in range(num_bits):
        row = i // cols
        col = i % cols
        # Embed each bit of the binary string into the LSB
        bit_planes[row, col, 0] = int(binary_string[i])

    # Recompose the image from the modified bit planes
    recomposed_image = np.zeros((rows, cols), dtype=np.uint8)
    for i in range(8):
        recomposed_image += bit_planes[:, :, i] * (2**i)

    return recomposed_image

# Function to extract the annotation from a recomposed image
def extract(recomposed_image, annotation_length):
    rows, cols = recomposed_image.shape
    num_bits = annotation_length * 8  # Each character is 8 bits
    binary_string = ""

    for i in range(num_bits):
        row = i // cols
        col = i % cols
        binary_string += str(recomposed_image[row, col] & 1) # Extract Binary String from LSB

    # Convert binary string back to text
    annotation = "".join(
        chr(int(binary_string[i:i+8], 2)) for i in range(0, len(binary_string), 8)
    )
    return annotation

# Loop over the images, process them, and evaluate the results
# Convert Binary to Decimal (Base 10):  int(binary_string[i:i+8], 2)
# Convert Decimal to Text: chr(int(binary_string[i:i+8], 2))

annotation = "This is an annotation"
for image_path in image_paths:
    original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if original_image is None:
        print(f"Image {image_path} not found. Skipping.")
        continue
    
    # Embed the annotation
    recomposed_image = embed(original_image, annotation)
    
    # Extract the annotation
    extracted_annotation = extract(recomposed_image, len(annotation))
    
    # Display the original and modified image
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(original_image, cmap="gray")
    plt.title(f"Original Image: {image_path}")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(recomposed_image, cmap="gray")
    plt.title(f"Recomposed Image: {image_path}")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

    # Evaluation results
    print(f"Original Annotation: {annotation}")
    print(f"Extracted Annotation: {extracted_annotation}")
    if annotation == extracted_annotation:
        print(f"Annotation successfully embedded and extracted from {image_path}.\n")
    else:
        print(f"Error in annotation extraction for {image_path}.\n")

    plt.subplot(1, 2, 1)
    plt.imshow(recomposed_image, cmap="gray")
    plt.title(f"Image Post Extraction: {image_path}")
    plt.axis("off")
    plt.tight_layout()
    plt.show()