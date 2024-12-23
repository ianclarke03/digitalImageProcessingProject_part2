import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
    


# Explicitly point to the 'task3_tumor_images' directory
image_folder = "task3_tumor_images"

# Ensure correct paths to images in the folder
image_paths = [os.path.join(image_folder, filename) 
               for filename in os.listdir(image_folder) 
               if filename.endswith('.png')]

print("Images to process:", image_paths)  # Debugging line







# List of 10 dummy images as placeholders
#image_paths = [f"tumor{i}.png" for i in range(0, 10)]

# Function to embed an annotation into an image
def embed(original_image, annotation):
    binary_string = ''.join([format(ord(char), '08b') for char in annotation])

    # Extract all bit planes
    rows, cols = original_image.shape
    bit_planes = np.zeros((rows, cols, 8), dtype=np.uint8)

    # Print the LSB of the original image
    print("LSB of Original Image (Before Embedding):")
    print(bit_planes[:, :, 0])

    for i in range(8):
        bit_planes[:, :, i] = (original_image >> i) & 1
        # print(bit_planes[:,:,0])

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

    # Print the LSB of the recomposed image
    print("LSB After Embedding:")
    print(recomposed_image & 1)  # Extract and print the LSB

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
for image_path in image_paths:
    original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if original_image is None:
        print(f"Image {image_path} not found. Skipping.")
        continue

    # Determine annotation based on the file name
    if "giloma" in image_path.lower():
        annotation = "This is a giloma tumor"
    elif "meningioma" in image_path.lower():
        annotation = "This is a meningioma tumor"
    elif "pituitary" in image_path.lower():
        annotation = "This is a pituitary tumor"
    elif "notumor" in image_path.lower():
        annotation = "This is not a tumor"
    else:
        annotation = "Unknown tumor type"

    # Debug: Print annotation for each image
    print(f"Embedding annotation for {image_path}: {annotation}")

    # Embed the annotation into the image
    recomposed_image = embed(original_image, annotation)

    # Extract the annotation to verify correctness
    extracted_annotation = extract(recomposed_image, len(annotation))
    print(f"Extracted annotation: {extracted_annotation}")

    # Display the images (optional, remove if not needed)
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


     # Extract the annotation
    extracted_annotation = extract(recomposed_image, len(annotation))
    
    
    
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