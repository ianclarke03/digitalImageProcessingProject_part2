# Part 3: Annotation Hiding (third team member)
# In this part of the project, you will develop a technique to embed annotations in the images. The annotations
# must be hidden within the image in such a way that the visual appearance of the image is not significantly
# altered.
# 3

# 3.1 Embedding Annotations
# The goal is to hide the annotation within each of the 10 images used in Part 1 and Part 2. The annotation
# should be embedded in the image using a suitable image processing technique, such as:
# • Least Significant Bit (LSB) Embedding: Embed the annotation in the least significant bits of the
# image pixels, ensuring minimal visual disruption.

# 3.2 Evaluation
# Evaluate the effectiveness of the annotation hiding technique by checking the following criteria:
# • Visual Impact: The image should not show visible signs of annotation presence after embedding.
# • Reversibility: Ensure that the annotation can be extracted from the image with minimal loss. Need
# an embedding extraction step to extract the annotations in the LSB back to its original text. Hint:
# Binary (base 2) to decimal (base 10) and to text (using ASCII).
# For each group (composed of three students), apply the annotation hiding technique to the 10 images
# and present the results.

import numpy as np
import cv2
import matplotlib.pyplot as plt





#Hide the annotation within the images

image_path = 'tumor.png'
original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
original_image = original_image.astype(np.uint8)

# Convert the annotation text to binary and combine each char into one long binary string
text = "This is an annotation"
words = text.split()
binary_string = ''.join([format(ord(char), '08b') for char in text])


#Extract all bit planes
rows, cols = original_image.shape
bit_planes = np.zeros((rows, cols, 8), dtype=np.uint8)

for i in range(8):
    bit_planes[:,:,i] = (original_image >> i) & 1

# Zero out the LSB (bit plane 0)
bit_planes[:,:,0] = 0

# Embed binary_string into the LSB bit plane
num_bits = len(binary_string)
if num_bits > rows * cols:
    raise ValueError('The image is too small to hide the annotation.')

for i in range(num_bits):
    row = i // cols
    col = i % cols
    # Embed each bit of the binary string into the LSB
    bit_planes[row, col, 0] = int(binary_string[i])




#extract the annotation from the images with minimal loss