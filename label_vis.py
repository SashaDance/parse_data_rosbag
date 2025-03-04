import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Define class names and corresponding colors
class_names = ['wet road', 'loose snow', 'packed snow', 'dry road', 'ice']
class_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255)] # Red, Green, Blue, Yellow

# Paths to your images and labels
image_paths = [
    'selected/1740145201997976729/image_1740145201997976729.png',
    'selected/1740145651629282926/image_1740145651629282926.png'
]
label_paths = [
    'rcs/labels/train/image_1740145201997976729.txt',
    'rcs/labels/train/image_1740145651629282926.txt'
]

# Create a figure with three subplots
fig, axs = plt.subplots(2, 1, figsize=(15, 5))
img_to_dataset = {
    0: 'fish',
    1: 'mini',
    2: 'wp'
}

# Process each image and its corresponding label
for idx, (image_path, label_path) in enumerate(zip(image_paths, label_paths)):
    # Load the image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB for matplotlib
    height, width, _ = image.shape

    # Create an empty mask image
    mask_image = np.zeros_like(image)

    # Read the ground truth labels
    with open(label_path, 'r') as file:
        lines = file.readlines()

    # Process each line in the label file
    for line in lines:
        parts = line.strip().split()
        class_id = int(parts[0])  # Class ID
        polygon = list(map(float, parts[1:]))  # Normalized polygon coordinates

        # Convert normalized polygon coordinates to absolute coordinates
        polygon = np.array(polygon).reshape(-1, 2) * np.array([width, height])
        polygon = polygon.astype(np.int32)

        # Create a binary mask for the current instance
        binary_mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(binary_mask, [polygon], 1)

        # Color the mask based on the class
        color = class_colors[class_id]
        colored_mask = np.zeros_like(image)
        colored_mask[binary_mask == 1] = color

        # Add borders to the mask
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(colored_mask, contours, -1, (255, 255, 255), 5)  # White border

        # Add the colored mask to the mask image
        mask_image = cv2.add(mask_image, colored_mask)

    # Combine the original image and the mask image
    combined_image = cv2.addWeighted(image, 0.7, mask_image, 0.3, 0)

    # Display the combined image in the subplot
    axs[idx].imshow(combined_image)
    # axs[idx].set_title(f'{img_to_dataset[idx]}')
    axs[idx].axis('off')

# Add labels on the left side of the row
for idx, ax in enumerate(axs):
    # Add a title for each image
    # ax.set_title(f'{img_to_dataset[idx]}', fontsize=12, pad=10)

    # Add class labels above the image
    for i, (class_name, color) in enumerate(zip(class_names, class_colors)):
        # Create a colored rectangle and text
        rect = plt.Rectangle((0.02 + i * 0.2, 0.02), 0.15, 0.05, color=np.array(color) / 255, transform=ax.transAxes)
        ax.add_patch(rect)
        ax.text(0.1 + i * 0.2, 0.04, class_name, color='black', fontsize=10, ha='center', va='center', transform=ax.transAxes)

plt.tight_layout()
plt.show()