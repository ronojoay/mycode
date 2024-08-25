import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def load_image(file_path):
    return np.array(Image.open(file_path))

def display_images(original, neighborhood_image,adjacency_image):
    fig, axs = plt.subplots(1, 3, figsize=(20, 5))

    axs[0].imshow(original, cmap='gray')
    axs[0].set_title('Original Image')
    axs[0].axis('off')
    
    axs[1].imshow(neighborhood_image, cmap='gray')
    axs[1].set_title('Neighborhood of pixels')
    axs[1].axis('off')

    axs[2].imshow(adjacency_image, cmap='gray')
    axs[2].set_title('Adjacency between pixels')
    axs[2].axis('off')
       
    plt.tight_layout()
    plt.show()

def calculate_pixel_neighborhood(image):
  # Example: Calculate average difference between a pixel and its neighbors
  result = np.zeros_like(image, dtype=np.float32)
  for i in range(1, image.shape[0] - 1):
    for j in range(1, image.shape[1] - 1):
      neighbors = [
          image[i-1, j-1], image[i-1, j], image[i-1, j+1],
          image[i, j-1],   image[i, j],   image[i, j+1],
          image[i+1, j-1], image[i+1, j], image[i+1, j+1]
      ]
      result[i, j] = np.mean(neighbors) 
  # Normalize to 0-255 range for display
  result = (result / result.max() * 255).astype(np.uint8)
  return result

def calculate_pixel_adjacency(image):
  # Example: Calculate average difference between a pixel and its neighbors
  result = np.zeros_like(image, dtype=np.float32)
  for i in range(1, image.shape[0] - 1):
    for j in range(1, image.shape[1] - 1):
      neighbors = [image[i-1, j], image[i+1, j], image[i, j-1], image[i, j+1]]
      result[i, j] = np.sum(np.abs(image[i, j] - np.array(neighbors))) 
  # Normalize to 0-255 range for display
  result = (result / result.max() * 255).astype(np.uint8)
  return result

# Main program
file_path ="C:/Users/kusum/OneDrive/Pictures/Screenshots/Screenshot 2024-03-31 022002.png"  # Update with your image path
original_image = load_image(file_path)
# Convert to grayscale if the image is in color
if len(original_image.shape) == 3:
    original_image = np.mean(original_image, axis=2).astype(np.uint8)

adjacency_image = calculate_pixel_adjacency(original_image)
neighborhood_image = calculate_pixel_neighborhood(original_image)

display_images(original_image,neighborhood_image,adjacency_image)