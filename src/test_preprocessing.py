import sys
import os


# Add the project root to the python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) 






from src.preprocessing import preprocess_image
import matplotlib.pyplot as plt
import numpy as np

#  Use any image from your dataset
image_path = 'data/NEU-CLS/Crazing/crazing_10.jpg'

img = preprocess_image(image_path)

print("Shape:", img.shape)
print("Data type:", img.dtype)
print("Pixel range:", img.min(), "to", img.max())

# Convert back to 0-255 for saving
img_uint8 = (img * 255).astype(np.uint8)

# save the image to disk(so i can view it from windows)
plt.imsave("preprocessed_output.png", img_uint8)


print("Preprocessed image saved as preprocessed_output.png")

# Optional: Show the image
plt.imshow(img)
plt.title("Preprocessed Image")
plt.axis('off')
plt.show()
