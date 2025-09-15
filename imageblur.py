import cv2
import numpy as np
from rembg import remove, new_session
from PIL import Image
import argparse
import os
import random
import glob

# --- Argument Parsing ---
parser = argparse.ArgumentParser(description="Randomly select an image from a dataset, detect and blur people with a specialized high-accuracy model.")
parser.add_argument('--datapath', type=str, required=True, help="Path to the dataset directory containing images.")
parser.add_argument('--output_dir', type=str, default='output_perfect', help="Directory to save the perfectly blurred output image.")
args = parser.parse_args()

# --- Find and Select a Random Image ---
print(f"Searching for images in: {args.datapath}")
image_extensions = ['*.jpg', '*.jpeg', '*.png'] # Added .jpeg for broader support
image_files = []
for ext in image_extensions:
    image_files.extend(glob.glob(os.path.join(args.datapath, ext)))

if not image_files:
    print(f"Error: No images found in the directory: {args.datapath}")
    exit()

random_image_path = random.choice(image_files)
print(f"Randomly selected image: {random_image_path}")

# --- Load Image ---
image_cv = cv2.imread(random_image_path)
if image_cv is None:
    print(f"Error: Could not read the selected image: {random_image_path}")
    exit()

# --- High-Accuracy Mask Generation using rembg ---
print("Generating perfect human mask with 'u2net_human_seg' model...")
session = new_session("u2net_human_seg")

image_pil = Image.fromarray(cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB))
output_pil = remove(image_pil, session=session, only_mask=True)
mask = np.array(output_pil) # This is the binary (0 or 255) mask

# --- Applying the Blur ---
if np.any(mask):
    print("Applying strong blur using the perfect mask with feathered edges...")
    
    blur_kernel_size = (151, 151)
    blurred_image_full = cv2.GaussianBlur(image_cv, blur_kernel_size, 0)

    # NEW: Feather the mask to create a smooth transition
    # A small blur on the mask itself will create semi-transparent edges.
    # The kernel size (e.g., 21x21) here is for the mask, not the image blur.
    # Adjust this if you need a wider or narrower feathering effect.
    feathered_mask = cv2.GaussianBlur(mask, (21, 21), 0) # Apply blur to the mask
    
    # Normalize the feathered mask to be between 0.0 and 1.0 (float)
    alpha = feathered_mask.astype(float) / 255.0

    # Expand alpha to 3 channels for blending
    alpha_3ch = np.stack([alpha]*3, axis=-1)
    
    # Blend the original and blurred images using the feathered alpha mask
    # output = (alpha * blurred_region) + ((1 - alpha) * original_region)
    final_image = (alpha_3ch * blurred_image_full) + ((1 - alpha_3ch) * image_cv)
    
    # Convert back to uint8 after blending
    final_image = np.clip(final_image, 0, 255).astype(np.uint8)
else:
    print("No human detected by the model.")
    final_image = image_cv

# --- Save the Final Image ---
os.makedirs(args.output_dir, exist_ok=True)
base_name = os.path.basename(random_image_path)
name, ext = os.path.splitext(base_name)
output_path = os.path.join(args.output_dir, f"{name}_perfect_feathered_blur{ext}")

cv2.imwrite(output_path, final_image)
print(f"✅ Successfully saved perfectly blurred image to: {output_path}")
