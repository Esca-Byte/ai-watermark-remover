import cv2
import easyocr
import numpy as np
import os
import glob
from simple_lama_inpainting import SimpleLama
from PIL import Image

def enhance_contrast_clahe(image):
    """
    Applies Contrast Limited Adaptive Histogram Equalization (CLAHE) to enhance local contrast.
    Good for detecting text in dark or complex backgrounds.
    """
    try:
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L-channel
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        
        # Merge and convert back to BGR
        limg = cv2.merge((cl, a, b))
        enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        return enhanced
    except Exception as e:
        print(f"Warning: CLAHE failed ({e}), using original image.")
        return image

def expand_bbox(bbox, scale=0.1):
    """
    Expands a bounding box by a certain scale factor relative to its height/width.
    bbox: list of 4 points [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
    """
    points = np.array(bbox, dtype=np.float32)
    center = np.mean(points, axis=0)
    
    # Calculate box dimensions to determine expansion amount
    # Height approx: dist between p0 and p3, or p1 and p2
    height = np.linalg.norm(points[0] - points[3])
    width = np.linalg.norm(points[0] - points[1])
    
    # Expand from center
    # Add a fixed pixel amount + percentage of size
    expansion_amount = max(5, int(min(height, width) * scale))
    
    vectors = points - center
    # Normalize vectors and push out
    # A simpler way: just scale the vectors from center
    # But for a reliable 'padding', let's just use the Simple Geometric scale
    
    # Logic: Offset each corner outwards
    # We can just scale the points relative to center
    # If we want 10% padding, we multiply distance from center by 1.1 roughly
    
    # Let's do a uniform expansion for robustness
    expanded_points = []
    for p in points:
        vec = p - center
        # normalize
        dist = np.linalg.norm(vec)
        if dist == 0: 
            expanded_points.append(p)
            continue
        
        # New distance = old_distance + expansion_amount
        new_dist = dist + expansion_amount 
        new_p = center + (vec / dist) * new_dist
        expanded_points.append(new_p)
        
    return np.array(expanded_points, dtype=np.int32)

def remove_watermark(image_path, output_path, reader, lama):
    """
    Removes text watermarks from an image using EasyOCR and LaMa inpainting
    with enhanced preprocessing and dynamic masking.
    """
    print(f"Processing: {image_path}")
    
    # 1. Load Image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image {image_path}")
        return

    # 2. Preprocessing & Detection (Double Pass)
    # Pass 1: Original Image
    results_orig = reader.readtext(img)
    
    # Pass 2: Enhanced Image
    img_enhanced = enhance_contrast_clahe(img)
    results_enhanced = reader.readtext(img_enhanced)
    
    # Combine results
    # Each result is (bbox, text, conf)
    all_results = results_orig + results_enhanced
    
    if not all_results:
        print(f"No text detected in {image_path}. Skipping.")
        return

    print(f"  - detected {len(results_orig)} text regions (Original)")
    print(f"  - detected {len(results_enhanced)} text regions (Enhanced)")

    # 3. Create Mask with Dynamic Dilation
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    
    count = 0
    for (bbox, text, prob) in all_results:
        # Confidence threshold (optional, strict < 0.2 usually junk)
        if prob < 0.2:
            continue
            
        # Dynamically expand the box based on its size
        expanded_poly = expand_bbox(bbox, scale=0.15) # 15% expansion
        
        cv2.fillPoly(mask, [expanded_poly], 255)
        count += 1

    if count == 0:
        print("  - No valid text after filtering. Skipping.")
        return

    # 4. Final Safety Dilation
    # Just a small global dilation to merge overlapping characters or smooth edges
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)

    # 5. Inpaint with LaMa
    try:
        # Convert to PIL for SimpleLama
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        mask_pil = Image.fromarray(mask)
        
        result_pil = lama(img_pil, mask_pil)
        
        # Convert back to OpenCV format
        result = cv2.cvtColor(np.array(result_pil), cv2.COLOR_RGB2BGR)
        
        # 6. Save Output
        cv2.imwrite(output_path, result)
        print(f"  -> Saved to: {output_path}")
        
    except Exception as e:
        print(f"Error during inpainting: {e}")

def main():
    input_dir = "inputs"
    output_dir = "outputs"
    
    # Ensure directories exist
    if not os.path.exists(input_dir):
        os.makedirs(input_dir)
        print(f"Created '{input_dir}' directory. Please put images there.")
        # Create a test image if folder was empty/created just now
        return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Initialize EasyOCR Reader
    print("Initializing EasyOCR (GPU)...")
    try:
        reader = easyocr.Reader(['en'], gpu=True)
    except Exception as e:
        print(f"GPU initialization failed ({e}), falling back to CPU...")
        reader = easyocr.Reader(['en'], gpu=False)
    
    # Initialize LaMa
    print("Initializing LaMa Inpainting model...")
    lama = SimpleLama()

    # Get all images
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.webp']
    image_files = []
    for ext in extensions:
        image_files.extend(glob.glob(os.path.join(input_dir, ext)))
        # Case insensitive check for extensions like .JPG on linux entails more work, 
        # but glob on Windows is usually Case Insensitive.
        # Adding upper case just in case.
        image_files.extend(glob.glob(os.path.join(input_dir, ext.upper())))

    # Remove duplicates if any
    image_files = sorted(list(set(image_files)))

    if not image_files:
        print(f"No images found in '{input_dir}'.")
        return

    print(f"Found {len(image_files)} images.")

    for img_path in image_files:
        filename = os.path.basename(img_path)
        out_path = os.path.join(output_dir, filename)
        remove_watermark(img_path, out_path, reader, lama)

    print("Batch processing complete!")

if __name__ == "__main__":
    main()
