# Plate OCR · Star Detection · Non-Local Means Denoising

This repository contains a single Jupyter Notebook implementing three computer vision pipelines:

1. **License Plate OCR** — preprocessing and recognition of license plate text.  
2. **Star Detection & Centroid Localization** — object counting and centroid calculation from a binary mask.  
3. **Non-Local Means Denoising** — advanced noise reduction, including theoretical discussion, CPU vs GPU implementation, and quantitative evaluation.

---

## Data

The notebook expects:
- `car.jpg` → license plate image  
- `stars_mask.png` → binary mask of stars  
- `bird.jpg`, `woman.jpg`, `vegetables.jpg` → natural images for denoising experiments  

You can replace these with your own images.

---

## Task 1 — License Plate OCR

**Objective:** Extract text from a vehicle license plate.

**What was done:**
- The car image is loaded in grayscale.  
- Gaussian blur is applied to suppress small variations.  
- Thresholding creates a binary mask highlighting plate characters.  
- The image is resized (e.g., ×6) to enhance OCR accuracy.  
- Tesseract OCR is applied with a whitelist of digits and uppercase letters, and page segmentation optimized for single-line text.  

**Outputs:**
- Preprocessed images (grayscale, thresholded, upscaled).  
- Extracted license plate string printed to console.  

---

## Task 2 — Star Detection & Centroid Localization

**Objective:** From a binary mask of stars, determine the total number of stars and their centroid coordinates.

**What was done:**
- Binary mask image is read and thresholded to ensure values are strictly 0 or 255.  
- Connected components analysis (`cv2.connectedComponentsWithStats`) is used to identify individual stars.  
- For each star, centroid `(x, y)` is computed from component statistics.  
- Very small components can be filtered out to remove noise.  
- Results are reported as star count and list of centroids, with optional overlay visualization.  

**Outputs:**
- Total number of stars in the image.  
- Centroid coordinates for each star.  
- Annotated image showing centroids.  

---

## Task 3 — Non-Local Means (NLM) Denoising

**Objective:** Apply NLM denoising to noisy images, analyze theory and performance, and compare CPU vs GPU implementations.

### 3a. Theory
- **Non-Local Means formula**: For each pixel, weights are computed by comparing a small patch around the pixel with patches in a larger search window. Similar patches contribute more to the average.  
- **Comparison to averaging filter**:  
  - Averaging uses fixed weights in a neighborhood.  
  - NLM adapts weights based on similarity, preserving detail and edges.  
- **Advantages**: strong detail preservation, robust against Gaussian noise.  
- **Disadvantages**: high computational cost, sensitive to parameter choices.  

### 3b. Implementation
- Gaussian noise with different variances is added to test images (`bird.jpg`, `woman.jpg`, `vegetables.jpg`).  
- NLM denoising is implemented using PyTorch.  
- Both **CPU** and **GPU** versions are executed.  

### 3c. Evaluation
- **Runtime analysis**: CPU vs GPU times are measured and compared.  
- **Quality metrics**:  
  - **PSNR** (Peak Signal-to-Noise Ratio) for fidelity.  
  - **SSIM** (Structural Similarity Index) for perceptual similarity.  
- **Visualization**: Original, noisy, and denoised images displayed side by side.  

**Outputs:**
- Denoised images for each test case.  
- CPU vs GPU runtime comparison.  
- PSNR and SSIM scores for each noise level.  
- Observations about GPU speedup and quality improvement.  

---

## How to Run

1. Open `ImageProcessing.ipynb` in Colab or locally.  
2. Upload the required images.  
3. Run all cells sequentially:
   - Task 1 → License Plate OCR  
   - Task 2 → Star Detection  
   - Task 3 → NLM Denoising with evaluation  
