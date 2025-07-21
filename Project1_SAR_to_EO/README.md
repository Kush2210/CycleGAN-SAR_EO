# Project 1: SAR-to-EO Image Translation Using CycleGAN

## Team Members
- Name: <Your Name>
- Email: <Your Email>

## Project Overview
This project implements a CycleGAN to translate Sentinel-1 SAR images to Sentinel-2 EO images, exploring multiple output band configurations. The solution includes preprocessing, model training, evaluation, and visualization.

---

## How to Run

### 1. Preprocessing
Extract and normalize the SAR and EO images for all band configurations:
```bash
python preprocess.py --data_dir ../data --output_dir ./preprocessed
```
- Place `ROIs2017_winter_s1.tar.gz` and `ROIs2017_winter_s2.tar.gz` in a `data/` directory at the project root.
- The script will extract, normalize, and save data for all three EO band configurations.

### 2. Training CycleGAN
Train the CycleGAN for a specific EO band configuration (e.g., RGB):
```bash
python train_cycleGAN.py --data_dir ./preprocessed --config rgb --epochs 100 --out_dir ./generated_samples
```
- Replace `rgb` with `nir_swre` or `rgb_nir` for other configurations.
- Model checkpoints and sample outputs will be saved in `generated_samples/`.

### 3. Evaluation & Visualization
Generate EO images, compute metrics, and save sample results:
```bash
python evaluate_results.py --data_dir ./preprocessed --config rgb --model_path ./generated_samples/G_rgb_epoch100.pth --out_dir ./generated_samples
```
- Replace `rgb` and the model path as needed for other configs or checkpoints.
- At least 5 SAR→EO samples and 3 cloud mask predictions will be saved in `generated_samples/`.
- Metrics (SSIM, PSNR, NDVI, F1, IoU) will be reported in `metrics.txt`.

---

## Data Preprocessing Steps
- Extract SAR and EO images from the Sen12MS dataset.
- Normalize all images to [-1, 1].
- Organize data for PyTorch training.

## Models Used
- CycleGAN (PyTorch implementation)
- Configurable for different EO band outputs (RGB, NIR/SWIR/Red Edge, RGB+NIR)

## Key Findings / Observations
- (To be filled after experiments)

## Tools and Frameworks Used
- PyTorch, torchvision
- numpy, scikit-image, matplotlib, pillow, tqdm, opencv-python, scipy, pyyaml

## Sample Results
- At least 5 SAR→EO samples and 3 cloud mask predictions will be saved in `generated_samples/`.
- Metrics (SSIM, PSNR, NDVI, F1, IoU) will be reported in a table. 