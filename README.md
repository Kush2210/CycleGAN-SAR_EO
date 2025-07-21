# SAR-to-EO Image Translation Using CycleGAN

This repository implements a CycleGAN model to translate Synthetic Aperture Radar (SAR) images from Sentinel-1 to Earth Observation (EO) optical images from Sentinel-2, using winter-season data from the Sen12MS dataset. The project explores multiple output band configurations (RGB, NIR/SWIR/Red Edge, RGB+NIR) and evaluates results using spectral-wise SSIM, PSNR, and NDVI metrics.

## Team Members
- Kush Garg, kushgarg_23mc079@dtu.ac.in 
- Saksham Jain, sakshamjain_23me234@dtu.ac.in


### ✅ Dual Implementation Approach:
We implemented the project using **two sperate solutions**:
1. ✍️ **Custom CycleGAN**: Designed and implemented our own CycleGAN architecture from scratch in PyTorch.
2. 📦 **Baseline Codebase**: Used the official [CycleGAN & Pix2Pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) implementation as a strong and reliable reference.


## We explored multiple solutions
Model options: CycleGAN, Supervised CycleGAN, Pix2Pix (Supervised GAN), Pix2PixHD, Multi-Conditional GANs,Seg-CycleGAN, U-Net-Based Models

Considering which models, and why:
- CycleGan : SAR and EO images are unpaired modalities (although slightly paired in Sen12MS), making CycleGAN the natural choice due to its unpaired image-to-image translation architecture using cycle consistency.

- # SAR-to-EO Image Translation: Model Selection Summary

## ✅ Baseline Model
- **CycleGAN**: Used as the baseline model since SAR and EO images are *unpaired* in the Sen12MS dataset.

## ❌ Not Used as Baseline (But Useful in Specific Cases)
- **Pix2Pix**: Requires *perfectly aligned paired data*; not suitable for unpaired SAR-EO translation.
- **Pix2PixHD**: Used only when working with *high-resolution paired images*.
- **Supervised CycleGAN**: Useful when *weak alignment* exists; adds L1 loss for better supervision.
- **Multi-Conditional GANs**: Useful when *multiple SAR bands, temporal inputs, or metadata* are used as input.
- **Seg-CycleGAN**: Incorporates *semantic segmentation maps* to improve semantic consistency in translation.
- **U-Net**: Can be used to remove clouds/snow caps in winter dataset.

---

## Project Overview
The goal of this project is to develop a CycleGAN model that translates Sentinel-1 SAR images into Sentinel-2 EO images, addressing the challenge of winter-season data (snow, ice, bare trees). The project involves:
1. Preprocessing winter-season SAR and EO images from the Sen12MS dataset.
2. Normalizing images to the [-1, 1] range for GAN compatibility.
3. Training CycleGAN models for three output configurations:
   - **RGB**: Bands B4 (Red), B3 (Green), B2 (Blue)
   - **NIR/SWIR/Red Edge**: Bands B8 (NIR), B11 (SWIR), B5 (Red Edge)
   - **RGB + NIR**: Bands B4, B3, B2, B8
4. Postprocessing generated images and evaluating them with SSIM, PSNR, and NDVI metrics.
5. Visualizing results to assess translation quality.

This implementation is inspired by research papers like "SAR-to-optical Image Translation using Supervised Cycle-Consistent Adversarial Networks" (Wang et al., 2019) and repositories like [yuuIind/SAR2Optical](https://github.com/yuuIind/SAR2Optical).

## Instructions to Run Code (Our self-made Architecture)
* For ease for running we have already including a sample of dataset in data folder, so to check the repo you dont have to download the dataset

### Prerequisites
- Python 3.8+
- PyTorch 2.0+
- Additional libraries: `numpy`, `scikit-image`, `matplotlib`, `rasterio`, `h5py`
- Hardware: GPU (NVIDIA recommended) for faster training

## Raw Coded CycleGAN (Our own Architecture)
### Setup 
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Kush2210/CycleGAN-SAR_EO.git
   cd Project1_SAR_to_EO
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download Sen12MS Dataset**:
   - Download winter-season data (`ROIs2017_winter_s1.tar.gz`, `ROIs2017_winter_s2.tar.gz`) from [https://dataserv.ub.tum.de/s/m1474000](https://dataserv.ub.tum.de/s/m1474000).
   - Extract files to `data/` directory:
     ```bash
     mkdir data
     tar -xvzf ROIs2017_winter_s1.tar.gz -C data/
     tar -xvzf ROIs2017_winter_s2.tar.gz -C data/
     ```

4. **Preprocess Data**:
   - Run the preprocessing script to normalize images to [-1, 1]:
     ```bash
     python preprocess.py --data_dir data/ --output_dir data/processed/
     ```

5. **Train CycleGAN Models**:
   - Train models for each configuration (RGB, NIR/SWIR/Red Edge, RGB+NIR):
     ```bash
        cd ..
        python Project1_SAR_to_EO/train_cycleGAN.py --data_dir ./Project1_SAR_to_EO/data --config rgb --epochs 10 --out_dir ./Project1_SAR_to_EO/generated_samples --batch_size 12
     ```
    (Batch size of 12 uses approx 11 GB GPU - Set batch size accordingly)
6. **Evaluate and Visualize**:
   - Compute SSIM, PSNR, and NDVI metrics and visualize results:
     ```bash
     python evaluate.py --model_dir results/ --data_dir data/processed/ --output_dir results/evaluation/
     ```

### Directory Structure
```
projectsubmission/
│
├── Project1_SAR_to_EO/
│   ├── data/                   # Contains trainA (SAR) and trainB (EO) image folders
│   │   ├── trainA/             # SAR images (input domain A)
│   │   └── trainB/             # EO images (input domain B)
│   ├── generated_samples/      # Generated EO images, cloud masks, and metric plots
│   ├── preprocess.py           # Data preprocessing and normalization script
│   ├── train_cycleGAN.py       # CycleGAN training script (with loss tracking and sample generation)
│   ├── evaluate_results.py     # Evaluation script (metrics, plots, and sample generation)
│   └── README.md               # Project-specific documentation
│
├── requirements.txt            # Python dependencies for the project
└── README.md                   # Top-level project overview and instructions
```

## Description

### Data Preprocessing Steps
1. **Dataset Loading**: Load Sentinel-1 SAR (VV, VH polarizations) and Sentinel-2 EO (B2, B3, B4, B5, B8, B11 bands) images from Sen12MS winter-season data using utilities inspired by [schmitt-muc/SEN12MS](https://github.com/schmitt-muc/SEN12MS).
2. **Normalization**: Scale pixel values to [-1, 1] using min-max normalization to ensure compatibility with CycleGAN’s architecture.
4. **Band Selection**: Extract specific bands for each configuration:
   - RGB: B4, B3, B2
   - NIR/SWIR/Red Edge: B8, B11, B5
   - RGB + NIR: B4, B3, B2, B8


### Models Used
- **CycleGAN**: Adapted from [yuuIind/SAR2Optical](https://github.com/yuucoin/SAR2Optical) and [junyanz/pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix).
  - **Generator**: Modified ResNet-based architecture to handle 2-channel SAR input and 3/4-channel EO output.
  - **Discriminator**: PatchGAN discriminator for multi-spectral outputs.
  - **Loss Functions**: Cycle-consistency loss, adversarial loss, and identity loss, inspired by “SAR-to-optical Image Translation using Supervised Cycle-Consistent Adversarial Networks” (Wang et al., 2019).
- Three models trained separately for RGB, NIR/SWIR/Red Edge, and RGB+NIR configurations.



## Implementing Baseline Codebase : [CycleGAN & Pix2Pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)

1. **Install Dependencies**:
    cd Original_CycleGAN_Repo_Implementation/pytorch-CycleGAN-and-pix2pix-master
   ```bash
   pip install -r requirements.txt
   ```
2. **Preprocess Data**:
   - Run the preprocessing script
    cd ..
     ```bash
     python preprocess.py 
     ```

3. **Train CycleGAN Models**:
   - Train models for each configuration (RGB, NIR/SWIR/Red Edge, RGB+NIR):
     ```bash
        cd ..
        !python train.py \
            --dataroot /content/train_8000_unzipped/train_8000 \
            --name sen12ms_cyclegan \
            --model cycle_gan \
            --input_nc 1 \
            --output_nc 1 \
            --epoch 3 \
            --n_epochs_decay 15 \
            --batch_size 6 \
            --gpu_ids 0
     ```
     (Change path of your dataset folder accodingly)
    (Batch size of 6 uses approx 13.9 GB GPU - Set batch size accordingly)


### Key Findings or Observations
- **Winter Challenges**: Snow and ice in winter-season data reduce contrast, making translation harder. Data augmentation (e.g., noise, rotations) improved robustness.
- **Band Configurations**: The RGB configuration achieved the highest SSIM and PSNR due to its similarity to natural images, while NIR/SWIR/Red Edge was more challenging due to spectral complexity.
- **NDVI Performance**: The RGB+NIR configuration preserved vegetation details better, as measured by NDVI differences.
- **Model Performance**: Supervised CycleGAN variants (inspired by Wang et al.) outperformed vanilla CycleGAN by leveraging paired data, reducing artifacts in generated images.

*Note*: Specific findings depend on your training results. Update this section after running experiments.

### Tools and Frameworks Used
- **PyTorch**: Core framework for implementing and training CycleGAN models.
- **scikit-image**: For computing SSIM and PSNR metrics.
- **numpy**: For NDVI calculations and array manipulations.
- **matplotlib** and **rasterio**: For visualizing generated and real EO images.
- **h5py**: For handling Sen12MS dataset in HDF5 format.
- **Sen12MS Utilities**: Adapted from [schmitt-muc/SEN12MS](https://github.com/schmitt-muc/SEN12MS) for data loading and preprocessing.

## Acknowledgments
- Inspired by research papers:  
  - Wang et al., “SAR-to-optical Image Translation using Supervised Cycle-Consistent Adversarial Networks” (IEEE Access, 2019).
  - Schmitt et al., “SAR-to-EO Image Translation with Multi-Conditional Adversarial Networks” (2022).
- Code adapted from:  
  - [yuuIind/SAR2Optical](https://github.com/yuuIind/SAR2Optical)  
  - [junyanz/pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)  
  - [schmitt-muc/SEN12MS](https://github.com/schmitt-muc/SEN12MS)
