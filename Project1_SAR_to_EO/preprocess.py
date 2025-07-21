"""
Preprocessing script for SAR-to-EO CycleGAN project.
- Extracts SAR and EO images from Sen12MS dataset
- Normalizes images to [-1, 1]
- Saves preprocessed data for training
- Supports three EO band configurations:
    a) RGB (B4, B3, B2)
    b) NIR, SWIR, Red Edge (B8, B11, B5)
    c) RGB + NIR (B4, B3, B2, B8)
"""
import os
import tarfile
import numpy as np
from skimage import io
from tqdm import tqdm
import argparse

# --- Band indices for Sentinel-2 (Sen12MS order) ---
BAND_IDX = {
    'B1': 0, 'B2': 1, 'B3': 2, 'B4': 3, 'B5': 4, 'B6': 5, 'B7': 6, 'B8': 7,
    'B8A': 8, 'B9': 9, 'B10': 10, 'B11': 11, 'B12': 12
}

CONFIGS = {
    'rgb':      [BAND_IDX['B4'], BAND_IDX['B3'], BAND_IDX['B2']],
    'nir_swre': [BAND_IDX['B8'], BAND_IDX['B11'], BAND_IDX['B5']],
    'rgb_nir':  [BAND_IDX['B4'], BAND_IDX['B3'], BAND_IDX['B2'], BAND_IDX['B8']]
}

# --- Argument Parsing ---
def get_args():
    parser = argparse.ArgumentParser(description='Preprocess Sen12MS SAR and EO data')
    parser.add_argument('--data_dir', type=str, default='../data', help='Directory with .tar.gz files')
    parser.add_argument('--output_dir', type=str, default='./preprocessed', help='Where to save processed data')
    return parser.parse_args()

# --- Extraction ---
def extract_tar(tar_path, extract_path):
    if not os.path.exists(extract_path):
        os.makedirs(extract_path, exist_ok=True)
    with tarfile.open(tar_path, 'r:gz') as tar:
        tar.extractall(path=extract_path)

# --- Normalization ---
def normalize_img(img):
    return (img.astype(np.float32) / 127.5) - 1.0

# --- Main Preprocessing ---
def main():
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)
    s1_tar = os.path.join(args.data_dir, 'ROIs2017_winter_s1.tar.gz')
    s2_tar = os.path.join(args.data_dir, 'ROIs2017_winter_s2.tar.gz')
    s1_dir = os.path.join(args.output_dir, 'S1')
    s2_dir = os.path.join(args.output_dir, 'S2')
    # Extraction
    if not os.path.exists(s1_dir) or not os.listdir(s1_dir):
        print('Extracting SAR...')
        extract_tar(s1_tar, s1_dir)
    if not os.path.exists(s2_dir) or not os.listdir(s2_dir):
        print('Extracting EO...')
        extract_tar(s2_tar, s2_dir)
    # Find all SAR/EO pairs
    sar_files = sorted([os.path.join(dp, f) for dp, dn, filenames in os.walk(s1_dir) for f in filenames if f.endswith('.tif')])
    eo_files = sorted([os.path.join(dp, f) for dp, dn, filenames in os.walk(s2_dir) for f in filenames if f.endswith('.tif')])
    print(f'Found {len(sar_files)} SAR and {len(eo_files)} EO files.')
    # Output dirs for configs
    for config in CONFIGS:
        os.makedirs(os.path.join(args.output_dir, config, 'sar'), exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, config, 'eo'), exist_ok=True)
    # Process pairs
    for sar_path, eo_path in tqdm(zip(sar_files, eo_files), total=min(len(sar_files), len(eo_files)), desc='Processing pairs'):
        sar_img = io.imread(sar_path)
        eo_img = io.imread(eo_path)
        sar_img = normalize_img(sar_img)
        eo_img = normalize_img(eo_img)
        base = os.path.splitext(os.path.basename(sar_path))[0]
        # Save for each config
        for config, bands in CONFIGS.items():
            eo_sel = eo_img[..., bands]
            np.save(os.path.join(args.output_dir, config, 'sar', f'{base}.npy'), sar_img)
            np.save(os.path.join(args.output_dir, config, 'eo', f'{base}.npy'), eo_sel)
    print('Preprocessing complete. Data saved in:', args.output_dir)

if __name__ == '__main__':
    main() 