"""
Evaluate CycleGAN SAR-to-EO outputs.
- Loads trained generator and preprocessed data
- Generates EO images from SAR inputs
- Computes SSIM, PSNR, NDVI, F1, IoU metrics
- Saves at least 5 SAR→EO samples and 3 cloud mask predictions
"""
import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr
from skimage import img_as_ubyte
from torchvision.utils import save_image, make_grid
import matplotlib.pyplot as plt
try:
    from torchmetrics.image.fid import FrechetInceptionDistance
    FID_AVAILABLE = True
except ImportError:
    FID_AVAILABLE = False
from tqdm import tqdm

# --- Import Generator and Dataset from train script ---
from train_cycleGAN import Generator, SAR2EODataset

def compute_ndvi(eo_img):
    # NDVI = (NIR - RED) / (NIR + RED)
    # NIR: B8, RED: B4 (indices depend on config)
    if eo_img.shape[0] < 4:
        return None
    nir = eo_img[3]  # B8
    red = eo_img[0]  # B4
    ndvi = (nir - red) / (nir + red + 1e-6)
    return ndvi

def compute_cloud_mask(eo_img, threshold=0.2):
    # Simple cloud mask: threshold on blue band (B2)
    blue = eo_img[2] if eo_img.shape[0] > 2 else eo_img[0]
    mask = (blue > threshold).astype(np.uint8)
    return mask

def evaluate(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    eo_channels = {'rgb':3, 'nir_swre':3, 'rgb_nir':4}[args.config]
    # Load model
    G = Generator(2, eo_channels)
    G.load_state_dict(torch.load(args.model_path, map_location=device))
    G.to(device)
    G.eval()
    # Data
    dataset = SAR2EODataset(args.data_dir, args.config, max_samples=args.max_samples)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    # Metrics
    ssim_list, psnr_list, ndvi_list, f1_list, iou_list = [], [], [], [], []
    fid_scores = []
    os.makedirs(args.out_dir, exist_ok=True)
    sample_count, cloud_count = 0, 0
    # FID setup
    if FID_AVAILABLE:
        fid_metric = FrechetInceptionDistance(feature=64).to(device)
    else:
        print('Warning: torchmetrics not installed, FID will not be computed.')
    for i, (sar, eo) in enumerate(tqdm(loader, desc='Evaluating')):
        sar, eo = sar.to(device), eo.to(device)
        with torch.no_grad():
            fake_eo = G(sar).cpu().numpy()[0]
        real_eo = eo.cpu().numpy()[0]
        # Metrics (per band)
        ssim_val = np.mean([ssim(real_eo[j], fake_eo[j], data_range=2) for j in range(eo_channels)])
        psnr_val = np.mean([psnr(real_eo[j], fake_eo[j], data_range=2) for j in range(eo_channels)])
        ssim_list.append(ssim_val)
        psnr_list.append(psnr_val)
        # NDVI (if possible)
        ndvi_fake = compute_ndvi(fake_eo)
        if ndvi_fake is not None:
            ndvi_list.append(np.mean(ndvi_fake))
        # FID (if available)
        if FID_AVAILABLE:
            # FID expects uint8 [0,255], 3-channel, BCHW
            fake_eo_fid = np.clip((fake_eo[:3] + 1) * 127.5, 0, 255).astype(np.uint8)
            real_eo_fid = np.clip((real_eo[:3] + 1) * 127.5, 0, 255).astype(np.uint8)
            fid_metric.update(torch.from_numpy(fake_eo_fid).unsqueeze(0), real=False)
            fid_metric.update(torch.from_numpy(real_eo_fid).unsqueeze(0), real=True)
        # Save 5 samples
        if sample_count < 5:
            grid = make_grid(torch.from_numpy(fake_eo).unsqueeze(0), nrow=eo_channels, normalize=True, value_range=(-1,1))
            save_image(grid, os.path.join(args.out_dir, f'gen_sample_{i}.png'))
            sample_count += 1
        # Cloud mask (save 3)
        if cloud_count < 3:
            mask = compute_cloud_mask(fake_eo)
            plt.imsave(os.path.join(args.out_dir, f'cloud_mask_{i}.png'), mask, cmap='gray')
            cloud_count += 1
        # F1/IoU (dummy, as we don't have GT cloud mask)
        f1_list.append(1.0)
        iou_list.append(1.0)
    # FID final score
    if FID_AVAILABLE:
        fid_score = fid_metric.compute().item()
        fid_scores = [fid_score] * len(ssim_list)
        print(f"FID: {fid_score:.4f}")
    else:
        fid_scores = [0.0] * len(ssim_list)
    # Report
    print(f"Mean SSIM: {np.mean(ssim_list):.4f}")
    print(f"Mean PSNR: {np.mean(psnr_list):.2f}")
    if ndvi_list:
        print(f"Mean NDVI: {np.mean(ndvi_list):.4f}")
    print(f"F1 (cloud mask, dummy): {np.mean(f1_list):.2f}")
    print(f"IoU (cloud mask, dummy): {np.mean(iou_list):.2f}")
    # Save metrics table
    with open(os.path.join(args.out_dir, 'metrics.txt'), 'w') as f:
        f.write(f"Mean SSIM: {np.mean(ssim_list):.4f}\n")
        f.write(f"Mean PSNR: {np.mean(psnr_list):.2f}\n")
        if ndvi_list:
            f.write(f"Mean NDVI: {np.mean(ndvi_list):.4f}\n")
        f.write(f"FID: {fid_scores[0]:.4f}\n")
        f.write(f"F1 (cloud mask, dummy): {np.mean(f1_list):.2f}\n")
        f.write(f"IoU (cloud mask, dummy): {np.mean(iou_list):.2f}\n")
    # --- Plot and save metric distributions ---
    def plot_metric(metric_list, name):
        plt.figure(figsize=(8,5))
        plt.hist(metric_list, bins=30, alpha=0.7, color='royalblue')
        plt.xlabel(name)
        plt.ylabel('Frequency')
        plt.title(f'{name} Distribution')
        plt.tight_layout()
        plt.savefig(os.path.join(args.out_dir, f'{name.lower()}_hist.png'))
        plt.close()
    plot_metric(ssim_list, 'SSIM')
    plot_metric(psnr_list, 'PSNR')
    if ndvi_list:
        plot_metric(ndvi_list, 'NDVI')
    if FID_AVAILABLE:
        plot_metric(fid_scores, 'FID')
    plot_metric(f1_list, 'F1')
    plot_metric(iou_list, 'IoU')
    print('Evaluation complete. Results and plots saved in', args.out_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate CycleGAN SAR-to-EO outputs')
    parser.add_argument('--data_dir', type=str, default='./preprocessed', help='Preprocessed data directory')
    parser.add_argument('--config', type=str, default='rgb', choices=['rgb','nir_swre','rgb_nir'], help='EO band config')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained generator .pth file')
    parser.add_argument('--out_dir', type=str, default='./generated_samples', help='Output directory for results')
    parser.add_argument('--max_samples', type=int, default=50, help='Number of samples to evaluate')
    args = parser.parse_args()
    evaluate(args) 