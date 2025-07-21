"""
Train CycleGAN for SAR-to-EO image translation (PyTorch).
- Supports three EO band configurations:
    a) RGB (B4, B3, B2)
    b) NIR, SWIR, Red Edge (B8, B11, B5)
    c) RGB + NIR (B4, B3, B2, B8)
- Visualizes and saves sample outputs during training.
- Saves model checkpoints.
"""
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image, make_grid
from tqdm import tqdm
import random
from PIL import Image
import matplotlib.pyplot as plt

# --- CycleGAN Model Components ---
# (Simple ResNet-based generator and PatchGAN discriminator)
class ResnetBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, 3),
            nn.InstanceNorm2d(dim),
            nn.ReLU(True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, 3),
            nn.InstanceNorm2d(dim)
        )
    def forward(self, x):
        return x + self.block(x)

class Generator(nn.Module):
    def __init__(self, in_c, out_c, n_blocks=6):
        super().__init__()
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_c, 64, 7),
            nn.InstanceNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.InstanceNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 256, 3, 2, 1),
            nn.InstanceNorm2d(256),
            nn.ReLU(True)
        ]
        for _ in range(n_blocks):
            model += [ResnetBlock(256)]
        model += [
            nn.ConvTranspose2d(256, 128, 3, 2, 1, output_padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 3, 2, 1, output_padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(True),
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, out_c, 7),
            nn.Tanh()
        ]
        self.model = nn.Sequential(*model)
    def forward(self, x):
        return self.model(x)

class Discriminator(nn.Module):
    def __init__(self, in_c):
        super().__init__()
        model = [
            nn.Conv2d(in_c, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(256, 512, 4, 1, 1),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(512, 1, 4, 1, 1)
        ]
        self.model = nn.Sequential(*model)
    def forward(self, x):
        return self.model(x)

# --- Normalization Utility ---
def normalize_tensor(img):
    # Assumes input is float32 or float64, range [0, 255] or [0, 1]
    if img.max() > 1.1:
        img = img / 127.5 - 1.0
    return img

# --- Image Pool for Discriminator Training (to reduce model oscillation) ---
class ImagePool:
    def __init__(self, pool_size=50):
        self.pool_size = pool_size
        self.images = []
    def query(self, images):
        return_images = []
        for image in images:
            image = image.unsqueeze(0)
            if len(self.images) < self.pool_size:
                self.images.append(image)
                return_images.append(image)
            else:
                if random.random() > 0.5:
                    idx = random.randint(0, self.pool_size - 1)
                    tmp = self.images[idx].clone()
                    self.images[idx] = image
                    return_images.append(tmp)
                else:
                    return_images.append(image)
        return torch.cat(return_images, 0)

# --- Dataset Loader ---
class SAR2EODataset(Dataset):
    def __init__(self, root, config, max_samples=None):
        # Support both the original structure and trainA/trainB
        sar_dir = os.path.join(root, config, 'sar')
        eo_dir = os.path.join(root, config, 'eo')
        if os.path.exists(sar_dir) and os.path.exists(eo_dir):
            self.sar_dir = sar_dir
            self.eo_dir = eo_dir
            self.sar_files = sorted(os.listdir(self.sar_dir))
            self.eo_files = sorted(os.listdir(self.eo_dir))
        else:
            # Fallback to trainA/trainB
            self.sar_dir = os.path.join(root, 'trainA')
            self.eo_dir = os.path.join(root, 'trainB')
            self.sar_files = sorted(os.listdir(self.sar_dir))
            self.eo_files = sorted(os.listdir(self.eo_dir))
        # Pair by filename if possible, else by index
        if len(self.sar_files) != len(self.eo_files):
            print(f"Warning: trainA and trainB have different number of files ({len(self.sar_files)} vs {len(self.eo_files)}). Pairing by index.")
            self.pairs = list(zip(self.sar_files, self.eo_files))
        else:
            # Try to pair by filename
            pairs = []
            eo_set = set(self.eo_files)
            for sar_file in self.sar_files:
                if sar_file in eo_set:
                    pairs.append((sar_file, sar_file))
                else:
                    pairs.append((sar_file, None))
            # If any None, fallback to index pairing
            if any(eo is None for _, eo in pairs):
                print("Warning: Not all SAR filenames match EO filenames. Pairing by sorted order.")
                self.pairs = list(zip(self.sar_files, self.eo_files))
            else:
                self.pairs = pairs
        if max_samples:
            self.pairs = self.pairs[:max_samples]
    def __len__(self):
        return len(self.pairs)
    def __getitem__(self, idx):
        sar_file, eo_file = self.pairs[idx]
        sar_path = os.path.join(self.sar_dir, sar_file)
        if eo_file is not None:
            eo_path = os.path.join(self.eo_dir, eo_file)
        else:
            raise FileNotFoundError(f"No matching EO file for SAR file {sar_file}")
        # --- SAR robust 2-channel loading ---
        if sar_path.endswith('.npy'):
            sar = np.load(sar_path)
        elif sar_path.lower().endswith(('.jpg', '.jpeg')):
            sar_img = Image.open(sar_path)
            sar = np.array(sar_img)
            if sar.ndim == 3:
                # If RGB, use only first two channels
                sar = sar[..., :2]
            elif sar.ndim == 2:
                # If grayscale, duplicate to make 2 channels
                sar = np.stack([sar, sar], axis=-1)
            else:
                raise ValueError(f'Unexpected SAR image shape: {sar.shape}')
        else:
            raise ValueError(f'Unsupported SAR file type: {sar_path}')
        # --- EO loading (as before) ---
        if eo_path.endswith('.npy'):
            eo = np.load(eo_path)
        elif eo_path.lower().endswith(('.jpg', '.jpeg')):
            eo = np.array(Image.open(eo_path).convert('RGB'))
        else:
            raise ValueError(f'Unsupported EO file type: {eo_path}')
        # --- Convert to torch tensor and normalize ---
        if sar.ndim == 2:
            sar = sar[..., None]  # Add channel dim if grayscale (should not happen now)
        if eo.ndim == 2:
            eo = eo[..., None]
        sar = torch.from_numpy(sar).permute(2,0,1).float()
        eo = torch.from_numpy(eo).permute(2,0,1).float()
        sar = normalize_tensor(sar)
        eo = normalize_tensor(eo)
        return sar, eo

# --- Training Utilities ---
def save_samples(G, loader, device, out_dir, step):
    G.eval()
    os.makedirs(out_dir, exist_ok=True)
    with torch.no_grad():
        for i, (sar, _) in enumerate(loader):
            sar = sar.to(device)
            fake_eo = G(sar)
            grid = make_grid(fake_eo, nrow=4, normalize=True, value_range=(-1,1))
            save_image(grid, os.path.join(out_dir, f'sample_{step}_{i}.png'))
            if i >= 1: break
    G.train()

# --- Main Training Loop ---
def main():
    parser = argparse.ArgumentParser(description='Train CycleGAN for SAR-to-EO')
    parser.add_argument('--data_dir', type=str, default='./preprocessed', help='Preprocessed data directory')
    parser.add_argument('--config', type=str, default='rgb', choices=['rgb','nir_swre','rgb_nir'], help='EO band config')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--out_dir', type=str, default='./generated_samples', help='Output dir for samples and checkpoints')
    parser.add_argument('--max_samples', type=int, default=None, help='Limit dataset size for quick runs')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Infer channels
    sar_channels = 2  # Sentinel-1 (VV, VH)
    eo_channels = {'rgb':3, 'nir_swre':3, 'rgb_nir':4}[args.config]
    # Data
    dataset = SAR2EODataset(args.data_dir, args.config, args.max_samples)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    # Models
    G = Generator(sar_channels, eo_channels).to(device)
    F = Generator(eo_channels, sar_channels).to(device)
    D_X = Discriminator(sar_channels).to(device)
    D_Y = Discriminator(eo_channels).to(device)
    # Losses
    mse = nn.MSELoss()
    l1 = nn.L1Loss()
    # Optims
    g_opt = optim.Adam(list(G.parameters())+list(F.parameters()), lr=args.lr, betas=(0.5,0.999))
    d_x_opt = optim.Adam(D_X.parameters(), lr=args.lr, betas=(0.5,0.999))
    d_y_opt = optim.Adam(D_Y.parameters(), lr=args.lr, betas=(0.5,0.999))
    # Learning Rate Decay Schedulers
    lr_lambda = lambda epoch: 1.0 - max(0, epoch + 1 - args.epochs//2) / float(args.epochs//2 + 1)
    g_scheduler = optim.lr_scheduler.LambdaLR(g_opt, lr_lambda=lr_lambda)
    d_x_scheduler = optim.lr_scheduler.LambdaLR(d_x_opt, lr_lambda=lr_lambda)
    d_y_scheduler = optim.lr_scheduler.LambdaLR(d_y_opt, lr_lambda=lr_lambda)
    # Image Pools
    fake_X_pool = ImagePool(50)
    fake_Y_pool = ImagePool(50)
    # Labels
    real_label = 1.0
    fake_label = 0.0
    # Loss tracking lists
    g_losses, d_x_losses, d_y_losses = [], [], []
    # Training
    for epoch in range(args.epochs):
        for i, (sar, eo) in enumerate(tqdm(loader, desc=f'Epoch {epoch+1}/{args.epochs}')):
            sar, eo = sar.to(device), eo.to(device)
            # --- Train Discriminators ---
            # D_Y (EO)
            D_Y.zero_grad()
            fake_eo = G(sar)
            out_real = D_Y(eo)
            out_fake = D_Y(fake_Y_pool.query(fake_eo.detach()))
            d_y_loss = 0.5 * (mse(out_real, torch.ones_like(out_real)) + mse(out_fake, torch.zeros_like(out_fake)))
            d_y_loss.backward()
            d_y_opt.step()
            # D_X (SAR)
            D_X.zero_grad()
            out_real = D_X(sar)
            fake_sar = F(eo)
            out_fake = D_X(fake_X_pool.query(fake_sar.detach()))
            d_x_loss = 0.5 * (mse(out_real, torch.ones_like(out_real)) + mse(out_fake, torch.zeros_like(out_fake)))
            d_x_loss.backward()
            d_x_opt.step()
            # --- Train Generators ---
            G.zero_grad(); F.zero_grad()
            # Adversarial
            adv_y = mse(D_Y(G(sar)), torch.ones_like(out_real))
            adv_x = mse(D_X(F(eo)), torch.ones_like(out_real))
            # Cycle
            cyc_x = l1(F(G(sar)), sar)
            cyc_y = l1(G(F(eo)), eo)
            # Identity
            idt_x = l1(F(eo), sar)  # EO->SAR, compare to SAR
            idt_y = l1(G(sar), eo)  # SAR->EO, compare to EO
            g_loss = adv_y + adv_x + 10*(cyc_x + cyc_y) + 5*(idt_x + idt_y)
            g_loss.backward()
            g_opt.step()
            # --- Logging & Visualization ---
            if i % 100 == 0:
                print(f'Epoch {epoch+1} [{i}/{len(loader)}] D_Y: {d_y_loss.item():.3f} D_X: {d_x_loss.item():.3f} G: {g_loss.item():.3f}')
            # Track losses
            g_losses.append(g_loss.item())
            d_x_losses.append(d_x_loss.item())
            d_y_losses.append(d_y_loss.item())
        # Save samples and checkpoints
        save_samples(G, loader, device, os.path.join(args.out_dir, args.config), epoch)
        torch.save(G.state_dict(), os.path.join(args.out_dir, f'G_{args.config}_epoch{epoch+1}.pth'))
        # Step learning rate schedulers
        g_scheduler.step()
        d_x_scheduler.step()
        d_y_scheduler.step()
    print('Training complete.')
    # --- Plot and save loss curves ---
    plt.figure(figsize=(10,6))
    plt.plot(g_losses, label='Generator Loss')
    plt.plot(d_x_losses, label='Discriminator X Loss')
    plt.plot(d_y_losses, label='Discriminator Y Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Training Loss Curves')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, 'loss_curves.png'))
    plt.close()

if __name__ == '__main__':
    main() 