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

# --- Dataset Loader ---
class SAR2EODataset(Dataset):
    def __init__(self, root, config, max_samples=None):
        self.sar_dir = os.path.join(root, config, 'sar')
        self.eo_dir = os.path.join(root, config, 'eo')
        self.files = sorted(os.listdir(self.sar_dir))
        if max_samples:
            self.files = self.files[:max_samples]
    def __len__(self):
        return len(self.files)
    def __getitem__(self, idx):
        sar = np.load(os.path.join(self.sar_dir, self.files[idx]))
        eo = np.load(os.path.join(self.eo_dir, self.files[idx]))
        sar = torch.from_numpy(sar).permute(2,0,1).float()
        eo = torch.from_numpy(eo).permute(2,0,1).float()
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
    # Labels
    real_label = 1.0
    fake_label = 0.0
    # Training
    for epoch in range(args.epochs):
        for i, (sar, eo) in enumerate(tqdm(loader, desc=f'Epoch {epoch+1}/{args.epochs}')):
            sar, eo = sar.to(device), eo.to(device)
            # --- Train Discriminators ---
            # D_Y (EO)
            D_Y.zero_grad()
            out_real = D_Y(eo)
            out_fake = D_Y(G(sar).detach())
            d_y_loss = 0.5 * (mse(out_real, torch.ones_like(out_real)) + mse(out_fake, torch.zeros_like(out_fake)))
            d_y_loss.backward()
            d_y_opt.step()
            # D_X (SAR)
            D_X.zero_grad()
            out_real = D_X(sar)
            out_fake = D_X(F(eo).detach())
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
            idt_x = l1(F(sar), sar)
            idt_y = l1(G(eo), eo)
            g_loss = adv_y + adv_x + 10*(cyc_x + cyc_y) + 5*(idt_x + idt_y)
            g_loss.backward()
            g_opt.step()
            # --- Logging & Visualization ---
            if i % 100 == 0:
                print(f'Epoch {epoch+1} [{i}/{len(loader)}] D_Y: {d_y_loss.item():.3f} D_X: {d_x_loss.item():.3f} G: {g_loss.item():.3f}')
        # Save samples and checkpoints
        save_samples(G, loader, device, os.path.join(args.out_dir, args.config), epoch)
        torch.save(G.state_dict(), os.path.join(args.out_dir, f'G_{args.config}_epoch{epoch+1}.pth'))
    print('Training complete.')

if __name__ == '__main__':
    main() 