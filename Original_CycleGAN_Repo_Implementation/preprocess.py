import tifffile
import numpy as np
import os
from PIL import Image

# img = tifffile.imread('/home/saksham/Downloads/ROIs2017_winter_s1/ROIs2017_winter/s1_8/ROIs2017_winter_s1_8_p30.tif')
# img = img[:, :, 0]

case = input("Enter case (S1 or S2): ")


if case == 'S1':
    root_dir = 'ROIs2017_winter_s1/ROIs2017_winter'
    save_dir = './DATA_S1'

    os.makedirs(save_dir, exist_ok=True)

    b = os.listdir(root_dir)

    for i in b:
        fname = os.path.join(root_dir, i)
        for x, j in enumerate(os.listdir(fname)):
            if j.endswith('.tif'):
                img = tifffile.imread(os.path.join(fname, j))

                # Sentinel-2 typical: B4 (Red), B3 (Green), B2 (Blue) at indices 3, 2, 1
                img = img[:, :, 0]

                # Convert to 8-bit format (0-255) for JPEG
                rgb_8bit = (img * 255).astype(np.uint8)
                filename = f"S1_{i}_{x:03d}.jpeg"
                save_path = os.path.join(save_dir, filename)

                # Save as JPEG
                Image.fromarray(rgb_8bit).save(save_path, format='JPEG', quality=100)
                print(f"Saved {filename} to {save_dir}")

elif case == 'S2':
    root_dir = 'ROIs2017_winter_s2/ROIs2017_winter'
    save_dir = './DATA_S2'

    os.makedirs(save_dir, exist_ok=True)

    b = os.listdir(root_dir)

    for i in b:
        fname = os.path.join(root_dir, i)
        for x, j in enumerate(os.listdir(fname)):
            if j.endswith('.tif'):
                img = tifffile.imread(os.path.join(fname, j))
                if img.ndim == 3 and img.shape[2] >= 4:
                    # Sentinel-2 typical: B4 (Red), B3 (Green), B2 (Blue) at indices 3, 2, 1
                    rgb = img[:, :, [3, 2, 1]]

                    # Normalize each channel
                    def normalize(channel):
                        return (channel - np.min(channel)) / (np.max(channel) - np.min(channel) + 1e-6)

                    rgb_normalized = np.stack([normalize(rgb[:, :, i]) for i in range(3)], axis=-1)

                    # Convert to 8-bit format (0-255) for JPEG
                    rgb_8bit = (rgb_normalized * 255).astype(np.uint8)
                    filename = f"S2_{i}_{x:03d}.jpeg"
                    save_path = os.path.join(save_dir, filename)

                    # Save as JPEG
                    Image.fromarray(rgb_8bit).save(save_path, format='JPEG', quality=100)
                    print(f"Saved {filename} to {save_dir}")




