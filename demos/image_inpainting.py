# demos/image_inpainting.py

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from compressed_sensing import create_dct_dictionary, matching_pursuit
# For reconstruction, we just need the matrix multiplication Psi * s
# Since matching pursuit gives s, we can compute x = Theta_full * s logic manually or reuse logic

def generate_phantom(size=32):
    """Creates a simple 2D geometric shape."""
    img = np.zeros((size, size))
    # Draw a rectangle
    margin = size // 4
    img[margin:3*margin, margin:3*margin] = 1.0
    return img

def main():
    print("Running Image Inpainting Demo...")
    
    # 1. Setup Image
    N_side = 32
    N = N_side * N_side
    img = generate_phantom(N_side)
    x_true = img.flatten()
    
    # 2. Masking (50% Missing Data)
    M = int(N * 0.5)
    np.random.seed(101)
    
    # Indices we KEEP
    known_indices = np.sort(np.random.choice(N, M, replace=False))
    y_measurements = x_true[known_indices]
    
    # Create Masked Image for visualization
    img_masked = np.zeros_like(x_true)
    img_masked[:] = np.nan # Use NaN to show missing pixels
    img_masked[known_indices] = y_measurements
    
    # 3. Compressed Sensing Recovery
    print(f"Reconstructing {N} pixels from {M} samples using DCT Sparse Prior...")
    
    # Create Sensing Matrix (Slow step for larger images)
    Theta = create_dct_dictionary(known_indices, N)
    
    # Solve for sparse coefficients in DCT domain
    # Images are sparse in DCT (few coefficients describe the square)
    s_est = matching_pursuit(y_measurements, Theta, max_iterations=50, tolerance=1e-2)
    
    # Reconstruct Full Image
    # We need the full Inverse DCT matrix (Dictionary for all pixels)
    # Ideally we'd have a fast IDCT function, but we'll re-generate the full basis matrix 
    # using our util (Phi would be identity here)
    all_indices = np.arange(N)
    Psi = create_dct_dictionary(all_indices, N)
    
    x_recon = np.real(np.dot(Psi, s_est))
    
    # 4. Plotting
    os.makedirs("plots", exist_ok=True)
    
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    
    ax[0].imshow(img, cmap='gray', vmin=0, vmax=1)
    ax[0].set_title("Original Image (Ground Truth)")
    ax[0].axis('off')
    
    # Helper to handle NaN for imshow
    masked_view = img_masked.reshape((N_side, N_side)).copy()
    # Fill nan with 0.5 (gray) for visibility or keep distinct
    current_cmap = plt.cm.gray
    current_cmap.set_bad(color='red')
    
    ax[1].imshow(masked_view, cmap=current_cmap, vmin=0, vmax=1)
    ax[1].set_title(f"Corrupted Input ({int(M/N*100)}% pixels)")
    ax[1].axis('off')
    
    ax[2].imshow(x_recon.reshape((N_side, N_side)), cmap='gray', vmin=0, vmax=1)
    ax[2].set_title(f"CS Reconstruction")
    ax[2].axis('off')
    
    plt.tight_layout()
    plt.savefig("plots/10_image_inpainting.png")
    print("Demo complete. Result saved to plots/10_image_inpainting.png")

if __name__ == "__main__":
    main()
