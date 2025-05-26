import torch
import os, sys, json, time, warnings, inspect
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torchvision.transforms as transforms_tv
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import seaborn as sns

from config import CONFIG, device, SEED, TRANSFORMERS_AVAILABLE
from utils import load_data, load_models, instantiate_attack
from attacks import SAGFMA, SAGFMA3, SAGFMA2, FMIA, torchattacks

def get_visualization_attack_pool():
    """
    Modular Attack Pool: Easy to add/remove attack variants.
    Add new attacks here or comment out unwanted ones.
    """
    pool = {}

    # === Standard Torchattacks ===
    pool['FGSM'] = {
        'class': torchattacks.FGSM,
        'params': {'eps': CONFIG['epsilon']}
    }
    pool['PGD'] = {
        'class': torchattacks.PGD,
        'params': {
            'eps': CONFIG['epsilon'],
            'alpha': CONFIG['alpha'],
            'steps': CONFIG['steps'],
            'random_start': True
        }
    }
    pool['MIFGSM'] = {
        'class': torchattacks.MIFGSM,
        'params': {
            'eps': CONFIG['epsilon'],
            'alpha': CONFIG['alpha'],
            'steps': CONFIG['steps'],
            'decay': 1.0
        }
    }
    pool['TIM'] = {
        'class': torchattacks.TIFGSM,
        'params': {
            'eps': CONFIG['epsilon'],
            'alpha': CONFIG['alpha'],
            'steps': CONFIG['steps'],
            'decay': 1.0,
            'kernel_name': 'gaussian',
            'len_kernel': 15
        }
    }
    # pool['DeepFool'] = {
    #     'class': torchattacks.DeepFool,
    #     'params': {
    #         'steps': CONFIG['steps'],
    #         'overshoot': 0.02
    #     }
    # }

    # === SAGFMA Variants ===
    sagfma_params = {
        'epsilon': CONFIG['epsilon'],
        'steps': CONFIG['steps'],
        'decay': 1.0,
        'alpha': CONFIG['alpha'],
        'freq_bands': 4,
        'attention_iters': 3,
        'adaptive': True
    }
    pool['SAGFMA_SingleScale'] = {
        'class': SAGFMA,
        'params': sagfma_params
    }
    pool['SAGFMA_SingleScale_w'] = {
        'class': SAGFMA3,
        'params': sagfma_params
    }

    # === SAGFMA2 Variants ===
    sagfma2_base = {
        'epsilon': CONFIG['epsilon'],
        'steps': CONFIG['steps'],
        'alpha': CONFIG['alpha'],
        'decay': 1.0,
        'gamma': 2.0,
        'modulation_strength': 0.05,
        'adaptive': False
    }
    pool['SAGFMA2_Base'] = {
        'class': SAGFMA2,
        'params': {**sagfma2_base, 'variant': 'base'}
    }
    pool['SAGFMA2_NAG'] = {
        'class': SAGFMA2,
        'params': {**sagfma2_base, 'variant': 'nag', 'decay': 0.9}
    }
    pool['SAGFMA2_FreqUpdate'] = {
        'class': SAGFMA2,
        'params': {**sagfma2_base, 'variant': 'freq_update'}
    }
    no_decay = dict(sagfma2_base)
    no_decay.pop('decay', None)
    pool['SAGFMA2_NoFreqMomentum'] = {
        'class': SAGFMA2,
        'params': {**no_decay, 'variant': 'no_freq_momentum'}
    }
    pool['SAGFMA2_Base_Adaptive'] = {
        'class': SAGFMA2,
        'params': {**sagfma2_base, 'variant': 'base', 'adaptive': True}
    }
    
    # Conditional: Only add ensemble variant if transformers is available
    if TRANSFORMERS_AVAILABLE:
        pool['SAGFMA2_Ensemble'] = {
            'class': SAGFMA2,
            'params': {**sagfma2_base, 'variant': 'ensemble'}
        }

    # # === FMIA Variants ===
    fmia_base = {
        'epsilon': CONFIG['epsilon'],
        'steps': CONFIG['steps'],
        'alpha': CONFIG['alpha'],
        'decay': 1.0,
        'low_freq_percent': 0.2,
        'high_freq_percent': 0.3,
        'filter_type': 'low_pass',
        'di_prob': 0.7,
        'si_scales': 5
    }
    pool['Base_FMIA'] = {
        'class': FMIA, 
        'params': {**fmia_base, 'variant': 'base', 'filter_type': 'low_pass'}
    }
    pool['Nesterov_FMIA'] = {
        'class': FMIA, 
        'params': {**fmia_base, 'variant': 'nesterov', 'filter_type': 'low_pass'}
    }
    pool['DI_FMIA'] = {
        'class': FMIA, 
        'params': {**fmia_base, 'variant': 'di', 'filter_type': 'low_pass'}
    }
    pool['SI_FMIA'] = {
        'class': FMIA, 
        'params': {**fmia_base, 'variant': 'si', 'filter_type': 'low_pass'}
    }
    pool['F-Grad_FMIA'] = {
        'class': FMIA, 
        'params': {**fmia_base, 'variant': 'f-grad', 'filter_type': 'low_pass'}
    }
    pool['HighPass_FMIA'] = {
        'class': FMIA, 
        'params': {**fmia_base, 'variant': 'base', 'filter_type': 'high_pass'}
    }
    pool['DI_SI_FMIA'] = {
        'class': FMIA, 
        'params': {**fmia_base, 'variant': 'di', 'si_scales': fmia_base['si_scales'], 'filter_type': 'low_pass'}
    }
    pool['NES_DI_SI_FMIA'] = {
        'class': FMIA, 
        'params': {**fmia_base, 'variant': 'nesterov-di-si', 'filter_type': 'low_pass'}
    }

    return pool

def select_better_image_reproducible(dataset, seed_offset=42):
    """Select a better image for visualization while maintaining reproducibility"""
    # Use different seed offset to get different images
    np.random.seed(SEED + seed_offset)
    
    # Try multiple candidates and select one with good characteristics
    candidates = np.random.choice(len(dataset), min(50, len(dataset)), replace=False)
    
    best_idx = candidates[0]  # Default fallback
    
    # Simple heuristic: prefer images with more variation (higher std)
    best_std = 0
    for idx in candidates[:10]:  # Check first 10 candidates
        img, _ = dataset[idx]
        img_std = img.std().item()
        if img_std > best_std:
            best_std = img_std
            best_idx = idx
    
    print(f"Selected image index {best_idx} (std: {best_std:.4f})")
    return 42


def filter_visualization_attacks(pool, categories=None):
    """
    Filter attacks by category for selective visualization.
    Categories: 'standard', 'sagfma', 'sagfma2', 'fmia', 'all'
    """
    if categories is None or 'all' in categories:
        return pool
    
    filtered = {}
    for attack_name, attack_info in pool.items():
        if 'standard' in categories and attack_name in ['FGSM', 'PGD', 'MIFGSM', 'TIM']:
            filtered[attack_name] = attack_info
        elif 'sagfma' in categories and 'SAGFMA_' in attack_name and 'SAGFMA2_' not in attack_name:
            filtered[attack_name] = attack_info
        elif 'sagfma2' in categories and 'SAGFMA2_' in attack_name:
            filtered[attack_name] = attack_info
        elif 'fmia' in categories and 'FMIA' in attack_name:
            filtered[attack_name] = attack_info
    
    return filtered


def generate_proper_heatmaps(viz_base, img, adv_images, attack_objects):
    """Generate heatmaps the correct way with proper FFT and frequency bands"""
    
    n = len(adv_images)
    if n == 0:
        return
    
    # Create the figure with proper sizing
    fig2, axes2 = plt.subplots(n, 3, figsize=(9, 2.5*n), constrained_layout=True)
    if n == 1:
        axes2 = axes2.reshape(1, 3)

    for i, name in enumerate(adv_images.keys()):
        print(f"  Generating heatmap for {name}")
        
        try:
            # Get the adversarial image and compute delta
            adv_img = adv_images[name][0].cpu()
            clean_np = img[0].cpu().permute(1, 2, 0).numpy()
            adv_np = adv_img.permute(1, 2, 0).numpy()
            delta = adv_img - img[0].cpu()  # Perturbation
            
            # Compute FFT of the perturbation (delta)
            fft = torch.fft.fft2(delta, norm='ortho')
            mag = torch.abs(fft)
            mag_shift = torch.fft.fftshift(mag, dim=(-2, -1))
            mag_np = mag_shift.numpy().mean(axis=0)  # Average across channels
            
            # Plot clean image
            axes2[i, 0].imshow(clean_np)
            axes2[i, 0].axis('off')
            axes2[i, 0].set_title('Clean')
            
            # Plot adversarial image
            axes2[i, 1].imshow(adv_np)
            axes2[i, 1].axis('off')
            axes2[i, 1].set_title(name)
            
            # Plot FFT magnitude spectrum
            fft_vis = np.log1p(mag_np)  # Log scale for better visualization
            axes2[i, 2].imshow(fft_vis, cmap='inferno')
            
            # Get frequency bands for the attack
            attack_obj = attack_objects[name]
            mask_np = get_frequency_bands_for_attack(attack_obj, img.shape)
            
            # Enhanced band visualization (for attacks with frequency filtering)
            if mask_np is not None and mask_np.max() > 0:
                unique_bands = np.unique(mask_np)
                colors = ['cyan', 'yellow', 'red', 'green', 'blue', 'magenta', 'orange', 'lime']
                
                for idx, band_val in enumerate(unique_bands[unique_bands > 0]):
                    band_mask = (mask_np == band_val).astype(float)
                    if band_mask.sum() > 0:  # Only draw if band exists
                        color = colors[idx % len(colors)]
                        axes2[i, 2].contour(band_mask, levels=[0.5], 
                                          colors=[color], linewidths=1.5, alpha=0.8)
            
            # Update title to indicate attack type and frequency bands
            attack_type = ""
            if is_fmia_attack(attack_obj):
                attack_type = " (FMIA)"
            elif is_sagfma_attack(attack_obj):
                attack_type = " (SAGFMA)"
            
            title_suffix = ' + Bands' if mask_np is not None and mask_np.max() > 0 else ''
            axes2[i, 2].axis('off')
            axes2[i, 2].set_title(f'FFT{attack_type}{title_suffix}')
            
        except Exception as e:
            print(f"    ✗ Failed to generate heatmap for {name}: {e}")
            # Fill with placeholder if failed
            for j in range(3):
                axes2[i, j].text(0.5, 0.5, f'Error: {name}', 
                                ha='center', va='center', transform=axes2[i, j].transAxes)
                axes2[i, j].axis('off')

    #plt.suptitle('FFT Heatmaps with Frequency Band Analysis', fontsize=14, y=0.98)
    plt.savefig(os.path.join(viz_base, 'heatmaps_fft.png'), dpi=300, bbox_inches='tight')
    plt.close(fig2)
    print("✓ FFT heatmaps with frequency bands created")



# === Helper Functions for Frequency Analysis ===
def compute_radial_profile(fft_magnitude_numpy_array):
    """Compute radial energy profile of FFT magnitude"""
    H, W = fft_magnitude_numpy_array.shape
    center_y, center_x = H // 2, W // 2
    y_coords, x_coords = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    distances = np.sqrt((y_coords - center_y)**2 + (x_coords - center_x)**2).astype(np.int32)
    max_radius = distances.max() + 1
    
    radial_energy_profile = np.zeros(max_radius)
    pixel_counts_per_radius = np.zeros(max_radius)
    
    for r_val in range(max_radius):
        radius_mask = (distances == r_val)
        if np.any(radius_mask):
            radial_energy_profile[r_val] = fft_magnitude_numpy_array[radius_mask].sum()
            pixel_counts_per_radius[r_val] = radius_mask.sum()
            
    return radial_energy_profile / (pixel_counts_per_radius + 1e-9)

def is_fmia_attack(attack_obj):
    """Check if attack is FMIA type"""
    return hasattr(attack_obj, '_fft_2d') and hasattr(attack_obj, '_frequency_filter')

def is_sagfma_attack(attack_obj):
    """Check if attack is SAGFMA type"""
    return hasattr(attack_obj, '_precompute_frequency_masks') and hasattr(attack_obj, 'freq_bands')

def get_frequency_bands_for_attack(attack_obj, image_shape):
    """Universal frequency band extraction for different attack types"""
    if is_fmia_attack(attack_obj):
        return get_freq_bands_fmia(attack_obj, image_shape)
    elif is_sagfma_attack(attack_obj):
        return get_freq_bands_sagfma(attack_obj, image_shape)
    return None

def get_freq_bands_fmia(attack_obj, image_shape):
    """Extract frequency bands from FMIA attack"""
    try:
        if hasattr(attack_obj, '_last_band_mask') and attack_obj._last_band_mask is not None:
            return attack_obj._last_band_mask.cpu().squeeze().numpy()
        else:
            # Generate dummy bands if not available
            dummy_fft = torch.rand(image_shape, dtype=torch.complex64, device=attack_obj.device)
            attack_obj._frequency_filter(dummy_fft, image_shape)
            if hasattr(attack_obj, '_last_band_mask') and attack_obj._last_band_mask is not None:
                return attack_obj._last_band_mask.cpu().squeeze().numpy()
    except Exception as e:
        print(f"Warning: Could not extract FMIA frequency bands: {e}")
    return None

def get_freq_bands_sagfma(attack_obj, image_shape):
    """Extract frequency bands from SAGFMA-type attack"""
    try:
        H, W = image_shape[-2], image_shape[-1]
        device_to_use = getattr(attack_obj, 'freq_weights', torch.empty(0)).device
        
        if device_to_use.type == 'cpu' and hasattr(attack_obj, 'model'):
            model_params = list(attack_obj.model.parameters())
            if model_params:
                device_to_use = model_params[0].device

        # Get or generate frequency masks
        mask_key = (H, W, device_to_use.type)
        if hasattr(attack_obj, '_freq_masks_cache') and mask_key in attack_obj._freq_masks_cache:
            freq_masks = attack_obj._freq_masks_cache[mask_key]
        else:
            freq_masks = attack_obj._precompute_frequency_masks(H, W, device=device_to_use)
            if hasattr(attack_obj, '_freq_masks_cache'):
                attack_obj._freq_masks_cache[mask_key] = freq_masks
        
        # Combine masks for visualization
        combined_mask = torch.zeros((H, W), device=device_to_use, dtype=torch.long)
        for band_idx, mask_tensor in enumerate(freq_masks):
            combined_mask[mask_tensor.to(device_to_use) > 0.5] = band_idx + 1
        return combined_mask.cpu().numpy()
    except Exception as e:
        print(f"Warning: Could not extract SAGFMA frequency bands: {e}")
    return None

# === Feature Extraction for Penultimate Layer ===
class FeatureExtractor:
    def __init__(self, model):
        self.model = model
        self.features = {}
        self.hooks = []
        self._register_hooks()
    
    def _register_hooks(self):
        def hook_fn(name):
            def hook(module, input, output):
                # Extract penultimate features
                feat = output.detach().cpu().numpy()
                if feat.ndim > 2:
                    feat = feat.reshape(feat.shape[0], -1).mean(axis=1)
                self.features[name] = feat
            return hook
        
        # Find penultimate layer
        target_layer = None
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'layer4'):
            target_layer = self.model.model.layer4
        elif hasattr(self.model, 'layer4'):
            target_layer = self.model.layer4
        elif hasattr(self.model, 'features'):
            target_layer = self.model.features[-1]
        
        if target_layer is not None:
            handle = target_layer.register_forward_hook(hook_fn('penultimate'))
            self.hooks.append(handle)
    
    def extract_features(self, x):
        self.features.clear()
        with torch.no_grad():
            _ = self.model(x)
        return self.features.get('penultimate', None)
    
    def cleanup(self):
        for handle in self.hooks:
            handle.remove()
        self.hooks.clear()
def main_visualization_script():
    print("\n" + "="*60 + "\n Adversarial Attack Visualization Script \n" + "="*60)
    
    # Load data and models
    val_loader = load_data(CONFIG)
    models_dict = load_models(CONFIG)
    primary_model = models_dict.get('resnet50_clean', next(iter(models_dict.values())))
    viz_base = 'visualizations'
    os.makedirs(viz_base, exist_ok=True)
    
    print(f"Visualizations will be saved in: {viz_base}")

    # Get attack pool
    full_attack_pool = get_visualization_attack_pool()
    selected_categories = ['all']  # Change this to filter attacks
    attacks_to_run = filter_visualization_attacks(full_attack_pool, selected_categories)
    
    if not attacks_to_run:
        print("Warning: No attacks selected for visualization.")
        return
    
    print(f"Selected {len(attacks_to_run)} attacks for visualization:")
    for attack_id in attacks_to_run:
        print(f" - {attack_id}")

    # Select image with better reproducibility
    dataset = val_loader.dataset
    
    # CUSTOMIZATION POINT: Change seed_offset to get different images
    rand_idx = select_better_image_reproducible(dataset, seed_offset=42)  # Try 42, 100, 200, etc.
    
    img, lbl = dataset[rand_idx]
    img = img.unsqueeze(0).to(device)
    lbl = torch.tensor([lbl], device=device)

    # Storage for results
    clean_feats = []
    adv_feats = []
    all_profiles = []
    adv_images = {}
    attack_objects = {}

    successful_attacks = 0
    failed_attacks = 0

    # Process each attack
    for name, info in attacks_to_run.items():
        print(f"Processing attack: {name}")
        
        try:
            atk = instantiate_attack(info['class'], primary_model, info['params'])
            attack_objects[name] = atk
            
            adv = atk(img, lbl)
            adv_images[name] = adv.detach()
            
            attack_dir = os.path.join(viz_base, name)
            os.makedirs(attack_dir, exist_ok=True)

            # Individual attack visualizations
            # Side-by-side comparison
            fig, axes = plt.subplots(1, 2, figsize=(6, 3))
            axes[0].imshow(img[0].cpu().permute(1, 2, 0).numpy())
            axes[0].axis('off')
            axes[0].set_title('Clean')
            axes[1].imshow(adv[0].cpu().permute(1, 2, 0).numpy())
            axes[1].axis('off')
            axes[1].set_title(name)
            fig.savefig(os.path.join(attack_dir, 'comparison.png'), dpi=300, bbox_inches='tight')
            plt.close(fig)

            # FFT spectrum of perturbation
            delta = (adv - img)[0]
            fft = torch.fft.fft2(delta.cpu(), norm='ortho')
            mag = torch.abs(fft)
            mag_shift = torch.fft.fftshift(mag)
            mag_np = mag_shift.numpy().mean(axis=0)
            
            plt.figure(figsize=(4, 4))
            plt.imshow(np.log1p(mag_np), cmap='inferno')
            plt.axis('off')
            plt.title('FFT Spectrum')
            plt.savefig(os.path.join(attack_dir, 'fft_spectrum.png'), dpi=300, bbox_inches='tight')
            plt.close()

            # Radial profile for aggregate plot
            profile = compute_radial_profile(mag_np)
            for r, val in enumerate(profile[:100]):
                all_profiles.append({'variant': name, 'radius': r, 'energy': val})

            successful_attacks += 1
            print(f"  Successfully processed {name}")
            
        except Exception as e:
            print(f"  Failed to process {name}: {e}")
            failed_attacks += 1
            continue

    if successful_attacks == 0:
        print("Warning: No attacks succeeded. Cannot create visualizations.")
        return

    # Create aggregate visualizations
    print(f"\nCreating aggregate visualizations for {successful_attacks} successful attacks...")

    # 1. Radial overlay plot
    if all_profiles:
        try:
            df = pd.DataFrame(all_profiles)
            plt.figure(figsize=(10, 6))
            for variant in df['variant'].unique():
                subset = df[df['variant'] == variant]
                plt.plot(subset['radius'], subset['energy'], label=variant, alpha=0.8, linewidth=2)
            plt.title("Radial Frequency Energy Comparison Across Attack Variants")
            plt.xlabel("Frequency Radius")
            plt.ylabel("Average Magnitude")
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(viz_base, 'radial_energy_overlay.png'), dpi=300, bbox_inches='tight')
            df.to_csv(os.path.join(viz_base, 'radial_energy_profiles.csv'), index=False)
            plt.close()
            print("Radial energy overlay created")
        except Exception as e:
            print(f"Radial overlay failed: {e}")

    # 2. Combined comparisons (2xN)
    try:
        n = len(adv_images)
        if n > 0:
            fig, axs = plt.subplots(2, n, figsize=(3*n, 6), constrained_layout=True)
            if n == 1:
                axs = axs.reshape(2, 1)
            
            for j, name in enumerate(adv_images.keys()):
                axs[0, j].imshow(img[0].cpu().permute(1, 2, 0).numpy())
                axs[0, j].axis('off')
                axs[0, j].set_title('Clean')
                axs[1, j].imshow(adv_images[name][0].cpu().permute(1, 2, 0).numpy())
                axs[1, j].axis('off')
                axs[1, j].set_title(name)
            plt.savefig(os.path.join(viz_base, 'all_comparisons.png'), dpi=300, bbox_inches='tight')
            plt.close()
            print("Combined comparisons created")
    except Exception as e:
        print(f"Combined comparisons failed: {e}")
    
    # 3. FFT side-by-side comparison
    try:
        n = len(adv_images)
        if n > 0:
            fig_fft, axs_fft = plt.subplots(2, n, figsize=(3*n, 6), constrained_layout=True)
            if n == 1:
                axs_fft = axs_fft.reshape(2, 1)

            clean_fft = torch.fft.fft2(img[0].cpu(), norm='ortho')
            clean_fft_mag = torch.fft.fftshift(torch.abs(clean_fft)).numpy().mean(axis=0)

            for j, name in enumerate(adv_images.keys()):
                adv_fft = torch.fft.fft2(adv_images[name][0].cpu(), norm='ortho')
                adv_fft_mag = torch.fft.fftshift(torch.abs(adv_fft)).numpy().mean(axis=0)

                axs_fft[0, j].imshow(np.log1p(clean_fft_mag), cmap='inferno')
                axs_fft[0, j].axis('off')
                axs_fft[0, j].set_title('FFT Clean')

                axs_fft[1, j].imshow(np.log1p(adv_fft_mag), cmap='inferno')
                axs_fft[1, j].axis('off')
                axs_fft[1, j].set_title(f'FFT {name}')

            plt.savefig(os.path.join(viz_base, 'fft_comparisons.png'), dpi=300, bbox_inches='tight')
            plt.close()
            print("FFT comparisons created")
    except Exception as e:
        print(f"FFT comparisons failed: {e}")

    # 4. Perturbation heatmaps
    try:
        n = len(adv_images)
        if n > 0:
            fig1, axes1 = plt.subplots(n, 3, figsize=(9, 2.5*n), constrained_layout=True)
            if n == 1:
                axes1 = axes1.reshape(1, 3)

            for i, name in enumerate(adv_images.keys()):
                adv_img = adv_images[name][0].cpu()
                clean_np = img[0].cpu().permute(1, 2, 0).numpy()
                adv_np = adv_img.permute(1, 2, 0).numpy()
                diff = np.abs(adv_np - clean_np).sum(axis=2)

                axes1[i, 0].imshow(clean_np)
                axes1[i, 0].axis('off')
                axes1[i, 0].set_title('Clean')
                axes1[i, 1].imshow(adv_np)
                axes1[i, 1].axis('off')
                axes1[i, 1].set_title(name)
                axes1[i, 2].imshow(diff, cmap='inferno')
                axes1[i, 2].axis('off')
                axes1[i, 2].set_title('Perturbation')
                
            plt.savefig(os.path.join(viz_base, 'heatmaps.png'), dpi=300, bbox_inches='tight')
            plt.close()
            print("Perturbation heatmaps created")
    except Exception as e:
        print(f"Perturbation heatmaps failed: {e}")

    # 5. Enhanced FFT heatmaps with frequency bands (THE CORRECTED VERSION)
    try:
        generate_proper_heatmaps(viz_base, img, adv_images, attack_objects)
    except Exception as e:
        print(f"Enhanced FFT heatmaps failed: {e}")

    print(f"\n" + "="*60)
    print(f"VISUALIZATION COMPLETE")
    print(f"="*60)
    print(f"Successful attacks visualized: {successful_attacks}")
    print(f"Failed attacks: {failed_attacks}")
    print(f"Output directory: {viz_base}")
    print(f"="*60)


if __name__ == '__main__':
    warnings.filterwarnings("ignore", category=UserWarning, module='torchvision.transforms.functional_tensor')
    warnings.filterwarnings("ignore", category=UserWarning, module='torchvision.transforms._functional_video')
    warnings.filterwarnings("ignore", category=UserWarning, module='skimage.metrics._structural_similarity')
    warnings.filterwarnings("ignore", message=".*is a deprecated alias for the builtin `float`.*")
    main_visualization_script()
