import torch
import os, sys, json, time, math, random, warnings, io
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms_tv
import torchvision.models as models_tv
import torchvision.datasets as datasets_tv
from torch.utils.data import DataLoader, Subset
from PIL import Image
from scipy.stats import norm
import inspect
from skimage.metrics import structural_similarity as ssim_skimage
from skimage.metrics import peak_signal_noise_ratio as psnr_skimage
import pandas as pd

from config import device, SEED, CONFIG

# Dynamic imports for optional libraries
try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    lpips = None
    LPIPS_AVAILABLE = False

try:
    import open_clip
    OPEN_CLIP_AVAILABLE = True
except ImportError:
    open_clip = None
    OPEN_CLIP_AVAILABLE = False

try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    timm = None
    TIMM_AVAILABLE = False

try:
    from robustbench.utils import load_model as load_robust_model
    ROBUSTBENCH_AVAILABLE = True
except ImportError:
    load_robust_model = None
    ROBUSTBENCH_AVAILABLE = False


class NormalizedModel(nn.Module):
    """Wraps a model to include input normalization."""
    def __init__(self, model, mean, std):
        super().__init__()
        self.model = model
        self.mean = torch.tensor(mean, dtype=torch.float32).view(1, -1, 1, 1).to(device)
        self.std = torch.tensor(std, dtype=torch.float32).view(1, -1, 1, 1).to(device)

    def forward(self, x):
        x_normalized = (x - self.mean.to(x.device)) / self.std.to(x.device)
        return self.model(x_normalized)

    def get_original_model(self):
        return self.model

def load_data(config_dict):
    """Loads dataset and creates dataloader."""
    transform_eval = transforms_tv.Compose([
        transforms_tv.Resize(256), 
        transforms_tv.CenterCrop(224),
        transforms_tv.ToTensor(),
    ])
    try:
        val_dataset_full = datasets_tv.ImageFolder(root=config_dict['val_dir'], transform=transform_eval)
    except Exception as e:
        print(f"ERROR loading dataset from {config_dict['val_dir']}: {e}")
        print("Please ensure 'val_dir' in config.py points to your ImageNet validation set parent directory.")
        sys.exit(1)

    subset_size = config_dict.get('test_subset_size', 0)
    if subset_size > 0 and subset_size < len(val_dataset_full):
        rng = np.random.RandomState(SEED)
        idxs = rng.choice(len(val_dataset_full), subset_size, replace=False)
        val_dataset = Subset(val_dataset_full, idxs)
        print(f"Using a subset of {len(val_dataset)} images for evaluation.")
    else:
        val_dataset = val_dataset_full
        print(f"Using the full validation set ({len(val_dataset)} images) or subset size is invalid.")

    return DataLoader(
        val_dataset,
        batch_size=config_dict['batch_size'],
        shuffle=False,
        num_workers=config_dict['num_workers'],
        pin_memory=True
    )


def load_models(config_dict):
    """Loads pretrained models (clean, robust, transformer)."""
    models_loaded = {}
    tv_mean = [0.485, 0.456, 0.406]
    tv_std = [0.229, 0.224, 0.225]

    print("Loading standard torchvision CNN models...")
    try:
        resnet50 = models_tv.resnet50(weights=models_tv.ResNet50_Weights.IMAGENET1K_V2)
        models_loaded['resnet50_clean'] = NormalizedModel(resnet50, tv_mean, tv_std).to(device).eval()
        densenet121 = models_tv.densenet121(weights=models_tv.DenseNet121_Weights.IMAGENET1K_V1)
        models_loaded['densenet121_clean'] = NormalizedModel(densenet121, tv_mean, tv_std).to(device).eval()
    except Exception as e:
        print(f"Error loading standard torchvision models: {e}")

    if TIMM_AVAILABLE:
        print("Loading Vision Transformer model (ViT via timm)...")
        try:
            vit_model_name = 'vit_base_patch16_224'
            vit_model_timm = timm.create_model(vit_model_name, pretrained=True)
            vit_cfg = vit_model_timm.default_cfg
            vit_mean = vit_cfg['mean']
            vit_std = vit_cfg['std']
            models_loaded['vit_base_patch16_timm'] = NormalizedModel(vit_model_timm, vit_mean, vit_std).to(device).eval()
        except Exception as e:
            print(f"Could not load ViT model '{vit_model_name}' from timm: {e}")
    else: 
        print("Skipping ViT model (timm library not available).")

    if ROBUSTBENCH_AVAILABLE:
        print("Loading robust model (e.g., Standard_R50 from RobustBench)...")
        try:
            robust_model_name_rb = 'Standard_R50' 
            rb_model_raw = load_robust_model(model_name=robust_model_name_rb, dataset='imagenet', threat_model='Linf')
            
            rb_mean, rb_std = tv_mean, tv_std
            if hasattr(rb_model_raw, 'normalizer') and rb_model_raw.normalizer is not None:
                if hasattr(rb_model_raw.normalizer, 'mean') and hasattr(rb_model_raw.normalizer, 'std'):
                    rb_mean_tensor = rb_model_raw.normalizer.mean
                    rb_std_tensor = rb_model_raw.normalizer.std
                    rb_mean = rb_mean_tensor.tolist() if isinstance(rb_mean_tensor, torch.Tensor) else rb_mean_tensor
                    rb_std = rb_std_tensor.tolist() if isinstance(rb_std_tensor, torch.Tensor) else rb_std_tensor
                    rb_model_raw.normalizer = nn.Identity()
            else:
                 print(f"Warning (RobustBench): Normalization stats for {robust_model_name_rb} not explicitly found. Using ImageNet defaults.")

            models_loaded['resnet50_robust_linf'] = NormalizedModel(rb_model_raw, rb_mean, rb_std).to(device).eval()
        except Exception as e:
            print(f"Could not load robust model from RobustBench: {e}")
    else: 
        print("Skipping robust models (robustbench library not available).")

    for model_n, model_o in models_loaded.items():
        model_o.eval()
        print(f"Model loaded and set to eval: {model_n}")
    return models_loaded


def calculate_confidence_interval(n_success: int, n_total: int, confidence_level: float = 0.95) -> tuple[float, float]:
    if n_total == 0: return 0.0, 0.0
    p_hat = n_success / n_total
    z_score = norm.ppf(1 - (1 - confidence_level) / 2)
    center_adjusted = p_hat + z_score**2 / (2 * n_total)
    term_val = z_score * np.sqrt((p_hat * (1 - p_hat) / n_total) + z_score**2 / (4 * n_total**2))
    denominator = 1 + z_score**2 / n_total
    low_bound = max(0.0, (center_adjusted - term_val) / denominator) * 100
    high_bound = min(1.0, (center_adjusted + term_val) / denominator) * 100
    return low_bound, high_bound

def instantiate_attack(AttackClass, model_instance, attack_params):
    sig = inspect.signature(AttackClass.__init__)
    if 'model' in sig.parameters:
        return AttackClass(model=model_instance, **attack_params)
    else:
        return AttackClass(model_instance, **attack_params)

def batch_psnr(clean_imgs: torch.Tensor, adv_imgs: torch.Tensor, data_val_range=1.0) -> np.ndarray:
    mse_val = torch.mean((clean_imgs.float() - adv_imgs.float())**2, dim=(1,2,3))
    psnr_values = 10 * torch.log10(data_val_range**2 / (mse_val + 1e-10))
    return psnr_values.cpu().numpy()

def batch_ssim(clean_imgs: torch.Tensor, adv_imgs: torch.Tensor, data_val_range=1.0, default_win_size=7) -> np.ndarray:
    ssim_results = []
    for clean_single, adv_single in zip(clean_imgs, adv_imgs):
        clean_np = clean_single.cpu().permute(1, 2, 0).numpy()
        adv_np = adv_single.cpu().permute(1, 2, 0).numpy()
        
        win_size_adjusted = min(default_win_size, clean_np.shape[0], clean_np.shape[1])
        if win_size_adjusted % 2 == 0: win_size_adjusted -= 1
        if win_size_adjusted < 3 : win_size_adjusted = 3

        try:
            is_multichannel = clean_np.ndim == 3 and clean_np.shape[-1] == 3
            if is_multichannel:
                ssim_val = ssim_skimage(clean_np, adv_np, data_range=float(data_val_range), 
                                        channel_axis=-1, win_size=win_size_adjusted)
            else:
                 ssim_val = ssim_skimage(clean_np, adv_np, data_range=float(data_val_range), 
                                        win_size=win_size_adjusted)
            ssim_results.append(ssim_val)
        except Exception as e:
            ssim_results.append(np.nan)
    return np.array(ssim_results)

def batch_lpips(clean_imgs: torch.Tensor, adv_imgs: torch.Tensor, lpips_model_obj) -> np.ndarray:
    if not LPIPS_AVAILABLE or lpips_model_obj is None:
        return np.full(clean_imgs.size(0), np.nan)
    clean_norm = (clean_imgs.float() * 2) - 1
    adv_norm = (adv_imgs.float() * 2) - 1
    with torch.no_grad():
        lpips_distances = lpips_model_obj(clean_norm.to(device), adv_norm.to(device))
    return lpips_distances.squeeze().cpu().numpy()

def batch_clip_similarity(clean_imgs: torch.Tensor, adv_imgs: torch.Tensor, clip_model_obj, clip_preprocess_func) -> np.ndarray:
    if not OPEN_CLIP_AVAILABLE or clip_model_obj is None or clip_preprocess_func is None:
        return np.full(clean_imgs.size(0), np.nan)
    
    processed_clean_batch = []
    processed_adv_batch = []
    for c_img_tensor, a_img_tensor in zip(clean_imgs, adv_imgs):
        pil_c = transforms_tv.ToPILImage()(c_img_tensor.cpu())
        pil_a = transforms_tv.ToPILImage()(a_img_tensor.cpu())
        processed_clean_batch.append(clip_preprocess_func(pil_c))
        processed_adv_batch.append(clip_preprocess_func(pil_a))
    
    # Stack preprocessed images into batches
    batch_c_tensors = torch.stack(processed_clean_batch).to(device)
    batch_a_tensors = torch.stack(processed_adv_batch).to(device)
    
    with torch.no_grad():
        features_c = clip_model_obj.encode_image(batch_c_tensors)
        features_a = clip_model_obj.encode_image(batch_a_tensors)
    
    # L2 normalize features
    features_c_norm = features_c / features_c.norm(dim=-1, keepdim=True)
    features_a_norm = features_a / features_a.norm(dim=-1, keepdim=True)
    
    # Cosine similarity
    clip_sim_scores = (features_c_norm * features_a_norm).sum(dim=-1).cpu().numpy()
    return clip_sim_scores

def jpeg_compression_defense(images_batch_tensor: torch.Tensor, jpeg_quality_val: int = 75) -> torch.Tensor:
    defended_images_list = []
    for img_tensor_single in images_batch_tensor:
        pil_img_orig = transforms_tv.ToPILImage()(img_tensor_single.cpu())
        # Save to buffer as JPEG
        jpeg_buffer = io.BytesIO()
        pil_img_orig.save(jpeg_buffer, format="JPEG", quality=jpeg_quality_val)
        jpeg_buffer.seek(0) # Rewind buffer to the beginning
        # Load from buffer and convert back to tensor
        pil_img_jpeg = Image.open(jpeg_buffer).convert("RGB")
        tensor_img_jpeg = transforms_tv.ToTensor()(pil_img_jpeg)
        defended_images_list.append(tensor_img_jpeg)
    return torch.stack(defended_images_list, dim=0).to(images_batch_tensor.device)

def save_metrics_csv(df_to_save: pd.DataFrame, file_path: str):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    df_to_save.to_csv(file_path, index=False)
    print(f"Metrics saved to: {file_path}")

