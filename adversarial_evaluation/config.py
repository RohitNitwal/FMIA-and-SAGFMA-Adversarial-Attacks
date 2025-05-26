# ==============================================================================
# Configuration & Global Settings
# ==============================================================================
import torch
import random
import numpy as np
import os # Added for results directory creation

# Reproducibility & Device
SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Configuration
CONFIG = {
    'val_dir': '/mnt/combined/rohit/NLP/benchmarking/ImagenetSubset/ILSVRC2012/val', # <<< CHANGE THIS PATH
    'map_dir': '/mnt/combined/rohit/NLP/benchmarking/ImagenetSubset/imagenet_class_index.json', # <<< CHANGE THIS PATH (if used, else can be ignored)
    'results_base_dir': './results/robust_evaluation_results',
    'adv_examples_dir': './results/adv_examples_robust',
    'visualization_base_dir': './results/visualizations',
    'batch_size': 16, # Reduced for potentially lower memory GPUs, adjust as needed
    'num_workers': 2, # Adjust based on your system cores
    'test_subset_size': 2000,  # Keeping this small for quick tests; increase for full evaluation (e.g., 2000 or 0 for full dataset)
    'epsilon': 8/255,
    'alpha': 2/255,
    'steps': 10,
    'num_classes': 1000,
    'clip_model_name': 'ViT-B-32',
    'clip_pretrained': 'laion2b_s34b_b79k',
    'lpips_net': 'alex',
    'jpeg_quality': 75,
    'randomization_prob': 0.1,
    'resolutions_to_test': [224, 160, 112],
    'epsilon_range': [2/255, 4/255, 8/255, 12/255, 16/255],
    'confidence_level': 0.95
}

# Ensure results directories exist when config is loaded
os.makedirs(CONFIG['results_base_dir'], exist_ok=True)
os.makedirs(CONFIG['visualization_base_dir'], exist_ok=True)
os.makedirs(CONFIG['adv_examples_dir'], exist_ok=True)

# Check for optional libraries
try:
    from transformers import ViTForImageClassification, ViTImageProcessor
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: 'transformers' library not found. Some attack variants (e.g., SAGFMA2 ensemble) may not be available.")

print(f"Configuration Loaded. Using device: {device}")
if not TRANSFORMERS_AVAILABLE:
    print("TRANSFORMERS_AVAILABLE is False, SAGFMA2_Ensemble variant (if used) will raise an error.")
