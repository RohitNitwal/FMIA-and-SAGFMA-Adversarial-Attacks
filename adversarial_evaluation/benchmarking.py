import torch
import os, sys, json, time, warnings
import numpy as np
import pandas as pd

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from config import CONFIG, device, SEED, TRANSFORMERS_AVAILABLE
from utils import (
    load_data, load_models, instantiate_attack, 
    jpeg_compression_defense, save_metrics_csv,
    LPIPS_AVAILABLE, OPEN_CLIP_AVAILABLE
)
from evaluations import (
    evaluate_attack_robust,
    evaluate_transferability_robust,
    evaluate_defenses_robust,
    evaluate_resolution_robust,
    evaluate_epsilon_sensitivity
)
from attacks import SAGFMA, SAGFMA3, SAGFMA2, FMIA, torchattacks

if LPIPS_AVAILABLE:
    import lpips
if OPEN_CLIP_AVAILABLE:
    import open_clip

def get_attack_pool():
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
    # pool['SAGFMA_SingleScale_w'] = {
    #     'class': SAGFMA3,
    #     'params': sagfma_params
    # }

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

    # === FMIA Variants ===
    fmia_base = {
        'epsilon': CONFIG['epsilon'],
        'steps': CONFIG['steps'],
        'alpha': CONFIG['alpha'],
        'decay': 0.9,
        'low_freq_percent': 0.5,
        'high_freq_percent': 0.5,
        'filter_type': 'low_pass',
        'di_prob': 0.3,
        'si_scales': 3
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

def filter_attacks_by_category(pool, categories=None):
    """
    Filter attacks by category for selective evaluation.
    Categories: 'standard', 'sagfma', 'sagfma2', 'fmia', 'all'
    """
    if categories is None or 'all' in categories:
        return pool
    
    filtered = {}
    for attack_name, attack_info in pool.items():
        if 'standard' in categories and attack_name in ['FGSM', 'PGD', 'MIFGSM', 'TIM', 'DeepFool']:
            filtered[attack_name] = attack_info
        elif 'sagfma' in categories and 'SAGFMA_' in attack_name and 'SAGFMA2_' not in attack_name:
            filtered[attack_name] = attack_info
        elif 'sagfma2' in categories and 'SAGFMA2_' in attack_name:
            filtered[attack_name] = attack_info
        elif 'fmia' in categories and 'FMIA' in attack_name:
            filtered[attack_name] = attack_info
    
    return filtered

def main_benchmark_script():
    print("\n" + "="*60 + "\n Adversarial Attack Benchmarking Script\n" + "="*60)
    print(f"Results will be saved in: {CONFIG['results_base_dir']}")

    # Load Data & Models
    print("\n--- Step 1: Loading Data & Models ---")
    val_loader = load_data(CONFIG)
    models_collection = load_models(CONFIG)
    if not models_collection:
        print("CRITICAL ERROR: No models were loaded. Benchmarking cannot proceed.")
        sys.exit(1)

    # Select Primary & Transfer Models
    primary_model_key = 'resnet50_clean'
    if primary_model_key not in models_collection:
        primary_model_key = next(iter(models_collection))
        print(f"Warning: Default primary model not found. Using '{primary_model_key}' instead.")
    
    primary_target_model = models_collection[primary_model_key]
    print(f"Primary model for direct attack evaluation: {primary_model_key}")

    transfer_target_models = {name: m for name, m in models_collection.items() if name != primary_model_key}
    if transfer_target_models:
        print(f"Models for transferability tests: {list(transfer_target_models.keys())}")
    else:
        print("No additional models available for transferability tests.")

    # Initialize LPIPS & CLIP Metric Models
    lpips_metric_obj = None
    if LPIPS_AVAILABLE and lpips is not None:
        try:
            lpips_metric_obj = lpips.LPIPS(net=CONFIG['lpips_net']).to(device).eval()
            print(f"LPIPS metric model ({CONFIG['lpips_net']}) initialized.")
        except Exception as e_lpips:
            print(f"Failed to load LPIPS model: {e_lpips}. LPIPS scores will be NaN.")

    clip_metric_obj, clip_preprocess_obj = None, None
    if OPEN_CLIP_AVAILABLE and open_clip is not None:
        try:
            clip_metric_obj, _, clip_preprocess_obj = open_clip.create_model_and_transforms(
                CONFIG['clip_model_name'], pretrained=CONFIG['clip_pretrained'], device=device)
            clip_metric_obj.eval()
            print(f"CLIP metric model ({CONFIG['clip_model_name']}) initialized.")
        except Exception as e_clip:
            print(f"Failed to load CLIP model: {e_clip}. CLIP similarity scores will be NaN.")

    # Get Attack Pool and Filter (Modular Selection)
    print("\n--- Step 2: Defining Attacks for Evaluation ---")
    full_attack_pool = get_attack_pool()
    
    # CUSTOMIZATION 
    # Options: ['all'], ['standard'], ['sagfma'], ['sagfma2'], ['fmia'], or combinations
    selected_categories = ['all']  # Change this to filter attacks
    attacks_to_run = filter_attacks_by_category(full_attack_pool, selected_categories)
    
    if not attacks_to_run:
        print("Warning: No attacks selected for evaluation.")
        return
    
    print(f"Selected {len(attacks_to_run)} attacks for evaluation:")
    for attack_id in attacks_to_run:
        print(f" - {attack_id}")

    # Prepare Summary Containers
    master_eval_summaries = {}
    master_transfer_summaries = {}
    master_defense_summaries = {}
    master_resolution_summaries = {}
    master_epsilon_summaries = {}

    # Main Evaluation Loop with Robust Error Handling
    print("\n--- Step 3: Executing Attack Evaluations ---")
    successful_attacks = 0
    failed_attacks = 0
    
    for attack_id_key, attack_details in attacks_to_run.items():
        print(f"\n===== Evaluating Attack: {attack_id_key} on Model: {primary_model_key} =====")
        attack_specific_results_dir = os.path.join(CONFIG['results_base_dir'], attack_id_key)
        os.makedirs(attack_specific_results_dir, exist_ok=True)

        # Robust Attack Instantiation
        try:
            current_attack_object = instantiate_attack(
                attack_details['class'], 
                primary_target_model, 
                attack_details['params']
            )
            print(f"Successfully instantiated {attack_id_key}")
        except Exception as e_instantiate:
            print(f"[SKIP] Could not instantiate {attack_id_key}: {e_instantiate}")
            master_eval_summaries[attack_id_key] = {
                'error': str(e_instantiate),
                'status': 'instantiation_failed'
            }
            failed_attacks += 1
            continue

        # Robust Primary Evaluation
        try:
            print(f"--- Stage 1: Primary evaluation of {attack_id_key} ---")
            current_eval_summary = evaluate_attack_robust(
                primary_target_model, current_attack_object, val_loader, attack_id_key, CONFIG, 
                lpips_metric_obj, clip_metric_obj, clip_preprocess_obj
            )
            master_eval_summaries[attack_id_key] = current_eval_summary
            
            with open(os.path.join(attack_specific_results_dir, 'summary_primary_evaluation.json'), 'w') as f_json:
                json.dump(current_eval_summary, f_json, indent=4, 
                         default=lambda o: o.item() if hasattr(o, 'item') else str(o))
            print(f"Primary evaluation completed for {attack_id_key}")
            
        except Exception as e_eval:
            print(f"[SKIP] Primary evaluation failed for {attack_id_key}: {e_eval}")
            master_eval_summaries[attack_id_key] = {
                'error': str(e_eval),
                'status': 'evaluation_failed'
            }
            failed_attacks += 1
            continue

        # Robust Transferability Evaluation
        if transfer_target_models:
            try:
                print(f"--- Stage 2: Transferability evaluation of {attack_id_key} ---")
                current_transfer_summary = evaluate_transferability_robust(
                    primary_target_model, transfer_target_models, current_attack_object, 
                    val_loader, attack_id_key, CONFIG
                )
                master_transfer_summaries[attack_id_key] = current_transfer_summary
                
                with open(os.path.join(attack_specific_results_dir, 'summary_transferability.json'), 'w') as f_json:
                    json.dump(current_transfer_summary, f_json, indent=4, 
                             default=lambda o: o.item() if hasattr(o, 'item') else str(o))
                print(f"Transferability evaluation completed for {attack_id_key}")
                
            except Exception as e_transfer:
                print(f"[SKIP] Transferability evaluation failed for {attack_id_key}: {e_transfer}")
                master_transfer_summaries[attack_id_key] = {
                    'error': str(e_transfer),
                    'status': 'transferability_failed'
                }

        # Conditional Evaluations for Key Attacks
        key_attacks_for_defense = ['PGD', 'SAGFMA2_Base', 'Base_FMIA']
        if attack_id_key in key_attacks_for_defense:
            # Defense Evaluation
            try:
                print(f"--- Stage 3: Defense evaluation against {attack_id_key} ---")
                defense_mechanisms = {
                    'JPEG_Compression_Q75': lambda x: jpeg_compression_defense(x, CONFIG['jpeg_quality'])
                }
                current_defense_summary = evaluate_defenses_robust(
                    primary_target_model, current_attack_object, val_loader, 
                    defense_mechanisms, attack_id_key, CONFIG
                )
                master_defense_summaries[attack_id_key] = current_defense_summary
                
                with open(os.path.join(attack_specific_results_dir, 'summary_defenses.json'), 'w') as f_json:
                    json.dump(current_defense_summary, f_json, indent=4, 
                             default=lambda o: o.item() if hasattr(o, 'item') else str(o))
                print(f"✓ Defense evaluation completed for {attack_id_key}")
                
            except Exception as e_defense:
                print(f"✗ [SKIP] Defense evaluation failed for {attack_id_key}: {e_defense}")
                master_defense_summaries[attack_id_key] = {
                    'error': str(e_defense),
                    'status': 'defense_failed'
                }

            # Resolution Sensitivity
            try:
                print(f"--- Stage 4: Resolution sensitivity of {attack_id_key} ---")
                current_resolution_summary = evaluate_resolution_robust(
                    primary_target_model, current_attack_object, val_loader, 
                    CONFIG['resolutions_to_test'], attack_id_key, CONFIG
                )
                master_resolution_summaries[attack_id_key] = current_resolution_summary
                
                with open(os.path.join(attack_specific_results_dir, 'summary_resolution_sensitivity.json'), 'w') as f_json:
                    json.dump(current_resolution_summary, f_json, indent=4, 
                             default=lambda o: o.item() if hasattr(o, 'item') else str(o))
                print(f"✓ Resolution sensitivity completed for {attack_id_key}")
                
            except Exception as e_resolution:
                print(f"✗ [SKIP] Resolution sensitivity failed for {attack_id_key}: {e_resolution}")
                master_resolution_summaries[attack_id_key] = {
                    'error': str(e_resolution),
                    'status': 'resolution_failed'
                }

        # Epsilon Sensitivity (for L-inf attacks)
        if ('eps' in attack_details['params'] and 
            attack_details['class'] != torchattacks.DeepFool):
            try:
                print(f"--- Stage 5: Epsilon sensitivity of {attack_id_key} ---")
                eps_sens_base_params = {k: v for k, v in attack_details['params'].items() if k != 'eps'}
                current_epsilon_summary = evaluate_epsilon_sensitivity(
                    primary_target_model, attack_details['class'], val_loader,
                    CONFIG['epsilon_range'], eps_sens_base_params,
                    attack_id_key, CONFIG
                )
                master_epsilon_summaries[attack_id_key] = current_epsilon_summary
                
                with open(os.path.join(attack_specific_results_dir, 'summary_epsilon_sensitivity_curve.json'), 'w') as f_json:
                    json.dump(current_epsilon_summary, f_json, indent=4, 
                             default=lambda o: o.item() if hasattr(o, 'item') else str(o))
                print(f"✓ Epsilon sensitivity completed for {attack_id_key}")
                
            except Exception as e_epsilon:
                print(f"✗ [SKIP] Epsilon sensitivity failed for {attack_id_key}: {e_epsilon}")
                master_epsilon_summaries[attack_id_key] = {
                    'error': str(e_epsilon),
                    'status': 'epsilon_failed'
                }

        successful_attacks += 1

    # Save Overall Aggregated Summary
    print("\n--- Step 4: Aggregating and Saving All Benchmark Results ---")
    full_benchmark_report = {
        'benchmark_config_summary': {
            'primary_model': primary_model_key,
            'dataset_subset_size': CONFIG['test_subset_size'],
            'default_epsilon': CONFIG['epsilon'],
            'default_steps': CONFIG['steps'],
            'total_attacks_attempted': len(attacks_to_run),
            'successful_attacks': successful_attacks,
            'failed_attacks': failed_attacks
        },
        'all_primary_evaluations': master_eval_summaries,
        'all_transfer_evaluations': master_transfer_summaries,
        'all_defense_evaluations': master_defense_summaries,
        'all_resolution_evaluations': master_resolution_summaries,
        'all_epsilon_sensitivity_evaluations': master_epsilon_summaries
    }
    
    master_summary_file_path = os.path.join(CONFIG['results_base_dir'], 'MASTER_BENCHMARK_SUMMARY_REPORT.json')
    with open(master_summary_file_path, 'w') as f_master_json:
        json.dump(full_benchmark_report, f_master_json, indent=4, 
                 default=lambda o: o.item() if hasattr(o, 'item') else str(o))

    # Print Summary
    print(f"\n" + "="*60)
    print(f"BENCHMARKING COMPLETE")
    print(f"="*60)
    print(f"Successful attacks: {successful_attacks}")
    print(f"Failed attacks: {failed_attacks}")
    print(f"Master summary: {master_summary_file_path}")
    print(f"Detailed results: {CONFIG['results_base_dir']}")
    print(f"="*60)

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning, module='torchvision.transforms.functional_tensor')
    warnings.filterwarnings("ignore", category=UserWarning, module='torchvision.transforms._functional_video')
    main_benchmark_script()
