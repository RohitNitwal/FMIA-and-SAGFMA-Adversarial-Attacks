import torch
import os
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
import torchvision.transforms as transforms_tv

from config import device, CONFIG
from utils import (
    calculate_confidence_interval, save_metrics_csv,
    batch_psnr, batch_ssim, batch_lpips, batch_clip_similarity,
    instantiate_attack
)

def evaluate_attack_robust(target_model, attack_obj, val_data_loader, current_attack_name,
                           global_config, lpips_metric_model=None,
                           clip_metric_model=None, clip_preprocess_transform=None):
    target_model.eval()
    detailed_records = []
    summary_stats = {'total_samples_proc':0, 'clean_correct_count':0, 'adv_correct_count':0, 
                     'successful_fool_count':0, 'batch_attack_times':[]}

    for batch_num, (clean_images, true_labels) in enumerate(tqdm(val_data_loader, desc=f"Evaluating {current_attack_name}")):
        clean_images, true_labels = clean_images.to(device), true_labels.to(device)
        current_batch_size = clean_images.size(0)
        summary_stats['total_samples_proc'] += current_batch_size

        with torch.no_grad():
            clean_model_outputs = target_model(clean_images)
            clean_model_preds = clean_model_outputs.argmax(1)
        clean_correct_mask_batch = clean_model_preds.eq(true_labels)

        time_start_attack = time.time()
        adv_images_generated = attack_obj(clean_images, true_labels)
        attack_duration_batch = time.time() - time_start_attack
        summary_stats['batch_attack_times'].append(attack_duration_batch)

        with torch.no_grad():
            adv_model_outputs = target_model(adv_images_generated)
            adv_model_preds = adv_model_outputs.argmax(1)
        adv_correct_mask_batch = adv_model_preds.eq(true_labels)

        fooled_mask_batch = clean_correct_mask_batch & ~adv_correct_mask_batch
        summary_stats['clean_correct_count'] += clean_correct_mask_batch.sum().item()
        summary_stats['adv_correct_count']   += adv_correct_mask_batch.sum().item()
        summary_stats['successful_fool_count']  += fooled_mask_batch.sum().item()

        # Perceptual metrics for the batch
        psnr_batch_vals = batch_psnr(clean_images, adv_images_generated)  
        ssim_batch_vals = batch_ssim(clean_images, adv_images_generated) 
        lpips_batch_vals = batch_lpips(clean_images, adv_images_generated, lpips_metric_model)        
        clip_batch_vals = batch_clip_similarity(clean_images, adv_images_generated, clip_metric_model, clip_preprocess_transform)       

        # Store per-sample records for detailed CSV
        batch_start_idx = batch_num * val_data_loader.batch_size
        for i in range(current_batch_size):
            detailed_records.append({
                'attack_name': current_attack_name,
                'sample_id': batch_start_idx + i,
                'is_clean_correct': int(clean_correct_mask_batch[i]),
                'is_adv_correct':   int(adv_correct_mask_batch[i]),
                'was_fooled':        int(fooled_mask_batch[i]),
                'metric_psnr':          psnr_batch_vals[i] if psnr_batch_vals is not None and i < len(psnr_batch_vals) else np.nan,
                'metric_ssim':          ssim_batch_vals[i] if ssim_batch_vals is not None and i < len(ssim_batch_vals) else np.nan,
                'metric_lpips':         lpips_batch_vals[i] if lpips_batch_vals is not None and i < len(lpips_batch_vals) else np.nan,
                'metric_clip_sim':      clip_batch_vals[i] if clip_batch_vals is not None and i < len(clip_batch_vals) else np.nan,
                'img_attack_time_avg': attack_duration_batch / current_batch_size if current_batch_size > 0 else 0
            })

    df_records_detailed = pd.DataFrame(detailed_records)
    results_path_attack = os.path.join(global_config['results_base_dir'], current_attack_name)
    save_metrics_csv(df_records_detailed, os.path.join(results_path_attack, 'attack_metrics_per_sample_detailed.csv'))

    # Calculate overall summary statistics
    total_samples = summary_stats['total_samples_proc']
    clean_accuracy_val = (summary_stats['clean_correct_count'] / total_samples * 100) if total_samples > 0 else 0
    adv_accuracy_val   = (summary_stats['adv_correct_count'] / total_samples * 100) if total_samples > 0 else 0
    attack_success_rate_val = (summary_stats['successful_fool_count'] / summary_stats['clean_correct_count'] * 100) if summary_stats['clean_correct_count'] > 0 else 0.0

    # Confidence Intervals
    ci_clean = calculate_confidence_interval(summary_stats['clean_correct_count'], total_samples, global_config['confidence_level'])
    ci_adv   = calculate_confidence_interval(summary_stats['adv_correct_count'], total_samples, global_config['confidence_level'])
    ci_asr   = calculate_confidence_interval(summary_stats['successful_fool_count'], summary_stats['clean_correct_count'], global_config['confidence_level']) if summary_stats['clean_correct_count'] > 0 else (0.0, 0.0)

    # Average perceptual metrics
    avg_psnr_val = np.nanmean([rec['metric_psnr'] for rec in detailed_records]) if detailed_records else np.nan
    avg_ssim_val = np.nanmean([rec['metric_ssim'] for rec in detailed_records]) if detailed_records else np.nan
    avg_lpips_val = pd.Series([rec['metric_lpips'] for rec in detailed_records]).dropna().mean() if detailed_records else np.nan
    avg_clip_val  = pd.Series([rec['metric_clip_sim'] for rec in detailed_records]).dropna().mean() if detailed_records else np.nan
    avg_attack_time_per_img = np.mean(summary_stats['batch_attack_times']) / val_data_loader.batch_size if val_data_loader.batch_size > 0 and summary_stats['batch_attack_times'] else 0

    final_summary_dict = {
        'attack_name_eval': current_attack_name,
        'total_samples_eval': total_samples,
        'clean_accuracy_pct': clean_accuracy_val,
        'clean_acc_ci_low': ci_clean[0], 'clean_acc_ci_high': ci_clean[1],
        'adv_accuracy_pct': adv_accuracy_val,
        'adv_acc_ci_low': ci_adv[0], 'adv_acc_ci_high': ci_adv[1],
        'attack_success_rate_pct': attack_success_rate_val,
        'asr_ci_low': ci_asr[0], 'asr_ci_high': ci_asr[1],
        'avg_psnr': avg_psnr_val,
        'avg_ssim': avg_ssim_val,
        'avg_lpips': avg_lpips_val,
        'avg_clip_similarity': avg_clip_val,
        'avg_processing_time_per_image_sec': avg_attack_time_per_img
    }
    return final_summary_dict

def evaluate_transferability_robust(source_model_for_attack, target_models_collection, attack_obj_on_source,
                                   val_data_loader, source_attack_name, global_config):
    source_model_for_attack.eval()
    for _tm_name, tm_obj in target_models_collection.items():
        tm_obj.eval()

    transfer_records_detailed = []
    target_model_stats = {tm_name:{'clean_correct_tm':0, 'adv_correct_tm':0, 'successful_transfers_tm':0, 'total_samples_tm':0} 
                          for tm_name in target_models_collection}
    
    print(f"Generating adversarial examples using {source_attack_name} on source model for transfer tests...")
    for batch_num, (clean_images, true_labels) in enumerate(tqdm(val_data_loader, desc=f"Transfer Test ({source_attack_name})")):
        clean_images, true_labels = clean_images.to(device), true_labels.to(device)
        current_batch_size = clean_images.size(0)

        adv_images_from_source = attack_obj_on_source(clean_images, true_labels)
        batch_start_idx = batch_num * val_data_loader.batch_size

        for tm_name_key, tm_obj_val in target_models_collection.items():
            target_model_stats[tm_name_key]['total_samples_tm'] += current_batch_size
            with torch.no_grad():
                clean_outputs_on_target = tm_obj_val(clean_images)
                clean_preds_on_target = clean_outputs_on_target.argmax(1)
                clean_correct_mask_on_target = clean_preds_on_target.eq(true_labels)
                
                adv_outputs_on_target = tm_obj_val(adv_images_from_source)
                adv_preds_on_target = adv_outputs_on_target.argmax(1)
                adv_correct_mask_on_target = adv_preds_on_target.eq(true_labels)
            
            transfer_fooled_mask = clean_correct_mask_on_target & ~adv_correct_mask_on_target

            target_model_stats[tm_name_key]['clean_correct_tm'] += clean_correct_mask_on_target.sum().item()
            target_model_stats[tm_name_key]['adv_correct_tm']   += adv_correct_mask_on_target.sum().item()
            target_model_stats[tm_name_key]['successful_transfers_tm']  += transfer_fooled_mask.sum().item()

            for i in range(current_batch_size):
                transfer_records_detailed.append({
                    'source_attack_name': source_attack_name,
                    'target_model_name': tm_name_key,
                    'sample_id': batch_start_idx + i,
                    'target_model_clean_correct': int(clean_correct_mask_on_target[i]),
                    'target_model_adv_correct': int(adv_correct_mask_on_target[i]),
                    'is_successful_transfer_sample': int(transfer_fooled_mask[i])
                })

    df_transfer_detailed = pd.DataFrame(transfer_records_detailed)
    results_path_attack_transfer = os.path.join(global_config['results_base_dir'], source_attack_name)
    save_metrics_csv(df_transfer_detailed, os.path.join(results_path_attack_transfer, 'transfer_metrics_per_sample_detailed.csv'))

    transfer_summary_output = {}
    for tm_name_key, stats_tm in target_model_stats.items():
        total_tm_eval_samples = stats_tm['total_samples_tm']
        clean_acc_tm = (stats_tm['clean_correct_tm'] / total_tm_eval_samples * 100) if total_tm_eval_samples > 0 else 0
        adv_acc_tm = (stats_tm['adv_correct_tm'] / total_tm_eval_samples * 100) if total_tm_eval_samples > 0 else 0
        transfer_asr_tm = (stats_tm['successful_transfers_tm'] / stats_tm['clean_correct_tm'] * 100) if stats_tm['clean_correct_tm'] > 0 else 0.0
        
        ci_clean_tm = calculate_confidence_interval(stats_tm['clean_correct_tm'], total_tm_eval_samples, global_config['confidence_level'])
        ci_adv_tm = calculate_confidence_interval(stats_tm['adv_correct_tm'], total_tm_eval_samples, global_config['confidence_level'])
        ci_tasr_tm = calculate_confidence_interval(stats_tm['successful_transfers_tm'], stats_tm['clean_correct_tm'], global_config['confidence_level']) if stats_tm['clean_correct_tm'] > 0 else (0.0, 0.0)
        
        transfer_summary_output[tm_name_key] = {
            'target_model_clean_accuracy_pct': clean_acc_tm, 
            'target_clean_acc_ci_low': ci_clean_tm[0], 'target_clean_acc_ci_high': ci_clean_tm[1],
            'target_model_adv_accuracy_pct': adv_acc_tm, 
            'target_adv_acc_ci_low': ci_adv_tm[0], 'target_adv_acc_ci_high': ci_adv_tm[1],
            'transfer_attack_success_rate_pct': transfer_asr_tm, 
            'transfer_asr_ci_low': ci_tasr_tm[0], 'transfer_asr_ci_high': ci_tasr_tm[1]
        }
    return transfer_summary_output

def evaluate_defenses_robust(model_victim, attack_obj_for_adv_gen, val_data_loader, defense_fn_map,
                             source_attack_name_for_adv, global_config):
    model_victim.eval()
    adv_images_all_cpu, true_labels_all_cpu = [], []
    print(f"Generating adversarial examples with {source_attack_name_for_adv} for defense evaluation...")
    for clean_images, true_labels in tqdm(val_data_loader, desc=f"Adv Gen for Defenses ({source_attack_name_for_adv})"):
        clean_images, true_labels = clean_images.to(device), true_labels.to(device)
        adv_batch_generated = attack_obj_for_adv_gen(clean_images, true_labels)
        adv_images_all_cpu.append(adv_batch_generated.cpu()) 
        true_labels_all_cpu.append(true_labels.cpu())
    adv_images_full_tensor = torch.cat(adv_images_all_cpu)
    true_labels_full_tensor = torch.cat(true_labels_all_cpu)

    defense_summary_output = {}
    total_adv_samples_for_defense = len(true_labels_full_tensor)

    for defense_id, defense_func in defense_fn_map.items():
        print(f"Applying defense: {defense_id} to adv examples from {source_attack_name_for_adv}...")
        correct_after_defense_count = 0
        
        adv_dataset = torch.utils.data.TensorDataset(adv_images_full_tensor, true_labels_full_tensor)
        adv_loader_for_defense = torch.utils.data.DataLoader(adv_dataset, batch_size=val_data_loader.batch_size, shuffle=False)

        for adv_batch_from_loader, labels_batch_from_loader in tqdm(adv_loader_for_defense, desc=f"Testing {defense_id}"):
            adv_batch_from_loader = adv_batch_from_loader.to(device)
            labels_batch_from_loader = labels_batch_from_loader.to(device)

            defended_batch = defense_func(adv_batch_from_loader)
            
            with torch.no_grad():
                preds_after_defense = model_victim(defended_batch).argmax(1)
            
            correct_mask_after_defense = preds_after_defense.eq(labels_batch_from_loader)
            correct_after_defense_count += correct_mask_after_defense.sum().item()
        
        accuracy_post_defense = (correct_after_defense_count / total_adv_samples_for_defense * 100) if total_adv_samples_for_defense > 0 else 0
        ci_defense = calculate_confidence_interval(correct_after_defense_count, total_adv_samples_for_defense, global_config['confidence_level'])
        defense_summary_output[defense_id] = {
            'accuracy_after_defense_pct': accuracy_post_defense, 
            'acc_after_defense_ci_low': ci_defense[0], 
            'acc_after_defense_ci_high': ci_defense[1]
        }
    return defense_summary_output

def evaluate_resolution_robust(model_victim, attack_obj_for_adv_gen, val_data_loader, resolutions_list,
                               source_attack_name_for_adv, global_config):
    model_victim.eval()
    adv_images_all_cpu, true_labels_all_cpu = [], []
    print(f"Generating adversarial examples with {source_attack_name_for_adv} for resolution robustness test...")
    for clean_images, true_labels in tqdm(val_data_loader, desc=f"Adv Gen for Resolution ({source_attack_name_for_adv})"):
        clean_images, true_labels = clean_images.to(device), true_labels.to(device)
        adv_batch_generated = attack_obj_for_adv_gen(clean_images, true_labels)
        adv_images_all_cpu.append(adv_batch_generated.cpu())
        true_labels_all_cpu.append(true_labels.cpu())
    adv_images_full_tensor = torch.cat(adv_images_all_cpu)
    true_labels_full_tensor = torch.cat(true_labels_all_cpu)

    resolution_summary_output = {}
    total_adv_samples_for_res = len(true_labels_full_tensor)

    adv_dataset_res = torch.utils.data.TensorDataset(adv_images_full_tensor, true_labels_full_tensor)
    adv_loader_for_res = torch.utils.data.DataLoader(adv_dataset_res, batch_size=val_data_loader.batch_size, shuffle=False)

    for res_value in resolutions_list:
        print(f"Testing resolution: {res_value}x{res_value} for adv examples from {source_attack_name_for_adv}...")
        correct_at_res_count = 0
        resize_op = transforms_tv.Resize((res_value, res_value), antialias=True)

        for adv_batch_res, labels_batch_res in tqdm(adv_loader_for_res, desc=f"Testing Res {res_value}"):
            adv_batch_res = adv_batch_res.to(device)
            labels_batch_res = labels_batch_res.to(device)
            
            resized_adv_batch = resize_op(adv_batch_res)
            with torch.no_grad():
                preds_at_res = model_victim(resized_adv_batch).argmax(1)
            correct_at_res_count += preds_at_res.eq(labels_batch_res).sum().item()
        
        accuracy_at_res = (correct_at_res_count / total_adv_samples_for_res * 100) if total_adv_samples_for_res > 0 else 0
        ci_res = calculate_confidence_interval(correct_at_res_count, total_adv_samples_for_res, global_config['confidence_level'])
        resolution_summary_output[f"res_{res_value}x{res_value}"] = {
            'accuracy_at_resolution_pct': accuracy_at_res, 
            'acc_at_res_ci_low': ci_res[0], 
            'acc_at_res_ci_high': ci_res[1]
        }
    return resolution_summary_output

def evaluate_epsilon_sensitivity(target_model, AttackClassType, val_data_loader,
                                 epsilon_values_list, base_attack_creation_params,
                                 attack_base_identifier, global_config):
    target_model.eval()
    epsilon_sensitivity_summary = {}

    for eps_val_test in tqdm(epsilon_values_list, desc=f"Epsilon Sensitivity for {attack_base_identifier}"):
        current_iter_attack_params = {**base_attack_creation_params}
        if AttackClassType.__name__ != 'DeepFool':
            current_iter_attack_params['epsilon'] = eps_val_test 
        
        try:
            attack_obj_for_eps = instantiate_attack(AttackClassType, target_model, current_iter_attack_params)
        except Exception as e:
            print(f"Failed to instantiate {AttackClassType.__name__} for eps={eps_val_test}: {e}. Skipping this epsilon.")
            epsilon_sensitivity_summary[eps_val_test] = {'attack_success_rate_pct': np.nan, 'error': str(e)}
            continue
        
        total_samples_for_this_eps = 0
        clean_correct_count_for_asr = 0
        successful_fool_count_for_eps = 0

        for clean_images, true_labels in tqdm(val_data_loader, desc=f"Eps: {eps_val_test:.4f}", leave=False):
            clean_images, true_labels = clean_images.to(device), true_labels.to(device)
            total_samples_for_this_eps += clean_images.size(0)
            
            with torch.no_grad():
                clean_preds_batch = target_model(clean_images).argmax(1)
            clean_correct_mask_batch_eps = clean_preds_batch.eq(true_labels)
            clean_correct_count_for_asr += clean_correct_mask_batch_eps.sum().item()

            adv_images_eps_generated = attack_obj_for_eps(clean_images, true_labels)
            with torch.no_grad():
                adv_preds_batch_eps = target_model(adv_images_eps_generated).argmax(1)
            
            fooled_mask_batch_eps = clean_correct_mask_batch_eps & ~adv_preds_batch_eps.eq(true_labels)
            successful_fool_count_for_eps += fooled_mask_batch_eps.sum().item()

        asr_at_this_eps = (successful_fool_count_for_eps / clean_correct_count_for_asr * 100) if clean_correct_count_for_asr > 0 else 0.0
        ci_asr_at_eps = calculate_confidence_interval(successful_fool_count_for_eps, clean_correct_count_for_asr, global_config['confidence_level']) if clean_correct_count_for_asr > 0 else (0.0, 0.0)
        
        epsilon_sensitivity_summary[eps_val_test] = {
            'attack_success_rate_pct': asr_at_this_eps, 
            'asr_ci_low': ci_asr_at_eps[0], 
            'asr_ci_high': ci_asr_at_eps[1],
            'total_samples_tested': total_samples_for_this_eps,
            'clean_correct_for_asr_base': clean_correct_count_for_asr,
            'successful_attacks_count': successful_fool_count_for_eps
        }
    return epsilon_sensitivity_summary
