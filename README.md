
# Frequency domain attacks FMIA and SAGFMA

A framework for evaluating adversarial attacks and defenses with advanced frequency-domain analysis capabilities.

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

## Features

- **Novel Frequency-Domain Attacks**: SAGFMA, SAGFMA2, FMIA with spatial-frequency analysis
- **Comprehensive Evaluation**: Primary attacks, transferability, defenses, resolution sensitivity
- **Extensive Ablation Studies**:SAGFMA2 configurations, SAGFMA configurations, FMIA configurations
- **Rich Visualizations**: FFT analysis, frequency band overlays, radial energy profiles




## Supported Attacks

### Standard Attacks
- FGSM, PGD, MI-FGSM, TI-FGSM and all of the attacks available on torchattacks library

### Novel Frequency-Domain Attacks
- **SAGFMA**: Self-Attention Guided Frequency Momentum Attack
- **SAGFMA2**: Another version with variants (base, nag, ensemble, freq_update)
- **FMIA**: Frequency Momentum Iterative Attack 

## Evaluations 

Each attack is evaluated across 5 dimensions:
1. **Primary Evaluation**: Attack success rates with confidence intervals
2. **Transferability**: Cross-model attack success
3. **Defense Robustness**: JPEG compression, noise filtering
4. **Resolution Sensitivity**: Multi-scale robustness
5. **Epsilon Sensitivity**: Perturbation budget analysis




## File Structure

```
adversarial_robustness_suite/
├── config.py                          # Configuration settings
├── utils.py                           # Utility functions
├── evaluations.py                     # Evaluation functions
├── attacks/                           # Attack implementations
├── run_benchmarking.py                # Main benchmarking
├── run_sagfma2_simple_ablation.py     # SAGFMA2 ablation (63+ configs)
├── run_sagfma_ablation.py             # SAGFMA ablation (112+ configs)
├── run_fmia_ablation.py             # SAGFMA ablation (112+ configs)
└── run_visualizations.py              # Visualization generation
```

---

**Note**: This framework provides state-of-the-art adversarial attack implementations with comprehensive evaluation capabilities for research purposes.
