import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class FMIA:
    """
    Frequency Momentum Iterative Attack (FMIA)

    This attack combines momentum-based adversarial updates with frequency-domain filtering.
    It optionally supports:
        - Frequency domain gradient accumulation ('f-grad')
        - Nesterov accelerated gradients ('nesterov')
        - Input diversity ('di')
        - Scale invariance ('si')

    Args:
        model (nn.Module): Target model.
        variant (str): Attack variant (e.g., 'base', 'f-grad', 'nesterov', 'di-si').
        epsilon (float): Maximum L∞ perturbation.
        steps (int): Number of iterations.
        alpha (float): Step size per iteration.
        decay (float): Momentum decay factor.
        low_freq_percent (float): Lower bound for frequency masking (0-1).
        high_freq_percent (float): Upper bound for band-pass filtering (0-1).
        filter_type (str): One of ['low_pass', 'high_pass', 'band_pass', 'none'].
        di_prob (float): Probability of applying input diversity.
        si_scales (int): Number of scales to use for scale-invariant gradient averaging.
    """
    def __init__(self, model, variant='base', epsilon=8/255, steps=10, alpha=2/255,
                 decay=1.0, low_freq_percent=0.3, high_freq_percent=0.6,
                 filter_type='low_pass', di_prob=0.7, si_scales=5):
        self.model = model.eval()
        self.epsilon = epsilon
        self.steps = steps
        self.alpha = alpha
        self.decay = decay
        self.variant = variant.lower()
        self.loss_fn = torch.nn.CrossEntropyLoss()

        # Set device safely
        try:
            self.device = next(model.parameters()).device
        except StopIteration:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.low_freq_percent = low_freq_percent
        self.high_freq_percent = high_freq_percent
        self.filter_type = filter_type
        self.di_prob = di_prob
        self.si_scales = si_scales if 'si' in self.variant or 'di-si' in self.variant else 1

        self._last_band_mask = None  # Stores last mask for visualization

    def _fft_2d(self, x_input):
        """Apply FFT and shift zero frequency to center."""
        return torch.fft.fftshift(torch.fft.fft2(x_input, dim=(-2, -1), norm='ortho'), dim=(-2, -1))

    def _ifft_2d(self, X_coeffs, original_spatial_shape=None):
        """Apply inverse FFT and unshift to recover spatial signal."""
        X_unshifted = torch.fft.ifftshift(X_coeffs, dim=(-2, -1))
        spatial_result = torch.fft.ifft2(X_unshifted, dim=(-2, -1), norm='ortho')
        if original_spatial_shape is not None:
            return spatial_result[..., :original_spatial_shape[-2], :original_spatial_shape[-1]]
        return spatial_result

    def _frequency_filter(self, fft_coeffs_input, original_image_spatial_shape):
        """
        Applies frequency mask based on `filter_type`.
        Returns filtered FFT coefficients and records mask for visualization.
        """
        if self.filter_type == 'none':
            self._last_band_mask = torch.ones_like(fft_coeffs_input[:1, :1, ..., 0].real)
            return fft_coeffs_input

        _b, _c, h_fft, w_fft = fft_coeffs_input.shape
        center_y, center_x = h_fft // 2, w_fft // 2

        yy_grid, xx_grid = torch.meshgrid(
            torch.arange(h_fft, device=fft_coeffs_input.device, dtype=torch.float32),
            torch.arange(w_fft, device=fft_coeffs_input.device, dtype=torch.float32),
            indexing='ij'
        )
        dist_from_center = torch.sqrt((yy_grid - center_y)**2 + (xx_grid - center_x)**2)
        max_fft_radius = torch.sqrt(torch.tensor((h_fft/2)**2 + (w_fft/2)**2, device=fft_coeffs_input.device))

        spatial_filter_mask = torch.zeros_like(dist_from_center)
        viz_band_identifier_mask = torch.zeros_like(dist_from_center, dtype=torch.long)

        if self.filter_type == 'low_pass':
            radius_cutoff = self.low_freq_percent * max_fft_radius
            spatial_filter_mask[dist_from_center <= radius_cutoff] = 1.0
            viz_band_identifier_mask[dist_from_center <= radius_cutoff] = 1

        elif self.filter_type == 'high_pass':
            radius_cutoff = self.high_freq_percent * max_fft_radius
            spatial_filter_mask[dist_from_center > radius_cutoff] = 1.0
            viz_band_identifier_mask[dist_from_center > radius_cutoff] = 2

        elif self.filter_type == 'band_pass':
            radius_cutoff_low = self.low_freq_percent * max_fft_radius
            radius_cutoff_high = self.high_freq_percent * max_fft_radius
            condition = (dist_from_center >= radius_cutoff_low) & (dist_from_center <= radius_cutoff_high)
            spatial_filter_mask[condition] = 1.0
            viz_band_identifier_mask[condition] = 3

        else:
            spatial_filter_mask = torch.ones_like(dist_from_center)
            viz_band_identifier_mask = torch.ones_like(dist_from_center, dtype=torch.long) * 4

        self._last_band_mask = viz_band_identifier_mask.unsqueeze(0).unsqueeze(0)
        return fft_coeffs_input * spatial_filter_mask.view(1, 1, h_fft, w_fft)

    def _normalize_grad_l1(self, grad_tensor):
        """L1-normalize the gradient per image."""
        norm_val = grad_tensor.view(grad_tensor.size(0), -1).abs().sum(dim=1, keepdim=True).view(-1, 1, 1, 1) + 1e-12
        return grad_tensor / norm_val

    def _project_perturbation(self, adv_images_perturbed, original_images_clean):
        """Project onto L∞ ball and clip to [0,1]."""
        perturbation = torch.clamp((adv_images_perturbed - original_images_clean), -self.epsilon, self.epsilon)
        return torch.clamp(original_images_clean + perturbation, 0, 1)

    def _apply_input_diversity(self, x_input, resize_factor=1.15, diversity_application_prob=0.7):
        """
        Apply input diversity with random resizing and padding.
        """
        if torch.rand(1).item() >= diversity_application_prob:
            return x_input

        _b, _c, h_orig, w_orig = x_input.shape
        h_resized, w_resized = int(h_orig * resize_factor), int(w_orig * resize_factor)
        h_pad_total = h_resized - h_orig
        w_pad_total = w_resized - w_orig

        top_pad = torch.randint(0, h_pad_total + 1, (1,)).item() if h_pad_total >= 0 else 0
        left_pad = torch.randint(0, w_pad_total + 1, (1,)).item() if w_pad_total >= 0 else 0
        bottom_pad = h_pad_total - top_pad if h_pad_total >= 0 else 0
        right_pad = w_pad_total - left_pad if w_pad_total >= 0 else 0

        img_intermediate_resized = F.interpolate(x_input, size=(h_resized, w_resized), mode='bilinear', align_corners=False)
        img_padded = F.pad(img_intermediate_resized, [left_pad, right_pad, top_pad, bottom_pad], mode='constant', value=0)

        return F.interpolate(img_padded, size=(h_orig, w_orig), mode='bilinear', align_corners=False)

    def __call__(self, images_clean, labels_true):
        """
        Run FMIA attack.

        Args:
            images_clean (torch.Tensor): Original input images.
            labels_true (torch.Tensor): Ground-truth labels.

        Returns:
            torch.Tensor: Adversarial images.
        """
        images_clean, labels_true = images_clean.to(self.device), labels_true.to(self.device)
        original_images_cloned = images_clean.clone().detach()
        adv_images_current = images_clean.clone().detach()

        # Initialize momentum buffer
        if 'f-grad' in self.variant:
            fft_sample_shape = self._fft_2d(images_clean[:1]).shape
            momentum_val = torch.zeros((images_clean.size(0), *fft_sample_shape[1:]), dtype=torch.complex64, device=self.device)
        else:
            momentum_val = torch.zeros_like(images_clean, device=self.device)

        original_spatial_dims = images_clean.shape

        for _iter in range(self.steps):
            adv_images_current.requires_grad = True
            input_for_gradient_calc = adv_images_current

            # Nesterov lookahead (optional)
            if 'nesterov' in self.variant:
                if 'f-grad' in self.variant:
                    spatial_momentum_approx = self._ifft_2d(momentum_val, original_spatial_dims).real
                else:
                    spatial_momentum_approx = momentum_val
                lookahead_img = adv_images_current + self.alpha * self.decay * spatial_momentum_approx
                input_for_gradient_calc = torch.clamp(lookahead_img, 0, 1).detach().requires_grad_(True)

            total_grad_accumulator = torch.zeros_like(images_clean, device=self.device)

            # Apply Scale-Invariant gradients (optional)
            for scale_idx in range(self.si_scales):
                current_scaled_input = input_for_gradient_calc / (2 ** scale_idx) if self.si_scales > 1 else input_for_gradient_calc

                if 'di' in self.variant or 'di-si' in self.variant:
                    current_scaled_input = self._apply_input_diversity(current_scaled_input, diversity_application_prob=self.di_prob)

                current_scaled_input = current_scaled_input.detach().requires_grad_(True)
                self.model.zero_grad()
                model_outputs = self.model(current_scaled_input)
                loss_val = self.loss_fn(model_outputs, labels_true)
                loss_val.backward()

                grad_from_scale = current_scaled_input.grad if current_scaled_input.grad is not None else torch.zeros_like(current_scaled_input)
                total_grad_accumulator += grad_from_scale / (2 ** scale_idx) if self.si_scales > 1 else grad_from_scale

            final_batch_grad = self._normalize_grad_l1(total_grad_accumulator / self.si_scales)

            # Frequency-momentum update
            if 'f-grad' in self.variant:
                grad_in_freq_domain = self._fft_2d(final_batch_grad)
                momentum_val = self.decay * momentum_val + grad_in_freq_domain
                filtered_freq_momentum = self._frequency_filter(momentum_val, original_spatial_dims)
                spatial_perturb_direction = self._ifft_2d(filtered_freq_momentum, original_spatial_dims).real
            else:
                momentum_val = self.decay * momentum_val + final_batch_grad
                if self.filter_type != 'none':
                    momentum_in_freq_domain = self._fft_2d(momentum_val)
                    filtered_spatial_momentum_freq = self._frequency_filter(momentum_in_freq_domain, original_spatial_dims)
                    spatial_perturb_direction = self._ifft_2d(filtered_spatial_momentum_freq, original_spatial_dims).real
                else:
                    spatial_perturb_direction = momentum_val

            # Apply update step
            perturb_step = self.alpha * torch.sign(spatial_perturb_direction)
            adv_images_current = self._project_perturbation(adv_images_current.detach() + perturb_step, original_images_cloned)
            momentum_val = momentum_val.detach()

        return adv_images_current.detach()
