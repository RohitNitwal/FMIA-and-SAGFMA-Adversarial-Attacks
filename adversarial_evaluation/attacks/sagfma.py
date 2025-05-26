import torch
import torch.nn as nn
import torch.nn.functional as F

class SAGFMA(nn.Module):
    """
    Self-Attention Guided Frequency Momentum Attack (SA-GFMA)

    This attack generates adversarial examples by combining:
    - Gradient-based iterative updates
    - Frequency-domain filtering via FFT with band masks
    - Momentum to stabilize direction
    - Self-attention to spatially guide perturbations
    - Optional per-image adaptive epsilon and step size
    """
    def __init__(self, model: nn.Module, epsilon: float = 8/255, steps: int = 10,
                 decay: float = 1.0, alpha: float = None, freq_bands: int = 4,
                 attention_iters: int = 3, adaptive: bool = True):
        super().__init__()
        self.model = model
        self.epsilon = epsilon
        self.steps = steps
        self.decay = decay
        self.alpha = alpha if alpha is not None else epsilon * 1.25 / steps
        self.freq_bands = max(2, freq_bands)
        self.attention_iters = attention_iters
        self.adaptive = adaptive

        self.freq_weights = nn.Parameter(torch.ones(self.freq_bands) / self.freq_bands, requires_grad=False)
        self.criterion = nn.CrossEntropyLoss()
        self._freq_masks_cache = {}

    def forward(self, images: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Main forward function to generate adversarial examples.
        Args:
            images: Input batch of shape (B, C, H, W)
            labels: Ground-truth labels for loss computation
        Returns:
            x_adv: Adversarially perturbed images
        """
        was_training = self.model.training
        self.model.eval()

        x_adv = images.clone().detach()
        momentum = torch.zeros_like(images, device=images.device)
        batch_size, channels, H, W = images.shape
        attention_map = torch.ones(batch_size, 1, H, W, device=images.device)

        current_eps, current_alpha = self._get_current_parameters(images)

        mask_key = (H, W, images.device)
        if mask_key not in self._freq_masks_cache:
            self._freq_masks_cache[mask_key] = self._precompute_frequency_masks(H, W, device=images.device)
        freq_masks = self._freq_masks_cache[mask_key]

        for t in range(self.steps):
            x_adv.requires_grad_(True)
            outputs = self.model(x_adv)
            loss = self.criterion(outputs, labels)
            grad = torch.autograd.grad(loss, x_adv, retain_graph=False, create_graph=False)[0]

            new_attention = self._update_attention_map(grad, attention_map)
            filtered_grad = self._frequency_transform(grad, new_attention, freq_masks)

            # Normalize gradient using L1 norm per image
            grad_norm_val = filtered_grad.view(batch_size, -1).abs().sum(dim=1, keepdim=True)
            normalized_grad = filtered_grad / (grad_norm_val.view(batch_size, 1, 1, 1) + 1e-10)
            momentum = self.decay * momentum + normalized_grad

            # Optionally reduce step size near the end
            effective_alpha = current_alpha
            if self.adaptive and t >= 0.8 * self.steps:
                effective_alpha = current_alpha * 0.5

            step_val = effective_alpha.view(batch_size, 1, 1, 1) * momentum.sign() if isinstance(effective_alpha, torch.Tensor) else effective_alpha * momentum.sign()
            x_adv = x_adv.detach() + step_val

            # Project to epsilon-ball
            delta = torch.clamp(x_adv - images,
                                -current_eps.view(batch_size, 1, 1, 1) if isinstance(current_eps, torch.Tensor) else -current_eps,
                                 current_eps.view(batch_size, 1, 1, 1) if isinstance(current_eps, torch.Tensor) else current_eps)
            x_adv = torch.clamp(images + delta, 0.0, 1.0).detach()
            attention_map = new_attention.detach()

        if was_training:
            self.model.train()
        return x_adv

    def _get_current_parameters(self, images):
        """
        Returns either static or adaptive epsilon and alpha
        """
        if self.adaptive:
            return self._compute_adaptive_params(images)
        return self.epsilon, self.alpha

    def _compute_adaptive_params(self, images: torch.Tensor) -> tuple:
        """
        Adjust epsilon and alpha per image based on standard deviation.
        """
        with torch.no_grad():
            batch_size = images.size(0)
            img_std = images.view(batch_size, -1).std(dim=1)
            scale = (1.0 - img_std).clamp(0.8, 1.2)
            adaptive_eps = self.epsilon * scale
            adaptive_alpha = self.alpha * scale
        return adaptive_eps, adaptive_alpha

    def _precompute_frequency_masks(self, H: int, W: int, device=None):
        """
        Radial band masks in FFT domain split into freq_bands parts.
        """
        device = device or self.freq_weights.device
        masks = []
        center_y, center_x = H // 2, W // 2
        yy = torch.arange(H, device=device).unsqueeze(1).expand(H, W).float()
        xx = torch.arange(W, device=device).unsqueeze(0).expand(H, W).float()
        dist = torch.sqrt((yy - center_y)**2 + (xx - center_x)**2)
        max_radius = dist.max()
        band_edges = torch.linspace(0, max_radius, steps=self.freq_bands + 1, device=device)

        for i in range(self.freq_bands):
            inner, outer = band_edges[i], band_edges[i+1]
            mask = torch.zeros((H, W), device=device)
            mask[(dist >= inner) & (dist <= outer if i == self.freq_bands - 1 else dist < outer)] = 1.0
            if i == 0:
                mask[center_y, center_x] = 1.0
            masks.append(mask)
        return masks

    def _update_attention_map(self, grad: torch.Tensor, prev_attention: torch.Tensor) -> torch.Tensor:
        """
        Update spatial attention map from current gradient and previous attention.
        Applies refinement via average pooling.
        """
        with torch.no_grad():
            b, c_channels, H, W = grad.shape
            grad_abs_norm = grad.abs() / (grad.abs().sum(dim=(1, 2, 3), keepdim=True) + 1e-10)
            attention = grad_abs_norm.sum(dim=1, keepdim=True)

            for _ in range(self.attention_iters):
                smoothed_attention = F.avg_pool2d(attention, kernel_size=3, stride=1, padding=1)
                attention = attention + F.relu(smoothed_attention)
                attention = attention / (attention.sum(dim=(2, 3), keepdim=True) + 1e-10)

            if not self.adaptive:
                attention = 0.7 * prev_attention + 0.3 * attention
            else:
                diff = (attention - prev_attention).abs().mean(dim=(1, 2, 3), keepdim=True)
                adapt_weight = (torch.sigmoid(10 * diff - 2) * 0.6 + 0.2).clamp(0.2, 0.8)
                attention = (1 - adapt_weight) * prev_attention + adapt_weight * attention

            max_val = attention.amax(dim=(2, 3), keepdim=True)
            attention = attention / (max_val + 1e-10)
        return attention

    def _frequency_transform(self, grad: torch.Tensor, attention_map: torch.Tensor, freq_masks: list) -> torch.Tensor:
        """
        Applies FFT-based filtering on the gradient using attention-weighted frequency bands.
        """
        with torch.no_grad():
            b, c_channels, H, W = grad.shape
            att_expanded = attention_map.expand(b, c_channels, H, W) if attention_map.shape[1] == 1 and c_channels > 1 else attention_map
            grad_attended = grad * att_expanded

            grad_flat = grad_attended.view(b * c_channels, H, W)
            fft_coeffs = torch.fft.fft2(grad_flat, norm='ortho')
            fft_shifted = torch.fft.fftshift(fft_coeffs, dim=(-2, -1))

            freq_band_weights = F.softmax(self.freq_weights.to(grad.device), dim=0)
            filtered_fft_shifted = torch.zeros_like(fft_shifted)

            for weight, mask_tensor in zip(freq_band_weights, freq_masks):
                filtered_fft_shifted += weight * fft_shifted * mask_tensor.to(grad.device).unsqueeze(0)

            filtered_fft_unshifted = torch.fft.ifftshift(filtered_fft_shifted, dim=(-2, -1))
            filtered_grad_spatial = torch.fft.ifft2(filtered_fft_unshifted, norm='ortho').real
            filtered_grad_reshaped = filtered_grad_spatial.view(b, c_channels, H, W)

            return filtered_grad_reshaped * att_expanded



class SAGFMA3(nn.Module):
    """
    Self-Attention Guided Frequency Momentum Attack (SA-GFMA3)

    This adversarial attack combines:
    - Spatial attention from gradients to focus on sensitive regions
    - Frequency-domain filtering using learnable band masks
    - Momentum-based iterative optimization
    - Optional per-image adaptive epsilon and step size

    Args:
        model (nn.Module): Target model to attack.
        epsilon (float): L-infinity perturbation budget.
        steps (int): Number of optimization steps.
        decay (float): Momentum decay factor.
        alpha (float): Step size per iteration. Defaults to epsilon * 1.25 / steps.
        freq_bands (int): Number of radial frequency bands.
        attention_iters (int): Iterations to refine attention map.
        adaptive (bool): Whether to use adaptive ε and α per image.
    """
    def __init__(self, model: nn.Module, epsilon=8/255, steps=10,
                 decay=1.0, alpha=None, freq_bands=4,
                 attention_iters=3, adaptive=True):
        super().__init__()
        self.model = model
        self.epsilon = epsilon
        self.steps = steps
        self.decay = decay
        self.alpha = alpha if alpha is not None else epsilon * 1.25 / steps
        self.freq_bands = max(2, freq_bands)
        self.attention_iters = attention_iters
        self.adaptive = adaptive

        self.freq_weights = nn.Parameter(torch.ones(self.freq_bands) / self.freq_bands, requires_grad=True)
        self.criterion = nn.CrossEntropyLoss()
        self._freq_masks_cache = {}
    def forward(self, images, labels):
        """
        Performs the attack with optional frequency weight learning.

        Args:
            images (torch.Tensor): Input images (B, C, H, W)
            labels (torch.Tensor): Ground truth labels (B,)
    
        Returns:
            torch.Tensor: Adversarial examples
        """
        torch.cuda.empty_cache()
        device = images.device
        self.model.eval()
        was_training = self.model.training

        x_adv = images.clone().detach()
        momentum = torch.zeros_like(images, device=device)
        batch_size, c_channels, H, W = images.shape
        attention_map = torch.ones(batch_size, 1, H, W, device=device)

        eps_used, alpha_used = self._get_current_parameters(images)

        cache_key = (H, W, device)
        if cache_key not in self._freq_masks_cache:
            self._freq_masks_cache[cache_key] = self._precompute_frequency_masks(H, W, device)
        freq_masks_list = self._freq_masks_cache[cache_key]

        stacked_masks = torch.stack(freq_masks_list, dim=0).to(device)

        # Initialize optimizer only if freq_weights are trainable
        optimizer = torch.optim.Adam([self.freq_weights], lr=0.01) if self.freq_weights.requires_grad else None

        if not self.freq_weights.requires_grad:
            with torch.no_grad():
                weights = F.softmax(self.freq_weights.to(device), dim=0)
                combined_mask = (weights.view(-1, 1, 1) * stacked_masks).sum(dim=0)

        for t in range(self.steps):
            x_adv.requires_grad_()
            outputs = self.model(x_adv)
            loss = self.criterion(outputs, labels)

            # If learning weights, recompute mask + step optimizer
            if self.freq_weights.requires_grad:
                weights = F.softmax(self.freq_weights.to(device), dim=0)
                combined_mask = (weights.view(-1, 1, 1) * stacked_masks).sum(dim=0)
                optimizer.zero_grad()

            grad = torch.autograd.grad(loss, x_adv, retain_graph=False, create_graph=False)[0]

            if self.freq_weights.requires_grad:
                loss.backward(retain_graph=True)
                optimizer.step()

            # Attention-guided frequency filter
            new_attention = self._update_attention_map(grad, attention_map)
            filtered_grad = self._frequency_transform(grad, new_attention, combined_mask)

            # Momentum update
            norm_val = filtered_grad.view(batch_size, -1).abs().sum(dim=1).view(batch_size, 1, 1, 1) + 1e-10
            normalized_grad = filtered_grad / norm_val
            momentum = self.decay * momentum + normalized_grad

            # Step + Projection
            effective_alpha = alpha_used * 0.5 if self.adaptive and t >= 0.8 * self.steps else alpha_used
            step = effective_alpha.view(batch_size, 1, 1, 1) * momentum.sign() if isinstance(effective_alpha, torch.Tensor) else effective_alpha * momentum.sign()

            x_adv = x_adv.detach() + step
            delta = torch.clamp(x_adv - images, 
                            -eps_used.view(batch_size, 1, 1, 1) if isinstance(eps_used, torch.Tensor) else -eps_used, 
                             eps_used.view(batch_size, 1, 1, 1) if isinstance(eps_used, torch.Tensor) else eps_used)
            x_adv = torch.clamp(images + delta, 0.0, 1.0).detach()
            attention_map = new_attention.detach()

        if was_training:
            self.model.train()

        return x_adv


    def _get_current_parameters(self, images):
        """Returns (epsilon, alpha) either adaptively or as constants."""
        return self._compute_adaptive_params(images) if self.adaptive else (self.epsilon, self.alpha)

    def _compute_adaptive_params(self, images):
        """Compute ε and α based on per-image std deviation."""
        with torch.no_grad():
            batch_size = images.size(0)
            std_dev = images.view(batch_size, -1).std(dim=1)
            scale = (1.0 - std_dev).clamp(0.8, 1.2)
            eps_val = self.epsilon * scale
            alpha_val = self.alpha * scale
        return eps_val, alpha_val

    def _precompute_frequency_masks(self, H, W, device):
        """
        Returns a list of band masks partitioning the frequency space.
        Each mask isolates a concentric radial frequency ring.
        """
        center_y, center_x = H // 2, W // 2
        yy = torch.arange(H, device=device).unsqueeze(1).expand(H, W).float()
        xx = torch.arange(W, device=device).unsqueeze(0).expand(H, W).float()
        dist = torch.sqrt((yy - center_y)**2 + (xx - center_x)**2)
        max_r = dist.max()
        band_edges = torch.linspace(0, max_r, steps=self.freq_bands + 1, device=device)
        masks = []
        for i in range(self.freq_bands):
            mask = torch.zeros(H, W, device=device)
            inner, outer = band_edges[i], band_edges[i + 1]
            if i == self.freq_bands - 1:
                mask[(dist >= inner) & (dist <= outer)] = 1.0
            else:
                mask[(dist >= inner) & (dist < outer)] = 1.0
            if i == 0:
                mask[center_y, center_x] = 1.0
            masks.append(mask)
        return masks

    def _update_attention_map(self, grad, prev_attention):
        """
        Updates the spatial attention map using gradients.
        Applies average pooling and adaptive blending.
        """
        with torch.no_grad():
            b, c_channels, H, W = grad.shape
            grad_abs = grad.abs()
            grad_norm = grad_abs / (grad_abs.sum(dim=(1, 2, 3), keepdim=True) + 1e-10)
            attention = grad_norm.sum(dim=1, keepdim=True)

            for _ in range(self.attention_iters):
                attention_pooled = F.avg_pool2d(attention, 3, 1, 1)
                attention = attention + F.relu(attention_pooled)
                attention = attention / (attention.sum(dim=(2, 3), keepdim=True) + 1e-10)

            if not self.adaptive:
                attention = 0.7 * prev_attention + 0.3 * attention
            else:
                diff_val = (attention - prev_attention).abs().mean(dim=(1, 2, 3), keepdim=True)
                blend_weight = (torch.sigmoid(10 * diff_val - 2) * 0.6 + 0.2).clamp(0.2, 0.8)
                attention = (1 - blend_weight) * prev_attention + blend_weight * attention

            return attention / (attention.amax(dim=(2, 3), keepdim=True) + 1e-10)

    def _frequency_transform(self, grad, attention_map, combined_mask):
        """
        Applies attention-guided FFT filtering and inverse transform.
        """
        with torch.no_grad():
            b, c_channels, H, W = grad.shape
            att_expanded = attention_map.expand(b, c_channels, H, W) if attention_map.shape[1] == 1 else attention_map
            grad_att = grad * att_expanded

            grad_flat = grad_att.view(b * c_channels, H, W)
            fft = torch.fft.fft2(grad_flat, norm='ortho')
            fft_shifted = torch.fft.fftshift(fft, dim=(-2, -1))

            filtered_fft = fft_shifted * combined_mask.to(fft.device)

            fft_unshifted = torch.fft.ifftshift(filtered_fft, dim=(-2, -1))
            filtered_grad = torch.fft.ifft2(fft_unshifted, norm='ortho').real

            return filtered_grad.view(b, c_channels, H, W) * att_expanded

