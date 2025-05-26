import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision.transforms as transforms
from config import TRANSFORMERS_AVAILABLE

if TRANSFORMERS_AVAILABLE:
    from transformers import ViTForImageClassification, ViTImageProcessor


class SAGFMA2(nn.Module):
    """
    Unified Self-Attention Guided Frequency Momentum Attack (SA-GFMA2)

    Implements five variants of the attack:
    - 'base': Spatial gradient + momentum (frequency modulated image, loss computed after FFT modulation)
    - 'nag': Nesterov Accelerated Gradient (momentum lookahead)
    - 'ensemble': Combines CNN and ViT attention maps and logits
    - 'freq_update': Updates adversarial example directly in frequency amplitude domain
    - 'no_freq_momentum': Applies PGD-style spatial update without using momentum

    Args:
        model (nn.Module): The victim model (CNN).
        epsilon (float): Maximum L-infinity perturbation.
        alpha (float): Step size per iteration. Defaults to epsilon * 1.25 / steps.
        steps (int): Number of optimization steps.
        decay (float): Momentum decay factor.
        variant (str): Attack variant. One of ['base', 'nag', 'ensemble', 'freq_update', 'no_freq_momentum'].
        gamma (float): Power for attention modulation.
        modulation_strength (float): Scaling factor for frequency-domain modulation.
        adaptive (bool): Whether to use adaptive step size based on attention magnitude.
        ensemble_vit_model_name (str): HuggingFace model name for ViT (if using 'ensemble' variant).
        attn_implementation (str): Backend for ViT attention (e.g., 'eager', 'sdpa').
    """
    def __init__(
        self, model, epsilon=8/255, alpha=None, steps=10,
        decay=1.0, variant='base', gamma=2.0,
        modulation_strength=0.05, adaptive=False,
        ensemble_vit_model_name='google/vit-base-patch16-224',
        attn_implementation="eager"
    ):
        super().__init__()

        self.epsilon = epsilon
        self.alpha = alpha if alpha is not None else epsilon * 1.25 / steps
        self.steps = steps
        self.decay = decay
        self.variant = variant.lower()
        self.gamma = gamma
        self.modulation_strength = modulation_strength
        self.adaptive = adaptive

        self.supported_variants = ['base', 'nag', 'ensemble', 'freq_update', 'no_freq_momentum']
        if self.variant not in self.supported_variants:
            raise ValueError(f"Unsupported variant '{self.variant}'. Choose from: {self.supported_variants}")

        # self.device = next(model.parameters()).device if next(model.parameters(), None) else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = next(model.parameters(), torch.empty(0, device='cuda' if torch.cuda.is_available() else 'cpu')).device

        self.model = model.to(self.device).eval()

        self.vit_model = None
        self.vit_processor = None
        if self.variant == 'ensemble':
            if not TRANSFORMERS_AVAILABLE:
                raise ImportError("Install 'transformers' library to use 'ensemble' variant.")
            self.vit_model = ViTForImageClassification.from_pretrained(
                ensemble_vit_model_name, output_attentions=True, attn_implementation=attn_implementation
            ).to(self.device).eval()
            self.vit_processor = ViTImageProcessor.from_pretrained(ensemble_vit_model_name)

        self.feature_map = None
        self._hook_handle = None
        self._register_hooks()


    def _register_hooks(self):
        """Registers forward hook to extract CNN feature map for attention."""
        if self._hook_handle:
            self._hook_handle.remove()
        def hook_fn(module, input, output):
            self.feature_map = output.detach()
        model_to_hook = getattr(self.model, 'get_original_model', lambda: self.model)()
        target_layer = None
        if hasattr(model_to_hook, 'layer4'):
            target_layer = model_to_hook.layer4[-1]
        elif hasattr(model_to_hook, 'features'):
            for layer in reversed(model_to_hook.features):
                if isinstance(layer, nn.Conv2d):
                    target_layer = layer
                    break
        if not target_layer:
            convs = [m for m in model_to_hook.modules() if isinstance(m, nn.Conv2d)]
            target_layer = convs[-1] if convs else model_to_hook
        self._hook_handle = target_layer.register_forward_hook(hook_fn)


    def get_attention_mask(self, x_input):
        """Returns normalized attention map from CNN feature map."""
        self.model.to(x_input.device)
        with torch.no_grad():
            _ = self.model(x_input)
        if self.feature_map is None:
            return torch.ones(x_input.shape[0], x_input.shape[2], x_input.shape[3], device=x_input.device)
        fmap_val = self.feature_map.float()
        if fmap_val.ndim != 4:
            return torch.ones_like(x_input[:, 0])
        attn_map = fmap_val.mean(dim=1, keepdim=True)
        attn_map = F.interpolate(attn_map, size=x_input.shape[2:], mode='bilinear', align_corners=False).squeeze(1)
        min_vals, max_vals = attn_map.amin(dim=(1,2), keepdim=True), attn_map.amax(dim=(1,2), keepdim=True)
        return (attn_map - min_vals) / (max_vals - min_vals + 1e-8)


    def get_vit_attention(self, x_input):
        """Returns normalized attention map from ViT model."""
        pil_images = [transforms.ToPILImage()(img.cpu()) for img in x_input]
        vit_inputs = self.vit_processor(images=pil_images, return_tensors='pt').to(self.device)
        vit_outputs = self.vit_model(**vit_inputs)
        last_attn = vit_outputs.attentions[-1].mean(dim=1)  # (B, L, L)
        B, _, _ = last_attn.shape
        maps = []
        for i in range(B):
            cls_attn = last_attn[i, 0, 1:]
            P = int(np.sqrt(cls_attn.shape[0]))
            if P * P != cls_attn.shape[0]:
                maps.append(torch.full((x_input.shape[2], x_input.shape[3]), cls_attn.mean().item(), device=self.device))
            else:
                reshaped = cls_attn.view(1, 1, P, P)
                upsampled = F.interpolate(reshaped, size=x_input.shape[2:], mode='bilinear', align_corners=False)
                maps.append((upsampled[0, 0] - upsampled[0, 0].min()) / (upsampled[0, 0].max() - upsampled[0, 0].min() + 1e-8))
        return torch.stack(maps)


    def fft_image(self, img):
        fft = torch.fft.fft2(img, dim=(-2, -1), norm='ortho')
        return torch.abs(fft), torch.angle(fft)

    def ifft_image(self, amp, phase):
        return torch.clamp(torch.fft.ifft2(amp * torch.exp(1j * phase), dim=(-2, -1), norm='ortho').real, 0, 1)

    def modulate_amplitude(self, amp, attn_mask):
        if attn_mask.ndim == 3:
            attn_mask = attn_mask.unsqueeze(1)
        modulator = 1 + self.modulation_strength * (attn_mask.expand_as(amp).clamp(0, 1) ** self.gamma)
        return amp * modulator

    def _compute_loss(self, logits, targets):
        if targets is None:
            return -F.cross_entropy(logits, logits.argmax(dim=1), reduction='mean')
        return F.cross_entropy(logits, targets, reduction='mean')

    def _normalize_gradient(self, grad, p_norm=1):
        flat = grad.view(grad.size(0), -1)
        norm = flat.abs().sum(dim=1, keepdim=True) if p_norm == 1 else flat.norm(p=2, dim=1, keepdim=True)
        return grad / (norm.view(-1, 1, 1, 1) + 1e-8)

    def forward(self, x_input, targets=None):
        x_input = x_input.to(self.device)
        if targets is not None:
            targets = targets.to(self.device).long()

        x_adv = x_input.clone().detach()
        momentum_spatial = torch.zeros_like(x_adv)
        momentum_freq = torch.zeros_like(x_adv)

        for _ in range(self.steps):
            if self.variant == 'nag':
                velocity = self.decay * momentum_spatial
                x_for_grad = (x_adv + self.alpha * velocity).detach().requires_grad_(True)
            else:
                x_for_grad = x_adv.detach().requires_grad_(True)

            cnn_attn = self.get_attention_mask(x_for_grad)
            attn = cnn_attn
            if self.variant == 'ensemble':
                vit_attn = self.get_vit_attention(x_for_grad)
                attn = (cnn_attn + vit_attn) / 2.0

            amp, phase = self.fft_image(x_for_grad)
            modulated_amp = self.modulate_amplitude(amp, attn)
            x_modulated = self.ifft_image(modulated_amp, phase)

            logits = self.model(x_modulated)
            if self.variant == 'ensemble':
                vit_inputs = self.vit_processor(
                    images=[transforms.ToPILImage()(img.cpu()) for img in x_modulated],
                    return_tensors='pt'
                ).to(self.device)
                vit_logits = self.vit_model(**vit_inputs).logits
                logits = (logits + vit_logits) / 2.0

            loss = self._compute_loss(logits, targets)
            self.model.zero_grad()
            if self.variant == 'ensemble' and self.vit_model:
                self.vit_model.zero_grad()
            if x_for_grad.grad is not None:
                x_for_grad.grad.zero_()
            loss.backward()
            grad = x_for_grad.grad.detach()

            step_size = self.alpha
            if self.adaptive:
                attn_mean = attn.view(attn.size(0), -1).mean(dim=1).clamp(0.5, 1.5)
                step_size = (self.alpha * (1.0 - attn_mean)).view(-1, 1, 1, 1)
                step_size = torch.clamp(step_size, min=1e-5)


            if self.variant == 'freq_update':
                grad_amp, _ = self.fft_image(grad)
                norm_amp = self._normalize_gradient(grad_amp, p_norm=1)
                momentum_freq = self.decay * momentum_freq + norm_amp
                cur_amp, cur_phase = self.fft_image(x_adv)
                new_amp = torch.clamp(cur_amp + step_size * torch.sign(momentum_freq), 0)
                x_adv = self.ifft_image(new_amp, cur_phase).detach()
            elif self.variant == 'no_freq_momentum':
                x_adv = (x_adv + step_size * torch.sign(grad)).detach()
            else:
                grad_norm = self._normalize_gradient(grad, p_norm=1)
                momentum_spatial = self.decay * momentum_spatial + grad_norm
                x_adv = (x_adv + step_size * torch.sign(momentum_spatial)).detach()

            delta = torch.clamp(x_adv - x_input, -self.epsilon, self.epsilon)
            x_adv = torch.clamp(x_input + delta, 0, 1).detach()

        return x_adv

    def __del__(self):
        """Remove the forward hook on deletion to avoid memory leaks."""
        if hasattr(self, '_hook_handle') and self._hook_handle:
            try:
                self._hook_handle.remove()
            except Exception:
                pass
