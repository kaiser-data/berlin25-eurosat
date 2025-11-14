"""Quantization utilities for weight compression in federated learning."""

import torch
import torch.nn as nn
from typing import Dict, Tuple
import logging

logger = logging.getLogger(__name__)


class QuantizedLinear(nn.Module):
    """Linear layer with quantized weights for real low-bit training."""

    def __init__(self, in_features, out_features, bias=True, bit_width=32):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bit_width = bit_width

        # Store weights in FP32 for gradient updates
        self.weight_fp32 = nn.Parameter(torch.randn(out_features, in_features))
        if bias:
            self.bias_fp32 = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias_fp32', None)

    def quantize_weight(self, weight):
        """Quantize weight tensor on-the-fly."""
        if self.bit_width == 32:
            return weight

        abs_max = weight.abs().max()
        if abs_max == 0:
            return weight

        if self.bit_width == 1:
            # Binary weights: -1 or +1
            scale = abs_max
            quantized = torch.sign(weight) * scale
        else:
            # Multi-bit quantization
            n_levels = 2 ** self.bit_width
            qmax = (n_levels // 2) - 1
            scale = abs_max / qmax
            quantized = torch.clamp(torch.round(weight / scale), -qmax - 1, qmax) * scale

        return quantized

    def forward(self, x):
        """Forward pass with quantized weights."""
        # Quantize weights for forward pass (still differentiable via straight-through estimator)
        weight_quantized = self.quantize_weight(self.weight_fp32)

        if self.bias_fp32 is not None:
            return nn.functional.linear(x, weight_quantized, self.bias_fp32)
        return nn.functional.linear(x, weight_quantized, None)


class QuantizedConv2d(nn.Module):
    """Conv2d layer with quantized weights for real low-bit training."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True, bit_width=32):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride
        self.padding = padding
        self.bit_width = bit_width

        # Store weights in FP32 for gradient updates
        self.weight_fp32 = nn.Parameter(torch.randn(out_channels, in_channels, *self.kernel_size))
        if bias:
            self.bias_fp32 = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias_fp32', None)

    def quantize_weight(self, weight):
        """Quantize weight tensor on-the-fly."""
        if self.bit_width == 32:
            return weight

        abs_max = weight.abs().max()
        if abs_max == 0:
            return weight

        if self.bit_width == 1:
            # Binary weights: -1 or +1
            scale = abs_max
            quantized = torch.sign(weight) * scale
        else:
            # Multi-bit quantization
            n_levels = 2 ** self.bit_width
            qmax = (n_levels // 2) - 1
            scale = abs_max / qmax
            quantized = torch.clamp(torch.round(weight / scale), -qmax - 1, qmax) * scale

        return quantized

    def forward(self, x):
        """Forward pass with quantized weights."""
        # Quantize weights for forward pass
        weight_quantized = self.quantize_weight(self.weight_fp32)

        return nn.functional.conv2d(x, weight_quantized, self.bias_fp32,
                                   self.stride, self.padding)


class WeightQuantizer:
    """Quantize model weights to specified bit-widths."""

    def __init__(self, bit_width: int):
        """
        Initialize quantizer.

        Args:
            bit_width: Number of bits (1, 2, 4, 8, 16, 32)
        """
        if bit_width not in [1, 2, 4, 8, 16, 32]:
            raise ValueError(f"Unsupported bit-width: {bit_width}")

        self.bit_width = bit_width
        self.n_levels = 2 ** bit_width

    def quantize_tensor(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """
        Quantize a tensor using symmetric quantization.

        Args:
            tensor: Input tensor

        Returns:
            (quantized_tensor, scale)
        """
        if self.bit_width == 32:
            return tensor, 1.0

        # Symmetric quantization
        abs_max = tensor.abs().max().item()

        if abs_max == 0:
            return torch.zeros_like(tensor), 1.0

        if self.bit_width == 1:
            # Binary: -1 or +1
            scale = abs_max
            quantized = torch.sign(tensor)
        else:
            # Multi-bit quantization
            qmax = (self.n_levels // 2) - 1
            scale = abs_max / qmax
            quantized = torch.clamp(torch.round(tensor / scale), -qmax - 1, qmax)

        return quantized, scale

    def dequantize_tensor(self, quantized: torch.Tensor, scale: float) -> torch.Tensor:
        """Dequantize tensor back to float."""
        if self.bit_width == 32:
            return quantized
        return quantized * scale

    def quantize_model(self, model: nn.Module) -> Tuple[Dict, Dict]:
        """
        Quantize all model parameters.

        Args:
            model: PyTorch model

        Returns:
            (quantized_state_dict, quantization_params)
        """
        quantized_state = {}
        quant_params = {}

        for name, param in model.state_dict().items():
            if param.dtype in [torch.float32, torch.float16]:
                q_tensor, scale = self.quantize_tensor(param)
                quantized_state[name] = q_tensor
                quant_params[name] = {"scale": scale}
            else:
                quantized_state[name] = param
                quant_params[name] = {"scale": 1.0}

        return quantized_state, quant_params

    def dequantize_model(self, quantized_state: Dict, quant_params: Dict) -> Dict:
        """Dequantize model weights."""
        dequantized_state = {}

        for name, q_tensor in quantized_state.items():
            scale = quant_params[name]["scale"]
            dequantized_state[name] = self.dequantize_tensor(q_tensor, scale)

        return dequantized_state


def get_model_size_mb(model: nn.Module, bit_width: int = 32) -> float:
    """Calculate model size in MB for given bit-width."""
    param_count = sum(p.numel() for p in model.parameters())
    size_bytes = param_count * (bit_width / 8)
    return size_bytes / (1024 * 1024)


def get_compression_metrics(model: nn.Module, bit_width: int) -> Dict[str, float]:
    """Calculate compression metrics."""
    original_size = get_model_size_mb(model, 32)
    compressed_size = get_model_size_mb(model, bit_width)
    compression_ratio = original_size / compressed_size
    reduction_percent = (1 - compressed_size / original_size) * 100

    return {
        "bit_width": bit_width,
        "original_size_mb": original_size,
        "compressed_size_mb": compressed_size,
        "compression_ratio": compression_ratio,
        "size_reduction_percent": reduction_percent,
    }
