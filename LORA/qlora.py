"""
QLoRA (Quantized Low-Rank Adaptation) Implementation
Based on:
    - Paper: https://arxiv.org/pdf/2305.14314.pdf
    - Repository: https://github.com/artidoro/qlora

This implementation builds on the LoRA implementation and adds quantization techniques
to further reduce memory usage during fine-tuning.
"""

"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                   QUANTIZED LOW-RANK ADAPTATION (QLoRA)                      ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  Key Idea: Apply LoRA to 4-bit quantized pre-trained models                  ║
║                                                                              ║
║  QLoRA architecture:                                                         ║
║    1. Keep frozen model weights in 4-bit precision                           ║
║    2. Compute activations in BFloat16 (higher precision)                     ║
║    3. Use Low-Rank Adaptation (LoRA) for trainable parameters                ║
║    4. Apply NF4 quantization with double quantization                        ║
║                                                                              ║
║  Components:                                                                 ║
║    • NF4 (Normal Float 4): 4-bit data type optimized for normal distribution ║
║    • Double Quantization: Quantize the quantization constants                ║
║    • Paged Optimizers: Better memory management with CPU offloading          ║
║    • Adapter Modules: Low-rank trainable matrices (LoRA)                     ║
║                                                                              ║
║  Memory Optimizations:                                                       ║
║    • 4-bit quantized model weights (drastically reduces memory usage)        ║
║    • Activation gradient checkpointing (trade computation for memory)        ║
║    • Paged Adamw optimizer (offload optimizer states to CPU)                 ║
║                                                                              ║
║  Advantages:                                                                 ║
║    • Fine-tune models up to 70B parameters on a single GPU                   ║
║    • Equal or better performance compared to full fine-tuning                ║
║    • Compatible with LoRA adapters and workflows                             ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Optional, Tuple, Union, Literal, List, Dict
from safetensors.torch import save_file
from enum import Enum, auto

# Importing bitsandbytes for quantization
try:
    import bitsandbytes as bnb
    from bitsandbytes.nn import Linear4bit, LinearNF4
    BITSANDBYTES_AVAILABLE = True
except ImportError:
    BITSANDBYTES_AVAILABLE = False
    print("Warning: bitsandbytes not available. Quantization features will be disabled.")

# Defining quantization types
class QuantizationType(Enum):
    NONE = auto()  # No quantization
    FP4 = auto()   # Float 4-bit
    NF4 = auto()   # Normal Float 4-bit (optimized for normally distributed weights)
    INT8 = auto()  # Integer 8-bit
    INT4 = auto()  # Integer 4-bit

# Double quantization
class DoubleQuantization:
    """
    Double Quantization: Further reduces memory usage by quantizing 
    the quantization constants themselves
    
    This compresses the memory footprint of the quantization constants,
    which can be significant when using 4-bit quantization
    """
    def __init__(self, enable=True):
        self.enable = enable

class QLoRABaseLayer:
    """Base class for all QLoRA layers with common functionality"""

    def __init__(self,
                 rank=8,
                 lora_alpha=8,
                 lora_dropout=0.0,
                 use_rslora=True,
                 quantization=QuantizationType.NF4,
                 double_quant=True):
        
        # Standard LoRA parameters
        self.rank = rank
        self.lora_alpha = lora_alpha
        self.lora_dropout = nn.Dropout(lora_dropout) if lora_dropout > 0 else lambda x: x
        self.use_rslora = use_rslora
        
        # Quantization parameters
        self.quantization = quantization
        self.double_quant = double_quant
        
        # Set scaling factor based on whether we use rank-stabilized LoRA
        self.scaling = self.lora_alpha / self.rank**0.5 if self.use_rslora else self.lora_alpha / self.rank
    
    def _load_pretrained_weights(self, state_dict):
        """Load pretrained weights into the layer"""
        self.weight.data = state_dict["weight"]
        if "bias" in state_dict and hasattr(self, "bias") and self.bias is not None:
            self.bias.data = state_dict["bias"]
            
    def _use_4bit(self):
        """Check if we're using 4-bit quantization"""
        return self.quantization in [QuantizationType.FP4, QuantizationType.NF4]
    
    def _use_8bit(self):
        """Check if we're using 8-bit quantization"""
        return self.quantization == QuantizationType.INT8


class QLoRALinear(nn.Module, QLoRABaseLayer):
    """QLoRA implementation for linear layers"""
    
    def __init__(self,
                 in_features,
                 out_features,
                 bias=True,
                 rank=8,
                 lora_alpha=8,
                 lora_dropout=0.0,
                 use_rslora=True,
                 quantization=QuantizationType.NF4,
                 double_quant=True,
                 **kwargs):
        
        nn.Module.__init__(self)
        
        QLoRABaseLayer.__init__(self,
                               rank=rank,
                               lora_alpha=lora_alpha,
                               lora_dropout=lora_dropout,
                               use_rslora=use_rslora,
                               quantization=quantization,
                               double_quant=double_quant)
        
        # Use the appropriate linear layer based on quantization
        if BITSANDBYTES_AVAILABLE and self._use_4bit():
            # Create a 4-bit quantized layer
            compute_dtype = torch.float16  # Can be changed to bfloat16 for better numerical stability
            
            if self.quantization == QuantizationType.NF4:
                # Normal Float 4-bit - optimized for normally distributed weights
                self.base_layer = LinearNF4(
                    in_features, 
                    out_features, 
                    bias=bias, 
                    compute_dtype=compute_dtype,
                    double_quant=double_quant
                )
            else:
                # Regular Float 4-bit
                self.base_layer = Linear4bit(
                    in_features, 
                    out_features, 
                    bias=bias, 
                    compute_dtype=compute_dtype,
                    double_quant=double_quant
                )
        
        elif BITSANDBYTES_AVAILABLE and self._use_8bit():
            # Create an 8-bit quantized layer
            self.base_layer = bnb.nn.Linear8bitLt(
                in_features,
                out_features,
                bias=bias
            )
        
        else:
            # Fallback to regular nn.Linear
            self.base_layer = nn.Linear(in_features, out_features, bias=bias, **kwargs)
            self.base_layer.weight.requires_grad = False  # Freeze base layer weights
            
        # Add LoRA low-rank adaptation matrices
        self.lora_A = nn.Parameter(torch.zeros((in_features, rank)))
        self.lora_B = nn.Parameter(torch.zeros((rank, out_features)))
        
        # Initialize LoRA matrices
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)  # Zero initialization for B helps with training stability
        
    def forward(self, x):
        """Forward pass combining base layer output with LoRA adaptation"""
        
        # Get output from the quantized/base layer
        base_output = self.base_layer(x)
        
        # Apply LoRA adaptation
        lora_output = self.lora_dropout(x) @ self.lora_A @ self.lora_B
        
        # Scale and add LoRA output
        output = base_output + lora_output * self.scaling
        
        return output
        
    def _merge_weights(self):
        """Merge LoRA weights with base weights for inference"""
        # This operation is more complex with quantized weights
        # For simplicity, we'll dequantize, merge, and then quantize again
        
        if hasattr(self.base_layer, "weight"):
            # For regular nn.Linear
            merged_weights = self.base_layer.weight.data + (self.lora_A @ self.lora_B).T * self.scaling
            
            state_dict = {"weight": merged_weights}
            if hasattr(self.base_layer, "bias") and self.base_layer.bias is not None:
                state_dict["bias"] = self.base_layer.bias.data
                
            # Create new linear layer with merged weights
            merged_layer = nn.Linear(
                self.base_layer.in_features,
                self.base_layer.out_features,
                bias=self.base_layer.bias is not None
            )
            merged_layer.load_state_dict(state_dict)
            
            return merged_layer
        else:
            # For quantized layers, merging is more complex and generally not recommended
            # Would need to dequantize weights, merge, and then requantize
            # For now, return the module as is with a warning
            print("Warning: Weight merging for quantized layers is not fully supported")
            return self


class QLoRAEmbedding(nn.Module, QLoRABaseLayer):
    """QLoRA implementation for embedding layers"""
    
    def __init__(self,
                 num_embeddings,
                 embedding_dim,
                 padding_idx=None,
                 rank=8,
                 lora_alpha=8,
                 lora_dropout=0.0,
                 use_rslora=True,
                 quantization=QuantizationType.NONE,  # Quantization often not used for embeddings
                 **kwargs):
        
        nn.Module.__init__(self)
        
        QLoRABaseLayer.__init__(self,
                               rank=rank,
                               lora_alpha=lora_alpha,
                               lora_dropout=lora_dropout,
                               use_rslora=use_rslora,
                               quantization=quantization)
        
        # Embeddings are typically not quantized in QLoRA
        self.base_layer = nn.Embedding(
            num_embeddings, 
            embedding_dim, 
            padding_idx=padding_idx,
            **kwargs
        )
        self.base_layer.weight.requires_grad = False
        
        # LoRA matrices for embeddings
        self.lora_A = nn.Parameter(torch.zeros((num_embeddings, rank)))
        self.lora_B = nn.Parameter(torch.zeros((rank, embedding_dim)))
        
        # Initialize LoRA matrices
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)  # Zero initialization for B
        
        # Store embedding parameters
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        
    def forward(self, x):
        """Forward pass combining base embedding with LoRA adaptation"""
        
        # Get output from the base embedding layer
        base_output = self.base_layer(x)
        
        # For embeddings, we need to use embedding lookup for A
        # Then multiply by B to get the low-rank adaptation
        lora_A_output = F.embedding(
            input=x,
            weight=self.lora_A,
            padding_idx=self.padding_idx
        )
        
        # Matrix multiply with B and scale
        lora_output = lora_A_output @ self.lora_B
        
        # Combine outputs
        output = base_output + lora_output * self.scaling
        
        return output
        
    def _merge_weights(self):
        """Merge LoRA weights with base weights for inference"""
        
        merged_weights = self.base_layer.weight.data + (self.lora_A @ self.lora_B) * self.scaling
        
        state_dict = {"weight": merged_weights}
        merged_embeddings = nn.Embedding(
            self.num_embeddings, 
            self.embedding_dim,
            padding_idx=self.padding_idx
        )
        merged_embeddings.load_state_dict(state_dict)
        
        return merged_embeddings


@dataclass
class QLoRAConfig:
    """Configuration for QLoRA fine-tuning"""
    
    # LoRA parameters
    rank: int = 8
    target_modules: Optional[Union[List[str], str]] = None
    exclude_modules: Optional[Union[List[str], str]] = None
    lora_alpha: float = 8.0
    lora_dropout: float = 0.0
    bias: Literal["none", "all", "lora_only"] = "none"
    use_rslora: bool = True
    
    # Quantization parameters
    quantization: QuantizationType = QuantizationType.NF4
    double_quant: bool = True
    
    # Memory optimization parameters
    gradient_checkpointing: bool = False
    paged_adamw: bool = False


class QLoRAModel(nn.Module):
    """QLoRA model wrapper that applies low-rank adapters to a base model"""
    
    def __init__(self, model, config):
        """
        Initialize QLoRA model
        
        Args:
            model: The base model to adapt (usually a HuggingFace transformer)
            config: QLoRAConfig object with adaptation parameters
        """
        super().__init__()
        
        self.model = model
        self.config = config
        
        # Process target and exclude modules
        if isinstance(config.target_modules, str):
            self.target_modules = [config.target_modules]
        else:
            self.target_modules = config.target_modules or []
            
        if isinstance(config.exclude_modules, str):
            self.exclude_modules = [config.exclude_modules]
        else:
            self.exclude_modules = config.exclude_modules or []
        
        # Disable gradient for all parameters before applying adapters
        self.disable_all_grads()
        
        # Apply QLoRA adapters to the model
        self._apply_qlora(self.model)
        
        # Enable gradient for bias terms if specified
        self._toggle_bias_grad()
        
        # Enable gradient checkpointing if specified
        if config.gradient_checkpointing and hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable()
        
        # Print trainable parameter stats
        trainable_params, all_params = self._compute_trainable_parameters()
        print(f"Trainable parameters: {trainable_params:,} ({trainable_params/all_params:.2%} of {all_params:,} total parameters)")

    def forward(self, *inputs, **kwargs):
        """Forward pass through the adapted model"""
        return self.model(*inputs, **kwargs)
    
    def _exclude_module_name_check(self, name):
        """Check if a module should be excluded from adaptation"""
        return any(exclude_name in name for exclude_name in self.exclude_modules)
    
    def _target_module_name_check(self, name):
        """Check if a module should be targeted for adaptation"""
        if not self.target_modules:
            return True  # Target all modules if none specified
        return any(target_name in name for target_name in self.target_modules)
    
    def _apply_qlora(self, module, module_name=""):
        """Recursively apply QLoRA adapters to eligible modules"""
        
        # Process all child modules first
        for name, child in list(module.named_children()):
            full_name = f"{module_name}.{name}" if module_name else name
            
            # Skip excluded modules
            if self._exclude_module_name_check(full_name):
                print(f"Excluding module: {full_name}")
                continue
                
            # Recursively process child modules
            self._apply_qlora(child, full_name)
        
        # Check if current module is a candidate for replacement
        if isinstance(module, nn.Linear) and self._target_module_name_check(module_name):
            # Replace with QLoRA adapter
            parent_name, child_name = module_name.rsplit(".", 1) if "." in module_name else ("", module_name)
            parent = self.model if not parent_name else _get_module_by_name(self.model, parent_name)
            
            # Create QLoRA adapter
            qlora_module = QLoRALinear(
                in_features=module.in_features,
                out_features=module.out_features,
                bias=module.bias is not None,
                rank=self.config.rank,
                lora_alpha=self.config.lora_alpha,
                lora_dropout=self.config.lora_dropout,
                use_rslora=self.config.use_rslora,
                quantization=self.config.quantization,
                double_quant=self.config.double_quant
            )
            
            # Initialize the QLoRA module with the original weights
            if hasattr(module, "weight"):
                # For regular Linear layers
                state_dict = {
                    "weight": module.weight.data,
                }
                if module.bias is not None:
                    state_dict["bias"] = module.bias.data
                qlora_module.base_layer.load_state_dict(state_dict)
            
            # Replace the original module with the QLoRA module
            setattr(parent, child_name, qlora_module)
            print(f"Replaced {module_name} with QLoRA adapter")
        
        # Handle embeddings separately
        elif isinstance(module, nn.Embedding) and self._target_module_name_check(module_name):
            # Replace with QLoRA embedding
            parent_name, child_name = module_name.rsplit(".", 1) if "." in module_name else ("", module_name)
            parent = self.model if not parent_name else _get_module_by_name(self.model, parent_name)
            
            # Create QLoRA adapter for embedding
            qlora_module = QLoRAEmbedding(
                num_embeddings=module.num_embeddings,
                embedding_dim=module.embedding_dim,
                padding_idx=module.padding_idx,
                rank=self.config.rank,
                lora_alpha=self.config.lora_alpha,
                lora_dropout=self.config.lora_dropout,
                use_rslora=self.config.use_rslora
            )
            
            # Initialize with original embedding weights
            qlora_module.base_layer.weight.data = module.weight.data
            
            # Replace the original module
            setattr(parent, child_name, qlora_module)
            print(f"Replaced {module_name} with QLoRA embedding adapter")
    
    def _compute_trainable_parameters(self):
        """Compute the number of trainable parameters versus total parameters"""
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        all_params = sum(p.numel() for p in self.parameters())
        return trainable_params, all_params
    
    def disable_all_grads(self):
        """Disable gradient computation for all parameters"""
        for param in self.model.parameters():
            param.requires_grad = False
    
    def _toggle_bias_grad(self):
        """Enable gradient computation for bias terms based on config"""
        if self.config.bias == "none":
            return
            
        for n, p in self.model.named_parameters():
            if "bias" in n:
                if self.config.bias == "all" or (
                    self.config.bias == "lora_only" and 
                    any(target_name in n for target_name in self.target_modules)
                ):
                    p.requires_grad = True
    
    def _merge_weights(self):
        """Merge adapter weights with base weights for inference"""
        
        # Create a new model with merged weights
        merged_model = type(self.model)(self.model.config)
        merged_model.load_state_dict(self.model.state_dict())
        
        # Process all modules that might have QLoRA adapters
        for name, module in list(self.model.named_modules()):
            if isinstance(module, (QLoRALinear, QLoRAEmbedding)):
                # Get the parent module
                parent_name, child_name = name.rsplit(".", 1) if "." in name else ("", name)
                parent = merged_model if not parent_name else _get_module_by_name(merged_model, parent_name)
                
                # Merge weights and replace the module
                merged_layer = module._merge_weights()
                setattr(parent, child_name, merged_layer)
        
        return merged_model
    
    def save_model(self, path, merge_weights=False):
        """Save model weights to a file"""
        
        if merge_weights:
            # Create model with merged weights
            print("Merging weights for export...")
            merged_model = self._merge_weights()
            model_state_dict = _detach_cpu(merged_model.state_dict())
        else:
            # Save adapter weights only
            print("Saving adapter weights only...")
            model_state_dict = _detach_cpu(self.model.state_dict())
        
        # Save model to file
        save_file(model_state_dict, path)
        print(f"Model saved to {path}")


def _get_module_by_name(model, name):
    """Get a module from a model by its name"""
    for n, m in model.named_modules():
        if n == name:
            return m
    raise ValueError(f"Module {name} not found in model")


def _detach_cpu(state_dict):
    """Detach tensors and move them to CPU"""
    return {k: v.detach().cpu() if isinstance(v, torch.Tensor) else v 
            for k, v in state_dict.items()}


# Helper function to get paged optimizer if bitsandbytes is available
def get_paged_optimizer(model, lr=2e-5, weight_decay=0.0, **kwargs):
    """Create a paged AdamW optimizer that offloads states to CPU"""
    if not BITSANDBYTES_AVAILABLE:
        print("Warning: bitsandbytes not available, using standard AdamW")
        return torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr,
            weight_decay=weight_decay,
            **kwargs
        )
    
    return bnb.optim.PagedAdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=weight_decay,
        **kwargs
    ) 