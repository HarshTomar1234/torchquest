"""
LoRA Implementation that follows:
    - Paper: https://arxiv.org/pdf/2106.09685
    - Repo: https://github.com/microsoft/LoRA/

Also, implementation by michaelnny was super helpful!
    - https://github.com/michaelnny/QLoRA-LLM/
"""

"""
╔══════════════════════════════════════════════════════════════════╗
║                   LOW-RANK ADAPTATION (LoRA)                     ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  Key Idea: Replace full fine-tuning with low-rank updates        ║
║                                                                  ║
║  Original computation:   y = xW                                  ║
║  LoRA computation:       y = xW + x(AB)                          ║
║                            = x(W + AB)                           ║
║                                                                  ║
║  Where:                                                          ║
║    • W ∈ ℝᵈˣᵏ: Pre-trained weight matrix (frozen)                ║
║    • A ∈ ℝᵈˣʳ: Low-rank adaptation matrix                        ║
║    • B ∈ ℝʳˣᵏ: Low-rank adaptation matrix                        ║
║    • r ≪ min(d,k): Rank of the update (typically 8, 16, etc.)    ║
║                                                                  ║
║  Final weight update:                                            ║
║    W' = W + α·AB/r     (Standard LoRA)                           ║
║    W' = W + α·AB/√r    (Rank-Stabilized LoRA)                    ║
║                                                                  ║
║  Benefits:                                                       ║
║    • Significantly fewer trainable parameters                    ║
║    • Memory efficient fine-tuning                                ║
║    • Adaptations can be composed or switched                     ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
"""
import math
import random
from mpmath import norm
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple, Union, Literal
from safetensors.torch import save_file
from torch.nn.modules import padding

class LoRABaseLayer:

    def __init__(self,
                 rank = 8,
                 lora_alpha = 8,
                 lora_dropout =  0.0,
                 use_rslora = True):
        self.rank = rank
        self.lora_alpha = lora_alpha
        self.lora_dropout = nn.Dropout(lora_dropout) if lora_dropout > 0 else lambda x: x 

        self.use_rslora = use_rslora

        self.scaling = self.lora_alpha / self.rank**0.5 if self.use_rslora else self.lora_alpha / self.rank
        
    def _load_pretrained_weights(self, state_dict):
        self.weight.data = state_dict["weight"]
        if "bias" in state_dict:
            self.bias.data = state_dict["bias"]  


class LoRALinear(nn.Linear, LoRABaseLayer):

    def __init__(self,
                 in_features,
                 out_features,
                 bias = True,
                 rank = 8,
                 lora_alpha = 8,
                 lora_dropout = 0.0,
                 use_rslora = True,
                 **kwargs):


        nn.Linear.__init__(self, in_features, out_features, bias = bias, **kwargs)

        LoRABaseLayer.__init__(self,
                               rank = rank,
                               lora_alpha = lora_alpha,
                               lora_dropout = lora_dropout,
                               use_rslora = use_rslora)

        # disabling gradients for original weight matrix and bias matrix
        self.weight.requires_grad = False
        # self.bias.requires_grad = False

        self.lora_A = nn.Parameter(torch.zeros((in_features, rank)))
        self.lora_B = nn.Parameter(torch.zeros((rank, out_features)))

        nn.init.kaiming_uniform_(self.lora_A, a = math.sqrt(5))  # In original paper, they use Gaussian Distribution but in Microsoft released paper, they use Normal Distribution and in implementation they use kaiming uniform distribution
   
    def _merge_weights(self):

        """
        xW^T + xAB = x(W^T + AB)

        """
        
        merged_weights = self.weight.data + self.scaling * (self.lora_A @ self.lora_B).T  

        state_dict = {"weight": merged_weights}
        if self.bias is not None:
            state_dict["bias"] = self.bias

        merged_linear_layer = nn.Linear(self.in_features,
                                        self.out_features,
                                        bias = True if self.bias is not None else False)
        merged_linear_layer.load_state_dict(state_dict)

        return merged_linear_layer 

    
    def forward(self, x):

        original_layer_output = F.linear(x, self.weight, bias = self.bias)

        lora_mutliplier = (self.lora_A @ self.lora_B) * self.scaling
        # print(lora_mutliplier.shape)
        # print(self.weight.shape)

        lora_low_rank_output = self.lora_dropout(x) @ lora_mutliplier
        # print(lora_low_rank_output.shape)
        # print(original_layer_output.shape)

        output = original_layer_output + lora_low_rank_output

        return output


class LoRAEmbedding(nn.Embedding, LoRABaseLayer): 

    def __init__(self,
                 num_embeddings,
                 embedding_dim,
                 rank = 8,
                 lora_alpha = 8,
                 lora_dropout = 0.0,
                 use_rslora = True,
                 **kwargs):

        nn.Embedding.__init__(self, num_embeddings, embedding_dim, **kwargs)

        LoRABaseLayer.__init__(self,
                               rank = rank,
                               lora_alpha = lora_alpha,
                               lora_dropout = lora_dropout,
                               use_rslora = use_rslora)

        self.weight.requires_grad = False

        self.lora_A = nn.Parameter(torch.zeros((num_embeddings, rank)))
        self.lora_B = nn.Parameter(torch.zeros((rank, embedding_dim)))

        nn.init.kaiming_uniform_(self.lora_A, a = math.sqrt(5))

    def _merge_weights(self):
        """
        xW^T + xAB = x(W^T + AB)
        """

        merged_weights = self.weight.data + (self.lora_A @ self.lora_B) * self.scaling

        state_dict = {"weight": merged_weights}
        merged_embeddings = nn.Embedding(self.num_embeddings, self.embedding_dim)
        merged_embeddings.load_state_dict(state_dict)

        return merged_embeddings


    def forward(self, x):

        original_layer_output = F.embedding(input=x,
                                            weight = self.weight,
                                            padding_idx = self.padding_idx,
                                            max_norm = self.max_norm,
                                            norm_type = self.norm_type,
                                            scale_grad_by_freq = self.scale_grad_by_freq,
                                            sparse = self.sparse)

        lora_low_rank_A_output = F.embedding(input=x,
                                           weight = self.lora_A,
                                           padding_idx = self.padding_idx,
                                           max_norm = self.max_norm,
                                           norm_type = self.norm_type,
                                           scale_grad_by_freq = self.scale_grad_by_freq,
                                           sparse = self.sparse)

        lora_low_rank_output = (lora_low_rank_A_output @ self.lora_B )* self.scaling  # dropout is not applied here because x is basically tokens (raw outputs from tokenizer)

        output = original_layer_output + lora_low_rank_output      

        return output                            

                                           
class LoRAConv2d(nn.Conv2d, LoRABaseLayer):

    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 kernel_size, 
                 stride=1, 
                 padding=0, 
                 bias=True,
                 rank=8, 
                 lora_alpha=8, 
                 lora_dropout=0.0,
                 use_rslora=True,
                 **kwargs):
        
       
        nn.Conv2d.__init__(self,
                           in_channels=in_channels, 
                           out_channels=out_channels, 
                           kernel_size=kernel_size, 
                           stride=stride, 
                           padding=padding, 
                           bias=bias, 
                           **kwargs)


        LoRABaseLayer.__init__(self,
                               rank=rank, 
                               lora_alpha=lora_alpha, 
                               lora_dropout=lora_dropout, 
                               use_rslora=use_rslora)


        self.weight.requires_grad = False


        self.lora_A = nn.Parameter(torch.zeros(rank, self.in_channels, *self.kernel_size))   # (out_channels, in_channels, kernel_size)
        self.lora_B = nn.Parameter(torch.zeros(rank,self.out_channels))   # (rank,out_channels kernel_size) using linear layer
        # self.lora_B = nn.Parameter(torch.zeros(self.out_channels,rank, 1, 1)) using kernel size 1x1
        
        nn.init.kaiming_uniform_(self.lora_A, a = math.sqrt(5))

    def _merge_weights(self):
        """
        xW^T + xAB = x(W^T + AB)
        """

        """
        Intuition: 
                
                A --> (8, 256, 4, 4)
                B --> (8, 384)

                B^T --> (384, 8)
                A --> (8, 256 * 4 * 4) Flattening so that we can multiply with B^T
                B^T x A --> (384, 8) X (8, 256 * 4 * 4) = (384, 256 * 4 * 4)
                
                and then Unflattening it to get the original shape of the weight matrix
                B^T x A --> (384, 256 * 4 * 4) --> (384, 256, 4, 4)
                  


        """
        # (rank x in_channels x k_h, k_w) -> (rank x in_channels* k_h*k_w)     
        lora_A_flatten = self.lora_A.flatten(1)

        # Matmul with lora_B Transposed (lora_B is rank x out_channels) -> (out_channels x rank) 
        lora_multiplier = (self.lora_B.T @ lora_A_flatten ) * self.scaling

        # placing back into Conv Weight Shape: (out_channels x in_channels*k_h*k_w) -> (out_channels x in_channels x k_h x k_w) 
        lora_multiplier = lora_multiplier.reshape(self.out_channels, self.in_channels, *self.kernel_size)

        merged_weights = self.weight.data + lora_multiplier

        state_dict = {"weight": merged_weights}
        if self.bias is not None:
            state_dict["bias"] = self.bias

        merged_conv_layer = nn.Conv2d(self.in_channels,
                                      self.out_channels,
                                      kernel_size=self.kernel_size,
                                      stride=self.stride,
                                      padding=self.padding,
                                      bias=True if self.bias is not None else False)
        merged_conv_layer.load_state_dict(state_dict)

        return merged_conv_layer


    def forward(self, x):

        original_layer_output = F.conv2d(x,
                                         weight = self.weight,
                                         bias = self.bias,
                                         stride = self.stride,
                                         padding = self.padding)

        lora_low_rank_A_output = F.conv2d(x,
                                         weight = self.lora_A,
                                         bias = None,
                                         stride = self.stride,
                                         padding = self.padding)  # in lora adaptation, we basically changing output channels and keeping stride and padding same

        """ 
        Intuition:
             256 channels ---> 4 channels(rank) ---> 256 channels

             B x 256 x H x W ---> B X 4 x H/2 X W/2 (keeping stride as 2x2 ) ----> B x H/2 x W/2 x 4 (permuting dimensions) * (4 x 256) (lora_B dimensions)----> B x H/2 x W/2 x 256 ---> B x 256 x H/2 x W/2 (permuting dimensions)

                                              OR

             B x 256 x H x W ---> B X 4 x H/2 X W/2 (keeping stride as 2x2 ) (now using 1x1 conv layer) ----> B x 256 x H/2 x W/2                                
         
        """                                         

        lora_low_rank_A_output = lora_low_rank_A_output.permute(0,2,3,1)

        lora_low_rank_output = self.lora_dropout(lora_low_rank_A_output) @ self.lora_B * self.scaling

        lora_low_rank_output = lora_low_rank_output.permute(0,3,1,2)


        output = original_layer_output + lora_low_rank_output

        return output

   

        

@dataclass
class LoRAConfig:
    
    rank: int = 8
    target_modules: Optional[Union[list[str], str]] = None
    exclude_modules: Optional[Union[list[str], str]] = None
    lora_alpha: float = 8.0
    lora_dropout: float = 0.0
    bias: Literal["none", "all", "lora_only"] = "none"
    use_rslora: bool = True

class LoRAModel(nn.Module):

    def __init__(self, model, confIg):

        super(LoRAModel, self).__init__()
        self.lora_model = model
        self.config = config

        if self.config.target_modules is None:
            self.config.target_modules = []
        elif isinstance(self.config.target_modules, str):
            self.config.target_modules = [self.config.target_modules]

        if self.config.exclude_modules is None:
            self.config.exclude_modules = []
        elif isinstance(self.config.exclude_modules, str):
            self.config.exclude_modules = [self.config.exclude_modules]


        original_trainable_params =  self._compute_trainable_parameters()
                
        self.disable_all_grads()
        # print(f"Total trainable parameters: {original_trainable_params}")
        # print(f"Total trainable parameters after shutting gradients: {self._compute_trainable_parameters()}")

        self._apply_lora(self.lora_model)

        self._toggle_bias_grad()

        lora_trainable_params = self._compute_trainable_parameters()


        param_stats = {
            "Initial Parameters": f"{original_trainable_params:,}",
            "LoRA Parameters": f"{lora_trainable_params:,}",
            "Trainable Proportion": f"{round(lora_trainable_params*100/original_trainable_params, 2)}%"
        }
        
        
        print("\n" + "="*50)
        print(f"{'PARAMETER STATISTICS':^50}")
        print("="*50)
        for key, value in param_stats.items():
            print(f"{key:.<30} {value:.>20}")
        print("="*50 + "\n")

    def forward(self, *inputs, **kwargs):
        return self.lora_model(*inputs, **kwargs)     


    def _exclude_module_name_check(self, name):
        return any([ex in name for ex in self.config.exclude_modules])  

    def _target_module_name_check(self, name):
        return any([tgt in name for tgt in self.config.target_modules])    
     
    def _apply_lora(self, module):

        """
        Method to recursively replace all the layers in a model with LoraLayers
        """

        for name, child in module.named_children():

            if self._target_module_name_check(name):
                if isinstance(child, nn.Linear):
                    new_layer = LoRALinear(in_features = child.in_features,
                                           out_features = child.out_features,
                                           rank = self.config.rank,
                                           bias = True if child.bias is not None else False,
                                           lora_alpha = self.config.lora_alpha,
                                           lora_dropout = self.config.lora_dropout,
                                           use_rslora = self.config.use_rslora)

                    new_layer._load_pretrained_weights(child.state_dict())
                    setattr(module, name, new_layer)                       
                    

                elif isinstance(child, nn.Embedding): 
                    new_layer = LoRAEmbedding(num_embeddings = child.num_embeddings,
                                              embedding_dim = child.embedding_dim,
                                              rank = self.config.rank,
                                              lora_alpha = self.config.lora_alpha,
                                              lora_dropout = self.config.lora_dropout,
                                              use_rslora = self.config.use_rslora)

                    new_layer._load_pretrained_weights(child.state_dict())
                    setattr(module, name, new_layer)   

                elif isinstance(child, nn.Conv2d):
                    new_layer = LoRAConv2d(in_channels = child.in_channels,
                                           out_channels = child.out_channels,
                                           kernel_size = child.kernel_size,
                                           stride = child.stride,
                                           padding = child.padding,
                                           bias = True if child.bias is not None else False,
                                           rank = self.config.rank,
                                           lora_alpha = self.config.lora_alpha,
                                           lora_dropout = self.config.lora_dropout,
                                           use_rslora = self.config.use_rslora)

                    new_layer._load_pretrained_weights(child.state_dict())
                    setattr(module, name, new_layer)                                                 
                   

            if (len(list(child.children())) > 0) and not self._exclude_module_name_check(name):
                self._apply_lora(child)        


    def _compute_trainable_parameters(self):

        total_learnable_params = 0
        for param in self.lora_model.parameters():
            if param.requires_grad:
                total_learnable_params += param.numel()

        return total_learnable_params

    def disable_all_grads(self):
        for name, param in self.lora_model.named_parameters():
            if not self._exclude_module_name_check(name):
                param.requires_grad = False

    def _toggle_bias_grad(self):
        for name, param in self.lora_model.named_parameters():

            if not self._exclude_module_name_check(name):
                if "bias" in name:
            
                    if self.config.bias == "none":
                        param.requires_grad = False

                    elif self.config.bias == "all":
                        param.requires_grad = True

                    elif self.config.bias == "lora_only" and (self._target_module_name_check(name)):
                        param.requires_grad = True

    def _merge_weights(self, module):
        """
        Recursively trigger weight merging and replace in model 
        """

        for name, child in module.named_children():

            if isinstance(child, (LoRALinear, LoRAEmbedding, LoRAConv2d)):
                 
                 # merging the layer 
                 merged_layer = child._merge_weights()

                 # replacing LoRA Layer with Merged layer 
                 setattr(module, name, merged_layer)

            else:

                if len(list(child.children())) > 0:
                    self._merge_weights(child) 

                   
    def save_model(self, path, merge_weights = False):
        """
        Method to save the model with LoRA weights
        """    

        """
        Method to save model safetensors to the given path
            - merge_weights -> True: Merge LoRA weights and save
            - merge_weights -> False: Only save trainable weights
        """    

        def _detach_cpu(param):
            # Helper function to move tensors to CPU and detach from computation graph
            # This ensures tensors can be saved without GPU memory dependencies
            return param.detach().cpu()

        # Two saving strategies based on the merge_weights flag:
        if merge_weights: 
            # Option 1: Merge LoRA weights with base model weights
            # This creates a standalone model that doesn't require LoRA architecture
            self._merge_weights(self.lora_model)

            # Save all parameters (both original and merged LoRA weights)
            # Useful when you want to save the entire model, including the base weights and the adaptations
            # Removes the "lora_model." prefix from parameter names for cleaner state dict keys
            state_dict = {
                name.replace("lora_model.", ""): _detach_cpu(param) for name, param in self.named_parameters() 
            }

        else:
            # Option 2: Save only the trainable parameters (LoRA weights)
            # This is much smaller in size as it only contains the low-rank matrices
            # Useful when you want to distribute just the adaptation weights
            state_dict = {
                name.replace("lora_model.", ""): _detach_cpu(param) for name, param in self.lora_model.named_parameters() if (param.requires_grad)
            }

        save_file(state_dict, path)
    

if __name__ == "__main__":
    
   from transformers import AutoModelForSequenceClassification
   target_modules = ["query", "key", "value", "dense", "word_embeddings"]
   exclude_modules = ["classifier"]

   config = LoRAConfig(target_modules = target_modules,
                       exclude_modules = exclude_modules,
                       bias = "lora_only")

   model = AutoModelForSequenceClassification.from_pretrained("FacebookAI/roberta-base")

  
   lora_model = LoRAModel(model, config)
   lora_model.save_model("path", merge_weights = True)
#  print(lora_model)


