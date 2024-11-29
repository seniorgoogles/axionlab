from brevitas.core.quant import SparseRescalingIntQuant, SparseThresholdRescalingIntQuant, IntQuant
from brevitas.core.bit_width import BitWidthParameter
from brevitas.core.scaling import IntScaling, ConstScaling, ParameterScaling
from brevitas.core.zero_point import ZeroZeroPoint
from brevitas.core.function_wrapper import Identity

import torch
from colorama import Fore, Style
import numpy as np

# set seed for notebook
torch.manual_seed(0)

"""_summary_
        self,
        int_quant: Module,
        scaling_impl: Module,
        int_scaling_impl: Module,
        zero_point_impl: Module,
        bit_width_impl: Module,
        sparse_eps: float = 0.1,
        disable_quant: bool = True,
        disable_sparse: bool = True,
        prune_first: bool = True):
    """
    
quantile = 0.7
threshold = 0.7

bit_width = 8

input_view = Identity()
int_quant = IntQuant(narrow_range=True, signed=False, input_view_impl=input_view)
scaling_impl = ParameterScaling(1.0)
int_scaling_impl = IntScaling(signed=True, narrow_range=False)
zero_point_impl = ZeroZeroPoint()
bit_width_impl = BitWidthParameter(bit_width=bit_width)    



inp_tensor = torch.randn(6, 6)

prune_only = SparseRescalingIntQuant(int_quant=int_quant, 
                                    scaling_impl=scaling_impl, 
                                    int_scaling_impl=int_scaling_impl, 
                                    zero_point_impl=zero_point_impl, 
                                    bit_width_impl=bit_width_impl,
                                    quantile=quantile,
                                    disable_prune=False,
                                    disable_quant=True,
                                    prune_first=True)

thres_prune_only = SparseThresholdRescalingIntQuant(int_quant=int_quant, 
                                    scaling_impl=scaling_impl, 
                                    int_scaling_impl=int_scaling_impl, 
                                    zero_point_impl=zero_point_impl, 
                                    bit_width_impl=bit_width_impl,
                                    threshold=0.7,
                                    disable_prune=False,
                                    disable_quant=True,
                                    prune_first=True)

quant_only = SparseRescalingIntQuant(int_quant=int_quant, 
                                    scaling_impl=scaling_impl, 
                                    int_scaling_impl=int_scaling_impl, 
                                    zero_point_impl=zero_point_impl, 
                                    bit_width_impl=bit_width_impl,
                                    quantile=quantile,
                                    disable_prune=True,
                                    disable_quant=False,
                                    prune_first=True)

prune_quant = SparseRescalingIntQuant(int_quant=int_quant, 
                                    scaling_impl=scaling_impl, 
                                    int_scaling_impl=int_scaling_impl, 
                                    zero_point_impl=zero_point_impl, 
                                    bit_width_impl=bit_width_impl,
                                    quantile=quantile,
                                    disable_prune=False,
                                    disable_quant=False,
                                    prune_first=True)

quant_prune = SparseRescalingIntQuant(int_quant=int_quant, 
                                    scaling_impl=scaling_impl, 
                                    int_scaling_impl=int_scaling_impl, 
                                    zero_point_impl=zero_point_impl, 
                                    bit_width_impl=bit_width_impl,
                                    quantile=quantile,
                                    disable_prune=False,
                                    disable_quant=False,
                                    prune_first=False)


def print_tensor_differences(tensor1, tensor2, tol=1e-6):
    """
    Print differences between two tensors with color highlighting:
    - Green for unchanged values
    - Red for the old value if changed, Yellow for the new value if changed
    - Previous and new values displayed side by side for comparison.
    
    Parameters:
        tensor1: First tensor (old values).
        tensor2: Second tensor (new values).
        tol: Tolerance for considering two floats as equal.
    """
    # Ensure tensors are on the same device and have the same dtype
    tensor1 = tensor1.to(dtype=torch.float32, device='cpu')
    tensor2 = tensor2.to(dtype=torch.float32, device='cpu')
    
    # Convert tensors to numpy arrays
    tensor1_np = tensor1.numpy()
    tensor2_np = tensor2.numpy()
    
    #print(tensor1_np)
    #print(tensor2_np)

    # Debugging: Print shapes and values
    #print(f"Tensor1 shape: {tensor1_np.shape}, Tensor2 shape: {tensor2_np.shape}")
    #print("Tensor1 values:", tensor1_np)
    #print("Tensor2 values:", tensor2_np)
    
    
    # Ensure tensors are of the same shape
    if tensor1_np.shape != tensor2_np.shape:
        print("Tensors have different shapes and cannot be compared element-wise.")
        return
    

    # Debugging: Compute and log differences
    diff = np.abs(tensor1_np - tensor2_np)
    print("Absolute differences:", diff)

    # Determine the max width needed for alignment
    max_width = max(len(f"{val:.6f}") for row in tensor2_np for val in row) + 2

    # Iterate over tensor elements
    for i in range(tensor1_np.shape[0]):
        row_output = []
        for j in range(tensor1_np.shape[1]):
            old_value = tensor1_np[i, j]
            new_value = tensor2_np[i, j]
            close = np.isclose(old_value, new_value, atol=tol, rtol=0)

            if close:
                # Green color for unchanged values
                formatted = f"{Fore.GREEN}{f'{old_value:.6f}'.center(max_width)} | {f'{new_value:.6f}'.center(max_width)}{Style.RESET_ALL}"
            else:
                # Red for old value, Yellow for new value when changed
                formatted = f"{Fore.RED}{f'{old_value:.6f}'.center(max_width)}{Style.RESET_ALL} | {Fore.YELLOW}{f'{new_value:.6f}'.center(max_width)}{Style.RESET_ALL}"
                #print(f"Mismatch at ({i}, {j}): Old={old_value}, New={new_value}, Diff={abs(old_value - new_value)}")
            
            row_output.append(formatted)
        
        # Print each row
        print(" ".join(row_output))
    print("-----------------------------------")
    print(f"{tensor1_np[0][0]:.50f}")
    print(f"{tensor2_np[0][0]:.50f}")
        
out_prune = prune_only(inp_tensor)
out_thres_prune = thres_prune_only(inp_tensor)
out_quant = quant_only(inp_tensor)
prune_quant = prune_quant(inp_tensor)
quant_prune = quant_prune(inp_tensor)


print("Input")
print("-----------------------------------")
print_tensor_differences(inp_tensor, inp_tensor)
print("-----------------------------------")
print()

print("Prune Only")
print("-----------------------------------")
print_tensor_differences(inp_tensor, out_prune[0].detach())
print("-----------------------------------")
print()

print("Thres. Prune Only")
print("-----------------------------------")
print_tensor_differences(inp_tensor, out_thres_prune[0].detach())
print("-----------------------------------")
print()

"""
print(" Prune vs. Thres. Prune")
print("-----------------------------------")
print_tensor_differences(out_prune[0].detach(), out_thres_prune[0].detach())
print("-----------------------------------")
print()

print("Quant Only")
print("-----------------------------------")
print_tensor_differences(inp_tensor, out_quant[0].detach())
print("-----------------------------------")
print()

print("Quant and Prune")
print("-----------------------------------")
print_tensor_differences(inp_tensor, prune_quant[0].detach())
print("-----------------------------------")
print()

print("Prune and Quant")
print_tensor_differences(inp_tensor, quant_prune[0].detach())
print("-----------------------------------")
print()


print("Prune-Quant vs. Quant-Prune")
print("-----------------------------------")
print_tensor_differences(prune_quant[0].detach(), quant_prune[0].detach())
print("-----------------------------------")
print()
"""