from brevitas.core.quant import SparseRescalingIntQuant, IntQuant
from brevitas.core.bit_width import BitWidthParameter
from brevitas.core.scaling import IntScaling, ConstScaling, ParameterScaling
from brevitas.core.zero_point import ZeroZeroPoint
import torch
from colorama import Fore, Style

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

int_quant = IntQuant(narrow_range=True, signed=False)
scaling_impl = ParameterScaling(1.0)
int_scaling_impl = IntScaling(signed=True, narrow_range=False)
zero_point_impl = ZeroZeroPoint()
bit_width_impl = BitWidthParameter(bit_width=16)    


inp_tensor = torch.randn(6, 6)

prune_only = SparseRescalingIntQuant(int_quant=int_quant, 
                                              scaling_impl=scaling_impl, 
                                              int_scaling_impl=int_scaling_impl, 
                                              zero_point_impl=zero_point_impl, 
                                              bit_width_impl=bit_width_impl,
                                              sparse_eps=0.3,
                                              disable_sparse=False,
                                              disable_quant=True,
                                              prune_first=True)

quant_only = SparseRescalingIntQuant(int_quant=int_quant, 
                                              scaling_impl=scaling_impl, 
                                              int_scaling_impl=int_scaling_impl, 
                                              zero_point_impl=zero_point_impl, 
                                              bit_width_impl=bit_width_impl,
                                              sparse_eps=0.3,
                                              disable_sparse=False,
                                              disable_quant=True,
                                              prune_first=True)

prune_quant = SparseRescalingIntQuant(int_quant=int_quant, 
                                              scaling_impl=scaling_impl, 
                                              int_scaling_impl=int_scaling_impl, 
                                              zero_point_impl=zero_point_impl, 
                                              bit_width_impl=bit_width_impl,
                                              sparse_eps=0.3,
                                              disable_sparse=False,
                                              disable_quant=False,
                                              prune_first=True)
quant_prune = SparseRescalingIntQuant(int_quant=int_quant, 
                                              scaling_impl=scaling_impl, 
                                              int_scaling_impl=int_scaling_impl, 
                                              zero_point_impl=zero_point_impl, 
                                              bit_width_impl=bit_width_impl,
                                              sparse_eps=0.3,
                                              disable_sparse=False,
                                              disable_quant=False,
                                              prune_first=False)



def print_tensor_differences(tensor1, tensor2):
    """
    Print differences between two tensors with color highlighting:
    - Red for the old value if changed, Yellow for the new value if changed
    - Green for unchanged values
    - Previous and new values displayed side by side for comparison
    """
    # Convert tensors to numpy arrays for element-wise comparison
    tensor1_np = tensor1.numpy()
    tensor2_np = tensor2.numpy()
    
    # Ensure tensors are of the same shape
    if tensor1_np.shape != tensor2_np.shape:
        print("Tensors have different shapes and cannot be compared element-wise.")
        return
    
    # Determine the max width needed for alignment, accounting for two values side by side
    max_width = max(len(f"{val:.4f}") for row in tensor2_np for val in row) + 2  # extra space for minus sign if needed
    
    # Iterate over tensor elements
    for i in range(tensor1_np.shape[0]):
        row_output = []
        for j in range(tensor1_np.shape[1]):
            old_value = f"{tensor1_np[i, j]:.4f}"
            new_value = f"{tensor2_np[i, j]:.4f}"
            if tensor1_np[i, j] == tensor2_np[i, j]:
                # Green color for unchanged values
                formatted = f"{Fore.GREEN}{old_value.center(max_width)} | {new_value.center(max_width)}{Style.RESET_ALL}"
            else:
                # Red for old value, Yellow for new value when changed
                formatted = f"{Fore.RED}{old_value.center(max_width)}{Style.RESET_ALL} | {Fore.YELLOW}{new_value.center(max_width)}{Style.RESET_ALL}"
            row_output.append(formatted)
        
        # Print each row
        print(" ".join(row_output))

        
out_prune = prune_only(inp_tensor)
out_quant = quant_only(inp_tensor)
prune_quant = prune_quant(inp_tensor)
quant_prune = quant_prune(inp_tensor)

print("Prune Only")
print("-----------------------------------")
print_tensor_differences(inp_tensor, out_prune[0].detach())
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