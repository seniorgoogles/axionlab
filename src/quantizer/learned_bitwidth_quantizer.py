import torch 
from brevitas.quant import Int8WeightPerTensorFloat
from brevitas.core.scaling import ScalingImplType
from brevitas.core.restrict_val import RestrictValueType
from brevitas.core.bit_width import BitWidthImplType

class LearnedBitWidthQuantizer(Int8WeightPerTensorFloat):
    scaling_per_output_channel = True
    scaling_impl_type = ScalingImplType.PARAMETER_FROM_STATS
    restrict_scaling_type = RestrictValueType.LOG_FP
    bit_width_impl_type = BitWidthImplType.PARAMETER
    bit_width = 16