import brevitas
import torch.nn as nn
import brevitas.nn as qnn
from brevitas.quant import Int8WeightPerTensorFixedPoint

import src.quantizer as quant

class Mapper:

    @staticmethod 
    def get_quantizer_from_conf(input_dict, key):
        """
        @todo haesslich
        """
        value = None
        if key in input_dict: 
            #print(f"{input_dict} {type(input_dict)=}")
            value = input_dict.pop(key)
        
        return key, value

    @staticmethod
    def get_module(module_class):
        if hasattr(nn, module_class):
            return getattr(nn, module_class)
        elif hasattr(qnn, module_class):
            return getattr(qnn, module_class)
        else:
            raise Exception(f"{module_class} not found in nn or qnn module.")
        
    @staticmethod
    def get_quantizer(quantizer_class):
        if hasattr(brevitas.quant, quantizer_class):
            return getattr(brevitas.quant, quantizer_class)
        elif hasattr(quant, quantizer_class):
            return getattr(quant, quantizer_class)

    @staticmethod
    def has_key(yaml, key):
        try:
            return True if key in yaml else False
        except yaml.YAMLError as exc:
            print(exc)
            return False

    @staticmethod
    def map_module_as_attr(module, obj, module_name):
        setattr(obj, module_name, module)
    @staticmethod
    def map_config_as_attr(config, module, obj, attr_name):
        layer_config = dict(config)
        #{'in_channels': 1, 'out_channels': 4, 'kernel_size': 5, 'stride': 1, 'padding': 0, 'bias': False, 'output_bit_width': 4, 'weight_quant': 'Int8WeightPerTensorFloat'}
        _, quantizer_str = Mapper.get_quantizer_from_conf(layer_config, 'weight_quant')

        if quantizer_str != None:
            setattr(obj, attr_name, module(**layer_config, weight_quant=Mapper.get_quantizer(quantizer_str)))
        else: 
            setattr(obj, attr_name, module(**layer_config))

        """
        if issubclass(module, qnn.QuantConv2d):
            in_channels = config[0]
            out_channels = config[1]
            kernel_size = (config[2], config[2])
            stride = (config[3], config[3])
            padding = (config[4], config[4])
            bias = config[5]
            dilation = config[6]

            # If has quant config
            if len(config) > 7:
                quant_config = config[7]
                setattr(obj, attr_name,
                        module(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                               stride=stride, padding=padding, bias=bias, dilation=dilation,
                               weight_bit_width=quant_config["weight_bit_width"]))

            setattr(obj, attr_name, module(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                           stride=stride, padding=padding, bias=bias, dilation=dilation))
        elif issubclass(module, nn.Conv2d):
            in_channels = config[0]
            out_channels = config[1]
            kernel_size = (config[2], config[2])
            stride = (config[3], config[3])
            padding = (config[4], config[4])
            bias = config[5]
            dilation = config[6]
            setattr(obj, attr_name, module(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                           stride=stride, padding=padding, bias=bias, dilation=dilation))

        elif issubclass(module, nn.BatchNorm2d):
            num_features = config[0]
            eps = float(config[1])
            momentum = config[2]
            affine = config[3]
            track_running_stats = config[4]
            setattr(obj, attr_name, module(num_features=num_features, eps=eps, momentum=momentum, affine=affine,
                                           track_running_stats=track_running_stats))
        elif issubclass(module, qnn.QuantReLU):
            config = config[0]
            return_quant_tensor = config['return_quant_tensor']
            setattr(obj, attr_name, module(return_quant_tensor=return_quant_tensor))
        else:
            setattr(obj, attr_name, module(*config))
        """
    @staticmethod
    def map_config_to_module(config, module):
        if issubclass(module, qnn.QuantConv2d):
            in_channels = config[0]
            out_channels = config[1]
            kernel_size = (config[2], config[2])
            stride = (config[3], config[3])
            padding = (config[4], config[4])
            bias = config[5]
            dilation = config[6]

            # If has quant config
            if len(config) > 7:
                quant_config = config[7]
                return module(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding, bias=bias, dilation=dilation,
                              weight_bit_width=quant_config["weight_bit_width"])

            return module(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                          stride=stride, padding=padding, bias=bias, dilation=dilation)

        elif issubclass(module, nn.Conv2d):
            in_channels = config[0]
            out_channels = config[1]
            kernel_size = (config[2], config[2])
            stride = (config[3], config[3])
            padding = (config[4], config[4])
            bias = config[5]
            dilation = config[6]

            return module(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                          padding=padding, bias=bias, dilation=dilation)

        elif issubclass(module, nn.BatchNorm2d):
            num_features = config[0]
            eps = float(config[1])
            momentum = config[2]
            affine = config[3]
            track_running_stats = config[4]
            return module(num_features=num_features, eps=eps, momentum=momentum, affine=affine,
                          track_running_stats=track_running_stats)
        elif issubclass(module, qnn.QuantReLU):
            config = config[0]
            return_quant_tensor = config['return_quant_tensor']
            act_bit_width = config['act_bit_width']

            return module(return_quant_tensor=return_quant_tensor, act_bit_width=act_bit_width)
        else:
            return module(*config)