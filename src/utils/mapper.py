import brevitas
import torch.nn as nn
import brevitas.nn as qnn
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
    def get_layer_by_name(layer_name):
        if hasattr(nn, layer_name):
            return getattr(nn, layer_name)
        elif hasattr(qnn, layer_name):
            return getattr(qnn, layer_name)
        else:
            raise Exception(f"{layer_name} not found in nn or qnn module.")
        
    @staticmethod
    def get_quantizer(quantizer_class):
        if hasattr(brevitas.quant, quantizer_class):
            return getattr(brevitas.quant, quantizer_class)
        # Fixed point quantizer
        elif hasattr(brevitas.quant.fixed_point, quantizer_class):
            return getattr(brevitas.quant.fixed_point, quantizer_class)
        # Own quantizer
        elif hasattr(quant, quantizer_class):
            return getattr(quant, quantizer_class)
        else:
            raise Exception(f"{quantizer_class} not found in brevitas.quant or quant module.")

    @staticmethod
    def has_key(yaml, key):
        try:
            return True if key in yaml else False
        except yaml.YAMLError as exc:
            print(exc)
            return False

    @staticmethod
    def add_model_to_object(model, obj, model_name):
        #@todo change order of inputs
        setattr(obj, model_name, model)
    @staticmethod
    def map_config_as_attr(config, module, obj, attr_name):
        layer_config = dict(config)
        #{'in_channels': 1, 'out_channels': 4, 'kernel_size': 5, 'stride': 1, 'padding': 0, 'bias': False, 'output_bit_width': 4, 'weight_quant': 'Int8WeightPerTensorFloat'}
        _, quantizer_str = Mapper.get_quantizer_from_conf(layer_config, 'weight_quant')
        _, out_quantizer_str = Mapper.get_quantizer_from_conf(layer_config, 'output_quant')

        if quantizer_str != None and out_quantizer_str != None:
            setattr(obj, attr_name, module(**layer_config, weight_quant=Mapper.get_quantizer(quantizer_str), output_quant=Mapper.get_quantizer(out_quantizer_str)))
        elif quantizer_str != None:
            setattr(obj, attr_name, module(**layer_config, weight_quant=Mapper.get_quantizer(quantizer_str)))
        else: 
            setattr(obj, attr_name, module(**layer_config))